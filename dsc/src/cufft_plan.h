// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// Batched FFT plans over cuFFT for the CUDA plugin. cuFFT is NOT linked: it is an optional library the plugin
// opens by dlopen on the first plan (so an install without it still loads the plugin), and every symbol is
// declared here (cuFFT's C ABI is stable across majors). A plan is built EAGERLY, never inside a stream capture
// (cuFFT allocates its work area at plan time), and executes on any stream, capture included (no allocation and
// no synchronisation at execution).
//
// Layout contract: dense row-major arrays of interleaved (re, im) pairs (float / double, trailing axis 2, which
// no extent here counts). `shape` is the COMPLEX shape of a complex transform; for a real transform it is the
// REAL side's shape (input of r2c, output of c2r) and the complex side has `shape[axes.back()] / 2 + 1` along
// the halved axis. Normalisation is numpy's `norm`, applied by a small PTX kernel loaded by the CUDA runtime.
#pragma once
#include <dlfcn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "aether/chunk/Chunk.h"

namespace aether_plugin {

enum FftKind { kC2C = 0, kR2C = 1, kC2R = 2 };
enum FftNorm { kBackward = 0, kForward = 1, kOrtho = 2 };

/// thrown when the cuFFT library cannot be opened: the seam reports it as "unavailable", not as a failure
struct CufftMissing : std::runtime_error { using std::runtime_error::runtime_error; };

namespace cufft {

using Handle = int;
enum { kForwardDir = -1, kInverseDir = 1 };
enum { kTypeR2C = 0x2a, kTypeC2R = 0x2c, kTypeC2C = 0x29, kTypeD2Z = 0x6a, kTypeZ2D = 0x6c, kTypeZ2Z = 0x69 };

struct Lib {
    int (*PlanMany)(Handle*, int, int*, int*, int, int, int*, int, int, int, int);
    int (*ExecC2C)(Handle, void*, void*, int);
    int (*ExecZ2Z)(Handle, void*, void*, int);
    int (*ExecR2C)(Handle, void*, void*);
    int (*ExecD2Z)(Handle, void*, void*);
    int (*ExecC2R)(Handle, void*, void*);
    int (*ExecZ2D)(Handle, void*, void*);
    int (*SetStream)(Handle, cudaStream_t);
    int (*Destroy)(Handle);
};

inline std::string moduleDir() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&moduleDir), &info) || !info.dli_fname) return ".";
    std::string p = info.dli_fname;
    auto cut = p.find_last_of('/');
    return cut == std::string::npos ? "." : p.substr(0, cut);
}

// cuFFT 12 ships with CUDA 13, cuFFT 11 with CUDA 12; the C ABI used here is the same, so whichever is installed
inline constexpr const char* kSonames[] = {"libcufft.so.12", "libcufft.so.11"};

/// The library, opened once. `AETHER_CUFFT_LIB` names one file explicitly (and is the only place looked at);
/// otherwise the soname through the loader, then the pip layout next to this plugin (`site-packages/nvidia/...`).
inline const Lib& lib() {
    static std::once_flag once;
    static Lib l{};
    static std::string failure;
    std::call_once(once, [] {
        std::vector<std::string> tries;
        if (const char* forced = std::getenv("AETHER_CUFFT_LIB")) tries = {forced};
        else {
            std::string dir = moduleDir() + "/../nvidia/";
            for (const char* so : kSonames) tries.push_back(so);
            for (const char* so : kSonames) tries.push_back(dir + "cu13/lib/" + so);
            for (const char* so : kSonames) tries.push_back(dir + "cufft/lib/" + so);
        }
        void* h = nullptr;
        std::string why;
        for (const auto& t : tries) {
            h = dlopen(t.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (h) break;
            const char* e = dlerror();
            why += "\n  " + t + ": " + (e ? e : "not loadable");
        }
        if (!h) { failure = "cuFFT is not installed (libcufft.so.12 / .11 not found)" + why; return; }
        auto sym = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, name));
            return fn != nullptr;
        };
        bool ok = sym(l.PlanMany, "cufftPlanMany") && sym(l.ExecC2C, "cufftExecC2C") && sym(l.ExecZ2Z, "cufftExecZ2Z") &&
                  sym(l.ExecR2C, "cufftExecR2C") && sym(l.ExecD2Z, "cufftExecD2Z") && sym(l.ExecC2R, "cufftExecC2R") &&
                  sym(l.ExecZ2D, "cufftExecZ2D") && sym(l.SetStream, "cufftSetStream") && sym(l.Destroy, "cufftDestroy");
        if (!ok) { failure = "the cuFFT library that was opened lacks the expected symbols"; l = Lib{}; }
    });
    if (!l.PlanMany)
        throw CufftMissing(failure + "\nGPU FFTs need cuFFT, an optional install: pip install 'aether-dsc[cufft12]' (CUDA 12) or 'aether-dsc[cufft13]' (CUDA 13)");
    return l;
}

}  // namespace cufft

// ---------------------------------------------------------------------------------------------- scale kernel
// y *= s over n reals (the numpy `norm` factor). Grid-stride, 64-bit counts.
template <class T> __global__ void scaleKernel(T* p, T s, std::uint64_t n) {
    const std::uint64_t stride = std::uint64_t(gridDim.x) * blockDim.x;
    for (std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) p[i] *= s;
}
inline void launchScale(bool dbl, void* data, double s, std::uint64_t n, cudaStream_t stream) {
    const unsigned blocks = unsigned(std::min<std::uint64_t>((n + 255) / 256, 1u << 20));
    if (dbl) scaleKernel<double><<<blocks, 256, 0, stream>>>(static_cast<double*>(data), s, n);
    else scaleKernel<float><<<blocks, 256, 0, stream>>>(static_cast<float*>(data), float(s), n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("fft scaling launch: ") + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------------------------- plans
class FftPlan {
public:
    /// Builds every cuFFT plan and the work area now. Refused while `stream` is capturing.
    FftPlan(int dev, int kind, bool dbl, int norm, bool forward, std::vector<std::int64_t> shape,
            std::vector<std::int32_t> axes, cudaStream_t stream)
        : dev_(dev), kind_(kind), dbl_(dbl), forward_(kind == kR2C ? true : kind == kC2R ? false : forward),
          shape_(std::move(shape)) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
            throw std::runtime_error("fft plan creation is not allowed inside a stream capture: build the plan before the capture begins");
        cudaGetLastError();
        if (axes.empty()) throw std::runtime_error("fft: no axes to transform");
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (axes[i] < 0 || axes[i] >= int(shape_.size())) throw std::runtime_error("fft: axis out of range");
            for (std::size_t j = 0; j < i; ++j) if (axes[j] == axes[i]) throw std::runtime_error("fft: repeated axis");
        }
        axes_ = std::move(axes);
        const auto& L = cufft::lib();   // throws CufftMissing
        try {

        OnDevice device(dev_);
        // plan creation may allocate: allow it even if another thread is capturing (it never joins that capture)
        RelaxedCapture relaxed;
        elem_ = (dbl_ ? 8 : 4);                                   // bytes per real
        std::int64_t points = 1;
        for (int a : axes_) points *= shape_[a];
        double s = 1.0;
        bool fwd = forward_;
        if (norm == kOrtho) s = 1.0 / std::sqrt(double(points));
        else if (fwd == (norm == kForward)) s = 1.0 / double(points);
        scale_ = s;

        std::vector<std::int64_t> half = shape_;
        if (kind_ != kC2C) half[axes_.back()] = shape_[axes_.back()] / 2 + 1;
        total_real_ = 1;
        for (auto v : shape_) total_real_ *= v;
        total_half_ = 1;
        for (auto v : half) total_half_ *= v;
        empty_ = total_real_ == 0;
        if (empty_) return;

        const std::size_t nc2c = (kind_ == kC2C) ? axes_.size() : axes_.size() - 1;
        // the c2c steps run on the complex-side shape
        const std::vector<std::int64_t>& cshape = kind_ == kC2C ? shape_ : half;
        for (std::size_t i = 0; i < nc2c; ++i) c2c_.push_back(makeStep(L, axes_[i], cshape, cufft::kTypeC2C + (dbl_ ? 0x40 : 0)));
        if (kind_ != kC2C) real_ = makeRealStep(L);
        if (kind_ == kC2R) {   // c2r destroys its input: work on a copy. Owned by the plan, so not tied to the caller's stream
            work_ = aether::Chunk::allocate(aether::Device(kDLCUDA, dev_), std::size_t(total_half_) * 2 * elem_, nullptr);
            cudaStreamSynchronize(nullptr);
        }
        } catch (...) {
            release();
            throw;
        }
    }

    ~FftPlan() { release(); }
    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;

    /// Runs on `stream`: no allocation, no synchronisation, legal inside a capture. `in` and `out` are device
    /// pointers; `in == out` is allowed for c2c only.
    void execute(const void* in, void* out, cudaStream_t stream) {
        if (empty_) return;
        const auto& L = cufft::lib();
        std::lock_guard<std::mutex> lock(mu_);
        OnDevice device(dev_);
        auto* src = const_cast<char*>(static_cast<const char*>(in));
        auto* dst = static_cast<char*>(out);
        auto check = [](int r, const char* what) {
            if (r != 0) throw std::runtime_error(std::string("cuFFT ") + what + " failed with code " + std::to_string(r));
        };
        const std::size_t cbytes = 2 * elem_;   // one complex element
        if (kind_ == kC2R) {
            auto* w = reinterpret_cast<char*>(work_.data());
            cudaError_t e = cudaMemcpyAsync(w, src, std::size_t(total_half_) * cbytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) throw std::runtime_error(std::string("fft input copy: ") + cudaGetErrorString(e));
            for (auto& st : c2c_) runC2c(L, st, w, w, stream, check);
            check(L.SetStream(real_.handle, stream), "set stream");
            for (std::int64_t o = 0; o < real_.loops; ++o) {
                void* a = w + o * real_.inLoopBytes;
                void* b = dst + o * real_.outLoopBytes;
                check(dbl_ ? L.ExecZ2D(real_.handle, a, b) : L.ExecC2R(real_.handle, a, b), "c2r");
            }
        } else if (kind_ == kR2C) {
            check(L.SetStream(real_.handle, stream), "set stream");
            for (std::int64_t o = 0; o < real_.loops; ++o) {
                void* a = src + o * real_.inLoopBytes;
                void* b = dst + o * real_.outLoopBytes;
                check(dbl_ ? L.ExecD2Z(real_.handle, a, b) : L.ExecR2C(real_.handle, a, b), "r2c");
            }
            for (auto& st : c2c_) runC2c(L, st, dst, dst, stream, check);
        } else {
            char* from = src;
            for (auto& st : c2c_) { runC2c(L, st, from, dst, stream, check); from = dst; }
        }
        if (scale_ != 1.0) {
            std::uint64_t n = std::uint64_t(kind_ == kC2R ? total_real_ : (kind_ == kR2C ? total_half_ : total_real_) * 2);
            launchScale(dbl_, dst, scale_, n, stream);
        }
    }

private:
    void release() {
        if (!cufftReady_) return;
        RelaxedCapture relaxed;
        const auto& L = cufft::lib();
        for (auto& st : c2c_) L.Destroy(st.handle);
        if (real_.handle) L.Destroy(real_.handle);
        c2c_.clear();
        real_ = Step{};
        cufftReady_ = false;
    }

    struct Step {
        cufft::Handle handle = 0;
        std::int64_t loops = 1;                 // exec repetitions (outer blocks when both sides of the axis are > 1)
        std::int64_t inLoopBytes = 0, outLoopBytes = 0;
    };
    struct OnDevice {
        int prev = 0;
        bool switched = false;
        explicit OnDevice(int dev) {
            cudaGetDevice(&prev);
            switched = prev != dev;
            if (switched) cudaSetDevice(dev);
        }
        ~OnDevice() { if (switched) cudaSetDevice(prev); }
    };
    struct RelaxedCapture {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        RelaxedCapture() { cudaThreadExchangeStreamCaptureMode(&mode); }
        ~RelaxedCapture() { cudaThreadExchangeStreamCaptureMode(&mode); }
    };

    template <class Check> void runC2c(const cufft::Lib& L, Step& st, void* in, void* out, cudaStream_t stream, Check& check) {
        check(L.SetStream(st.handle, stream), "set stream");
        for (std::int64_t o = 0; o < st.loops; ++o) {
            auto* a = static_cast<char*>(in) + o * st.inLoopBytes;
            auto* b = static_cast<char*>(out) + o * st.outLoopBytes;
            int dir = forward_ ? cufft::kForwardDir : cufft::kInverseDir;
            check(dbl_ ? L.ExecZ2Z(st.handle, a, b, dir) : L.ExecC2C(st.handle, a, b, dir), "c2c");
        }
    }

    void plan(const cufft::Lib& L, Step& st, int n, int inner, std::int64_t outer, int inDist, int outDist, int inEmbed,
              int outEmbed, int inStride, int outStride, int type, int batch) {
        int nn = n, ie = inEmbed, oe = outEmbed;
        cufft::Handle h = 0;
        int r = L.PlanMany(&h, 1, &nn, &ie, inStride, inDist, &oe, outStride, outDist, type, batch);
        if (r != 0) throw std::runtime_error("cuFFT plan creation failed with code " + std::to_string(r) + " (length " + std::to_string(n) + ")");
        st.handle = h;
        cufftReady_ = true;
        (void)inner; (void)outer;
    }

    Step makeStep(const cufft::Lib& L, int axis, const std::vector<std::int64_t>& cshape, int type) {
        std::int64_t n = cshape[axis], inner = 1, outer = 1;
        for (std::size_t k = axis + 1; k < cshape.size(); ++k) inner *= cshape[k];
        for (int k = 0; k < axis; ++k) outer *= cshape[k];
        Step st;
        const std::int64_t cbytes = std::int64_t(2 * elem_);
        if (inner == 1) {   // contiguous transforms, batched over `outer`
            plan(L, st, int(n), 1, outer, int(n), int(n), int(n), int(n), 1, 1, type, int(outer));
        } else {            // strided by `inner`; one exec per outer block
            plan(L, st, int(n), int(inner), outer, 1, 1, int(n * inner), int(n * inner), int(inner), int(inner), type, int(inner));
            st.loops = outer;
            st.inLoopBytes = st.outLoopBytes = n * inner * cbytes;
        }
        return st;
    }

    // the real axis: r2c reads reals shaped `shape_` and writes pairs shaped `half`; c2r the reverse
    Step makeRealStep(const cufft::Lib& L) {
        int axis = axes_.back();
        std::int64_t n = shape_[axis], h = n / 2 + 1, inner = 1, outer = 1;
        for (std::size_t k = axis + 1; k < shape_.size(); ++k) inner *= shape_[k];
        for (int k = 0; k < axis; ++k) outer *= shape_[k];
        Step st;
        const std::int64_t rbytes = std::int64_t(elem_), cbytes = 2 * rbytes;
        bool r2c = kind_ == kR2C;
        int type = r2c ? (dbl_ ? cufft::kTypeD2Z : cufft::kTypeR2C) : (dbl_ ? cufft::kTypeZ2D : cufft::kTypeC2R);
        int realDist = int(n), halfDist = int(h);
        if (inner == 1) {
            plan(L, st, int(n), 1, outer, r2c ? realDist : halfDist, r2c ? halfDist : realDist, r2c ? int(n) : int(h),
                 r2c ? int(h) : int(n), 1, 1, type, int(outer));
        } else {
            plan(L, st, int(n), int(inner), outer, 1, 1, int((r2c ? n : h) * inner), int((r2c ? h : n) * inner), int(inner),
                 int(inner), type, int(inner));
            st.loops = outer;
            st.inLoopBytes = (r2c ? n * rbytes : h * cbytes) * inner;
            st.outLoopBytes = (r2c ? h * cbytes : n * rbytes) * inner;
        }
        return st;
    }

    int dev_, kind_;
    bool dbl_, forward_;
    std::vector<std::int64_t> shape_;
    std::vector<std::int32_t> axes_;
    std::size_t elem_ = 4;
    double scale_ = 1.0;
    std::int64_t total_real_ = 0, total_half_ = 0;
    bool empty_ = false, cufftReady_ = false;
    std::vector<Step> c2c_;
    Step real_;
    aether::Chunk work_;
    std::mutex mu_;
};

}  // namespace aether_plugin
