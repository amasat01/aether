// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// The CUDA plugin: this TU, compiled by nvcc with `-cudart none` over aether's CUDA-mode headers, exports exactly
// `aether_memory_v1`, a thin C wrapper over aether's `Chunk` / `fill` / `StreamTransport` (the memory rules live in
// those headers) and the batched FFT plans. It needs `libcuda.so.1` and no CUDA runtime library: the runtime
// functions the headers call are defined by DriverRuntime.inl over the driver API. One plugin serves CUDA 12 and 13.
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Fill.h"
#include "aether/residency/Transport.h"
#include "cufft_plan.h"
#include "seam.h"

namespace {

thread_local std::string g_error;

int fail(const std::string& message) {
    g_error = message;
    return 1;
}
int cuda_fail(cudaError_t e) {
    cudaGetLastError();   // clear the sticky error of a refused call
    return fail(cudaGetErrorString(e));
}
template <class F> int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}

aether::Device device_of(int kind, int dev) {
    return kind == AETHER_MEM_CPU ? aether::Device(kDLCPU)
         : kind == AETHER_MEM_PINNED ? aether::Device(kDLCUDAHost) : aether::Device(kDLCUDA, dev);
}
inline aether::Stream S(void* s) { return static_cast<aether::Stream>(s); }

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

int probe(int* count) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return cuda_fail(e);
    if (n < 1) return cuda_fail(cudaErrorNoDevice);
    if (count) *count = n;
    return 0;
}
int runtime_version(int* v) {
    cudaError_t e = cudaRuntimeGetVersion(v);
    return e == cudaSuccess ? 0 : cuda_fail(e);
}
int current_device(int* d) {
    cudaError_t e = cudaGetDevice(d);
    return e == cudaSuccess ? 0 : cuda_fail(e);
}
const char* last_error() { return g_error.c_str(); }

int chunk_alloc(int kind, int dev, size_t bytes, void* stream, void** chunk, void** ptr) {
    return guarded([&] {
        auto c = std::make_unique<aether::Chunk>(aether::Chunk::allocate(device_of(kind, dev), bytes, S(stream)));
        *ptr = c->data();
        *chunk = c.release();
    });
}
void chunk_release(void* chunk) { delete static_cast<aether::Chunk*>(chunk); }
void drain() { aether::drainDeferredFrees(); }

int fill(int kind, int dev, void* ptr, size_t bytes, const void* value, size_t width, void* stream) {
    return guarded([&] {
        aether::Chunk c = aether::Chunk::borrow(ptr, bytes, device_of(kind, dev));
        aether::fill(c, std::span<const std::byte>(static_cast<const std::byte*>(value), width), S(stream));
    });
}

// One transport serves every copy: its bounce buffer is reused, so staged copies run one at a time. Never
// destroyed (the pinned buffer would otherwise be freed after the runtime is gone).
int copy(int dst_kind, int dst_dev, void* dst, int src_kind, int src_dev, const void* src, size_t bytes, void* stream) {
    static std::mutex mu;
    static auto* transport = new aether::StreamTransport();
    return guarded([&] {
        aether::Chunk d = aether::Chunk::borrow(dst, bytes, device_of(dst_kind, dst_dev));
        aether::Chunk s = aether::Chunk::borrow(const_cast<void*>(src), bytes, device_of(src_kind, src_dev));
        std::lock_guard<std::mutex> lock(mu);
        transport->copyOrdered(d, s, S(stream));
    });
}

int stream_wait(int dev, void* producer, void* consumer) {
    OnDevice d(dev);
    cudaEvent_t ev;
    cudaError_t e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (e != cudaSuccess) return cuda_fail(e);
    e = cudaEventRecord(ev, S(producer));
    if (e == cudaSuccess) e = cudaStreamWaitEvent(S(consumer), ev, 0);
    cudaEventDestroy(ev);
    return e == cudaSuccess ? 0 : cuda_fail(e);
}
int stream_sync(int dev, void* stream) {
    OnDevice d(dev);
    cudaError_t e = cudaStreamSynchronize(S(stream));
    return e == cudaSuccess ? 0 : cuda_fail(e);
}
int pointer_device(const void* p, int* dev) {
    cudaPointerAttributes a;
    cudaError_t e = cudaPointerGetAttributes(&a, p);
    if (e != cudaSuccess) return cuda_fail(e);
    if (a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged) return fail("not a device pointer");
    *dev = a.device;
    return 0;
}

int fft_plan(int dev, int kind, int is_double, int norm, int forward, int ndim, const int64_t* shape, int naxes,
             const int32_t* axes, void* stream, void** plan) {
    try {
        OnDevice d(dev);
        *plan = new aether_plugin::FftPlan(dev, kind, is_double != 0, norm, forward != 0,
                                           std::vector<std::int64_t>(shape, shape + ndim),
                                           std::vector<std::int32_t>(axes, axes + naxes), S(stream));
        return 0;
    } catch (const aether_plugin::CufftMissing& e) {
        g_error = e.what();
        return 2;
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}
int fft_exec(void* plan, const void* in, void* out, void* stream) {
    return guarded([&] { static_cast<aether_plugin::FftPlan*>(plan)->execute(in, out, S(stream)); });
}
void fft_destroy(void* plan) { delete static_cast<aether_plugin::FftPlan*>(plan); }

const AetherMemoryV1 table = {AETHER_MEMORY_ABI, probe, runtime_version, current_device, last_error, chunk_alloc,
                              chunk_release, drain, fill, copy, stream_wait, stream_sync, pointer_device,
                              fft_plan, fft_exec, fft_destroy};

}  // namespace

extern "C" __attribute__((visibility("default"))) const AetherMemoryV1* aether_memory_v1(void) { return &table; }
