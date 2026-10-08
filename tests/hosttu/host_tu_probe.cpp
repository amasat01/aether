// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tests/hosttu/host_tu_probe.cpp — THE HOST-COMPILER TRANSLATION UNIT.
//
// ★ THIS FILE IS COMPILED BY `g++`, INSIDE THE CUDA BUILD. ★  It is added to
// `aether_tests` with `LANGUAGE CXX` forced (tests/CMakeLists.txt), it does
// NOT define `AETHER_CPP_MODE` (that would flip the member set for one TU
// and be an ODR violation), and it includes the FULL `aether/aether.h`
// umbrella. Its existence in the CUDA-build link line is itself the
// acceptance: the eagle port compiles six such `.cpp` tests against an
// installed CUDA-mode aether, and without this fix they died at
// `aether/device/Device.h`'s default `Device()` constructor: '__host__' does not name a type`.
//
// It allocates, fills, uploads, downloads, views and indexes — everything a
// host TU is promised. It launches NOTHING; `<<<>>>` is nvcc grammar, and
// `aether::eval::runtimeEval` refuses to instantiate here by design (see
// `tests/sass/check_host_tu_includes.sh`, which proves that refusal).
//
// Deliberately NOT named `test_*.cpp`: the CPP_MODE build globs `test_*.cpp`
// into `aether_tests`, and this harness only means something where the two
// compilers coexist. All gtest cases live in the `.cu` partner, so only
// `tests/expected_tests_cuda.txt` gains rows.

#include "tests/hosttu/host_tu_probe.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace aether_r8 {

void hostFillLayoutTable(LayoutRow* out)
{
    // NON-INLINE, external linkage, and this body is the ONLY expansion the
    // host compiler makes. The `.cu` partner has its own separately named
    // filler; neither can be COMDAT-deduped onto the other, so the two
    // tables really are two compilers' measurements.
    std::size_t i = 0;
#define AETHER_R8_MEASURE_ONE(...)                                                                                     \
    out[i].name  = #__VA_ARGS__;                                                                                       \
    out[i].size  = static_cast<std::uint64_t>(sizeof(__VA_ARGS__));                                                    \
    out[i].align = static_cast<std::uint64_t>(alignof(__VA_ARGS__));                                                   \
    ++i;
    AETHER_R8_LAYOUT_TYPES(AETHER_R8_MEASURE_ONE)
#undef AETHER_R8_MEASURE_ONE
    static_cast<void>(i);
}

/** @brief Everything the host TU owns, constructed by `g++`. */
struct HostOwned {
    explicit HostOwned(std::size_t n)
        : scalar(n)
        , vector(n)
    {
    }

    aether::Array<float> scalar;
    aether::Array<float, 3> vector;
    aether::Stream stream = nullptr;
};

HostOwned* hostMakeArrays(std::size_t n, HostArraySummary* summary)
{
    auto* owned = new HostOwned(n);

    // Fill through the HOST view — plain host code, no CUDA vocabulary.
    auto sv = owned->scalar.hostView();
    for (std::size_t i = 0; i < n; ++i)
        sv(i) = static_cast<float>(i);

    auto vv = owned->vector.hostView();
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < n; ++i)
            vv(c, i) = static_cast<float>(1000 * c + i);

    // The `Stream` overloads of upload()/download() are part of what a host
    // TU is allowed to reach (a runtime API call, not device grammar).
    cudaStreamCreate(&owned->stream);
    owned->scalar.upload(owned->stream);
    owned->vector.upload(owned->stream);
    cudaStreamSynchronize(owned->stream);

    if (summary != nullptr) {
        const auto& map = vv.mapping();
        summary->scalarHostData   = owned->scalar.hostView().data();
        summary->scalarDeviceData = owned->scalar.deviceView().data();
        summary->vectorHostData   = vv.data();
        summary->vectorDeviceData = owned->vector.deviceView().data();
        summary->samples          = static_cast<std::uint64_t>(owned->vector.samples());
        summary->capacity         = static_cast<std::uint64_t>(owned->vector.capacity());
        summary->vectorRank       = static_cast<std::uint64_t>(vv.rank());
        summary->vectorStride0    = static_cast<std::int64_t>(map.stride(0));
        summary->vectorStride1    = static_cast<std::int64_t>(map.stride(1));
        summary->vectorExtent0    = static_cast<std::uint64_t>(vv.extent(0));
        summary->vectorExtent1    = static_cast<std::uint64_t>(vv.extent(1));
        summary->arrayFloatSize   = static_cast<std::uint64_t>(sizeof(aether::Array<float>));
        summary->arrayVec3Size    = static_cast<std::uint64_t>(sizeof(aether::Array<float, 3>));
    }
    return owned;
}

void hostDestroyArrays(HostOwned* owned)
{
    if (owned == nullptr)
        return;
    if (owned->stream != nullptr)
        cudaStreamDestroy(owned->stream);
    delete owned;
}

aether::Array<float, 3>::ViewT hostVectorDeviceView(HostOwned* owned) { return owned->vector.deviceView(); }

aether::Array<float, 3>::ViewT hostVectorHostView(HostOwned* owned) { return owned->vector.hostView(); }

void hostDownloadVector(HostOwned* owned)
{
    owned->vector.download(owned->stream);
    cudaStreamSynchronize(owned->stream);
}

aether::Device hostMakeCudaDevice() { return aether::Device(kDLCUDA, 0); }

aether::Stream hostStream(HostOwned* owned) { return owned->stream; }

} // namespace aether_r8
