// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::copyAsync(View, View, Stream)` tests (CUDA build). CUDA-only,
// no .cpp twin: `aether::Stream`/`copyAsync` (both the existing
// Chunk-level pair in chunk/Copy.h and this file's view-level wrapper in
// view/Copy.h) exist only under AETHER_HAS_CUDA -- there is no
// AETHER_CPP_MODE stream/async concept anywhere in aether to widen into
// (see view/Copy.h's own docstring). The API-compiles-and-CPU-pair-throws
// row lives here as CpuPairThrowsLikeChunkLevelCopyAsync, inside this .cu
// file, rather than a separate test_ViewCopyAsync.cpp -- a genuine
// AETHER_CPP_MODE TU calling this API cannot exist without inventing a
// shim Stream type nothing else in aether has. The row itself needs no
// GPU to execute (kDLCPU Chunk::allocate, same trick test_Copy.cu's own
// IllegalCpuCudaDirectAsyncPathThrows uses) -- only the file it lives in
// needs a CUDA-mode build to exist at all.
//
// NVCC host-pass quirk: every `aether::Array<double, N>` construction
// below spells the component count as a literal (`Array<double, 3>`),
// never a same-named local `constexpr std::size_t` fed in as the NTTP --
// see test_ViewComponent.cu's own header note (same TU-wide file, same
// root cause) for the full nvcc repro/citation. `N` (an ordinary
// constructor argument) is unaffected and stays a per-function local.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class ViewCopyAsyncTest : public ::testing::Test { };

TEST_F(ViewCopyAsyncTest, CpuPairThrowsLikeChunkLevelCopyAsync)
{
    // Genuine kDLCPU views (NOT Array's hostChunk_, which is kDLCUDAHost
    // in a CUDA-mode build -- CUDAHost<->CUDAHost IS async-capable) --
    // CPU<->CPU is exactly the pair aether::copyAsync(Chunk&, const
    // Chunk&, Stream) itself refuses (chunk/Copy.h: "no async path for
    // plain (pageable) CPU memory"). This IS the file header's "CPP row":
    // it asserts the view-level API compiles (this whole file only builds
    // under AETHER_HAS_CUDA) and that the CPU pair throws exactly as
    // Copy.h's own Chunk-level copyAsync does -- no widening.
    constexpr std::size_t N = 16;
    auto srcChunk = aether::Chunk::allocate(aether::Device(kDLCPU), N * sizeof(double));
    auto dstChunk = aether::Chunk::allocate(aether::Device(kDLCPU), N * sizeof(double));
    auto src       = aether::make_view<double, aether::dyn>(srcChunk, N);
    auto dst       = aether::make_view<double, aether::dyn>(dstChunk, N);

    aether::Stream stream = 0; // default stream -- never reached, the pair is refused first
    EXPECT_THROW(aether::copyAsync(dst, src, stream), aether::Error);
}

TEST_F(ViewCopyAsyncTest, ExtentsMismatchThrows)
{
    aether::Array<double, 3> dst(10);
    aether::Array<double, 3> src(8);
    ASSERT_NE(dst.samples(), src.samples());

    aether::Stream stream = 0;
    EXPECT_THROW(aether::copyAsync(dst.hostView(), src.hostView(), stream), aether::Error);
}

TEST_F(ViewCopyAsyncTest, StridesMismatchThrows)
{
    // Two Arrays with the same samples() (equal extents) but different
    // capacity(): hostView()'s pitch is capacity(), not samples(), so this
    // is an extents-equal, strides-unequal pair -- the case the
    // extents check alone would miss.
    constexpr std::size_t N = 10;
    aether::Array<double, 3> dst(N);
    aether::Array<double, 3> src(N);
    src.reserve(128);
    ASSERT_EQ(dst.samples(), src.samples());
    ASSERT_NE(dst.capacity(), src.capacity());

    aether::Stream stream = 0;
    EXPECT_THROW(aether::copyAsync(dst.hostView(), src.hostView(), stream), aether::Error);
}

TEST_F(ViewCopyAsyncTest, HostToDeviceThenDeviceToHostRoundTripIsBitExact)
{
    // Bit-identity vs Array::upload()/download(): a clobber-and-recover
    // round trip through copyAsync ALONE (no Array::upload()/download()
    // call in between) -- the final read can only match the ORIGINAL host
    // values if the device-resident bytes copyAsync produced (H2D) and the
    // host-resident bytes it read back (D2H) are both exactly right.
    constexpr std::size_t C = 3, N = 40;
    aether::Array<double, 3> arr(N);
    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 137 + i) * 0.25;

    std::vector<double> original(C * N);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            original[c * N + i] = hv(c, i);

    aether::Stream stream = 0;
    aether::copyAsync(arr.deviceView(), arr.hostView(), stream); // H2D via the view-level API
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (std::size_t c = 0; c < C; ++c) // poison the host copy
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = -1.0;

    aether::copyAsync(arr.hostView(), arr.deviceView(), stream); // D2H via the view-level API
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(verify(c, i), original[c * N + i]) << "c=" << c << " i=" << i;
}

TEST_F(ViewCopyAsyncTest, HostToDeviceMatchesArrayUploadBitExact)
{
    // The literal differential form of the "Bit-identity test vs
    // Array::upload/download for H2D/D2H": the SAME host source fed
    // through TWO separate H2D paths (Array::upload() vs the view-level
    // copyAsync), each downloaded back and compared byte-for-byte.
    constexpr std::size_t C = 3, N = 20;
    aether::Array<double, 3> reference(N);
    aether::Array<double, 3> viaCopyAsync(N);

    auto refHv = reference.hostView();
    auto viaHv = viaCopyAsync.hostView();
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            const double v  = static_cast<double>(c * 91 + i) * 0.75 - 3.0;
            refHv(c, i)     = v;
            viaHv(c, i)     = v; // identical host source on both sides
        }
    }

    reference.upload(); // the REFERENCE H2D transfer: Array::upload()

    aether::Stream stream = 0;
    aether::copyAsync(viaCopyAsync.deviceView(), viaCopyAsync.hostView(), stream); // the H2D path under test
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    reference.download();
    viaCopyAsync.download();

    auto r = reference.hostView();
    auto v = viaCopyAsync.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(v(c, i), r(c, i)) << "c=" << c << " i=" << i;
}

TEST_F(ViewCopyAsyncTest, DeviceToDeviceMatchesADirectChunkLevelCopyAsync)
{
    // Same N => same quantised capacity() (a pure function of N), so
    // `src`/`viaView`/`oracle` all pitch identically below -- comparing
    // through each Array's own hostView() needs no manual pitch math.
    constexpr std::size_t C = 3, N = 30;
    aether::Array<double, 3> src(N);
    aether::Array<double, 3> viaView(N);
    aether::Array<double, 3> oracle(N);
    ASSERT_EQ(src.capacity(), viaView.capacity());
    ASSERT_EQ(src.capacity(), oracle.capacity());

    auto hv = src.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 53 + i) * 1.5;
    src.upload();

    const std::size_t bytes = static_cast<std::size_t>(src.deviceView().mapping().required_span_size()) * sizeof(double);

    // The ORACLE path: a DIRECT Chunk-level D2D copyAsync, bypassing View
    // entirely -- Chunk::borrow wraps each Array's own device span (same
    // technique Array::copyRowsPitch_ already uses internally).
    auto srcChunk    = aether::Chunk::borrow(src.deviceView().data(), bytes, src.deviceView().device());
    auto oracleChunk = aether::Chunk::borrow(oracle.deviceView().data(), bytes, oracle.deviceView().device());

    aether::Stream stream = 0;
    aether::copyAsync(oracleChunk, srcChunk, stream);                  // the oracle path (Chunk-level)
    aether::copyAsync(viaView.deviceView(), src.deviceView(), stream); // the path under test (View-level)
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    oracle.download();
    viaView.download();

    auto o = oracle.hostView();
    auto v = viaView.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(v(c, i), o(c, i)) << "c=" << c << " i=" << i;
}

TEST_F(ViewCopyAsyncTest, CopyIsOrderedOnTheGivenStream)
{
    // "the copy is on the given stream": record an event AFTER each
    // copyAsync call, wait on that event, then read -- a stream-ordering
    // bug (e.g. the copy silently running synchronously or on the wrong
    // stream) would only coincidentally read correct data, but this
    // pattern is the house idiom for asserting ordering explicitly
    // (mirrors test_Array.cu's own async lifecycle test).
    constexpr std::size_t C = 3, N = 25;
    aether::Array<double, 3> arr(N);
    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 7 + i);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    aether::copyAsync(arr.deviceView(), arr.hostView(), stream); // H2D
    cudaEvent_t afterH2D;
    ASSERT_EQ(cudaEventCreate(&afterH2D), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(afterH2D, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(afterH2D), cudaSuccess);

    for (std::size_t c = 0; c < C; ++c) // poison the host copy
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = -1.0;

    aether::copyAsync(arr.hostView(), arr.deviceView(), stream); // D2H
    cudaEvent_t afterD2H;
    ASSERT_EQ(cudaEventCreate(&afterD2H), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(afterD2H, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(afterD2H), cudaSuccess);

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(verify(c, i), static_cast<double>(c * 7 + i)) << "c=" << c << " i=" << i;

    ASSERT_EQ(cudaEventDestroy(afterH2D), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(afterD2H), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
} // namespace aether_tests
