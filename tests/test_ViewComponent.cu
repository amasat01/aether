// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `View::component<I>()` tests (CUDA build; test_ViewComponent.cpp covers
// the same fixture). Adds the component<I>() kernel: a device __global__
// that builds a component<I>() view inside device code, over an Array's
// real device-resident storage, and writes through it --
// upload()/kernel/download(), the same lifecycle test_Array.cu's
// addOneKernel exercises for the plain View.
//
// NVCC host-pass quirk (not previously documented elsewhere): every
// `aether::Array<double, N>` construction below spells the component
// count as a literal (`Array<double, 3>`), never a same-named local
// `constexpr std::size_t` fed in as the NTTP -- a minimal repro (nvcc from
// CUDA 12.x, host pass) showed that a second TEST_F in one .cu TU
// instantiating the same `Array<double, X>` specialization via a named
// local (any name, even a fresh one per function) fails with a spurious
// "template argument 2 is invalid" / "request for member ... in ..., which
// is of non-class type 'int'" -- unless an earlier use in the same TU
// already instantiated that specialization via a literal argument
// (test_Array.cu's `SamplesAndSize` -- `Array<double, 3> arr(10);` --
// happens to do this before its own two named-local "Lifecycle" tests,
// which is why that file was never hit by this). Sibling in spirit to
// test_PacketOps.cpp's own documented nvcc host-pass quirk (`kPacketItemW`)
// but a different trigger (class-template NTTP naming, not
// member-template-call naming) -- confirmed by a standalone bisection,
// `.template component<I>()` does not fix it, only literal NTTPs do. `N`
// (an ordinary constructor argument, not a template argument) is
// unaffected and stays a per-function named local throughout.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class ViewComponentTest : public ::testing::Test { };

// Array::hostView()/deviceView() return a layout_stride view
// (pitch = capacity(), shape = samples()) -- same convention as
// test_Array.cu's own Vec3dStridedView alias.
using Vec3dStridedView = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;

/** @brief THE component<I>() KERNEL: builds `pos.component<1>()` INSIDE
 *  device code and writes `+= 1.0` through it -- the "writes land in the
 *  parent" contract requires, exercised on
 *  the device side (the CPP arm's WriteThroughComponentUploadThenRead...
 *  test covers the host-backend half). */
__global__ void addOneToComponentOneKernel(Vec3dStridedView pos)
{
    auto i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    auto y = pos.component<1>();
    if (i.global() >= y.samples())
        return;
    y(i.global()) += 1.0;
}

TEST_F(ViewComponentTest, WriteThroughDeviceBuiltComponentUploadKernelDownloadBitExact)
{
    constexpr std::size_t C = 3, N = 100;
    aether::Array<double, 3> arr(N);

    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 1000 + i);

    arr.upload();

    constexpr unsigned int blockDim = 32;
    const unsigned int numBlocks    = static_cast<unsigned int>((N + blockDim - 1) / blockDim);
    addOneToComponentOneKernel<<<numBlocks, blockDim>>>(arr.deviceView());
    aether::cuda::checkLastLaunch("addOneToComponentOneKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    arr.download();

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            const double expect = static_cast<double>(c * 1000 + i) + (c == 1 ? 1.0 : 0.0);
            EXPECT_EQ(verify(c, i), expect) << "c=" << c << " i=" << i;
        }
    }
}

/** @brief Plain read-through kernel (no component<I>() on the device side)
 *  -- pairs with the host-write half below to cover the acceptance test:
 *  "write via the component view on host, upload(), read back through the
 *  parent's device view in a kernel." */
__global__ void readComponentTwoIntoFlatBufferKernel(Vec3dStridedView pos, double* out, std::size_t n)
{
    auto i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= n)
        return;
    out[i.global()] = pos(2, i.global());
}

TEST_F(ViewComponentTest, HostBuiltComponentWriteIsVisibleThroughTheParentsDeviceViewInAKernel)
{
    constexpr std::size_t N = 64;
    aether::Array<double, 3> arr(N);

    auto z = arr.hostView().component<2>();
    for (std::size_t i = 0; i < N; ++i)
        z(i) = static_cast<double>(i) * 2.0 + 5.0;

    arr.upload();

    aether::Array<double> out(N);
    constexpr unsigned int blockDim = 32;
    const unsigned int numBlocks    = static_cast<unsigned int>((N + blockDim - 1) / blockDim);
    readComponentTwoIntoFlatBufferKernel<<<numBlocks, blockDim>>>(arr.deviceView(), out.deviceView().data(), N);
    aether::cuda::checkLastLaunch("readComponentTwoIntoFlatBufferKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    out.download();

    auto outHv = out.hostView();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(outHv(i), static_cast<double>(i) * 2.0 + 5.0) << "i=" << i;
}

TEST_F(ViewComponentTest, ShapeAndPitchMatchTheSourceMappingOverADeviceView)
{
    // Pointer-ARITHMETIC / accessor-only (no dereference of device
    // memory) -- same pattern as test_Chunk.cu's CudaDeviceAllocationIsAligned.
    constexpr std::size_t N = 12;
    aether::Array<double, 3> arr(N);
    auto dv = arr.deviceView();
    auto x  = dv.component<0>();

    EXPECT_EQ(x.rank(), 1u);
    EXPECT_EQ(x.samples(), dv.samples());
    EXPECT_EQ(x.mapping().stride(0), dv.mapping().stride(1));
    EXPECT_EQ(x.data(), dv.data());
    EXPECT_EQ(x.device(), dv.device());
}

} // namespace
} // namespace aether_tests
