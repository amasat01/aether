// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Array tests (CUDA build). Paired with test_Array.cpp. This file carries
// a device __global__ that mutates an Array's device-resident data through
// View + SampleIndex, driven by an upload()/kernel/download() lifecycle
// exercised in BOTH blocking and stream (async) forms.
//
// GPU EXECUTION NOTE: this file is built and its tests LISTED
// (--gtest_list_tests, which does not execute test bodies); the suite is
// never RUN here — that happens elsewhere, on the GPU.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class ArrayTest : public ::testing::Test { };

TEST_F(ArrayTest, SamplesAndSize)
{
    aether::Array<double, 3> arr(10);
    EXPECT_EQ(arr.samples(), 10u);
    EXPECT_EQ(arr.size(), 30u);
}

TEST_F(ArrayTest, ScalarArraySamplesAndSize)
{
    aether::Array<double> arr(7);
    EXPECT_EQ(arr.samples(), 7u);
    EXPECT_EQ(arr.size(), 7u);
}

TEST_F(ArrayTest, HostAndDeviceViewAreDistinctChunksInCudaMode)
{
    aether::Array<double, 3> arr(4);
    auto hv = arr.hostView();
    auto dv = arr.deviceView();
    EXPECT_NE(hv.data(), dv.data());
}

// `Array::hostView()`/`deviceView()` return a `layout_stride` view
// (pitch = capacity(), shape = samples()) rather than the compact
// `layout_right` `VecView<double,3>` alias — same convention as
// `test_Partition.cu`'s own `Block3dView`.
using Vec3dStridedView = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;

/** @brief THE FIRST REAL KERNEL: `pos(c,i) += 1` via View + SampleIndex. */
__global__ void addOneKernel(Vec3dStridedView pos)
{
    auto i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= pos.samples())
        return;
    for (std::size_t c = 0; c < 3; ++c)
        pos(c, i.global()) += 1.0;
}

TEST_F(ArrayTest, LifecycleFillUploadKernelDownloadBitExactBlocking)
{
    constexpr std::size_t C = 3, N = 100;
    aether::Array<double, C> arr(N);

    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 1000 + i);

    arr.upload();

    constexpr unsigned int blockDim = 32;
    const unsigned int numBlocks    = static_cast<unsigned int>((N + blockDim - 1) / blockDim);
    addOneKernel<<<numBlocks, blockDim>>>(arr.deviceView());
    aether::cuda::checkLastLaunch("addOneKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    arr.download();

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(verify(c, i), static_cast<double>(c * 1000 + i + 1));
}

TEST_F(ArrayTest, LifecycleFillUploadKernelDownloadBitExactAsyncStream)
{
    constexpr std::size_t C = 3, N = 64;
    aether::Array<double, C> arr(N);

    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 10 + i);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    arr.upload(stream);

    constexpr unsigned int blockDim = 32;
    const unsigned int numBlocks    = static_cast<unsigned int>((N + blockDim - 1) / blockDim);
    addOneKernel<<<numBlocks, blockDim, 0, stream>>>(arr.deviceView());
    aether::cuda::checkLastLaunch("addOneKernel");

    arr.download(stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(verify(c, i), static_cast<double>(c * 10 + i + 1));
}

} // namespace
} // namespace aether_tests
