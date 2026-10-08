// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// SampleIndex tests (CUDA build; test_SampleIndex.cpp covers the same
// host-semantics cases). This file additionally adds a device-side kernel
// check exercising `SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x)`
// from device code — the host build can only assert the *formula*, not
// that the real CUDA built-ins feed it correctly.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/backend/cuda/Launch.h> // for aether::cuda::checkLastLaunch
#include <aether/index/SampleIndex.h>

namespace aether_tests {
namespace {

class SampleIndexTest : public ::testing::Test { };

TEST_F(SampleIndexTest, HostMakeFromFlatIndexGlobalEqualsWork)
{
    auto i = aether::SampleIndex::make(7);
    EXPECT_EQ(i.global(), 7u);
    EXPECT_EQ(i.work(), 7u);
}

TEST_F(SampleIndexTest, ThreadBlockDimComposesGlobalIndex)
{
    // global = blockIdx*blockDim + threadIdx; work = threadIdx.
    auto i = aether::SampleIndex::make(/*threadIdx*/ 3, /*blockIdx*/ 2, /*blockDim*/ 10);
    EXPECT_EQ(i.global(), 23u);
    EXPECT_EQ(i.work(), 3u);
}

TEST_F(SampleIndexTest, EnumeratedGridMatchesFlatIndex)
{
    constexpr std::size_t blockDim  = 8;
    constexpr std::size_t numBlocks = 4;
    for (std::size_t b = 0; b < numBlocks; ++b) {
        for (std::size_t t = 0; t < blockDim; ++t) {
            auto i = aether::SampleIndex::make(t, b, blockDim);
            EXPECT_EQ(i.global(), b * blockDim + t);
            EXPECT_EQ(i.work(), t);
        }
    }
}

TEST_F(SampleIndexTest, ConstexprEvaluatesAtCompileTime)
{
    constexpr auto i = aether::SampleIndex::make(5);
    static_assert(i.global() == 5, "SampleIndex::make(flat) must be constexpr");
    constexpr auto j = aether::SampleIndex::make(1, 2, 4);
    static_assert(j.global() == 9 && j.work() == 1, "SampleIndex::make(t,b,d) must be constexpr");
    EXPECT_EQ(i.global(), 5u);
    EXPECT_EQ(j.global(), 9u);
}

TEST_F(SampleIndexTest, PacketIndexFullAndTail)
{
    auto full = aether::PacketIndex<4>::make(8);
    EXPECT_TRUE(full.full());
    EXPECT_EQ(full.base_, 8u);
    EXPECT_EQ(full.active_, 4u);

    auto tail = aether::PacketIndex<4>::makeTail(8, 2);
    EXPECT_FALSE(tail.full());
    EXPECT_EQ(tail.base_, 8u);
    EXPECT_EQ(tail.active_, 2u);
}

TEST_F(SampleIndexTest, PacketIndexScalarExtractsPerLaneSampleIndex)
{
    auto packet = aether::PacketIndex<4>::make(8);
    for (std::size_t k = 0; k < 4; ++k) {
        auto s = packet.scalar(k);
        EXPECT_EQ(s.global(), 8u + k);
        EXPECT_EQ(s.work(), 8u + k);
    }
}

constexpr unsigned int kBlockDim  = 8;
constexpr unsigned int kNumBlocks = 4;
constexpr unsigned int kTotal     = kBlockDim * kNumBlocks;

/** @brief Writes `global()`/`work()`, composed from the REAL CUDA built-ins,
 *         one entry per lane — the device-side check asks for. */
__global__ void sampleIndexKernel(unsigned int* globalOut, unsigned int* workOut)
{
    auto i              = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    globalOut[i.global()] = static_cast<unsigned int>(i.global());
    workOut[i.global()]   = static_cast<unsigned int>(i.work());
}

TEST_F(SampleIndexTest, DeviceKernelComposesGlobalAndWorkFromRealCudaBuiltins)
{
    unsigned int* d_global = nullptr;
    unsigned int* d_work   = nullptr;
    ASSERT_EQ(cudaMalloc(&d_global, kTotal * sizeof(unsigned int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_work, kTotal * sizeof(unsigned int)), cudaSuccess);

    sampleIndexKernel<<<kNumBlocks, kBlockDim>>>(d_global, d_work);
    aether::cuda::checkLastLaunch("sampleIndexKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    unsigned int h_global[kTotal];
    unsigned int h_work[kTotal];
    ASSERT_EQ(
        cudaMemcpy(h_global, d_global, kTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_work, d_work, kTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);
    (void)cudaFree(d_global);
    (void)cudaFree(d_work);

    for (unsigned int b = 0; b < kNumBlocks; ++b) {
        for (unsigned int t = 0; t < kBlockDim; ++t) {
            const unsigned int g = b * kBlockDim + t;
            EXPECT_EQ(h_global[g], g);
            EXPECT_EQ(h_work[g], t);
        }
    }
}

} // namespace
} // namespace aether_tests
