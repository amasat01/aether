// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// SampleIndex tests (host / AETHER_CPP_MODE build; test_SampleIndex.cu
// covers the same host-semantics cases, plus a device-side kernel check) —
// the host build can only assert the *formula*, not that real CUDA
// built-ins feed it correctly.

#include <cstddef>

#include <gtest/gtest.h>

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

} // namespace
} // namespace aether_tests
