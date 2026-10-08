// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk copy tests (host / AETHER_CPP_MODE build; test_Copy.cu is the
// CUDA-build twin): the CUDA build adds a GPU-touching
// H<->CUDAHost<->D<->D round-trip that this file cannot cover (no CUDA
// backend here). "Illegal path" is exercised via a 0-byte kDLCUDA-labelled
// Chunk (see test_Chunk.cpp's
// ZeroByteChunkValidEvenForAKindThisBuildCannotAllocate) rather than a real
// CUDA allocation.

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Copy.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>

namespace aether_tests {
namespace {

class CopyTest : public ::testing::Test { };

TEST_F(CopyTest, CpuToCpuBlockingIsBitExact)
{
    aether::Device cpu(kDLCPU);
    auto src = aether::Chunk::allocate(cpu, 64);
    auto dst = aether::Chunk::allocate(cpu, 64);
    for (std::size_t i = 0; i < src.size(); ++i)
        src.data()[i] = static_cast<std::byte>((i * 37 + 11) & 0xFF);

    aether::copy(dst, src);

    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}

TEST_F(CopyTest, ZeroByteCopyIsANoOp)
{
    aether::Device cpu(kDLCPU);
    auto src = aether::Chunk::allocate(cpu, 0);
    auto dst = aether::Chunk::allocate(cpu, 0);
    EXPECT_NO_THROW(aether::copy(dst, src));
}

TEST_F(CopyTest, SizeMismatchThrows)
{
    aether::Device cpu(kDLCPU);
    auto src = aether::Chunk::allocate(cpu, 64);
    auto dst = aether::Chunk::allocate(cpu, 32);
    EXPECT_THROW(aether::copy(dst, src), aether::Error);
}

TEST_F(CopyTest, IllegalCpuCudaDirectPathThrows)
{
    // bytes==0 lets us construct a kDLCUDA-labelled Chunk without a real
    // CUDA allocation; copy()'s illegal-path check runs BEFORE the
    // zero-byte short-circuit, so this still exercises the guard.
    auto cpu = aether::Chunk::allocate(aether::Device(kDLCPU), 0);
    auto fakeCuda = aether::Chunk::allocate(aether::Device(kDLCUDA), 0);
    EXPECT_THROW(aether::copy(fakeCuda, cpu), aether::Error);
    EXPECT_THROW(aether::copy(cpu, fakeCuda), aether::Error);
}

} // namespace
} // namespace aether_tests
