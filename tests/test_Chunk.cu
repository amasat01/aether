// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk tests (CUDA build; test_Chunk.cpp is the host-build twin over the
// same fixture), but the covered kind set differs by construction: this
// build has kDLCPU, kDLCUDAHost and kDLCUDA available, so it covers all
// three where test_Chunk.cpp covers only kDLCPU (plus the
// AETHER_CPP_MODE-only "CUDA kind throws" contract, which does not apply
// here since CUDA is available).

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>

namespace aether_tests {
namespace {

class ChunkTest : public ::testing::Test { };

TEST_F(ChunkTest, CpuAllocateAndFree)
{
    aether::Device cpu(kDLCPU);
    auto chunk = aether::Chunk::allocate(cpu, 256);
    EXPECT_NE(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 256u);
    EXPECT_EQ(chunk.device(), cpu);
}

TEST_F(ChunkTest, CudaHostAllocateAndFree)
{
    aether::Device host(kDLCUDAHost);
    auto chunk = aether::Chunk::allocate(host, 256);
    EXPECT_NE(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 256u);
    EXPECT_EQ(chunk.device(), host);
}

TEST_F(ChunkTest, CudaDeviceAllocateAndFree)
{
    aether::Device dev(kDLCUDA);
    auto chunk = aether::Chunk::allocate(dev, 256);
    EXPECT_NE(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 256u);
    EXPECT_EQ(chunk.device(), dev);
}

TEST_F(ChunkTest, CpuAllocationIsAligned)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % 256, 0u);
}

TEST_F(ChunkTest, CudaHostAllocationIsAligned)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % 256, 0u);
}

TEST_F(ChunkTest, CudaDeviceAllocationIsAligned)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % 256, 0u);
}

TEST_F(ChunkTest, ZeroByteChunkIsValidAndEmpty)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 0);
    EXPECT_EQ(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 0u);
}

TEST_F(ChunkTest, MoveConstructionLeavesSourceEmpty)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCUDA), 128);
    void* originalPtr = a.data();
    aether::Chunk b(std::move(a));
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(b.data(), originalPtr);
}

TEST_F(ChunkTest, MoveAssignmentFreesOwnMemoryAndLeavesSourceEmpty)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 64);
    auto b = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 32);
    void* bPtr = b.data();
    a = std::move(b);
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(a.data(), bPtr);
}

TEST_F(ChunkTest, UnsupportedDeviceKindAlwaysThrows)
{
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLOpenCL), 64), aether::Error);
}

} // namespace
} // namespace aether_tests
