// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk tests (host / AETHER_CPP_MODE build; test_Chunk.cu is the
// CUDA-build twin over the same fixture), but the covered kind set differs
// by construction: this build has only kDLCPU available, so it
// additionally covers the "CUDA kind throws in AETHER_CPP_MODE" contract
// that a CUDA build cannot exercise.

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
    // Freed by the destructor at scope exit — no crash is the pass condition.
}

TEST_F(ChunkTest, CpuAllocationIsAligned)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 4096);
    EXPECT_EQ(chunk.alignment(), 256u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % chunk.alignment(), 0u);
}

TEST_F(ChunkTest, ZeroByteChunkIsValidAndEmpty)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 0);
    EXPECT_EQ(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 0u);
}

TEST_F(ChunkTest, ZeroByteChunkValidEvenForAKindThisBuildCannotAllocate)
{
    // bytes==0 never touches the underlying allocator — even a device
    // kind unavailable in this AETHER_CPP_MODE build produces a valid empty
    // chunk rather than throwing. This is what lets test_Copy.cpp construct
    // an "illegal pair" Chunk without ever needing a real CUDA allocation.
    aether::Device fakeCuda(kDLCUDA);
    auto chunk = aether::Chunk::allocate(fakeCuda, 0);
    EXPECT_EQ(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 0u);
    EXPECT_EQ(chunk.device(), fakeCuda);
}

TEST_F(ChunkTest, MoveConstructionLeavesSourceEmpty)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCPU), 128);
    void* originalPtr = a.data();
    aether::Chunk b(std::move(a));
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(b.data(), originalPtr);
    EXPECT_EQ(b.size(), 128u);
    // `a`'s destructor at scope exit must be a no-op (data()==nullptr) — no
    // double-free of `b`'s memory when both go out of scope.
}

TEST_F(ChunkTest, MoveAssignmentFreesOwnMemoryAndLeavesSourceEmpty)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    auto b = aether::Chunk::allocate(aether::Device(kDLCPU), 32);
    void* bPtr = b.data();
    a = std::move(b); // frees a's original 64-byte allocation first
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0u);
    EXPECT_EQ(a.data(), bPtr);
    EXPECT_EQ(a.size(), 32u);
}

TEST_F(ChunkTest, CudaKindAllocateThrowsInCppMode)
{
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLCUDA), 64), aether::Error);
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLCUDAHost), 64), aether::Error);
}

TEST_F(ChunkTest, UnsupportedDeviceKindAlwaysThrows)
{
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLOpenCL), 64), aether::Error);
}

} // namespace
} // namespace aether_tests
