// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Stream-ordered memory (host / AETHER_CPP_MODE build; test_StreamMemory.cu is the CUDA-build twin): the
// same API with its synchronous host behaviour — a stream-ordered allocate is the plain allocation, nothing
// is ever capturing or deferred, `fill` and `copyOrdered` are host operations, and a CUDA kind is refused.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Fill.h>
#include <aether/chunk/StreamOrdered.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class StreamMemoryTest : public ::testing::Test { };

template<class T>
std::span<const std::byte> bytesOf(const T& v)
{
    return std::as_bytes(std::span<const T, 1>(&v, 1));
}

TEST_F(StreamMemoryTest, StreamOrderedAllocateOnCpuIsThePlainAllocation)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 1024, aether::Stream{});
    EXPECT_NE(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 1024u);
    EXPECT_EQ(chunk.alignment(), 256u);
    EXPECT_TRUE(chunk.streamOrdered());
    EXPECT_EQ(chunk.stream(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % 256, 0u);
}

TEST_F(StreamMemoryTest, PlainAllocateIsNotStreamOrdered)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    EXPECT_FALSE(chunk.streamOrdered());
}

TEST_F(StreamMemoryTest, StreamOrderedFlagFollowsAMove)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCPU), 64, aether::Stream{});
    aether::Chunk b = std::move(a);
    EXPECT_TRUE(b.streamOrdered());
    EXPECT_FALSE(a.streamOrdered());
}

TEST_F(StreamMemoryTest, StreamOrderedAllocateOfACudaKindThrowsWithoutABackend)
{
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLCUDA), 64, aether::Stream{}), aether::Error);
    EXPECT_THROW(aether::Chunk::allocate(aether::Device(kDLCUDAHost), 64, aether::Stream{}), aether::Error);
}

TEST_F(StreamMemoryTest, NothingIsCapturingOrDeferredWithoutABackend)
{
    EXPECT_FALSE(aether::streamCapturing(aether::Stream{}));
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
    aether::drainDeferredFrees();
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
}

TEST_F(StreamMemoryTest, FillCpuValueWidths)
{
    for (std::size_t width : { std::size_t{ 1 }, std::size_t{ 2 }, std::size_t{ 4 }, std::size_t{ 8 } }) {
        std::vector<std::byte> value(width);
        for (std::size_t j = 0; j < width; ++j)
            value[j] = std::byte(0x10 + 3 * j);   // distinct bytes: the plane path on a device
        auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 8 * width);
        aether::fill(chunk, value, aether::Stream{});
        for (std::size_t i = 0; i < 8; ++i)
            EXPECT_EQ(std::memcmp(chunk.data() + i * width, value.data(), width), 0) << "width " << width;
    }
}

TEST_F(StreamMemoryTest, FillCpuUniformValue)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    const std::uint32_t v = 0x07070707u;
    aether::fill(chunk, bytesOf(v), aether::Stream{});
    for (std::size_t i = 0; i < 64; ++i)
        EXPECT_EQ(chunk.data()[i], std::byte(7));
}

TEST_F(StreamMemoryTest, FillRejectsAWidthThatDoesNotDivideTheChunk)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 10);
    const std::uint32_t v = 1;
    EXPECT_THROW(aether::fill(chunk, bytesOf(v), aether::Stream{}), aether::Error);
    EXPECT_THROW(aether::fill(chunk, std::span<const std::byte>{}, aether::Stream{}), aether::Error);
}

TEST_F(StreamMemoryTest, FillOfAnEmptyChunkIsANoOp)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 0);
    const std::uint64_t v = 1;
    EXPECT_NO_THROW(aether::fill(chunk, bytesOf(v), aether::Stream{}));
}

TEST_F(StreamMemoryTest, CopyOrderedBetweenHostChunksCopiesTheBytes)
{
    aether::StreamTransport transport;
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), 96);
    auto dst = aether::Chunk::allocate(aether::Device(kDLCPU), 96);
    for (std::size_t i = 0; i < 96; ++i)
        src.data()[i] = std::byte(i);
    transport.copyOrdered(dst, src, aether::Stream{});
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 96), 0);
}

} // namespace
} // namespace aether_tests
