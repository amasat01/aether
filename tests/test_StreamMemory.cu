// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Stream-ordered memory (CUDA build; test_StreamMemory.cpp is the host-build twin): stream-ordered
// allocate/free, a free requested during a capture deferred and drained, fill of every value width on a
// device and inside a capture, and the ordered host<->device copy (pinned direct, pageable staged).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
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

#define CUDA_OK(call) ASSERT_EQ((call), cudaSuccess)

class StreamMemoryTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(aether::deferredFreeCount(), 0u);
    }
    void TearDown() override
    {
        aether::drainDeferredFrees();
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    /** @brief Bytes of the device chunk, downloaded on `stream_` and waited for. */
    std::vector<std::byte> download(const aether::Chunk& dev)
    {
        std::vector<std::byte> out(dev.size());
        cudaMemcpyAsync(out.data(), dev.data(), dev.size(), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return out;
    }

    cudaStream_t stream_ = nullptr;
};

std::vector<std::byte> pattern(std::size_t width, int seed)
{
    std::vector<std::byte> v(width);
    for (std::size_t j = 0; j < width; ++j)
        v[j] = std::byte(seed + 3 * j);
    return v;
}

TEST_F(StreamMemoryTest, StreamOrderedAllocateAndFreeOnADevice)
{
    {
        auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 1024, stream_);
        EXPECT_NE(chunk.data(), nullptr);
        EXPECT_TRUE(chunk.streamOrdered());
        EXPECT_EQ(chunk.stream(), stream_);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(chunk.data()) % 256, 0u);
        CUDA_OK(cudaMemsetAsync(chunk.data(), 0x5a, chunk.size(), stream_));
        for (std::byte b : download(chunk))
            ASSERT_EQ(b, std::byte(0x5a));
    }   // freed on stream_, behind the work queued there
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
    CUDA_OK(cudaStreamSynchronize(stream_));
}

TEST_F(StreamMemoryTest, StreamOrderedAllocateOnPinnedHost)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 256, stream_);
    EXPECT_TRUE(chunk.streamOrdered());
    std::memset(chunk.data(), 3, chunk.size());
    EXPECT_EQ(chunk.data()[255], std::byte(3));
}

TEST_F(StreamMemoryTest, StreamOrderedChunkMovesKeepTheirStream)
{
    auto a = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    aether::Chunk b = std::move(a);
    EXPECT_TRUE(b.streamOrdered());
    EXPECT_EQ(b.stream(), stream_);
    EXPECT_FALSE(a.streamOrdered());
}

TEST_F(StreamMemoryTest, PlainAllocateStaysSynchronous)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 256);
    EXPECT_FALSE(chunk.streamOrdered());
}

TEST_F(StreamMemoryTest, StreamCapturingReportsTheCaptureState)
{
    EXPECT_FALSE(aether::streamCapturing(stream_));
    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    EXPECT_TRUE(aether::streamCapturing(stream_));
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphDestroy(graph);
    EXPECT_FALSE(aether::streamCapturing(stream_));
}

TEST_F(StreamMemoryTest, FreeDuringCaptureIsDeferredAndTheGraphReplays)
{
    auto out    = aether::Chunk::allocate(aether::Device(kDLCUDA), 64);
    auto victim = aether::Chunk::allocate(aether::Device(kDLCUDA), 4096, stream_);
    const std::uint64_t value = 0x0102030405060708ull;

    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    aether::fill(out, std::as_bytes(std::span<const std::uint64_t, 1>(&value, 1)), stream_);
    { aether::Chunk dying = std::move(victim); }   // freed while stream_ captures: queued, not issued
    EXPECT_EQ(aether::deferredFreeCount(), 1u);
    aether::drainDeferredFrees();                  // still captured: stays queued
    EXPECT_EQ(aether::deferredFreeCount(), 1u);
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));   // the capture survived the free
    cudaGraphExec_t exec = nullptr;
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
        CUDA_OK(cudaMemsetAsync(out.data(), 0, out.size(), stream_));
        CUDA_OK(cudaGraphLaunch(exec, stream_));
        const auto bytes = download(out);
        for (std::size_t i = 0; i < 8; ++i)
            ASSERT_EQ(std::memcmp(bytes.data() + 8 * i, &value, 8), 0) << "replay " << replay;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);

    EXPECT_EQ(aether::deferredFreeCount(), 1u);   // nothing outside the capture has run since
    aether::drainDeferredFrees();
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
}

TEST_F(StreamMemoryTest, ALaterStreamOrderedAllocationDrainsTheQueue)
{
    auto victim = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    { aether::Chunk dying = std::move(victim); }
    EXPECT_EQ(aether::deferredFreeCount(), 1u);
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphDestroy(graph);
    auto next = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
}

TEST_F(StreamMemoryTest, PinnedFreeDuringANonBlockingCaptureIsIssuedAtOnce)
{
    auto pinned = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 256, stream_);
    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    // a capture on the legacy stream is what blocks a pinned free: this one is a non-blocking stream, so the
    // free is legal and issued at once
    { aether::Chunk dying = std::move(pinned); }
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphDestroy(graph);
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
}

TEST_F(StreamMemoryTest, ExitDrainReleasesTheQueueAndThenDropsFrees)
{
    auto victim = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    { aether::Chunk dying = std::move(victim); }
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphDestroy(graph);
    ASSERT_EQ(aether::deferredFreeCount(), 1u);

    auto& queue = aether::detail::DeferredFrees::instance();
    queue.drain(true, true);                       // what the atexit hook runs
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
    EXPECT_TRUE(queue.exiting());
    { auto late = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_); }   // left to the driver
    EXPECT_EQ(aether::deferredFreeCount(), 0u);
    queue.resumeAfterExit();
    EXPECT_FALSE(queue.exiting());
}

TEST_F(StreamMemoryTest, FillDeviceValueWidths)
{
    for (std::size_t width : { std::size_t{ 1 }, std::size_t{ 2 }, std::size_t{ 4 }, std::size_t{ 8 } }) {
        const auto value = pattern(width, 0x11);   // distinct bytes: one strided memset per byte plane
        auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 16 * width, stream_);
        aether::fill(chunk, value, stream_);
        const auto bytes = download(chunk);
        for (std::size_t i = 0; i < 16; ++i)
            ASSERT_EQ(std::memcmp(bytes.data() + i * width, value.data(), width), 0) << "width " << width;
    }
}

TEST_F(StreamMemoryTest, FillDeviceUniformMultiByteValueIsOneMemset)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 64, stream_);
    const std::uint32_t v = 0x07070707u;
    aether::fill(chunk, std::as_bytes(std::span<const std::uint32_t, 1>(&v, 1)), stream_);
    for (std::byte b : download(chunk))
        ASSERT_EQ(b, std::byte(7));
}

TEST_F(StreamMemoryTest, FillHostKindsValueWidths)
{
    for (auto kind : { kDLCPU, kDLCUDAHost }) {
        for (std::size_t width : { std::size_t{ 1 }, std::size_t{ 2 }, std::size_t{ 4 }, std::size_t{ 8 } }) {
            const auto value = pattern(width, 0x21);
            auto chunk = aether::Chunk::allocate(aether::Device(kind), 8 * width);
            aether::fill(chunk, value, stream_);
            for (std::size_t i = 0; i < 8; ++i)
                ASSERT_EQ(std::memcmp(chunk.data() + i * width, value.data(), width), 0);
        }
    }
}

TEST_F(StreamMemoryTest, FillInsideACaptureReplays)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 8 * 8);
    const auto value = pattern(8, 0x31);
    CUDA_OK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    aether::fill(chunk, value, stream_);
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
        CUDA_OK(cudaMemsetAsync(chunk.data(), 0, chunk.size(), stream_));
        CUDA_OK(cudaGraphLaunch(exec, stream_));
        const auto bytes = download(chunk);
        for (std::size_t i = 0; i < 8; ++i)
            ASSERT_EQ(std::memcmp(bytes.data() + 8 * i, value.data(), 8), 0) << "replay " << replay;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
}

TEST_F(StreamMemoryTest, FillRejectsAWidthThatDoesNotDivideTheChunk)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 10, stream_);
    const auto value = pattern(4, 1);
    EXPECT_THROW(aether::fill(chunk, value, stream_), aether::Error);
}

TEST_F(StreamMemoryTest, CopyOrderedPinnedGoesDirect)
{
    aether::StreamTransport transport;
    auto pinned = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 512);
    auto back   = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 512);
    auto dev    = aether::Chunk::allocate(aether::Device(kDLCUDA), 512, stream_);
    for (std::size_t i = 0; i < 512; ++i)
        pinned.data()[i] = std::byte(i * 7);
    transport.copyOrdered(dev, pinned, stream_);
    transport.copyOrdered(back, dev, stream_);
    CUDA_OK(cudaStreamSynchronize(stream_));
    EXPECT_EQ(std::memcmp(back.data(), pinned.data(), 512), 0);
}

TEST_F(StreamMemoryTest, CopyOrderedPageableIsStagedBothWays)
{
    aether::StreamTransport transport;
    auto src  = aether::Chunk::allocate(aether::Device(kDLCPU), 4096);
    auto back = aether::Chunk::allocate(aether::Device(kDLCPU), 4096);
    auto dev  = aether::Chunk::allocate(aether::Device(kDLCUDA), 4096, stream_);
    for (std::size_t i = 0; i < 4096; ++i)
        src.data()[i] = std::byte(i * 13);
    transport.copyOrdered(dev, src, stream_);
    transport.copyOrdered(back, dev, stream_);   // device -> pageable blocks: complete on return
    EXPECT_EQ(std::memcmp(back.data(), src.data(), 4096), 0);
}

TEST_F(StreamMemoryTest, CopyOrderedStagedUploadsDoNotClobberTheBounceBuffer)
{
    aether::StreamTransport transport;
    auto a    = aether::Chunk::allocate(aether::Device(kDLCPU), 1 << 16);
    auto b    = aether::Chunk::allocate(aether::Device(kDLCPU), 1 << 16);
    auto devA = aether::Chunk::allocate(aether::Device(kDLCUDA), 1 << 16, stream_);
    auto devB = aether::Chunk::allocate(aether::Device(kDLCUDA), 1 << 16, stream_);
    std::memset(a.data(), 0xA1, a.size());
    std::memset(b.data(), 0xB2, b.size());
    transport.copyOrdered(devA, a, stream_);
    transport.copyOrdered(devB, b, stream_);   // reuses the bounce chunk: waits for the first upload
    for (std::byte x : download(devA))
        ASSERT_EQ(x, std::byte(0xA1));
    for (std::byte x : download(devB))
        ASSERT_EQ(x, std::byte(0xB2));
}

TEST_F(StreamMemoryTest, CopyOrderedWaitsForWorkAlreadyQueuedOnTheStream)
{
    aether::StreamTransport transport;
    auto dev  = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    auto host = aether::Chunk::allocate(aether::Device(kDLCPU), 256);
    CUDA_OK(cudaMemsetAsync(dev.data(), 0x6c, dev.size(), stream_));
    transport.copyOrdered(host, dev, stream_);
    for (std::size_t i = 0; i < 256; ++i)
        ASSERT_EQ(host.data()[i], std::byte(0x6c));
}

TEST_F(StreamMemoryTest, CopyOrderedDeviceToDevice)
{
    aether::StreamTransport transport;
    auto a = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    auto b = aether::Chunk::allocate(aether::Device(kDLCUDA), 256, stream_);
    CUDA_OK(cudaMemsetAsync(a.data(), 0x42, a.size(), stream_));
    transport.copyOrdered(b, a, stream_);
    for (std::byte x : download(b))
        ASSERT_EQ(x, std::byte(0x42));
}

TEST_F(StreamMemoryTest, CopyOrderedBetweenHostChunksWaitsForTheStreamThenCopies)
{
    aether::StreamTransport transport;
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), 96);
    auto dst = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 96);
    std::memset(src.data(), 9, 96);
    transport.copyOrdered(dst, src, stream_);
    EXPECT_EQ(dst.data()[95], std::byte(9));
}

TEST_F(StreamMemoryTest, CopyOrderedRejectsASizeMismatch)
{
    aether::StreamTransport transport;
    auto a = aether::Chunk::allocate(aether::Device(kDLCUDA), 64, stream_);
    auto b = aether::Chunk::allocate(aether::Device(kDLCUDA), 32, stream_);
    EXPECT_THROW(transport.copyOrdered(a, b, stream_), aether::Error);
}

} // namespace
} // namespace aether_tests
