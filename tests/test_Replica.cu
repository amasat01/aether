// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// ReplicaSet tests (CUDA build; test_Replica.cpp is the host-build twin
// over the same fixture). Adds the CUDAHost<->CUDA broadcast path
// (bit-exact) and confirms `ReplicaSet::broadcast` propagates
// `aether/chunk/Copy.h`'s legal-pair matrix verbatim (a plain-CPU source
// broadcasting into a CUDA replica throws, exactly as a direct
// `aether::copy` would).

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Copy.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/residency/Replica.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class ReplicaTest : public ::testing::Test { };

TEST_F(ReplicaTest, BroadcastAcrossCudaHostAndCudaReplicasIsBitExact)
{
    constexpr std::size_t N = 256;
    std::vector<aether::Device> devices{ aether::Device(kDLCUDAHost), aether::Device(kDLCUDA) };
    aether::ReplicaSet rs(devices, N);

    auto src = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    for (std::size_t i = 0; i < N; ++i)
        src.data()[i] = static_cast<std::byte>((i * 53 + 17) & 0xFF);

    rs.broadcast(src);

    // Replica 0 (CUDAHost) is host-addressable directly.
    EXPECT_EQ(std::memcmp(rs.chunk(0).data(), src.data(), N), 0);

    // Replica 1 (CUDA) needs a device -> host readback to compare.
    auto verify = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(verify, rs.chunk(1));
    EXPECT_EQ(std::memcmp(verify.data(), src.data(), N), 0);
}

TEST_F(ReplicaTest, BroadcastAsyncAcrossCudaHostAndCudaReplicasIsBitExact)
{
    constexpr std::size_t N = 256;
    std::vector<aether::Device> devices{ aether::Device(kDLCUDAHost), aether::Device(kDLCUDA) };
    aether::ReplicaSet rs(devices, N);

    auto src = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    for (std::size_t i = 0; i < N; ++i)
        src.data()[i] = static_cast<std::byte>((i * 89 + 5) & 0xFF);

    aether::Stream stream = 0; // default stream
    rs.broadcast(src, stream);
    cudaStreamSynchronize(stream);

    EXPECT_EQ(std::memcmp(rs.chunk(0).data(), src.data(), N), 0);
    auto verify = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(verify, rs.chunk(1));
    EXPECT_EQ(std::memcmp(verify.data(), src.data(), N), 0);
}

TEST_F(ReplicaTest, BroadcastFromPlainCpuToACudaReplicaThrows)
{
    std::vector<aether::Device> devices{ aether::Device(kDLCUDA) };
    aether::ReplicaSet rs(devices, 64);
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    EXPECT_THROW(rs.broadcast(src), aether::Error);
}

// `StreamTransport::copyAsync` — CUDA-backend-only member — must
// actually perform a correct async copy (the concept's own `static_assert`
// in `aether/residency/Transport.h` only proves the SIGNATURE exists).
TEST_F(ReplicaTest, StreamTransportCopyAsyncDelegatesToChunkCopyAsync)
{
    constexpr std::size_t N = 128;
    auto src = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    auto dst = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    for (std::size_t i = 0; i < N; ++i)
        src.data()[i] = static_cast<std::byte>((i * 3 + 1) & 0xFF);

    aether::Stream stream = 0; // default stream
    aether::StreamTransport transport;
    transport.copyAsync(dst, src, stream);
    cudaStreamSynchronize(stream);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), N), 0);
}

// `Replica<T, Transport>` bit-identity vs the PRE-CHANGE `ReplicaSet`
// reference, over CUDAHost<->CUDA (single
// GPU -- device 0 only, no second physical device needed for this claim;
// see test_Replica.cpp's own docstring for the reference's provenance).
TEST_F(ReplicaTest, TemplatedReplicaDefaultTransportBitIdenticalToPreChangeReplicaSetCudaHostAndCuda)
{
    constexpr std::size_t N = 256;
    std::vector<aether::Device> devices{ aether::Device(kDLCUDAHost), aether::Device(kDLCUDA) };

    std::vector<unsigned char> pattern(N);
    for (std::size_t i = 0; i < N; ++i)
        pattern[i] = static_cast<unsigned char>((i * 61 + 23) & 0xFF);
    auto src = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    std::memcpy(src.data(), pattern.data(), N);

    aether::ReplicaSet oldPath(devices, N); // the pre-existing reference implementation
    oldPath.broadcast(src);

    aether::Replica<unsigned char> newPath(devices, N); // default Transport = StreamTransport
    newPath.broadcast(src);

    // Replica 0 (CUDAHost) is host-addressable directly.
    EXPECT_EQ(std::memcmp(newPath.chunk(0).data(), oldPath.chunk(0).data(), N), 0);
    EXPECT_EQ(std::memcmp(newPath.chunk(0).data(), pattern.data(), N), 0);

    // Replica 1 (CUDA) needs a device -> host readback to compare.
    auto oldReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    auto newReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(oldReadback, oldPath.chunk(1));
    aether::copy(newReadback, newPath.chunk(1));
    EXPECT_EQ(std::memcmp(newReadback.data(), oldReadback.data(), N), 0)
        << "replica 1 (CUDA) vs pre-change ReplicaSet";
    EXPECT_EQ(std::memcmp(newReadback.data(), pattern.data(), N), 0) << "replica 1 (CUDA) vs source";
}

} // namespace
} // namespace aether_tests
