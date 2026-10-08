// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// ReplicaSet tests (host / AETHER_CPP_MODE build; test_Replica.cu is the
// CUDA-build twin over the same fixture): construction from an explicit
// Device list, `chunk(i)` accessor, and `broadcast()` — bit-exact into
// every replica.

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/residency/Replica.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class ReplicaTest : public ::testing::Test { };

TEST_F(ReplicaTest, ConstructsOneChunkPerDeviceWithTheRequestedSize)
{
    // Distinct DEVICE IDS stand in for distinct devices — Chunk::allocate
    // dispatches on device KIND, not id, so this exercises the "N chunks on
    // N devices" contract entirely within an AETHER_CPP_MODE (kDLCPU-only)
    // build.
    std::vector<aether::Device> devices{
        aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1), aether::Device(kDLCPU, 2)
    };
    aether::ReplicaSet rs(devices, 256);
    EXPECT_EQ(rs.size(), 3u);
    for (std::size_t i = 0; i < devices.size(); ++i) {
        EXPECT_EQ(rs.chunk(i).size(), 256u);
        EXPECT_EQ(rs.chunk(i).device(), devices[i]);
        EXPECT_NE(rs.chunk(i).data(), nullptr);
        EXPECT_TRUE(rs.chunk(i).owns());
    }
}

TEST_F(ReplicaTest, EmptyDeviceListYieldsAnEmptySet)
{
    std::vector<aether::Device> devices;
    aether::ReplicaSet rs(devices, 128);
    EXPECT_EQ(rs.size(), 0u);
}

TEST_F(ReplicaTest, BroadcastCopiesSourceBytesIntoEveryReplicaBitExact)
{
    constexpr std::size_t N = 64;
    std::vector<aether::Device> devices{ aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1) };
    aether::ReplicaSet rs(devices, N);

    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), N);
    for (std::size_t i = 0; i < N; ++i)
        src.data()[i] = static_cast<std::byte>((i * 37 + 11) & 0xFF);

    rs.broadcast(src);

    for (std::size_t i = 0; i < devices.size(); ++i)
        EXPECT_EQ(std::memcmp(rs.chunk(i).data(), src.data(), N), 0) << "replica " << i;
}

// `StreamTransport` — the single v1 `transport` implementation — must
// actually perform a correct copy, not merely satisfy the concept
// (`aether/residency/Transport.h`'s own `static_assert` already proves
// the latter at compile time).
TEST_F(ReplicaTest, StreamTransportCopyDelegatesToChunkCopy)
{
    constexpr std::size_t N = 32;
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), N);
    auto dst = aether::Chunk::allocate(aether::Device(kDLCPU), N);
    for (std::size_t i = 0; i < N; ++i)
        src.data()[i] = static_cast<std::byte>(i);

    aether::StreamTransport transport;
    transport.copy(dst, src);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), N), 0);
}

// `Replica<T, Transport>` bit-identity vs the pre-existing `ReplicaSet`
// reference. `ReplicaSet::broadcast` (`aether/residency/Replica.h`) is a
// pure per-replica `aether::copy(dst, src)` -- for CPU<->CPU that is
// `std::memcpy` (`aether/chunk/Copy.h`'s CpuCpu leg), so the golden is the
// deterministic closed form "every replica's bytes == src's bytes"; this
// test also cross-checks live against `ReplicaSet` directly, for
// redundancy against a golden that is itself wrong.
TEST_F(ReplicaTest, TemplatedReplicaDefaultTransportBitIdenticalToPreChangeReplicaSet)
{
    constexpr std::size_t N = 96;
    std::vector<aether::Device> devices{
        aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1), aether::Device(kDLCPU, 2)
    };

    std::vector<unsigned char> pattern(N);
    for (std::size_t i = 0; i < N; ++i)
        pattern[i] = static_cast<unsigned char>((i * 73 + 19) & 0xFF);
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), N);
    std::memcpy(src.data(), pattern.data(), N);

    aether::ReplicaSet oldPath(devices, N); // the pre-existing reference implementation
    oldPath.broadcast(src);

    aether::Replica<unsigned char> newPath(devices, N); // default Transport = StreamTransport
    newPath.broadcast(src);

    ASSERT_EQ(oldPath.size(), newPath.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        EXPECT_EQ(std::memcmp(newPath.chunk(i).data(), oldPath.chunk(i).data(), N), 0)
            << "replica " << i << " vs pre-change ReplicaSet";
        EXPECT_EQ(std::memcmp(newPath.chunk(i).data(), pattern.data(), N), 0)
            << "replica " << i << " vs source (closed form)";
    }
}

} // namespace
} // namespace aether_tests
