// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// CUDA-only PartitionedArray transport test (no `.cpp` companion; see
// tests/test_PeerAccess.cpp for the CPP_MODE moveTo/exchange
// compile+correctness proof over kDLCPU stand-in devices). Covers
// PartitionedArray::moveTo/exchange (aether/residency/PartitionedArray.h)
// — device-to-device partition movement built on Copy.h's CUDA<->CUDA
// pair through a transport's route.
//
// Every test uses N=64 samples over 2 ranks at pad_to=32, so
// padded_==32==realCount(r) for both ranks (no short trailing block) —
// scatter()/gather() move exactly `padded` samples per rank with nothing
// left uninitialized, so a round trip through hostView() is a clean,
// bit-exact channel for verifying moveTo/exchange without dereferencing
// device memory from host code (deviceView()'s data is CUDA-resident;
// only a kernel or a host<->device aether::copy may touch it, mirroring
// test_PartitionedArray.cu's own convention of never reading a device
// View directly from a host test body).
//
// Single-GPU rows (same transport route, both ranks pinned to device 0)
// run for real in the normal gate; *TwoGpu* rows relocate/exchange across
// devices 0 and 1 and skip with a printed reason when
// cudaGetDeviceCount() < 2.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/residency/PeerAccess.h>

namespace aether_tests {
namespace {

using aether::Device;
using aether::PartitionedArray;
using aether::StreamTransport;

class PartitionedMoveTest : public ::testing::Test { };

TEST_F(PartitionedMoveTest, MoveToSameDeviceBitIdenticalRoundTrip)
{
    constexpr std::size_t N = 64;
    // Both ranks pinned to the SAME physical device (id 0) -- Route::SAME
    // for the transport's own decision, exercisable on a single GPU.
    std::vector<Device> devices{ Device(kDLCUDA, 0), Device(kDLCUDA, 0) };
    PartitionedArray<double, aether::dyn> pa(devices, N, 32);
    ASSERT_EQ(pa.padded(), 32u);
    ASSERT_EQ(pa.realCount(0), 32u);
    ASSERT_EQ(pa.realCount(1), 32u);

    std::vector<double> hostIn(N);
    for (std::size_t i = 0; i < N; ++i)
        hostIn[i] = static_cast<double>(i) * 1.25 - 5.0;
    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = hostIn[i];
    }
    pa.scatter();

    StreamTransport transport;
    EXPECT_EQ(transport.route(pa.device(0), Device(kDLCUDA, 0)), aether::Route::SAME);
    pa.moveTo(0, Device(kDLCUDA, 0), transport); // fresh allocation, SAME device id -- exercises the full relocate machinery
    EXPECT_EQ(pa.device(0), Device(kDLCUDA, 0));

    // Clobber the host canonical copy first, so a passing gather() PROVES
    // the relocated chunk still holds the right bytes (not that hostView()
    // never changed).
    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = -1.0;
    }
    pa.gather();
    auto hAfter = pa.hostView();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(hAfter(i), hostIn[i]) << "moveTo (same device) round trip: i=" << i;
}

TEST_F(PartitionedMoveTest, ExchangeSameDeviceBitIdenticalContentSwap)
{
    constexpr std::size_t N = 64;
    std::vector<Device> devices{ Device(kDLCUDA, 0), Device(kDLCUDA, 0) };
    PartitionedArray<double, aether::dyn> pa(devices, N, 32);

    std::vector<double> hostIn(N);
    for (std::size_t i = 0; i < 32; ++i) {
        hostIn[i]      = static_cast<double>(i); // rank 0's block
        hostIn[32 + i] = 1000.0 + static_cast<double>(i); // rank 1's block
    }
    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = hostIn[i];
    }
    pa.scatter();

    StreamTransport transport;
    pa.exchange(0, 1, transport);
    // exchange never reassigns devices -- both ranks stay on device 0.
    EXPECT_EQ(pa.device(0), Device(kDLCUDA, 0));
    EXPECT_EQ(pa.device(1), Device(kDLCUDA, 0));

    pa.gather();
    auto h = pa.hostView();
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(h(i), hostIn[32 + i]) << "exchange: rank0 slot now holds rank1's data, i=" << i;
        EXPECT_DOUBLE_EQ(h(32 + i), hostIn[i]) << "exchange: rank1 slot now holds rank0's data, i=" << i;
    }
}

TEST_F(PartitionedMoveTest, MoveToTwoGpuBitIdenticalRoundTrip)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    constexpr std::size_t N = 64;
    std::vector<Device> devices{ Device(kDLCUDA, 0), Device(kDLCUDA, 1) };
    PartitionedArray<double, aether::dyn> pa(devices, N, 32);

    std::vector<double> hostIn(N);
    for (std::size_t i = 0; i < N; ++i)
        hostIn[i] = static_cast<double>(i) * -0.75 + 12.0;
    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = hostIn[i];
    }
    pa.scatter();

    StreamTransport transport;
    EXPECT_EQ(transport.route(pa.device(0), Device(kDLCUDA, 1)), aether::Route::P2P); // P2P both ways on this box
    pa.moveTo(0, Device(kDLCUDA, 1), transport); // rank 0's block: device 0 -> device 1, genuinely cross-device
    EXPECT_EQ(pa.device(0), Device(kDLCUDA, 1));
    EXPECT_EQ(pa.device(1), Device(kDLCUDA, 1)); // unchanged -- moveTo touches only the named rank

    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = -1.0;
    }
    pa.gather();
    auto hAfter = pa.hostView();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(hAfter(i), hostIn[i]) << "moveTo (0 -> 1) round trip: i=" << i;
}

TEST_F(PartitionedMoveTest, ExchangeTwoGpuBitIdenticalContentSwap)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    constexpr std::size_t N = 64;
    std::vector<Device> devices{ Device(kDLCUDA, 0), Device(kDLCUDA, 1) };
    PartitionedArray<double, aether::dyn> pa(devices, N, 32);

    std::vector<double> hostIn(N);
    for (std::size_t i = 0; i < 32; ++i) {
        hostIn[i]      = 7.0 * static_cast<double>(i) + 1.0; // rank 0 (device 0)
        hostIn[32 + i] = -3.0 * static_cast<double>(i) - 2.0; // rank 1 (device 1)
    }
    {
        auto h = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            h(i) = hostIn[i];
    }
    pa.scatter();

    StreamTransport transport;
    pa.exchange(0, 1, transport); // device 0 <-> device 1, content swap; each rank's OWN device unchanged
    EXPECT_EQ(pa.device(0), Device(kDLCUDA, 0));
    EXPECT_EQ(pa.device(1), Device(kDLCUDA, 1));

    pa.gather();
    auto h = pa.hostView();
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_DOUBLE_EQ(h(i), hostIn[32 + i]) << "exchange (0<->1): rank0 slot, i=" << i;
        EXPECT_DOUBLE_EQ(h(32 + i), hostIn[i]) << "exchange (0<->1): rank1 slot, i=" << i;
    }
}

} // namespace
} // namespace aether_tests
