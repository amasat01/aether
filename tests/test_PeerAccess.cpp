// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tests/test_PeerAccess.cpp (host / AETHER_CPP_MODE build).
// Paired with test_PeerAccess.cu (house convention — see test_Chunk.*).
// Covers aether::Route / aether::peerRoute in a build with NO CUDA backend:
// the API compiles, routes are always HOST, and the D2D rows are SKIP with
// reason.
//
// PartitionedArray::moveTo/exchange (aether/residency/PartitionedArray.h)
// have NO dedicated .cpp test file
// (tests/test_PartitionedMove*.cu covers this for CUDA only)
// — this file also carries their CPP_MODE compile+correctness proof, over
// kDLCPU stand-in "devices" (same convention as test_Replica.cpp's
// ConstructsOneChunkPerDeviceWithTheRequestedSize: Chunk::allocate
// dispatches on device KIND, not id, and this build has only ONE kind),
// routed through the default StreamTransport's Route::HOST path
// (aether::copy's CpuCpu leg — a plain memcpy).

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/residency/PartitionedArray.h>
#include <aether/residency/PeerAccess.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class PeerAccessTest : public ::testing::Test { };

TEST_F(PeerAccessTest, RouteIsAlwaysHostInCppModeRegardlessOfDeviceIds)
{
    // Distinct DEVICE IDS stand in for distinct devices (same convention
    // as test_Replica.cpp) — this build has only kDLCPU, so peerRoute's
    // AETHER_CPP_MODE branch must apply unconditionally, cross-device AND
    // same-device alike.
    EXPECT_EQ(aether::peerRoute(aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1)), aether::Route::HOST);
    EXPECT_EQ(aether::peerRoute(aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 0)), aether::Route::HOST);
}

TEST_F(PeerAccessTest, ForceStagingIsIgnoredInCppModeRouteStillHost)
{
    // forceStaging is a CUDA<->CUDA-pair-only override — this build has
    // no such pair to force; peerRoute's AETHER_CPP_MODE
    // branch ignores the argument entirely.
    EXPECT_EQ(aether::peerRoute(aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1), /*forceStaging=*/true),
        aether::Route::HOST);
}

TEST_F(PeerAccessTest, D2dMovementIsSkippedInCppMode)
{
    GTEST_SKIP() << "CPP_MODE (host-only) build: no CUDA backend, so genuine device-to-device "
                    "peer/staged movement does not apply here -- aether::peerRoute always returns "
                    "Route::HOST (see RouteIsAlwaysHostInCppModeRegardlessOfDeviceIds); the real "
                    "P2P/STAGED/SAME exercise lives in tests/test_PeerAccess.cu + "
                    "tests/test_PartitionedMove.cu, CUDA builds only.";
}

// PartitionedArray::moveTo/exchange CPP_MODE proof (see file header). N=64
// over 2 ranks at pad_to=32 gives padded_==64/2==32 EXACTLY, so
// realCount(r)==padded() for BOTH ranks (no short trailing block) — every
// deviceView(r) is then a full, fully-INITIALIZED 32-element window, safe
// for a straight memcmp bit-identity check (no uninitialized padding tail
// to accidentally compare, unlike a short-block partition).
TEST_F(PeerAccessTest, PartitionedArrayMoveToAndExchangeCompileAndWorkOverCpuStandInDevices)
{
    using aether::Device;
    using aether::PartitionedArray;
    using aether::StreamTransport;

    constexpr std::size_t N = 64;
    std::vector<Device> devices{ Device(kDLCPU, 0), Device(kDLCPU, 1) };
    PartitionedArray<double, aether::dyn> pa(devices, N, 32);
    ASSERT_EQ(pa.padded(), 32u);
    ASSERT_EQ(pa.realCount(0), 32u);
    ASSERT_EQ(pa.realCount(1), 32u);

    {
        auto host = pa.hostView();
        for (std::size_t i = 0; i < N; ++i)
            host(i) = static_cast<double>(i) * 2.5 - 3.0;
    }
    pa.scatter();

    std::vector<double> rank0Before(32), rank1Before(32);
    {
        auto v0 = pa.deviceView(0);
        auto v1 = pa.deviceView(1);
        for (std::size_t i = 0; i < 32; ++i) {
            rank0Before[i] = v0(i);
            rank1Before[i] = v1(i);
        }
    }

    StreamTransport transport;
    EXPECT_EQ(transport.route(devices[0], devices[1]), aether::Route::HOST);

    // moveTo: relocate rank 0's ENTIRE block onto a THIRD stand-in device — bytes preserved.
    pa.moveTo(0, Device(kDLCPU, 2), transport);
    EXPECT_EQ(pa.device(0), Device(kDLCPU, 2));
    {
        auto v0 = pa.deviceView(0);
        ASSERT_EQ(v0.samples(), 32u);
        for (std::size_t i = 0; i < 32; ++i)
            EXPECT_DOUBLE_EQ(v0(i), rank0Before[i]) << "moveTo: i=" << i;
    }

    // exchange: swap ranks 0/1's CONTENTS; each rank's OWN device is unchanged by exchange.
    pa.exchange(0, 1, transport);
    EXPECT_EQ(pa.device(0), Device(kDLCPU, 2)); // exchange never reassigns devices (moveTo already did, above)
    EXPECT_EQ(pa.device(1), devices[1]);
    {
        auto v0 = pa.deviceView(0);
        auto v1 = pa.deviceView(1);
        for (std::size_t i = 0; i < 32; ++i) {
            EXPECT_DOUBLE_EQ(v0(i), rank1Before[i]) << "exchange: rank0 slot, i=" << i;
            EXPECT_DOUBLE_EQ(v1(i), rank0Before[i]) << "exchange: rank1 slot, i=" << i;
        }
    }

    // exchange(r, r) is a documented no-op.
    pa.exchange(1, 1, transport);
    {
        auto v1 = pa.deviceView(1);
        for (std::size_t i = 0; i < 32; ++i)
            EXPECT_DOUBLE_EQ(v1(i), rank0Before[i]) << "exchange(r,r) no-op: i=" << i;
    }
}

TEST_F(PeerAccessTest, MoveToOutOfRangeRankThrows)
{
    using aether::Device;
    using aether::PartitionedArray;
    using aether::StreamTransport;

    std::vector<Device> devices{ Device(kDLCPU, 0) };
    PartitionedArray<double, aether::dyn> pa(devices, 8);
    StreamTransport transport;
    EXPECT_THROW(pa.moveTo(1, Device(kDLCPU, 1), transport), aether::Error);
}

TEST_F(PeerAccessTest, ExchangeOutOfRangeRankThrows)
{
    using aether::Device;
    using aether::PartitionedArray;
    using aether::StreamTransport;

    std::vector<Device> devices{ Device(kDLCPU, 0) };
    PartitionedArray<double, aether::dyn> pa(devices, 8);
    StreamTransport transport;
    EXPECT_THROW(pa.exchange(0, 1, transport), aether::Error);
}

} // namespace
} // namespace aether_tests
