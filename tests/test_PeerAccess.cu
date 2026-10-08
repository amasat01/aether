// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// PeerAccess tests (CUDA build; test_PeerAccess.cpp is the host-build
// twin). Covers aether::Route / aether::peerRoute /
// StreamTransport::route() over real CUDA devices: route enum for (0,0);
// forced-staging bit-identity vs a direct copy on one GPU (Route::SAME
// forced to Route::STAGED); the *TwoGpu* P2P route verdict and the
// forced-staging-despite-P2P bit-identity check across devices 0 and 1.
//
// Every *TwoGpu* row self-skips (GTEST_SKIP) when
// cudaGetDeviceCount() < 2.

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Copy.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/residency/PeerAccess.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class PeerAccessTest : public ::testing::Test { };

TEST_F(PeerAccessTest, RouteForSameDeviceIsSame)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const aether::Device dev0(kDLCUDA, 0);
    EXPECT_EQ(aether::peerRoute(dev0, dev0), aether::Route::SAME);

    aether::StreamTransport transport;
    EXPECT_EQ(transport.route(dev0, dev0), aether::Route::SAME);
}

// Single-GPU forced-staging exercise: forceStaging overrides SAME ->
// STAGED even for a single device, letting the STAGED
// path (device -> pinned host bounce -> device, StreamTransport's own
// BounceBuffer) be bit-compared against a direct aether::copy WITHOUT a
// second physical GPU.
TEST_F(PeerAccessTest, ForcedStagingOnOneDeviceIsBitIdenticalToADirectCopy)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr std::size_t N = 4096;
    const aether::Device dev0(kDLCUDA, 0);

    std::vector<unsigned char> pattern(N);
    for (std::size_t i = 0; i < N; ++i)
        pattern[i] = static_cast<unsigned char>((i * 41 + 7) & 0xFF);

    auto srcHost = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    std::memcpy(srcHost.data(), pattern.data(), N);
    auto src = aether::Chunk::allocate(dev0, N);
    aether::copy(src, srcHost); // seed the device source

    // --- Reference: a direct aether::copy (Route bypassed entirely). ---
    auto refDst = aether::Chunk::allocate(dev0, N);
    aether::copy(refDst, src);
    auto refReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(refReadback, refDst);

    // --- Forced-staging transport: SAME device, but forceStaging=true routes STAGED. ---
    aether::StreamTransport transport(/*forceStaging=*/true);
    ASSERT_EQ(transport.route(dev0, dev0), aether::Route::STAGED);
    auto stagedDst = aether::Chunk::allocate(dev0, N);
    transport.copy(stagedDst, src);
    auto stagedReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(stagedReadback, stagedDst);

    EXPECT_EQ(std::memcmp(stagedReadback.data(), refReadback.data(), N), 0)
        << "forced-staging (device->pinned bounce->device) must be bit-identical to a direct copy";
    EXPECT_EQ(std::memcmp(stagedReadback.data(), pattern.data(), N), 0);
}

TEST_F(PeerAccessTest, RouteForDistinctDevicesTwoGpuIsP2p)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    const aether::Device dev0(kDLCUDA, 0);
    const aether::Device dev1(kDLCUDA, 1);
    // Two-GPU test box: cudaDeviceCanAccessPeer(0->1)=1, (1->0)=1.
    EXPECT_EQ(aether::peerRoute(dev0, dev1), aether::Route::P2P);
    EXPECT_EQ(aether::peerRoute(dev1, dev0), aether::Route::P2P);
}

TEST_F(PeerAccessTest, ForcedStagingTwoGpuBitIdenticalDespiteP2p)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    constexpr std::size_t N = 4096;
    const aether::Device dev0(kDLCUDA, 0);
    const aether::Device dev1(kDLCUDA, 1);

    std::vector<unsigned char> pattern(N);
    for (std::size_t i = 0; i < N; ++i)
        pattern[i] = static_cast<unsigned char>((i * 97 + 3) & 0xFF);

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto srcHost = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    std::memcpy(srcHost.data(), pattern.data(), N);
    auto src = aether::Chunk::allocate(dev0, N);
    aether::copy(src, srcHost);

    // --- Reference: default transport, no forceStaging -> Route::P2P. ---
    aether::StreamTransport direct;
    ASSERT_EQ(direct.route(dev0, dev1), aether::Route::P2P);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    auto refDst = aether::Chunk::allocate(dev1, N);
    direct.copy(refDst, src);
    auto refReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(refReadback, refDst);

    // --- forceStaging=true: STAGED despite P2P being available. ---
    aether::StreamTransport staged(/*forceStaging=*/true);
    ASSERT_EQ(staged.route(dev0, dev1), aether::Route::STAGED);
    auto stagedDst = aether::Chunk::allocate(dev1, N);
    staged.copy(stagedDst, src);
    auto stagedReadback = aether::Chunk::allocate(aether::Device(kDLCUDAHost), N);
    aether::copy(stagedReadback, stagedDst);

    EXPECT_EQ(std::memcmp(stagedReadback.data(), refReadback.data(), N), 0)
        << "forced-staging across devices must be bit-identical to the P2P direct path despite P2P being available";
    EXPECT_EQ(std::memcmp(stagedReadback.data(), pattern.data(), N), 0);
}

} // namespace
} // namespace aether_tests
