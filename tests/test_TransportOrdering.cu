// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// CUDA-only test for StreamTransport's ordering primitives (cudaEvent /
// wait() / fence()), which do not exist outside a CUDA backend.
// StreamTransport records a cudaEvent after every async movement and
// exposes wait(event)/fence(); the red-first witness is a test that
// removes the wait and observes a stale read.
//
// Two independent things are tested here:
//  1. EventFenceOrdersMultipleStreamsOnOneDevice -- deterministic,
//     always green: proves lastEvent()/wait()/fence() correctly order
//     several async moves issued on different streams through the same
//     transport.
//  2. The red-first ordering witness itself (OrderingWitness*): a
//     deliberately racy schedule -- a slow kernel (busy-spins on
//     clock64() for ~250ms) on a source stream overwrites a sentinel with
//     the "real" value, while an independent stream's copyAsync races to
//     read that same source concurrently. A single runtime env-var switch,
//     TRANSPORTS_RACY_ORDERING (any non-empty value = racy), selects
//     between the two arms of one test body:
//       - unset (the committed, default gate state): the test calls
//         transport.wait(producerEvent) before issuing the copy -- the
//         slow kernel is guaranteed complete first, so the read is
//         correct. Expected: green.
//       - set (TRANSPORTS_RACY_ORDERING=1): the same test body skips that
//         wait -- the copy on the independent stream starts within
//         microseconds of the kernel launch, long before a ~250ms busy
//         spin can finish, so the read observes the stale sentinel.
//         Expected: red (the EXPECT_DOUBLE_EQ below fails). Flipping the
//         arm is a separate, user-declared run of this same binary with
//         the environment variable set -- never by editing this file.
//     `*TwoGpu*`-tagged rows repeat the witness across devices 0 and 1
//     (P2P-routed copy), skipping with a printed reason when
//     cudaGetDeviceCount() < 2.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Copy.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/residency/Transport.h>

namespace aether_tests {
namespace {

class TransportOrderingTest : public ::testing::Test { };

/** @brief Busy-spins on `clock64()` for `spinCycles` device-clock ticks
 *         (~250ms on this box's Pascal sm_61 parts at their ~1.3GHz base
 *         clock) before writing
 *         `value` to `*ptr` -- this is the "slow kernel on the source
 *         stream" used as the ordering witness's schedule. */
__global__ void slowFillKernel(double* ptr, double value, long long spinCycles)
{
    const long long start = clock64();
    while (clock64() - start < spinCycles) {
        // deliberately busy-spin: simulates "producer still computing"
    }
    *ptr = value;
}

/** @brief ~250ms of spin at a conservative 1.0GHz device clock lower
 *         bound -- several orders of magnitude longer than a
 *         cudaMemcpyAsync's enqueue-to-start latency on an independent
 *         stream, so the race this file engineers is reliable, not
 *         probabilistic (per the workspace's own "idle GPU green is not
 *         evidence -- MANUFACTURE the adversarial schedule" lesson). */
constexpr long long kSpinCycles = 250'000'000LL;

constexpr double kSentinel = -1.0;
constexpr double kCorrect  = 42.5;

/** @brief `true` iff this run has been flipped to the RACY
 *         arm (TRANSPORTS_RACY_ORDERING set to any non-empty value). */
bool racyOrderingRequested()
{
    const char* v = std::getenv("TRANSPORTS_RACY_ORDERING");
    return v != nullptr && v[0] != '\0';
}

// ---------------------------------------------------------------------
// 1. Deterministic fence semantics (single-GPU row).
// ---------------------------------------------------------------------
TEST_F(TransportOrderingTest, EventFenceOrdersMultipleStreamsOnOneDevice)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const aether::Device dev0(kDLCUDA, 0);
    const aether::Device host(kDLCUDAHost);

    constexpr std::size_t N = 512;
    std::vector<unsigned char> patternA(N), patternB(N);
    for (std::size_t i = 0; i < N; ++i) {
        patternA[i] = static_cast<unsigned char>((i * 13 + 1) & 0xFF);
        patternB[i] = static_cast<unsigned char>((i * 29 + 5) & 0xFF);
    }

    auto srcAHost = aether::Chunk::allocate(host, N);
    auto srcBHost = aether::Chunk::allocate(host, N);
    std::memcpy(srcAHost.data(), patternA.data(), N);
    std::memcpy(srcBHost.data(), patternB.data(), N);
    auto srcA = aether::Chunk::allocate(dev0, N);
    auto srcB = aether::Chunk::allocate(dev0, N);
    aether::copy(srcA, srcAHost);
    aether::copy(srcB, srcBHost);

    auto dstA = aether::Chunk::allocate(dev0, N);
    auto dstB = aether::Chunk::allocate(dev0, N);

    cudaStream_t streamA, streamB;
    ASSERT_EQ(cudaStreamCreate(&streamA), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streamB), cudaSuccess);

    aether::StreamTransport transport;
    EXPECT_EQ(transport.lastEvent(), nullptr) << "no event before any move";
    transport.copyAsync(dstA, srcA, streamA);
    cudaEvent_t evA = transport.lastEvent();
    ASSERT_NE(evA, nullptr);
    transport.copyAsync(dstB, srcB, streamB);
    cudaEvent_t evB = transport.lastEvent();
    ASSERT_NE(evB, nullptr);
    EXPECT_NE(evA, evB) << "each async move records its OWN event";

    transport.fence(); // must wait on BOTH streams' events, not just the last

    auto readA = aether::Chunk::allocate(host, N);
    auto readB = aether::Chunk::allocate(host, N);
    aether::copy(readA, dstA);
    aether::copy(readB, dstB);
    EXPECT_EQ(std::memcmp(readA.data(), patternA.data(), N), 0) << "stream A's move";
    EXPECT_EQ(std::memcmp(readB.data(), patternB.data(), N), 0) << "stream B's move";

    ASSERT_EQ(cudaStreamDestroy(streamA), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streamB), cudaSuccess);
}

// ---------------------------------------------------------------------
// 2. The RED-first ordering witness.
// ---------------------------------------------------------------------
TEST_F(TransportOrderingTest, OrderingWitnessRedWithoutWaitGreenWithWait)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const aether::Device dev0(kDLCUDA, 0);
    const aether::Device host(kDLCUDAHost);
    const bool racy = racyOrderingRequested();

    auto srcHostSeed = aether::Chunk::allocate(host, sizeof(double));
    *reinterpret_cast<double*>(srcHostSeed.data()) = kSentinel;
    auto src = aether::Chunk::allocate(dev0, sizeof(double));
    aether::copy(src, srcHostSeed); // src now holds the SENTINEL, synchronously -- before the race begins

    cudaStream_t streamA, streamB; // A: the "producer" (slow kernel); B: the racing consumer copy
    ASSERT_EQ(cudaStreamCreate(&streamA), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streamB), cudaSuccess);

    cudaEvent_t producerEvent;
    ASSERT_EQ(cudaEventCreate(&producerEvent), cudaSuccess);

    slowFillKernel<<<1, 1, 0, streamA>>>(reinterpret_cast<double*>(src.data()), kCorrect, kSpinCycles);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "slowFillKernel launch (non-blocking check)";
    ASSERT_EQ(cudaEventRecord(producerEvent, streamA), cudaSuccess);

    aether::StreamTransport transport;
    if (!racy) {
        // GREEN arm (the committed default): block until the slow kernel
        // has genuinely finished before reading its output.
        transport.wait(producerEvent);
    }
    // RACY arm (TRANSPORTS_RACY_ORDERING set): no wait
    // -- streamB's copy is enqueued while streamA's kernel is almost
    // certainly still spinning.
    auto dstHost = aether::Chunk::allocate(host, sizeof(double));
    transport.copyAsync(dstHost, src, streamB);
    ASSERT_EQ(cudaStreamSynchronize(streamB), cudaSuccess); // wait for streamB's OWN copy to land, racy or not

    const double observed = *reinterpret_cast<const double*>(dstHost.data());
    EXPECT_DOUBLE_EQ(observed, kCorrect)
        << (racy ? "RACY arm (TRANSPORTS_RACY_ORDERING set): EXPECTED TO FAIL -- this demonstrates the "
                   "witness caught a stale read; observed the SENTINEL because the wait() was skipped."
                 : "GREEN arm (default): transport.wait(producerEvent) must make this deterministic.");

    ASSERT_EQ(cudaStreamSynchronize(streamA), cudaSuccess); // drain the slow kernel before teardown regardless of arm
    ASSERT_EQ(cudaEventDestroy(producerEvent), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streamA), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streamB), cudaSuccess);
}

TEST_F(TransportOrderingTest, OrderingWitnessAcrossDevicesTwoGpuRedWithoutWaitGreenWithWait)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    const aether::Device dev0(kDLCUDA, 0);
    const aether::Device dev1(kDLCUDA, 1);
    const aether::Device host(kDLCUDAHost);
    const bool racy = racyOrderingRequested();

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto srcHostSeed = aether::Chunk::allocate(host, sizeof(double));
    *reinterpret_cast<double*>(srcHostSeed.data()) = kSentinel;
    auto src = aether::Chunk::allocate(dev0, sizeof(double));
    aether::copy(src, srcHostSeed);

    cudaStream_t streamA; // producer, lives on device 0 (where the slow kernel runs)
    ASSERT_EQ(cudaStreamCreate(&streamA), cudaSuccess);
    cudaEvent_t producerEvent;
    ASSERT_EQ(cudaEventCreate(&producerEvent), cudaSuccess);

    slowFillKernel<<<1, 1, 0, streamA>>>(reinterpret_cast<double*>(src.data()), kCorrect, kSpinCycles);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "slowFillKernel launch (non-blocking check)";
    ASSERT_EQ(cudaEventRecord(producerEvent, streamA), cudaSuccess);

    aether::StreamTransport transport; // route(dev0,dev1) == P2P on this box
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    cudaStream_t streamB; // consumer, races the slow kernel from an INDEPENDENT device/stream
    ASSERT_EQ(cudaStreamCreate(&streamB), cudaSuccess);

    if (!racy)
        transport.wait(producerEvent);

    auto dstDev1 = aether::Chunk::allocate(dev1, sizeof(double));
    transport.copyAsync(dstDev1, src, streamB); // P2P device 0 -> device 1
    ASSERT_EQ(cudaStreamSynchronize(streamB), cudaSuccess);

    auto dstHost = aether::Chunk::allocate(host, sizeof(double));
    aether::copy(dstHost, dstDev1);
    const double observed = *reinterpret_cast<const double*>(dstHost.data());
    EXPECT_DOUBLE_EQ(observed, kCorrect)
        << (racy ? "RACY arm (TRANSPORTS_RACY_ORDERING set): EXPECTED TO FAIL across devices -- stale "
                   "P2P read because the wait() was skipped."
                 : "GREEN arm (default): transport.wait(producerEvent) must make the cross-device read deterministic.");

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streamA), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(producerEvent), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streamA), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streamB), cudaSuccess);
}

} // namespace
} // namespace aether_tests
