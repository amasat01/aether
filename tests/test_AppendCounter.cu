// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::accum::AppendCounter` tests (CUDA mode). Mirrors
// test_AppendCounter.cpp's host battery (house convention). Many
// threads race `View::tryAppend()` into an `Array`'s `deviceSpare()`:
//
//   * exact count and every slot written exactly once, within capacity;
//   * overflow -> the `DeviceFlag` is set, the raw count is CLAMPED at
//     `commit()`, and NO write ever lands past the DECLARED capacity — a
//     canary word placed one slot past it (still inside the `Array`'s own,
//     larger, real allocation) is checked untouched.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::accum::AppendCounter;
using aether::accum::AppendCounterView;
using aether::accum::npos;
using ScalarArrView = aether::Array<double>::ViewT;

AETHER_KERNEL()
void appendFillKernel(AppendCounterView counter, ScalarArrView spare, int* writeCounts, double base)
{
    const aether::SampleIndex i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    const aether::offset_t slot  = counter.tryAppend();
    if (slot == npos)
        return;
    spare(slot)          = base + static_cast<double>(i.global());
    atomicAdd(&writeCounts[slot], 1);
}

class AppendCounterTest : public ::testing::Test { };

TEST_F(AppendCounterTest, ManyThreadsAppendIntoSpareCapacityExactCount)
{
    constexpr std::size_t kCapacity = 4096;
    aether::Array<double> arr(0);
    arr.reserve(kCapacity); // capacity is already a multiple of kCapacityAlignment
    ASSERT_EQ(arr.capacity(), kCapacity);
    AppendCounter counter(arr.capacity());

    int* writeCounts;
    cudaMalloc(&writeCounts, kCapacity * sizeof(int));
    cudaMemset(writeCounts, 0, kCapacity * sizeof(int));

    const int threads     = static_cast<int>(kCapacity);
    const int blockSize    = 256;
    const int nBlocks      = (threads + blockSize - 1) / blockSize;
    appendFillKernel<<<nBlocks, blockSize>>>(counter.deviceView(), arr.deviceSpare(), writeCounts, 0.0);
    aether::cuda::checkLastLaunch("appendFillKernel");

    EXPECT_EQ(counter.rawCount(), kCapacity);
    EXPECT_FALSE(counter.overflowed());

    std::vector<int> hostCounts(kCapacity, 0);
    cudaMemcpy(hostCounts.data(), writeCounts, kCapacity * sizeof(int), cudaMemcpyDeviceToHost);
    for (std::size_t k = 0; k < kCapacity; ++k)
        EXPECT_EQ(hostCounts[k], 1) << "slot " << k;

    const std::size_t committed = counter.commit(arr);
    EXPECT_EQ(committed, kCapacity);
    EXPECT_EQ(arr.samples(), kCapacity);

    cudaFree(writeCounts);
}

TEST_F(AppendCounterTest, OverflowSetsFlagClampsCountAndCanaryUntouched)
{
    constexpr std::size_t kDeclaredCapacity = 32;
    constexpr int kThreads                    = 512; // deliberately far past kDeclaredCapacity
    constexpr double kCanarySentinel           = -999.0;

    aether::Array<double> arr(0);
    arr.reserve(kDeclaredCapacity + 1); // real capacity rounds UP past kDeclaredCapacity (kCapacityAlignment=32 -> 64)
    ASSERT_GT(arr.capacity(), kDeclaredCapacity)
        << "the test needs real allocated room past the DECLARED AppendCounter capacity for the canary";

    // Seed the canary at index kDeclaredCapacity of the spare tail (still
    // inside the array's real, larger allocation) and upload.
    {
        auto hsp                    = arr.hostSpare();
        hsp(kDeclaredCapacity)       = kCanarySentinel;
        arr.upload();
    }

    AppendCounter counter(kDeclaredCapacity); // deliberately LESS than arr.capacity()

    int* writeCounts;
    cudaMalloc(&writeCounts, kDeclaredCapacity * sizeof(int));
    cudaMemset(writeCounts, 0, kDeclaredCapacity * sizeof(int));

    appendFillKernel<<<2, 256>>>(counter.deviceView(), arr.deviceSpare(), writeCounts, 1000.0);
    aether::cuda::checkLastLaunch("appendFillKernel");
    ASSERT_EQ(kThreads, 2 * 256);

    EXPECT_TRUE(counter.overflowed());
    EXPECT_GT(counter.rawCount(), kDeclaredCapacity) << "every overflowing thread still incremented the raw counter";

    std::vector<int> hostCounts(kDeclaredCapacity, 0);
    cudaMemcpy(hostCounts.data(), writeCounts, kDeclaredCapacity * sizeof(int), cudaMemcpyDeviceToHost);
    for (std::size_t k = 0; k < kDeclaredCapacity; ++k)
        EXPECT_EQ(hostCounts[k], 1) << "slot " << k << " -- every declared slot written exactly once";

    const std::size_t committed = counter.commit(arr); // clamps to kDeclaredCapacity, does not throw
    EXPECT_EQ(committed, kDeclaredCapacity);
    EXPECT_EQ(arr.samples(), kDeclaredCapacity);

    // The canary, one slot PAST the declared capacity, must be untouched.
    arr.download();
    cudaDeviceSynchronize();
    auto after = arr.hostSpare(); // [samples(), capacity()) = [32, arr.capacity())
    EXPECT_EQ(after(0), kCanarySentinel) << "a write landed past the AppendCounter's DECLARED capacity";

    EXPECT_THROW(counter.checkOverflow("AppendCounterTest"), aether::Error);
    EXPECT_FALSE(counter.overflowed());

    cudaFree(writeCounts);
}

} // namespace
} // namespace aether_tests
