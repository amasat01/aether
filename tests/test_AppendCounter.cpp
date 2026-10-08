// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::accum::AppendCounter` tests on the CPU host path
// (AETHER_CPP_MODE). Mirrors test_AppendCounter.cu's device battery (house
// convention — the two files exercise the same contract through each
// mode's own concurrency primitive: OpenMP here, a CUDA kernel there).

#include <cstddef>
#include <vector>

#include <omp.h>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::accum::AppendCounter;
using aether::accum::npos;

class AppendCounterTest : public ::testing::Test { };

TEST_F(AppendCounterTest, SequentialTryAppendFillsSlotsInOrder)
{
    AppendCounter counter(8);
    auto view = counter.deviceView();
    for (std::size_t k = 0; k < 8; ++k) {
        const aether::offset_t slot = view.tryAppend();
        EXPECT_EQ(slot, static_cast<aether::offset_t>(k));
    }
    EXPECT_EQ(counter.rawCount(), 8u);
    EXPECT_FALSE(counter.overflowed());
}

TEST_F(AppendCounterTest, ConcurrentTryAppendProducesExactCountAndEverySlotOnce)
{
    constexpr std::size_t kCapacity = 4096;
    AppendCounter counter(kCapacity);
    auto view = counter.deviceView();
    std::vector<int> writes(kCapacity, 0);

#pragma omp parallel for
    for (std::size_t k = 0; k < kCapacity; ++k) {
        const aether::offset_t slot = view.tryAppend();
        EXPECT_NE(slot, npos);
        EXPECT_LT(static_cast<std::size_t>(slot), kCapacity);
#pragma omp atomic
        writes[slot] += 1;
    }

    EXPECT_EQ(counter.rawCount(), kCapacity);
    EXPECT_FALSE(counter.overflowed());
    for (std::size_t k = 0; k < kCapacity; ++k)
        EXPECT_EQ(writes[k], 1) << "slot " << k;
}

TEST_F(AppendCounterTest, OverflowSetsFlagAndReturnsNposWithoutExceedingCapacity)
{
    constexpr std::size_t kCapacity = 32;
    constexpr std::size_t kThreads  = 256; // deliberately > capacity
    AppendCounter counter(kCapacity);
    auto view = counter.deviceView();
    std::vector<int> writes(kCapacity, 0);
    int overflowCount = 0;

#pragma omp parallel for
    for (std::size_t k = 0; k < kThreads; ++k) {
        const aether::offset_t slot = view.tryAppend();
        if (slot == npos) {
#pragma omp atomic
            ++overflowCount;
        } else {
            EXPECT_LT(static_cast<std::size_t>(slot), kCapacity);
#pragma omp atomic
            writes[slot] += 1;
        }
    }

    EXPECT_TRUE(counter.overflowed());
    EXPECT_EQ(static_cast<std::size_t>(overflowCount), kThreads - kCapacity);
    for (std::size_t k = 0; k < kCapacity; ++k)
        EXPECT_EQ(writes[k], 1) << "slot " << k;
    EXPECT_THROW(counter.checkOverflow("AppendCounterTest"), aether::Error);
    EXPECT_FALSE(counter.overflowed()); // checkOverflow cleared it
}

TEST_F(AppendCounterTest, CommitClampsAndAdoptsIntoArray)
{
    aether::Array<double, 3> arr(0);
    arr.reserve(10); // -> capacity 32
    AppendCounter counter(arr.capacity());
    auto view = counter.deviceView();
    auto sp    = arr.hostSpare();

    for (int k = 0; k < 5; ++k) {
        const aether::offset_t slot = view.tryAppend();
        ASSERT_NE(slot, npos);
        sp(0, slot) = static_cast<double>(k);
        sp(1, slot) = static_cast<double>(k) * 2.0;
        sp(2, slot) = static_cast<double>(k) * 3.0;
    }

    const std::size_t committed = counter.commit(arr);
    EXPECT_EQ(committed, 5u);
    EXPECT_EQ(arr.samples(), 5u);

    auto v = arr.hostView();
    for (std::size_t k = 0; k < 5; ++k) {
        EXPECT_EQ(v(0, k), static_cast<double>(k));
        EXPECT_EQ(v(1, k), static_cast<double>(k) * 2.0);
        EXPECT_EQ(v(2, k), static_cast<double>(k) * 3.0);
    }
    // commit() resets the counter for the next step.
    EXPECT_EQ(counter.rawCount(), 0u);
}

TEST_F(AppendCounterTest, ResetZerosCounterForNextStep)
{
    AppendCounter counter(16);
    auto view = counter.deviceView();
    for (int k = 0; k < 4; ++k)
        view.tryAppend();
    ASSERT_EQ(counter.rawCount(), 4u);
    counter.reset();
    EXPECT_EQ(counter.rawCount(), 0u);
}

} // namespace
} // namespace aether_tests
