// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Race-correctness tests for `aether::accum::atomic` primitives on the CPU
// host path (AETHER_CPP_MODE). Mirrors test_Atomic.cu (house convention —
// see test_ArrayCapacity.cpp/.cu).

#include <cmath>
#include <vector>

#include <omp.h>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

/* ------------------------------------------------------------------ */
/* minAbsReal                                                          */
/* ------------------------------------------------------------------ */

class AtomicMinAbsRealTest : public ::testing::Test {
protected:
    static constexpr std::size_t N = 4096;
};

TEST_F(AtomicMinAbsRealTest, ConcurrentMinPreservesSign)
{
    double target = 100.0;
    std::vector<double> cand(N);
    for (std::size_t k = 0; k < N; ++k) {
        const double mag = 1.0 / static_cast<double>(k + 1);
        cand[k]          = (k % 2 == 0) ? mag : -mag;
    }
    cand[0] = -0.5;

#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k) {
        aether::accum::atomic::minAbsReal(&target, cand[k]);
    }

    const double expectedMag = 1.0 / static_cast<double>(N);
    EXPECT_DOUBLE_EQ(std::abs(target), expectedMag);
    EXPECT_GT(target, 0.0) << "Sign of initial value must be preserved";
}

TEST_F(AtomicMinAbsRealTest, NoUpdateWhenLargerMagnitude)
{
    double target = -1e-12;
    std::vector<double> cand(N);
    for (std::size_t k = 0; k < N; ++k)
        cand[k] = (k % 2 == 0) ? 1.0 : -2.0;

#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k) {
        aether::accum::atomic::minAbsReal(&target, cand[k]);
    }

    EXPECT_DOUBLE_EQ(target, -1e-12);
}

/* ------------------------------------------------------------------ */
/* orBool                                                              */
/* ------------------------------------------------------------------ */

class AtomicOrBoolTest : public ::testing::Test { };

TEST_F(AtomicOrBoolTest, SingleSlot_AllWritersTrue_ResultTrue)
{
    bool target = false;
#pragma omp parallel for
    for (std::size_t k = 0; k < 1024; ++k) {
        aether::accum::atomic::orBool(&target, true);
    }
    EXPECT_TRUE(target);
}

TEST_F(AtomicOrBoolTest, SingleSlot_AllWritersFalse_ResultFalse)
{
    bool target = false;
#pragma omp parallel for
    for (std::size_t k = 0; k < 1024; ++k) {
        aether::accum::atomic::orBool(&target, false);
    }
    EXPECT_FALSE(target);
}

TEST_F(AtomicOrBoolTest, AdjacentSlots_NoCrossCorruption)
{
    /* On the host path the byte is its own atomic store-of-1 (no word
     * widening), so neighbour stomping is structurally impossible — but
     * still exercise the same scenario for parity with the CUDA test, and
     * to guard against any future implementation drift. */
    constexpr std::size_t SLOTS            = 16;
    constexpr std::size_t WRITERS_PER_SLOT = 64;
    bool slots[SLOTS] = { false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false };

    const std::size_t total = SLOTS * WRITERS_PER_SLOT;
#pragma omp parallel for
    for (std::size_t i = 0; i < total; ++i) {
        const std::size_t slot = i % SLOTS;
        const bool writeTrue   = (slot % 2 == 0);
        aether::accum::atomic::orBool(slots + slot, writeTrue);
    }

    for (std::size_t k = 0; k < SLOTS; ++k) {
        if (k % 2 == 0)
            EXPECT_TRUE(slots[k]) << "even slot " << k << " should be true";
        else
            EXPECT_FALSE(slots[k]) << "odd slot " << k << " was corrupted by neighbour OR";
    }
}

/* ------------------------------------------------------------------ */
/* maxInt                                                              */
/* ------------------------------------------------------------------ */

class AtomicMaxIntTest : public ::testing::Test { };

TEST_F(AtomicMaxIntTest, ConcurrentMaxIsCorrect)
{
    constexpr std::size_t N = 4096;
    int target               = -1;
    std::vector<int> cand(N);
    for (std::size_t k = 0; k < N; ++k)
        cand[k] = static_cast<int>(k * 3 - 100);
    cand[N / 3] = 999999;

#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k) {
        aether::accum::atomic::maxInt(&target, cand[k]);
    }

    EXPECT_EQ(target, 999999);
}

} // namespace
} // namespace aether_tests
