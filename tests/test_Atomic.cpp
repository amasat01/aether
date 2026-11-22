// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Race-correctness tests for `aether::accum::atomic` primitives on the CPU
// host path (AETHER_CPP_MODE). Mirrors test_Atomic.cu (house convention —
// see test_ArrayCapacity.cpp/.cu).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
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

/* ------------------------------------------------------------------ */
/* element updates: minAt, maxAt, andAt, xorAt, fetchAddAt, exchangeAt, */
/* compareExchangeAt. Reference: the sequential fold with std::min,     */
/* std::max and the bit operators; the folds do not depend on the order */
/* the threads arrive in, so the result is exact.                       */
/* ------------------------------------------------------------------ */

class AtomicElementTest : public ::testing::Test {
protected:
    static constexpr std::size_t N = 4096;

    /** Pseudo-random candidates spread over both signs. */
    template<class E>
    static std::vector<E> candidates()
    {
        std::vector<E> out(N);
        std::uint64_t state = 88172645463325252ull;
        for (auto& x : out) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x = static_cast<E>(static_cast<std::int64_t>(state % 20001) - 10000);
        }
        return out;
    }

    template<class E>
    static void checkMinMax()
    {
        const auto cand = candidates<E>();
        E lo = static_cast<E>(5), hi = static_cast<E>(-5);
#pragma omp parallel for
        for (std::size_t k = 0; k < N; ++k) {
            aether::accum::atomic::minAt(lo, cand[k]);
            aether::accum::atomic::maxAt(hi, cand[k]);
        }
        EXPECT_EQ(lo, std::min(static_cast<E>(5), *std::min_element(cand.begin(), cand.end())));
        EXPECT_EQ(hi, std::max(static_cast<E>(-5), *std::max_element(cand.begin(), cand.end())));
    }

    template<class E>
    static void checkBits()
    {
        const auto cand = candidates<E>();
        E all = static_cast<E>(~E(0)), flip = E(0);
        E wantAll = all, wantFlip = flip;
        for (const E x : cand) {
            wantAll  = static_cast<E>(wantAll & x);
            wantFlip = static_cast<E>(wantFlip ^ x);
        }
#pragma omp parallel for
        for (std::size_t k = 0; k < N; ++k) {
            aether::accum::atomic::andAt(all, cand[k]);
            aether::accum::atomic::xorAt(flip, cand[k]);
        }
        EXPECT_EQ(all, wantAll);
        EXPECT_EQ(flip, wantFlip);
    }
};

TEST_F(AtomicElementTest, MinAndMaxEqualTheSequentialFoldOnEveryType)
{
    checkMinMax<float>();
    checkMinMax<double>();
    checkMinMax<std::int32_t>();
    checkMinMax<std::uint32_t>();
    checkMinMax<std::int64_t>();
}

TEST_F(AtomicElementTest, AndAndXorEqualTheSequentialFoldOnEveryInteger)
{
    checkBits<std::int32_t>();
    checkBits<std::uint32_t>();
    checkBits<std::int64_t>();
}

TEST_F(AtomicElementTest, ANaNAbsorbsInMinAndMaxWhateverTheOrder)
{
    for (std::size_t at : {std::size_t{0}, N / 2, N - 1}) {
        double lo = 1.0, hi = 1.0;
#pragma omp parallel for
        for (std::size_t k = 0; k < N; ++k) {
            const double v = (k == at) ? std::numeric_limits<double>::quiet_NaN() : static_cast<double>(k);
            aether::accum::atomic::minAt(lo, v);
            aether::accum::atomic::maxAt(hi, v);
        }
        EXPECT_TRUE(std::isnan(lo));
        EXPECT_TRUE(std::isnan(hi));
    }
}

TEST_F(AtomicElementTest, FetchAddReturnsEachPartialSumOnce)
{
    std::int64_t cell = 0;
    std::vector<std::int64_t> seen(N);
#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k)
        seen[k] = aether::accum::atomic::fetchAddAt(cell, 1);
    EXPECT_EQ(cell, static_cast<std::int64_t>(N));
    std::sort(seen.begin(), seen.end());
    for (std::size_t k = 0; k < N; ++k)
        EXPECT_EQ(seen[k], static_cast<std::int64_t>(k));
}

TEST_F(AtomicElementTest, ExchangeChainReturnsEveryValueOnceAndTheCellKeepsOne)
{
    // N threads exchange their own value into one cell that starts at -1: the old values returned, together with the
    // final cell, are exactly the initial value and the N values, each once.
    double cell = -1.0;
    std::vector<double> old(N);
#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k)
        old[k] = aether::accum::atomic::exchangeAt(cell, static_cast<double>(k));
    old.push_back(cell);
    std::sort(old.begin(), old.end());
    EXPECT_EQ(old[0], -1.0);
    for (std::size_t k = 0; k < N; ++k)
        EXPECT_EQ(old[k + 1], static_cast<double>(k));
}

TEST_F(AtomicElementTest, CompareExchangeOnAFreshCellHasExactlyOneWinner)
{
    std::uint32_t cell = 0;
    std::vector<std::uint32_t> seen(N);
#pragma omp parallel for
    for (std::size_t k = 0; k < N; ++k)
        seen[k] = aether::accum::atomic::compareExchangeAt(cell, 0u, static_cast<std::uint32_t>(k + 1));
    const auto winners = static_cast<std::size_t>(std::count(seen.begin(), seen.end(), 0u));
    EXPECT_EQ(winners, std::size_t{1});
    EXPECT_GE(cell, 1u);
    EXPECT_LE(cell, static_cast<std::uint32_t>(N));
}

} // namespace
} // namespace aether_tests
