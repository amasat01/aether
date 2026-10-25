// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// `aether::accum::run` on one arm (test_Run.cu: the device, test_Run.cpp: the
// host). The including file defines `runArm<T>(c, compensated, out, commits)`
// and `kDevice`, then includes this header. Each case is a keyed sum of
// terms in [-1, 1) whose error against a long-double reference, over
// eps * sum|t| per slot, is within 8 plain and 1 compensated (8 when runs are
// shorter than a warp: the in-warp tree before the one compensated commit is
// plain), and whose commit count is the one the arm's run structure implies.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace aether_tests {
namespace {

enum class Form { Aligned32, Aligned16, Tested, Sorted };

struct Case {
    std::string name;
    Form form;
    std::vector<long long> key;  // the slot of each lane
    std::size_t slots;
    long long own;               // offset of the lane's own sample (the alignment vote reads it)
    bool shortRuns = false;      // runs shorter than a warp
};

/** @brief Host lanes per serial range (one `open`/`close` each). */
constexpr std::size_t kHostRange = 4096;

/** @brief A uniform value in [0, 1) for lane `l`, the same on both arms. */
inline double unitOf(std::size_t l)
{
    std::uint64_t x = (static_cast<std::uint64_t>(l) + 1u) * 0x9E3779B97F4A7C15ull;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    return static_cast<double>(x >> 11) * 0x1.0p-53;
}

/** @brief The term of lane `l`, in [-1, 1). */
inline double termOf(std::size_t l)
{
    return 2.0 * unitOf(l) - 1.0;
}

inline Case byQuotient(std::string name, Form form, std::size_t n, long long run, long long own = 0)
{
    Case c{ std::move(name), form, std::vector<long long>(n), 0, own };
    for (std::size_t l = 0; l < n; ++l)
        c.key[l] = (static_cast<long long>(l) + own) / run;
    c.slots = static_cast<std::size_t>(c.key.back() + 1);
    return c;
}

/** @brief Sorted keys whose runs are 1..40 lanes long. */
inline Case mixedRuns(std::size_t n)
{
    Case c{ "sorted_mixed", Form::Sorted, std::vector<long long>(n), 0, 0 };
    long long k = 0;
    std::size_t left = 1;
    for (std::size_t l = 0; l < n; ++l) {
        if (--left == 0) {
            ++k;
            left = 1 + static_cast<std::size_t>(unitOf(l) * 1000.0) % 40;
        }
        c.key[l] = k;
    }
    c.slots     = static_cast<std::size_t>(k + 1);
    c.shortRuns = true;
    return c;
}

/** @brief Keys out of order: still a correct sum, only shorter runs. */
inline Case unsortedKeys(std::size_t n)
{
    Case c{ "sorted_unsorted", Form::Sorted, std::vector<long long>(n), 1024, 0 };
    for (std::size_t l = 0; l < n; ++l)
        c.key[l] = static_cast<long long>((l * 7919u) % c.slots);
    return c;
}

/** @brief (run commits, lane commits) the arm issues for `c`. */
inline std::pair<unsigned long long, unsigned long long> expectedCommits(const Case& c, bool device)
{
    const std::size_t n = c.key.size();
    unsigned long long run = 0, lane = 0;
    if (!device) {
        for (std::size_t l = 0; l < n; ++l)
            run += (l % kHostRange == 0 || c.key[l] != c.key[l - 1]) ? 1u : 0u;
        return { run, lane };
    }
    for (std::size_t w = 0; w < n; w += 32) {
        bool uniform = true;
        unsigned heads = 0;
        for (std::size_t l = w; l < w + 32; ++l) {
            uniform = uniform && c.key[l] == c.key[w];
            heads += (l == w || c.key[l] != c.key[l - 1]) ? 1u : 0u;
        }
        switch (c.form) {
        case Form::Aligned32:
        case Form::Aligned16: {
            const bool aligned = (c.own % 32) == 0;
            const unsigned width = c.form == Form::Aligned32 ? 32u : 16u;
            if (aligned)
                run += 32u / width;
            else
                lane += 32u;
            break;
        }
        case Form::Tested:
            if (uniform)
                run += 1u;
            else
                lane += 32u;
            break;
        case Form::Sorted:
            run += heads;
            break;
        }
    }
    return { run, lane };
}

template<class T>
void runCase(const Case& c, bool compensated)
{
    std::vector<long double> ref(c.slots, 0.0L), mass(c.slots, 0.0L);
    for (std::size_t l = 0; l < c.key.size(); ++l) {
        const auto t = static_cast<long double>(static_cast<T>(termOf(l)));
        ref[static_cast<std::size_t>(c.key[l])] += t;
        mass[static_cast<std::size_t>(c.key[l])] += std::fabs(t);
    }
    std::vector<long double> got;
    unsigned long long commits[2] = { 0, 0 };
    runArm<T>(c, compensated, got, commits);
    double worst = 0.0;
    for (std::size_t k = 0; k < c.slots; ++k) {
        if (mass[k] == 0.0L)
            continue;
        const double err = static_cast<double>(std::fabs(got[k] - ref[k])
                                                / (std::numeric_limits<T>::epsilon() * mass[k]));
        worst = err > worst ? err : worst;
    }
    const auto want = expectedCommits(c, kDevice);
    const std::string what = c.name + (compensated ? " compensated " : " plain ") + (sizeof(T) == 4 ? "f32" : "f64");
    EXPECT_LE(worst, compensated && !c.shortRuns ? 1.0 : 8.0) << what;
    EXPECT_EQ(commits[0], want.first) << what << ": run commits";
    EXPECT_EQ(commits[1], want.second) << what << ": per-lane commits";
}

void runAll(const Case& c)
{
    for (const bool compensated : { false, true }) {
        runCase<float>(c, compensated);
        runCase<double>(c, compensated);
    }
}

TEST(AccumRun, AlignedRunCommitsOncePerWarp)
{
    runAll(byQuotient("aligned_8192", Form::Aligned32, 64 * 8192, 8192));
}

TEST(AccumRun, HalfWarpRunCommitsOncePerHalf)
{
    runAll(byQuotient("aligned_48", Form::Aligned16, 1024 * 48, 48));
}

TEST(AccumRun, MisalignedWarpCommitsPerLane)
{
    runAll(byQuotient("misaligned", Form::Aligned32, 64 * 1024, 32, 1000));
}

TEST(AccumRun, TestedRunCommitsOncePerUniformWarp)
{
    runAll(byQuotient("tested_64", Form::Tested, 64 * 8192, 64));
    runAll(byQuotient("tested_8", Form::Tested, 64 * 1024, 8));
}

TEST(AccumRun, SortedRunCommitsOncePerRunInWarp)
{
    Case c = byQuotient("sorted_8192", Form::Sorted, 64 * 8192, 8192);
    runAll(c);
    runAll(mixedRuns(64 * 8192));
}

TEST(AccumRun, UnsortedKeysStayCorrect)
{
    runAll(unsortedKeys(64 * 1024));
}

} // namespace
} // namespace aether_tests
