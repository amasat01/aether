// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// random/ independence rows (host / AETHER_CPP_MODE build).
//
// Each (seed, global, counter) owns ONE Philox block and every draw reads only
// that block's words, so neighbouring counters are independent. These rows pin
// that: a lag-k fourth-moment check (for independent standard normals
// E[z^2 z'^2] = 1; the old word-offset positioning gave lag 2 (double) and
// lag 1 (float) sharing words), structural disjointness of the words each
// primitive reads, and the open-interval (0,1) range of `uniform01`.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

namespace rnd = aether::random::detail;

constexpr std::uint64_t kLags[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 64 };

template<class Real>
void expectLagMoment(std::uint64_t seedSalt)
{
    constexpr int N = 1000000;
    for (const std::uint64_t lag : kLags) {
        double sum = 0.0, sum2 = 0.0;
        for (int k = 0; k < N; ++k) {
            const std::uint64_t seed = rnd::splitmix64(seedSalt + static_cast<std::uint64_t>(k));
            const double z           = static_cast<double>(rnd::standardNormal<Real>(seed, 0, 5));
            const double y           = static_cast<double>(rnd::standardNormal<Real>(seed, 0, 5 + lag));
            const double q           = z * z * y * y;
            sum += q;
            sum2 += q * q;
        }
        const double mean = sum / N;
        const double se   = std::sqrt((sum2 / N - mean * mean) / N);
        EXPECT_LT(std::fabs(mean - 1.0), 5.0 * se) << "lag " << lag << " mean " << mean << " se " << se;
    }
}

TEST(RandomIndependenceTest, Double_Normal_LagFourthMoment)
{
    expectLagMoment<double>(0x1234567ull);
}

TEST(RandomIndependenceTest, Float_Normal_LagFourthMoment)
{
    expectLagMoment<float>(0x7654321ull);
}

TEST(RandomIndependenceTest, Blocks_AreWordDisjointAcrossCounters)
{
    for (const std::uint64_t seed : { 0ull, 0xC0FFEEull, 0xFFFFFFFFFFFFFFFFull }) {
        for (const aether::offset_t global : { 0u, 3u, 4294967295u }) {
            std::set<std::uint32_t> seen;
            std::size_t total = 0;
            for (std::uint64_t c = 0; c < 256; ++c) {
                const rnd::Uint4 b = rnd::philoxBlock(seed, global, c);
                for (const std::uint32_t w : { b.x, b.y, b.z, b.w }) {
                    seen.insert(w);
                    ++total;
                }
            }
            EXPECT_EQ(seen.size(), total) << "seed " << seed << " global " << global;
        }
    }
}

TEST(RandomIndependenceTest, Primitives_ReadOnlyTheirOwnBlock)
{
    for (const std::uint64_t seed : { 0ull, 0xC0FFEEull, 0xFFFFFFFFFFFFFFFFull }) {
        for (std::uint64_t c = 0; c < 512; ++c) {
            const rnd::Uint4 b = rnd::philoxBlock(seed, 7, c);
            // bits64: the first two words of the block, low word first.
            EXPECT_EQ(rnd::randomBits64(seed, 7, c), (static_cast<std::uint64_t>(b.y) << 32) | b.x);
            // uniform: word 0 of the block.
            EXPECT_EQ(rnd::uniform01<double>(seed, 7, c), rnd::uniformDoubleFromWord(b.x));
            EXPECT_EQ(rnd::uniform01<float>(seed, 7, c), rnd::uniformFloatFromWord(b.x));
            // Neighbouring counters never share a word at any position: the old
            // hi(c) == lo(c+1) overlap of randomBits64 is gone.
            const std::uint64_t cur = rnd::randomBits64(seed, 7, c);
            const std::uint64_t nxt = rnd::randomBits64(seed, 7, c + 1);
            EXPECT_NE(cur >> 32, nxt & 0xFFFFFFFFull);
        }
    }
}

TEST(RandomIndependenceTest, Uniform_IsOpenIntervalAtBoundaryWords)
{
    for (const std::uint32_t w : { 0u, 1u, 0x1FFu, 0x200u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFE00u, 0xFFFFFFFEu, 0xFFFFFFFFu }) {
        const double d = rnd::uniformDoubleFromWord(w);
        const float f  = rnd::uniformFloatFromWord(w);
        EXPECT_GT(d, 0.0) << w;
        EXPECT_LT(d, 1.0) << w;
        EXPECT_GT(f, 0.0f) << w;
        EXPECT_LT(f, 1.0f) << w;
    }
    // Monotone in the word (the float form keeps the top 23 bits).
    EXPECT_LE(rnd::uniformFloatFromWord(0u), rnd::uniformFloatFromWord(0xFFFFFFFFu));
    EXPECT_LE(rnd::uniformDoubleFromWord(0u), rnd::uniformDoubleFromWord(0xFFFFFFFFu));
    // log of both the draw and its complement stay finite at the extreme words.
    for (const std::uint32_t w : { 0u, 0xFFFFFFFFu }) {
        EXPECT_TRUE(std::isfinite(std::log(rnd::uniformDoubleFromWord(w))));
        EXPECT_TRUE(std::isfinite(std::log(1.0 - rnd::uniformDoubleFromWord(w))));
        EXPECT_TRUE(std::isfinite(std::log(1.0f - rnd::uniformFloatFromWord(w))));
    }
}

} // namespace
} // namespace aether_tests
