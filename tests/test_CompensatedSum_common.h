// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Shared compensated-sum building blocks in aether/accum/atomic.h: the
// Neumaier step, the symmetric pair-combine, the non-finite rule and the
// 32-slot tree (host twin of the warp butterfly). Host-side checks, included
// by test_CompensatedSum.cpp and test_CompensatedSum.cu.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace comp_sum {

namespace at = aether::accum::atomic::detail;

/** Terms of one fixture: pairs +-x (full mantissa) plus residual terms on the
 *  2^-10 grid. The exact sum is the grid sum, known by construction. */
template<class T>
struct Fixture {
    std::vector<T> terms;
    double exact;
};

template<class T>
Fixture<T> makeFixture(unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> big(1.0, 3.0);
    std::uniform_int_distribution<int> grid(1, 4);
    Fixture<T> f;
    constexpr int pairs = 64, residuals = 64;
    for (int i = 0; i < pairs; ++i) {
        const T x = static_cast<T>(big(rng));
        f.terms.push_back(x);
        f.terms.push_back(-x);
    }
    long long m = 0;
    for (int i = 0; i < residuals; ++i) {
        const int g = grid(rng);
        m += g;
        f.terms.push_back(static_cast<T>(g) * static_cast<T>(0.0009765625));
    }
    f.exact = static_cast<double>(m) * 0.0009765625;
    std::shuffle(f.terms.begin(), f.terms.end(), rng);
    return f;
}

/** Distance from `got` to `exact` in units of the spacing of T at `exact`. */
template<class T>
double ulpErr(T got, double exact)
{
    const T e   = static_cast<T>(exact);
    const T ulp = std::nextafter(e, std::numeric_limits<T>::infinity()) - e;
    return std::abs(static_cast<double>(got) - exact) / static_cast<double>(ulp);
}

/** 32 interleaved Neumaier walks, then the canonical 32-slot tree. */
template<class T>
T treeSum(const std::vector<T>& terms)
{
    T h[32] = {}, c[32] = {};
    for (std::size_t i = 0; i < terms.size(); ++i)
        at::neumaierAdd_(h[i % 32], c[i % 32], terms[i]);
    T oh, oc;
    at::reduce32Host_(h, c, oh, oc);
    return oh + oc;
}

template<class T>
T walkSum(const std::vector<T>& terms)
{
    T h = 0, c = 0;
    for (T t : terms)
        at::neumaierAdd_(h, c, t);
    return h + c;
}

template<class T>
T plainSum(const std::vector<T>& terms)
{
    T s = 0;
    for (T t : terms)
        s += t;
    return s;
}

template<class T>
bool sameBits(T a, T b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// One body per check, run as a float test and a double test.
#define COMP_SUM_TEST(Name)                                                    \
    template<class T>                                                          \
    void Name##Body();                                                         \
    TEST(CompensatedSumF32, Name) { Name##Body<float>(); }                     \
    TEST(CompensatedSumF64, Name) { Name##Body<double>(); }                    \
    template<class T>                                                          \
    void Name##Body()

COMP_SUM_TEST(NeumaierWalkAndTreeWithinOneUlp)
{
    for (unsigned seed = 1; seed <= 8; ++seed) {
        const auto f = makeFixture<T>(seed);
        EXPECT_LE(ulpErr(walkSum(f.terms), f.exact), 1.0) << "seed " << seed;
        EXPECT_LE(ulpErr(treeSum(f.terms), f.exact), 1.0) << "seed " << seed;
    }
}

COMP_SUM_TEST(PlainSumFailsTheFixture)
{
    int failures = 0;
    for (unsigned seed = 1; seed <= 8; ++seed) {
        const auto f = makeFixture<T>(seed);
        if (ulpErr(plainSum(f.terms), f.exact) > 1.0)
            ++failures;
    }
    EXPECT_GE(failures, 6);
}

COMP_SUM_TEST(PairCombineIsSymmetricBitwise)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
        const T ah = static_cast<T>(d(rng) * std::pow(2.0, i % 20 - 10));
        const T bh = static_cast<T>(d(rng) * std::pow(2.0, (i / 7) % 20 - 10));
        const T ac = static_cast<T>(d(rng) * 1e-6);
        const T bc = static_cast<T>(d(rng) * 1e-6);
        T xh, xc, yh, yc;
        at::pairCombine_(ah, ac, bh, bc, xh, xc);
        at::pairCombine_(bh, bc, ah, ac, yh, yc);
        ASSERT_TRUE(sameBits(xh, yh) && sameBits(xc, yc)) << "i " << i;
    }
}

COMP_SUM_TEST(PairCombineKeepsTheResidual)
{
    const T big = std::ldexp(T(1), std::numeric_limits<T>::digits + 3), tiny = static_cast<T>(0.0009765625);
    T oh, oc;
    at::pairCombine_(big, T(0), tiny, T(0), oh, oc);
    EXPECT_EQ(oh, big);
    EXPECT_EQ(oc, tiny);
}

COMP_SUM_TEST(LoneInfGivesInf)
{
    constexpr T inf = std::numeric_limits<T>::infinity();
    T h = 0, c = 0;
    at::neumaierAdd_(h, c, T(1.5));
    at::neumaierAdd_(h, c, inf);
    at::neumaierAdd_(h, c, T(2.25));
    EXPECT_EQ(h + c, inf);
    T oh, oc;
    at::pairCombine_(inf, T(0), T(3), T(0.5), oh, oc);
    EXPECT_EQ(oh + oc, inf);
}

COMP_SUM_TEST(OppositeInfsGiveNaN)
{
    constexpr T inf = std::numeric_limits<T>::infinity();
    T h = 0, c = 0;
    at::neumaierAdd_(h, c, inf);
    at::neumaierAdd_(h, c, T(1));
    at::neumaierAdd_(h, c, -inf);
    EXPECT_TRUE(std::isnan(h + c));
    T oh, oc;
    at::pairCombine_(inf, T(0), -inf, T(0), oh, oc);
    EXPECT_TRUE(std::isnan(oh + oc));
}

COMP_SUM_TEST(NaNPropagates)
{
    T h = 0, c = 0;
    at::neumaierAdd_(h, c, T(1));
    at::neumaierAdd_(h, c, std::numeric_limits<T>::quiet_NaN());
    at::neumaierAdd_(h, c, T(1));
    EXPECT_TRUE(std::isnan(h + c));
    T oh, oc;
    at::pairCombine_(std::numeric_limits<T>::quiet_NaN(), T(0), T(1), T(0), oh, oc);
    EXPECT_TRUE(std::isnan(oh + oc));
}

} // namespace comp_sum
} // namespace aether_tests
