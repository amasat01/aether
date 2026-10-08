// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_Random_common.h
 * @brief Shared fixture, fill helpers and statistical checkers for the paired
 *        `test_Random.{cpp,cu}` suites.
 *
 * Every tolerance is a generous multiple (6-8x) of the theoretical estimator
 * standard error, so a pass is reproducible rather than lucky.
 *
 * `kN` is a fixed constant rather than a mode-dependent value:
 * `AETHER_SANITIZE_MINIMAL` is explicitly documented in the root
 * CMakeLists.txt as "currently a no-op, no aether test reads it", so there
 * is no knob to read a smaller count from. The fills go through
 * `aether::Array<T,3>` + `hostView()` + `view[SampleIndex] = expr`
 * (`aether/view/View.h`). aether's uniform draws land in `(lo, hi]` on both
 * arms — one endpoint convention everywhere; the moments are identical to
 * 2^-32 either way.
 *
 * The suites this header serves check TWO different things, and it is worth
 * being explicit about which is which. The statistical tests below say the
 * stream is DISTRIBUTED correctly and is order-independent. They cannot say
 * it is the SAME stream as yesterday's, or as the other arm's — that claim
 * (one seed, one stream, host and device alike) is carried by the two
 * `Golden_*` tests against `tests/random/golden_random.h`.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "tests/random/golden_random.h"

namespace aether_tests {

/** @brief Fixed seed for reproducible test streams. */
inline constexpr std::uint64_t SEED = 0xC0FFEEu;

/** @brief Sample count for every statistical test. */
inline constexpr std::size_t kN = 65536;

/** @brief Shared fixture — the suite name both `.cpp` and `.cu` register under. */
class RandomTest : public ::testing::Test { };

/** @brief Fill `a` serially: `f(view, SampleIndex)` writes one sample. */
template<class ArrT, class F>
inline void fillSerial(ArrT& a, std::size_t n, F f)
{
    auto v = a.hostView();
    for (std::size_t k = 0; k < n; ++k)
        f(v, aether::SampleIndex::make(k));
}

/** @brief Fill `a` from an OpenMP parallel loop — the order-independence probe. */
template<class ArrT, class F>
inline void fillParallel(ArrT& a, std::size_t n, F f)
{
    auto v = a.hostView();
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(n); ++k)
        f(v, aether::SampleIndex::make(static_cast<std::size_t>(k)));
}

/** @brief Read component `d` (0..2) of sample `k` as a double. */
template<class ViewT>
inline double comp(const ViewT& v, std::size_t k, int d)
{
    return static_cast<double>(v(static_cast<std::size_t>(d), k));
}

/** @brief Every component of every sample is bit-identical between two views. */
template<class ViewA, class ViewB>
inline void expectIdentical(const ViewA& a, const ViewB& b, std::size_t n)
{
    for (std::size_t k = 0; k < n; ++k)
        for (int d = 0; d < 3; ++d)
            ASSERT_EQ(comp(a, k, d), comp(b, k, d)) << "k=" << k << " d=" << d;
}

/** @brief Per-component empirical mean / variance against `N(mean, sd^2)`. */
template<class ViewT>
inline void checkNormalMoments(const ViewT& v, std::size_t n, double mean, double sd)
{
    const double tolM = 6.0 * sd / std::sqrt(static_cast<double>(n));
    const double tolV = 8.0 * sd * sd * std::sqrt(2.0 / static_cast<double>(n));
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        double s2 = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double x = comp(v, k, d);
            s += x;
            s2 += x * x;
        }
        const double m = s / static_cast<double>(n);
        const double var = s2 / static_cast<double>(n) - m * m;
        EXPECT_NEAR(m, mean, tolM) << "normal mean dim " << d;
        EXPECT_NEAR(var, sd * sd, tolV) << "normal var dim " << d;
    }
}

/** @brief Per-component empirical mean / variance against `U(lo, hi]`. */
template<class ViewT>
inline void checkUniformMoments(const ViewT& v, std::size_t n, double lo, double hi)
{
    const double mean = 0.5 * (lo + hi);
    const double var = (hi - lo) * (hi - lo) / 12.0;
    const double tolM = 6.0 * std::sqrt(var / static_cast<double>(n));
    const double tolV = 8.0 * var * std::sqrt(2.0 / static_cast<double>(n));
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        double s2 = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double x = comp(v, k, d);
            ASSERT_GE(x, lo) << "uniform below lo";
            ASSERT_LT(x, hi + 1e-9) << "uniform above hi";
            s += x;
            s2 += x * x;
        }
        const double m = s / static_cast<double>(n);
        const double empVar = s2 / static_cast<double>(n) - m * m;
        EXPECT_NEAR(m, mean, tolM) << "uniform mean dim " << d;
        EXPECT_NEAR(empVar, var, tolV) << "uniform var dim " << d;
    }
}

/** @brief Bit pattern of a `double`, for exact golden comparisons. */
inline std::uint64_t bitsOfDouble(double v)
{
    std::uint64_t b = 0;
    static_assert(sizeof(b) == sizeof(v), "double is not 64-bit");
    __builtin_memcpy(&b, &v, sizeof(b));
    return b;
}

/** @brief Bit pattern of a `float`, for exact golden comparisons. */
inline std::uint32_t bitsOfFloat(float v)
{
    std::uint32_t b = 0;
    static_assert(sizeof(b) == sizeof(v), "float is not 32-bit");
    __builtin_memcpy(&b, &v, sizeof(b));
    return b;
}

/**
 * @brief TIER-TOL band for a golden normal: absolute AND relative, so a
 *        large-|z| tail value is not held to an absolute-only bound it
 *        cannot meet. `tol` is the per-type band from `golden_random.h`.
 */
inline double normalBand(double expected, double tol)
{
    return tol * (1.0 + std::fabs(expected));
}

/**
 * @brief Per-component empirical mean against a `lognormal(mu, sigma)`
 *        (tolerances derived from the estimator's standard error).
 */
template<class ViewT>
inline void checkLognormalMean(const ViewT& v, std::size_t n, double mu, double sigma)
{
    const double theoMean = std::exp(mu + 0.5 * sigma * sigma);
    const double theoVar = (std::exp(sigma * sigma) - 1.0) * std::exp(2.0 * mu + sigma * sigma);
    const double tolM = 8.0 * std::sqrt(theoVar / static_cast<double>(n));
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double x = comp(v, k, d);
            EXPECT_GT(x, 0.0) << "lognormal must be positive";
            s += x;
        }
        EXPECT_NEAR(s / static_cast<double>(n), theoMean, tolM) << "lognormal mean dim " << d;
    }
}

/**
 * @brief Per-component empirical mean against `exponential(lambda)`
 *        (tolerances derived from the estimator's standard error).
 */
template<class ViewT>
inline void checkExponentialMean(const ViewT& v, std::size_t n, double lambda)
{
    const double theoMean = 1.0 / lambda;
    const double tolM = 8.0 * theoMean / std::sqrt(static_cast<double>(n)); // sd == mean
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double x = comp(v, k, d);
            EXPECT_GE(x, 0.0) << "exponential must be non-negative";
            s += x;
        }
        EXPECT_NEAR(s / static_cast<double>(n), theoMean, tolM) << "exponential mean dim " << d;
    }
}

/**
 * @brief Per-component empirical success rate against `bernoulli(p)`
 *        (tolerances derived from the estimator's standard error).
 */
template<class ViewT>
inline void checkBernoulliRate(const ViewT& v, std::size_t n, double p)
{
    const double tolM = 8.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(n));
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double x = comp(v, k, d);
            EXPECT_TRUE(x == 0.0 || x == 1.0) << "bernoulli must be 0/1";
            s += x;
        }
        EXPECT_NEAR(s / static_cast<double>(n), p, tolM) << "bernoulli rate dim " << d;
    }
}

/**
 * @brief Empirical mean + full covariance against a multivariate normal
 *        (tolerances derived from the estimator's standard error).
 */
template<class ViewT>
inline void checkCovariance(const ViewT& v, std::size_t n, const double meanv[3], const double cov[3][3])
{
    double m[3] = { 0.0, 0.0, 0.0 };
    double c[3][3] = { { 0.0 } };
    for (std::size_t k = 0; k < n; ++k) {
        const double x[3] = { comp(v, k, 0), comp(v, k, 1), comp(v, k, 2) };
        for (int i = 0; i < 3; ++i) {
            m[i] += x[i];
            for (int j = 0; j < 3; ++j)
                c[i][j] += x[i] * x[j];
        }
    }
    for (int i = 0; i < 3; ++i) {
        m[i] /= static_cast<double>(n);
        EXPECT_NEAR(m[i], meanv[i], 6.0 * std::sqrt(cov[i][i] / static_cast<double>(n))) << "mvn mean dim " << i;
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double emp = c[i][j] / static_cast<double>(n) - m[i] * m[j];
            const double tol = 8.0 * std::sqrt((cov[i][i] * cov[j][j] + cov[i][j] * cov[i][j]) / static_cast<double>(n));
            EXPECT_NEAR(emp, cov[i][j], tol) << "mvn cov " << i << "," << j;
        }
}

/**
 * @brief Every integer draw lies in `[lo, hi]` and every value in that range
 *        is hit at least once (histogram gate).
 */
template<class ViewT>
inline void checkUniformIntRange(const ViewT& v, std::size_t n, int lo, int hi)
{
    const int span = hi - lo + 1;
    std::vector<std::size_t> hist(static_cast<std::size_t>(span), 0);
    for (std::size_t k = 0; k < n; ++k)
        for (int d = 0; d < 3; ++d) {
            const int val = static_cast<int>(comp(v, k, d));
            ASSERT_GE(val, lo);
            ASSERT_LE(val, hi);
            ++hist[static_cast<std::size_t>(val - lo)];
        }
    for (int b = 0; b < span; ++b)
        EXPECT_GT(hist[static_cast<std::size_t>(b)], std::size_t{ 0 }) << "uniformInt never produced " << (lo + b);
}

} // namespace aether_tests
