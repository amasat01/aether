// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_TrigTotal_common.h
 * @brief Corpus and scoring for the totality of FP64 `aether::math::sin`/
 *        `cos`/`sincos`: every double, finite or not, against the host libm.
 *
 * Corpus: an ENUMERATED special-value list (signed zeros, smallest and
 * largest subnormals, `DBL_MIN`, Inf, NaN, `2^31` and its neighbours — the
 * device fast/slow path cut — `1e22`, `1e300`, `DBL_MAX`, the hardest known
 * double for reduction modulo pi/2, and doubles next to `k * pi/2` for `k`
 * from `1e3` to `1e300`), plus a seeded sweep that is log-uniform in
 * magnitude over every binade from the smallest subnormal to `DBL_MAX`.
 *
 * Instrument: ULP distance to `std::sin`/`std::cos`, gated at `kUlpTol`
 * (`kPinned` lists the inputs where glibc is itself off, with the
 * correctly rounded value used instead on the device).
 * A NaN result must match a NaN reference exactly (and vice versa); a zero
 * result must carry the reference's sign bit (`sin(-0) == -0`).
 *
 * `test_TrigTotal.cu` evaluates on the device (the `detail/BoundedTrig.h`
 * route), `test_TrigTotal.cpp` on the host (the `std::` route).
 */

#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace aether_tests {
namespace TrigTotalTest {

/// Gate: 1 ULP against glibc (the documented bound is 2; every input of
/// this corpus measures within 1, as test_BoundedTrig also gates).
constexpr int64_t kUlpTol = 1;

inline uint64_t bitsOf(double x)
{
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

/// Monotone ULP distance; both arguments finite and non-NaN.
inline int64_t ulpDist(double a, double b)
{
    if (a == b)
        return 0;
    uint64_t ia = bitsOf(a), ib = bitsOf(b);
    ia = (ia >> 63) ? 0x8000000000000000ULL - ia : ia + 0x8000000000000000ULL;
    ib = (ib >> 63) ? 0x8000000000000000ULL - ib : ib + 0x8000000000000000ULL;
    return static_cast<int64_t>(ia > ib ? ia - ib : ib - ia);
}

inline void pushSigned(std::vector<double>& v, double x)
{
    v.push_back(x);
    v.push_back(-x);
}

inline std::vector<double> makeCorpus()
{
    std::vector<double> v;
    const double inf    = std::numeric_limits<double>::infinity();
    const double qnan   = std::numeric_limits<double>::quiet_NaN();
    const double minSub = std::numeric_limits<double>::denorm_min();

    /* Enumerated specials. */
    pushSigned(v, 0.0);
    pushSigned(v, minSub);
    pushSigned(v, 2.0 * minSub);
    pushSigned(v, DBL_MIN - minSub); /* largest subnormal */
    pushSigned(v, DBL_MIN);
    pushSigned(v, inf);
    pushSigned(v, qnan);
    pushSigned(v, 1e-300);
    pushSigned(v, 0x1p-27);
    pushSigned(v, 1e22);
    pushSigned(v, 1e300);
    pushSigned(v, DBL_MAX);
    pushSigned(v, std::nextafter(DBL_MAX, 0.0));
    pushSigned(v, std::ldexp(6381956970095103.0, 797)); /* hardest double mod pi/2 */
    for (double c : { 0x1p31, 0x1p32, 0x1p52, 0x1p53, 0x1p63, 0x1p64, 0x1p65, 0x1p100, 0x1p1000, 0x1p1023 }) {
        pushSigned(v, c);
        pushSigned(v, std::nextafter(c, 0.0));
        pushSigned(v, std::nextafter(c, inf));
    }
    pushSigned(v, 0x1p31 - 1.0);
    pushSigned(v, 0x1p31 + 1.0);

    /* Doubles next to k * pi/2 (reduced angle ~ulp(x)): k spans both device
     * reductions. The long double product is close enough to land within
     * a few ulps of the nearest double; the two neighbours either side add
     * the cancellation-heavy cases. */
    const long double halfPi = 1.570796326794896619231321691639751442L;
    for (long double k : { 1e3L, 7.0e5L, 1.234567e9L, 1.367e9L, 3e9L, 1e12L, 1e15L, 1e18L, 0x1p60L, 1e22L, 1e100L, 1e200L,
             1e300L / 1.6L }) {
        for (long double dk : { 0.0L, 1.0L, 2.0L, 3.0L }) {
            double x = static_cast<double>((k + dk) * halfPi);
            for (int s = 0; s < 2; ++s) {
                pushSigned(v, x);
                pushSigned(v, std::nextafter(x, 0.0));
                pushSigned(v, std::nextafter(x, inf));
                x = std::nextafter(std::nextafter(x, inf), inf);
            }
        }
    }

    /* Log-uniform sweep: a uniformly drawn biased exponent (0 = subnormal
     * through 0x7FE = DBL_MAX's binade), random mantissa and sign. */
    std::mt19937_64 rng(0x7219A5ULL);
    for (int i = 0; i < 1 << 16; ++i) {
        const uint64_t r    = rng();
        const uint64_t e    = rng() % 0x7FFULL;
        const uint64_t bits = (r & 0x800FFFFFFFFFFFFFULL) | (e << 52);
        double x;
        std::memcpy(&x, &bits, sizeof(x));
        v.push_back(x);
    }
    /* Plus a dense in-range block, the fast path's own domain. */
    std::uniform_real_distribution<double> inRange(-1e4, 1e4);
    for (int i = 0; i < 1 << 12; ++i)
        v.push_back(inRange(rng));
    return v;
}

/**
 * Inputs where glibc itself misses the true value by more than `kUlpTol`,
 * with the correctly rounded results (mpmath at 3000 bits) pinned instead.
 * The hardest double for reduction modulo pi/2: glibc's `cos` is 8 ULP off
 * there. Scoring by `|x|` covers both signs.
 */
struct PinnedRef {
    double x, sinValue, cosValue;
};
inline constexpr PinnedRef kPinned[] = {
    { 0x1.6ac5b262ca1ffp+849, 0x1.0p+0, -0x1.14ae72e6ba22fp-61 },
};

/// The reference for `x`: the pinned value if listed (and `usePinned`),
/// else the host libm.
inline double reference(double x, bool isSin, bool usePinned)
{
    for (const PinnedRef& p : kPinned) {
        if (usePinned && std::fabs(x) == p.x)
            return isSin ? std::copysign(p.sinValue, x) : p.cosValue;
    }
    return isSin ? std::sin(x) : std::cos(x);
}

/// Scores one output column; returns the worst ULP seen, adds failures.
/// The host route IS the libm, so the host twin scores against it as is
/// (`usePinned = false`); the device twin scores against `kPinned` too.
inline int64_t score(const std::vector<double>& xs, const std::vector<double>& got, bool isSin, bool usePinned,
    const std::string& label)
{
    int64_t worst = 0;
    int failures  = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double want = reference(xs[i], isSin, usePinned);
        const double have = got[i];
        bool ok;
        if (std::isnan(want) || std::isnan(have)) {
            ok = std::isnan(want) && std::isnan(have);
        } else if (want == 0.0 || have == 0.0) {
            ok = bitsOf(want) == bitsOf(have);
        } else {
            const int64_t d = ulpDist(want, have);
            worst           = d > worst ? d : worst;
            ok              = d <= kUlpTol;
        }
        if (!ok && ++failures <= 10) {
            ADD_FAILURE() << label << "(" << std::hexfloat << xs[i] << ") = " << have << ", libm " << want
                          << std::defaultfloat;
        }
    }
    std::printf("[ trig     ] %s: worst %lld ULP over %zu inputs\n", label.c_str(), static_cast<long long>(worst),
        xs.size());
    EXPECT_EQ(failures, 0) << label << ": " << failures << " of " << xs.size() << " inputs off";
    return worst;
}

constexpr bool kSin = true;
constexpr bool kCos = false;

/// One evaluation of the corpus: inputs and the four output columns.
struct Columns {
    std::vector<double> xs, sin, cos, sc, cc;
};

/**
 * The C standard's special cases, independent of any libm: `sin`/`cos` of
 * +-Inf and of NaN are NaN; `sin(+-0) == +-0` (same sign bit), `cos(+-0)
 * == 1`; `sin(x) == x` for every subnormal `x`.
 */
inline void expectCStandardSpecials(const Columns& r)
{
    int checked = 0;
    for (size_t i = 0; i < r.xs.size(); ++i) {
        const double x = r.xs[i];
        if (std::isinf(x) || std::isnan(x)) {
            ++checked;
            EXPECT_TRUE(std::isnan(r.sin[i]) && std::isnan(r.cos[i]) && std::isnan(r.sc[i]) && std::isnan(r.cc[i]))
                << "trig(" << x << ") = " << r.sin[i] << ", " << r.cos[i] << ", " << r.sc[i] << ", " << r.cc[i];
        } else if (std::fpclassify(x) == FP_ZERO || std::fpclassify(x) == FP_SUBNORMAL) {
            ++checked;
            EXPECT_EQ(bitsOf(r.sin[i]), bitsOf(x)) << std::hexfloat << "sin(" << x << ") = " << r.sin[i];
            EXPECT_EQ(bitsOf(r.sc[i]), bitsOf(x)) << std::hexfloat << "sincos.sin(" << x << ") = " << r.sc[i];
            EXPECT_EQ(r.cos[i], 1.0) << std::hexfloat << "cos(" << x << ") = " << r.cos[i];
            EXPECT_EQ(r.cc[i], 1.0) << std::hexfloat << "sincos.cos(" << x << ") = " << r.cc[i];
        }
    }
    EXPECT_GE(checked, 10) << "the corpus lost its enumerated specials";
}

} // namespace TrigTotalTest
} // namespace aether_tests
