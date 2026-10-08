// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCert.h
 * @brief Shared host-only certification harness for Band's scalar ops
 *        (abs/copysign/fmax/fmin/add/sub/mul): one driver per admission
 *        category (`certifyExactTotal` for 0-ULP exact ops,
 *        `certifySpineAdmitted` for bounded ops with a degraded/rejected
 *        tail), plus two non-vacuity arms on the checker itself
 *        (`countRedUnderCarrierPerturbation`, a one-ULP carrier nudge, and
 *        `countRedUnderIdentityMutant`, a forgot-the-op stand-in), each
 *        expected to fail. `bandToIEEE` has no device overload, so this
 *        header is host-only; `test_BandMathCert.cpp` and
 *        `test_BandMathCert.cu` both include it verbatim to register the
 *        same tests on both trees.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"
#include "tools/ulp_oracle/ulp.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <algorithm>
#include <vector>

namespace aether_tests {
namespace bandmath {

using aether::banded::Band;
namespace bd = aether::banded::detail;
namespace ulp = aether_tools::ulp;

// =========================================================================
//  double <-> Band, via the certified UNCHECKED ingest/egress
//  (aether::banded::bandFromIEEE / bandToIEEE) — NOT
//  BandedReal::fromDouble, which throws outside its tier-1 storage window
//  and would reject the very degraded/rejected corpus rows this harness
//  exists to exercise.
// =========================================================================
inline std::uint64_t doubleBitsOf(double x)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

inline double doubleOfBits(std::uint64_t b)
{
    double x = 0.0;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

inline Band bandFromDoubleBits(std::uint64_t bits)
{
    const std::uint32_t lo = static_cast<std::uint32_t>(bits);
    const std::uint32_t hi = static_cast<std::uint32_t>(bits >> 32);
    return bd::bandFromIEEE(lo, hi);
}

inline std::uint64_t bandToDoubleBits(Band b)
{
    std::uint32_t lo = 0, hi = 0;
    bd::bandToIEEE(b, lo, hi);
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

// One FP32-ULP nudge on the leading limb — the RED-first carrier
// perturbation (see file docstring, arm 1).
inline Band perturbHiByOneFloatUlp(Band b)
{
    // Push away from zero (magnitude increase) by exactly one FP32 ULP;
    // for hi == 0 there is no sign to read, so push toward +inf.
    const float target = (b.hi > 0.0f) ? std::numeric_limits<float>::infinity()
        : (b.hi < 0.0f) ? -std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::infinity();
    const float h = std::nextafterf(b.hi, target);
    return Band{ h, b.lo, b.tail };
}

// =========================================================================
//  Counters returned by every driver — feeds the report's "row counts"
//  contract directly.
// =========================================================================
struct CertCounts {
    std::size_t inDomainTotal   = 0;
    std::size_t inDomainPass    = 0;
    std::size_t degradedTotal   = 0;
    std::size_t degradedPass    = 0;
    std::size_t rejectedTotal   = 0;
    std::size_t rejectedPass    = 0;
    std::int64_t maxInDomainUlp = 0;
};

// =========================================================================
//  certifyExactTotal — abs/copysign/fmax/fmin. Bound is 0 ULP verbatim from
//  the contract table ("total, exact") for IN_DOMAIN rows; the op ITSELF
//  adds no error at any magnitude, but the round trip through the carrier
//  (`bandFromIEEE`/`bandToIEEE`) is only LOSSLESS inside the same admission
//  window every op shares (see gen_corpus.py's module note on this file's
//  own earlier, wrong, "unconditionally in_domain" draft) — so degraded/
//  rejected rows get the SAME weak claims certifySpineAdmitted uses,
//  parameterized by bound=0 here instead of 2.
// =========================================================================
template<typename Row, typename UnaryOp>
CertCounts certifyExactTotalUnary(const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    CertCounts c;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) ? (d <= static_cast<std::int64_t>(bound))
                                                     : (ulp::classify(rv) == ulp::classify(gv));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in=0x" << std::hex << r.in[0]
                               << std::dec << " ref=" << rv << " got=" << gv << " ulp=" << d;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref="
                                   << rv << " got=" << gv;
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch)
                || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    return c;
}

template<typename Row, typename BinaryOp>
CertCounts certifyExactTotalBinary(const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    CertCounts c;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band b    = bandFromDoubleBits(r.in[1]);
        const Band got  = op(a, b);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) ? (d <= static_cast<std::int64_t>(bound))
                                                     : (ulp::classify(rv) == ulp::classify(gv));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in0=0x" << std::hex << r.in[0]
                               << " in1=0x" << r.in[1] << std::dec << " ref=" << rv << " got=" << gv
                               << " ulp=" << d;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref="
                                   << rv << " got=" << gv;
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch)
                || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    return c;
}

// =========================================================================
//  certifySpineAdmitted — add/sub/mul(/fma). Label-branched per row.
// =========================================================================
// SAME constant gen_corpus.py's K_BAND_ADMITTED_LO derives (Band.h's own
// bd::kBandCarrierFloorExp + bd::kBandAdmissionMargin = -96 + 8 = -88).
inline constexpr int kBandAdmittedLo = bd::kBandCarrierFloorExp + bd::kBandAdmissionMargin;

// Base-2 exponent of the leading operand's OWN bit pattern, used only to
// BUCKET degraded rows for the monotonicity check below — independent of
// row emission order (a first draft tracked "the previous row's ulp",
// implicitly assuming generation order = severity order; that broke the
// moment several MANTISSA points for the SAME exponent step interleaved
// before the next step began, comparing within-step jitter as if it were
// between-step degradation). 0/inf/nan read as INT64_MIN (never matches a
// real bucket).
inline int floorBucketOf(double x)
{
    if (x == 0.0 || !std::isfinite(x))
        return -1000000;
    int e         = 0;
    const double m = std::frexp(x, &e);
    (void)m;
    return e - 1; // ldexp(mantissa in [1,2), e-1) convention, matches gen_corpus.py
}

template<typename Row, typename BinaryOp>
CertCounts certifySpineAdmittedBinary(const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    CertCounts c;
    // distance-past-the-admitted-floor-edge -> worst |ulp| seen at that
    // distance, aggregated over EVERY row/mantissa that lands there,
    // regardless of position in the array.
    std::map<int, std::int64_t> floorBucketMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band b    = bandFromDoubleBits(r.in[1]);
        const Band got  = op(a, b);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) && (d <= static_cast<std::int64_t>(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": ref=" << rv << " got=" << gv
                               << " ulp=" << d << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            // Weak claim (no per-op closed-form degradation curve exists
            // for add/sub/mul):
            // the result must stay FINITE — a degraded row is allowed to
            // lose accuracy, never to produce NaN/garbage from a
            // well-defined finite input pair.
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage "
                                   << "from a finite input: ref=" << rv << " got=" << gv;
            if (d != ulp::kMismatch) {
                const int fb = floorBucketOf(doubleOfBits(r.in[0]));
                if (fb > -1000000 && fb < kBandAdmittedLo) {
                    const int dist    = kBandAdmittedLo - fb;
                    auto [it, _]      = floorBucketMaxUlp.try_emplace(dist, 0);
                    it->second        = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            // The documented escape: EITHER the certified bound is
            // violated (result diverges — the failure the corpus exists to
            // exercise) OR the candidate itself goes non-finite (FP32
            // overflow past the hard ceiling). A silent in-bound PASS past
            // a hard limit is what this arm exists to catch.
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch)
                || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    // Monotonicity, aggregated by distance-past-the-floor-edge bucket (not
    // row order): non-decreasing worst-case |ulp| as distance increases,
    // one dip tolerated (adjacent-bucket noise), a LARGE regression is not.
    bool floorMonotoneOk        = true;
    std::int64_t prevBucketUlp  = -1;
    for (const auto& [dist, worstUlp] : floorBucketMaxUlp) {
        (void)dist;
        if (prevBucketUlp >= 0 && worstUlp < prevBucketUlp) {
            if (prevBucketUlp - worstUlp > prevBucketUlp / 2 + 4)
                floorMonotoneOk = false;
        }
        prevBucketUlp = worstUlp;
    }
    EXPECT_TRUE(floorMonotoneOk) << "floor-side degraded family is not monotone non-decreasing";
    return c;
}

// =========================================================================
//  RED-FIRST arms (file docstring). Applied to the IN-DOMAIN rows only —
//  the arms exist to prove the BOUND CHECK has teeth, which is only a
//  meaningful question where a bound is actually asserted.
// =========================================================================
template<typename Row, typename UnaryOp>
std::size_t countRedUnderCarrierPerturbationUnary(
    const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]);
        const Band got        = perturbHiByOneFloatUlp(op(a));
        const double gv       = doubleOfBits(bandToDoubleBits(got));
        const double rv       = doubleOfBits(r.ref);
        const std::int64_t d  = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row, typename BinaryOp>
std::size_t countRedUnderCarrierPerturbationBinary(
    const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]);
        const Band b          = bandFromDoubleBits(r.in[1]);
        const Band got        = perturbHiByOneFloatUlp(op(a, b));
        const double gv       = doubleOfBits(bandToDoubleBits(got));
        const double rv       = doubleOfBits(r.ref);
        const std::int64_t d  = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderIdentityMutantUnary(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]); // "dropped op": return a unchanged
        const double gv       = doubleOfBits(bandToDoubleBits(a));
        const double rv       = doubleOfBits(r.ref);
        const std::int64_t d  = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderIdentityMutantBinary(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]); // "dropped op": ignore b, return a
        const double gv       = doubleOfBits(bandToDoubleBits(a));
        const double rv       = doubleOfBits(r.ref);
        const std::int64_t d  = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

} // namespace bandmath
} // namespace aether_tests
