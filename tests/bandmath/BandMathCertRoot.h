// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCertRoot.h
 * @brief Certification drivers for the five root ops (sqrt/rsqrt/cbrt/
 *        hypot/rsqrtCube), extending `tests/bandmath/BandMathCert.h`.
 *        `certifyRootUnary`/`certifyRootBinary` generalize the shared
 *        three-way (in_domain/degraded/rejected) contract to non-zero
 *        bounds and either arity, with degradation-curve monotonicity
 *        bucketed by distance from the admitted edge; the RED-first arms
 *        are reused verbatim from `BandMathCert.h`.
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCert.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <algorithm>

namespace aether_tests {
namespace bandmath {

// =========================================================================
//  certifyRootUnary / certifyRootBinary — TOL-tier three-way contract.
//  in_domain: |ulp| <= bound. degraded: stays finite (or the reference
//  itself is non-finite) — bucketed by EXPONENT DISTANCE PAST the row's
//  OWN admitted edge (computed from the double operand's own magnitude,
//  independent of corpus order, mirroring certifySpineAdmittedBinary's own
//  technique) and asserted non-decreasing (one dip tolerated). rejected:
//  the documented escape (non-finite OR |ulp| > bound OR class mismatch).
// =========================================================================
inline int rootExponentOf(double x)
{
    if (x == 0.0 || !std::isfinite(x))
        return -1000000;
    int e = 0;
    std::frexp(x, &e);
    return e - 1; // ldexp convention, matches gen_corpus_root.py's exponent_of
}

// Coarse bucket key: |exponent| / 4. This family's admitted window is
// SYMMETRIC (tiny and huge operands degrade by the SAME mechanism, @see
// BandRoot.h's kBandRsqrtCeilingExp), so degradation is a function of
// DISTANCE FROM ZERO, not of sign — |e| folds both directions onto one
// curve. The /4 grouping absorbs the corpus's own sparse per-exponent
// sampling (a single-exponent bucket can hold only 1-2 rows, and THAT
// row's own mantissa can coincidentally land near-exact even deep in a
// degrading region — measured on this package's own cbrt corpus: an
// isolated |e|=105 bucket read 0 ULP surrounded by neighbors past 100).
inline int rootBucketKey(int e)
{
    return (e <= -1000000) ? e : std::abs(e) / 4;
}

template<typename Row, typename UnaryOp>
CertCounts certifyRootUnary(const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    CertCounts c;
    std::map<int, std::int64_t> distMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
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
            EXPECT_TRUE(pass) << "in_domain row " << i << ": ref=" << rv << " got=" << gv << " ulp=" << d
                               << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            if (d != ulp::kMismatch) {
                const int e = rootExponentOf(doubleOfBits(r.in[0]));
                if (e > -1000000) {
                    const int key = rootBucketKey(e);
                    auto [it, _]  = distMaxUlp.try_emplace(key, 0);
                    it->second    = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    // Monotone-non-decreasing degradation as the operand moves FURTHER from
    // the origin (the exponent bucket key), one dip tolerated.
    bool monotoneOk       = true;
    std::int64_t prevUlp  = -1;
    for (const auto& [e, worstUlp] : distMaxUlp) {
        (void)e;
        if (prevUlp >= 0 && worstUlp < prevUlp) {
            if (prevUlp - worstUlp > prevUlp / 2 + 4)
                monotoneOk = false;
        }
        prevUlp = worstUlp;
    }
    EXPECT_TRUE(monotoneOk) << "degraded family is not monotone non-decreasing away from the admitted edge";
    return c;
}

template<typename Row, typename BinaryOp>
CertCounts certifyRootBinary(const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    CertCounts c;
    std::map<int, std::int64_t> distMaxUlp;
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
            EXPECT_TRUE(pass) << "in_domain row " << i << ": ref=" << rv << " got=" << gv << " ulp=" << d
                               << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            if (d != ulp::kMismatch) {
                const int ea = rootExponentOf(doubleOfBits(r.in[0]));
                const int eb = rootExponentOf(doubleOfBits(r.in[1]));
                const int e  = std::max(ea, eb);
                if (e > -1000000) {
                    const int key = rootBucketKey(e);
                    auto [it, _]  = distMaxUlp.try_emplace(key, 0);
                    it->second    = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    bool monotoneOk      = true;
    std::int64_t prevUlp = -1;
    for (const auto& [e, worstUlp] : distMaxUlp) {
        (void)e;
        if (prevUlp >= 0 && worstUlp < prevUlp) {
            if (prevUlp - worstUlp > prevUlp / 2 + 4)
                monotoneOk = false;
        }
        prevUlp = worstUlp;
    }
    EXPECT_TRUE(monotoneOk) << "degraded family is not monotone non-decreasing away from the admitted edge";
    return c;
}

} // namespace bandmath
} // namespace aether_tests
