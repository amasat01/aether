// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCertRound.h
 * @brief Certification drivers for floor/ceil/round/trunc/fdim/fmod/fma,
 *        extending `tests/bandmath/BandMathCert.h`. `floor`/`ceil`/`round`/
 *        `trunc` (0-ULP exact) reuse `certifyExactTotalUnary`; `fdim`
 *        reuses `certifySpineAdmittedBinary`; `fmod`'s exact envelope
 *        reuses `certifyExactTotalBinary`'s three-way shape (its span leg
 *        is certified separately in `test_BandMathRound_common.h`). `fma`,
 *        the one ternary op here, gets a new `certifySpineAdmittedTernary`
 *        driver and matching ternary RED-first arms, reusing
 *        `BandMathCert.h`'s perturbation/mutant helpers.
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCert.h"

#include <cstdint>
#include <map>
#include <algorithm>

namespace aether_tests {
namespace bandmath {

// =========================================================================
//  certifySpineAdmittedTernary — fma. SAME contract as
//  certifySpineAdmittedBinary, generalized to three operands: the
//  degraded-row bucket key is the WORST (most negative) floor-distance
//  across all three operands, so a term that individually sits well
//  inside the envelope cannot mask another that does not.
// =========================================================================
template<typename Row, typename TernaryOp>
CertCounts certifySpineAdmittedTernary(const Row* rows, std::size_t n, double bound, TernaryOp op)
{
    CertCounts c;
    std::map<int, std::int64_t> floorBucketMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band b    = bandFromDoubleBits(r.in[1]);
        const Band c3   = bandFromDoubleBits(r.in[2]);
        const Band got  = op(a, b, c3);
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
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage "
                                   << "from a finite input: ref=" << rv << " got=" << gv;
            if (d != ulp::kMismatch) {
                const int fbA = floorBucketOf(doubleOfBits(r.in[0]));
                const int fbB = floorBucketOf(doubleOfBits(r.in[1]));
                const int fbC = floorBucketOf(doubleOfBits(r.in[2]));
                int fb        = fbA;
                if (fbB > -1000000 && (fb <= -1000000 || fbB < fb))
                    fb = fbB;
                if (fbC > -1000000 && (fb <= -1000000 || fbC < fb))
                    fb = fbC;
                if (fb > -1000000 && fb < kBandAdmittedLo) {
                    const int dist = kBandAdmittedLo - fb;
                    auto [it, _]   = floorBucketMaxUlp.try_emplace(dist, 0);
                    it->second     = std::max(it->second, d);
                }
            }
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
    bool floorMonotoneOk       = true;
    std::int64_t prevBucketUlp = -1;
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
//  RED-FIRST ternary arms — same two analogs BandMathCert.h's own
//  docstring names (arm 1: one-FP32-ULP carrier perturbation on the
//  leading limb; arm 2: the "dropped op" identity twin, here "ignore b
//  and c, return a").
// =========================================================================
template<typename Row, typename TernaryOp>
std::size_t countRedUnderCarrierPerturbationTernary(
    const Row* rows, std::size_t n, double bound, TernaryOp op)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]);
        const Band b          = bandFromDoubleBits(r.in[1]);
        const Band c3         = bandFromDoubleBits(r.in[2]);
        const Band got        = perturbHiByOneFloatUlp(op(a, b, c3));
        const double gv       = doubleOfBits(bandToDoubleBits(got));
        const double rv       = doubleOfBits(r.ref);
        const std::int64_t d  = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderIdentityMutantTernary(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a          = bandFromDoubleBits(r.in[0]); // "dropped op": ignore b, c, return a
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
