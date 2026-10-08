// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandLevers_common.h
 * @brief `BandAccum`/`BandAccumVec<N>` conformance — the accumulator slice
 *        of the `BandLevers` suite: `BandLevers.BandAccumKnownAnswerCorpus`,
 *        `BandLevers.BandAccumVecLaneIndependence`,
 *        `BandLevers.BandAccumVecExpressionFormMatchesPerLaneLoop`, plus
 *        `Atan2SectorConstantsAreAMatchedPairOnHost` (the `AtanSector.h`
 *        sector table's matched-pair property).
 *
 * This is a deliberate partial slice of the full `BandLevers` suite: rows
 * exercising `RsqrtCore`/
 * `AtanSector`/the transcendental family/`recip`/`div` composites need
 * shared corpus-builder and archived-reference test-support scaffolding
 * that does not exist here yet, and stay deferred so the narrowing is
 * visible rather than silent.
 *
 * @section swap The ingest swap
 * Every term is built via `BandedReal::fromDouble(t)` (the tier-1 HOST
 * ingest terminal) demoted to `Band` via its own implicit `operator Band()`.
 *
 * @section egress The egress terminals
 * The first two tests call `BandAccum::toDouble()` (the accumulator's
 * host-only double-sum egress) directly rather than reaching around it with
 * the free function `bd::bandToDouble`. The third
 * (`BandAccumVecExpressionFormMatchesPerLaneLoop`) uses
 * `BandAccum::toBandedReal().toBits()` instead — the accumulator's other
 * egress terminal (the codec pack, not the double sum) — for the identical
 * reason: both are bit-exact deterministic functions of the same certified
 * `Band` state, so either terminal decides the claim, and using the pack
 * terminal here exercises coverage `toDouble()` does not.
 */

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/backend/cuda/Launch.h>

#include "aether/banded/banded.h"
#include "aether/view/Item.h"

#include "tests/banded/cert_harness.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace aether_tests {
namespace BandLeversTest {

using aether::banded::Band;
using aether::banded::BandAccum;
using aether::banded::BandAccumVec;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;

// =========================================================================
//  A small local DD (double-double, ~106-bit) tan-near-zero reference,
//  reusing `aether_tests::cert::ddref`'s DD type/ops (`cert_harness.h`,
//  already shared by the Ff1/Ff2 certs) and adding only what THIS test
//  needs on top: `ddDiv` (Newton) and `ddTanSmall` (a term-recurrence
//  sin/cos, valid for `|r| <= pi`, then one DD divide). Pure `double`
//  arithmetic throughout.
// =========================================================================
namespace ddref = aether_tests::cert::ddref;
using DD        = ddref::DD;

inline DD ddQuickTwoSum(double a, double b)
{
    const double s = a + b;
    return DD{ s, b - (s - a) };
}

inline DD ddDiv(DD a, DD b)
{
    const double bv = b.hi + b.lo;
    const double q1 = (a.hi + a.lo) / bv;
    DD r            = ddref::add(a, ddref::neg(ddref::mul(DD{ q1, 0.0 }, b)));
    const double q2 = (r.hi + r.lo) / bv;
    r               = ddref::add(r, ddref::neg(ddref::mul(DD{ q2, 0.0 }, b)));
    const double q3 = (r.hi + r.lo) / bv;
    return ddref::add(ddQuickTwoSum(q1, q2), DD{ q3, 0.0 });
}

/// @brief `sin(r)`, `cos(r)` in double-double by TERM RECURRENCE, valid for
/// `|r| <= pi`. `ddDiv`, never the component-wise `DD{hi/d, lo/d}`, is
/// load-bearing at this magnitude.
inline void ddSinCosSmall(DD r, DD& sOut, DD& cOut)
{
    const DD u = ddref::mul(r, r);
    DD ts = r, ss = r;
    DD tc{ 1.0, 0.0 }, sc{ 1.0, 0.0 };
    for (int j = 1; j <= 34; j++) {
        const double ds = static_cast<double>(2 * j) * static_cast<double>(2 * j + 1);
        ts              = ddref::neg(ddDiv(ddref::mul(ts, u), DD{ ds, 0.0 }));
        ss              = ddref::add(ss, ts);
        const double dc = static_cast<double>(2 * j - 1) * static_cast<double>(2 * j);
        tc              = ddref::neg(ddDiv(ddref::mul(tc, u), DD{ dc, 0.0 }));
        sc              = ddref::add(sc, tc);
    }
    sOut = ss;
    cOut = sc;
}

inline DD ddTanSmall(DD z)
{
    DD sn, cs;
    ddSinCosSmall(z, sn, cs);
    return ddDiv(sn, cs);
}

/// @brief `Band` -> DD, the exact limb sum widened one term at a time (a
/// RAW carrier is not normalized, so this never routes through the store
/// terminal).
inline DD bandToDD(Band b)
{
    return ddref::add(
        ddref::add(DD{ static_cast<double>(b.hi), 0.0 }, DD{ static_cast<double>(b.lo), 0.0 }),
        DD{ static_cast<double>(b.tail), 0.0 });
}

/// @brief The float's bit pattern -- so -0 != +0 and NaN payloads compare.
inline std::uint32_t fbits(float f) { return static_cast<std::uint32_t>(bd::floatAsInt(f)); }

// =========================================================================
//  Bit helpers
// =========================================================================

inline std::uint64_t doubleBits(double d)
{
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

/// @brief `double` -> `Band`, tier-1 host ingest. @see the file docstring's
/// "swap" section.
inline Band bandFromDouble(double t) { return static_cast<Band>(BandedReal::fromDouble(t)); }

// =========================================================================
//  Fixture. Every row here is DEVICEHOST-callable but run from the HOST —
//  BandAccum's own state machine is exercised directly, no kernel needed for
//  these three (the device arm and the fp64-free SASS audit live in
//  test_BandLevers.cu, exercising a NEW aether-only row this port adds).
// =========================================================================

class BandLevers : public ::testing::Test {
};

/// @brief Fold a list of terms into a `BandAccum` via `addTerm`, in order,
/// then read it back through the exemplar's HOST-ONLY `double` egress.
inline double accumSum(const std::vector<double>& terms)
{
    BandAccum acc;
    for (double t : terms)
        acc.addTerm(bandFromDouble(t));
    return acc.toDouble();
}

// -------------------------------------------------------------------------
//  1. Known-answer corpus: sign mix, cancellation, spread magnitude.
// -------------------------------------------------------------------------
TEST_F(BandLevers, BandAccumKnownAnswerCorpus)
{
    struct Row {
        const char* name;
        std::vector<double> terms;
    };
    // Sign mix: small exact dyadic terms, both signs, no cancellation.
    // Cancellation + spread magnitude: +2^20, +2^-10, -2^20, +3.0 in that
    // ORDER -- the running partial sum passes through 2^20 (spread against
    // the 2^-10 term already folded in) before the exact same 2^20 is
    // subtracted back off, leaving the residual.
    // Spread magnitude (6 terms, alternating sign, ~50 binades of spread).
    const Row rows[] = {
        { "sign-mix", { 7.0, 3.5, 10.0, -2.0, -1.5, -0.25, 0.25 } },
        { "cancellation+spread", { 1048576.0, 0.0009765625, -1048576.0, 3.0 } },
        { "spread-magnitude-6",
            { std::ldexp(1.0, 30), -std::ldexp(1.0, 20), std::ldexp(1.0, 10), -1.0,
                std::ldexp(1.0, -10), -std::ldexp(1.0, -20) } },
    };

    for (const Row& r : rows) {
        double ref = 0.0;
        for (double t : r.terms)
            ref += t;
        const double got = accumSum(r.terms);
        std::printf(
            "[BandAccum] %-20s ref=%.17g got=%.17g\n", r.name, ref, got);
        EXPECT_EQ(doubleBits(got), doubleBits(ref))
            << r.name << ": BandAccum did not reproduce the exact double sum";
    }
}

// -------------------------------------------------------------------------
//  2. BandAccumVec<3> lane independence: three different known-answer
//     corpora, one per lane, folded through addTerm/subTerm -- each lane
//     must land on its OWN answer and not leak into the others.
// -------------------------------------------------------------------------
TEST_F(BandLevers, BandAccumVecLaneIndependence)
{
    BandAccumVec<3> vec;

    const double lane0[] = { 5.0, -2.0, 1.25 }; // addTerm only
    const double lane1[] = { 100.0, 0.5 };      // addTerm, then subTerm
    const double lane2Add = std::ldexp(1.0, 15);
    const double lane2Sub = std::ldexp(1.0, 15) - 1.0;

    for (double t : lane0)
        vec.lane(0).addTerm(bandFromDouble(t));

    vec.lane(1).addTerm(bandFromDouble(lane1[0]));
    vec.lane(1).addTerm(bandFromDouble(lane1[1]));
    vec.lane(1).subTerm(bandFromDouble(99.0));

    vec.lane(2).addTerm(bandFromDouble(lane2Add));
    vec.lane(2).subTerm(bandFromDouble(lane2Sub));

    const double ref0 = 5.0 - 2.0 + 1.25;
    const double ref1 = 100.0 + 0.5 - 99.0;
    const double ref2 = lane2Add - lane2Sub;

    const double got0 = vec.lane(0).toDouble();
    const double got1 = vec.lane(1).toDouble();
    const double got2 = vec.lane(2).toDouble();

    std::printf("[BandAccumVec] lane0 ref=%.17g got=%.17g\n", ref0, got0);
    std::printf("[BandAccumVec] lane1 ref=%.17g got=%.17g\n", ref1, got1);
    std::printf("[BandAccumVec] lane2 ref=%.17g got=%.17g\n", ref2, got2);

    EXPECT_EQ(doubleBits(got0), doubleBits(ref0));
    EXPECT_EQ(doubleBits(got1), doubleBits(ref1));
    EXPECT_EQ(doubleBits(got2), doubleBits(ref2))
        << "lane 2 must equal ITS OWN corpus, not lane 0's or lane 1's -- a "
           "cross-lane leak would still pass rows 1-2 above";
}

// -------------------------------------------------------------------------
//  3. BandAccumVec<N>::addTerm(Expression)/subTerm(Expression) -- the
//     vector-level overload must be BIT-IDENTICAL to the hand-rolled
//     per-lane loop. Zero coverage existed for this overload on any
//     carrier's *Band* sibling before this closure.
// -------------------------------------------------------------------------
TEST_F(BandLevers, BandAccumVecExpressionFormMatchesPerLaneLoop)
{
    constexpr int L = 4;
    using ItemN     = aether::Item<Band, L>;

    BandAccumVec<L> accVec;
    BandAccumVec<L> accLane;

    for (int s = 0; s < 20; s++) {
        ItemN item;
        [&]<std::size_t... d>(std::index_sequence<d...>) {
            ((item.template get<d>() = bandFromDouble(
                  std::sin(0.29 * s + 0.71 * static_cast<double>(d) + 0.4)
                  * std::ldexp(1.0, ((s + 2 * static_cast<int>(d)) % 9) - 4))),
                ...);
        }(std::make_index_sequence<L>{});

        accVec.addTerm(item);
        [&]<std::size_t... d>(std::index_sequence<d...>) {
            ((accLane.lane(static_cast<int>(d)).addTerm(item.template get<d>())), ...);
        }(std::make_index_sequence<L>{});
    }

    int mismatches = 0;
    for (int d = 0; d < L; d++) {
        const std::uint64_t vBits = accVec.lane(d).toBandedReal().toBits();
        const std::uint64_t lBits = accLane.lane(d).toBandedReal().toBits();
        if (vBits != lBits) {
            mismatches++;
            std::printf("[BandAccumVec/expr] lane %d MISMATCH: vec=0x%016llx "
                        "lane-loop=0x%016llx\n",
                d, static_cast<unsigned long long>(vBits),
                static_cast<unsigned long long>(lBits));
        }
    }
    EXPECT_EQ(mismatches, 0)
        << "BandAccumVec::addTerm(Expression) must recurse into the exact "
           "same per-lane addTerm sequence the hand-rolled loop runs -- "
           "same lane order, same Band adds, so the two are the same "
           "computation and must agree bit for bit, not merely closely";
}

// -------------------------------------------------------------------------
//  The sector table's matched-pair property: each stored `t_i` really is
//  `tan` of its paired stored `phi_i` (the derivation order the file header
//  describes), scored against the local DD tan-near-zero reference above,
//  plus the head sector's EXACT-zero row and the selector's monotonicity.
// -------------------------------------------------------------------------
TEST_F(BandLevers, Atan2SectorConstantsAreAMatchedPairOnHost)
{
    // The five sectors, reached through the SHIPPED selector at a
    // representative ratio inside each one (the head, then the four centres).
    const float probe[5] = { 0.02f, 0.17f, 0.36f, 0.57f, 0.83f };
    double worstRel       = 0.0;
    int worstSec           = -1;
    std::uint64_t seen     = 0;
    for (int i = 0; i < 5; i++) {
        const bd::AtanSector sec = bd::atanSector(probe[i], 1.0f);
        const DD phi = bandToDD(sec.phi);
        const DD tv  = bandToDD(sec.t);
        if (i == 0) {
            // the head is EXACTLY zero in both constants, and that is the
            // whole of the near-zero relative-accuracy argument
            EXPECT_EQ(fbits(sec.t.hi), 0u);
            EXPECT_EQ(fbits(sec.t.lo), 0u);
            EXPECT_EQ(fbits(sec.t.tail), 0u);
            EXPECT_EQ(fbits(sec.phi.hi), 0u);
            EXPECT_EQ(fbits(sec.phi.lo), 0u);
            EXPECT_EQ(fbits(sec.phi.tail), 0u);
            seen++;
            continue;
        }
        const DD tanPhi = ddTanSmall(phi);
        const DD d      = ddref::add(tanPhi, ddref::neg(tv));
        const double rel = std::fabs(d.hi + d.lo) / std::fabs(tv.hi + tv.lo);
        if (rel > worstRel) {
            worstRel = rel;
            worstSec = i;
        }
        seen++;
    }
    // 2^-70 is the carrier's own three-limb resolution; the pair may not be
    // worse than the constants themselves can be.
    const double kPairBound = std::ldexp(1.0, -70);
    std::printf("[BandLever/host] sector matched pairs over %llu sectors: "
                "worst |tan(phi_i) - T_i|/|T_i| = %.4e = 2^%.2f at sector %d "
                "(bound 2^-70)\n",
        static_cast<unsigned long long>(seen), worstRel,
        worstRel > 0.0 ? std::log2(worstRel) : -999.0, worstSec);
    EXPECT_EQ(seen, 5u);
    EXPECT_GT(worstRel, 0.0)
        << "every sector's tan(phi) matched its stored T to the last bit of a "
           "double-double, which means the comparison is not resolving "
           "anything at all";
    EXPECT_LT(worstRel, kPairBound)
        << "a sector's stored tangent is NOT the tangent of its stored base "
           "angle. This identity is exact for the pair it is given, so a "
           "mismatch here is a reduction error nothing downstream recovers";
    // and the selector must be monotone and must reach every sector
    EXPECT_TRUE(bd::kBandAtanSecB0 < bd::kBandAtanSecB1 && bd::kBandAtanSecB1 < bd::kBandAtanSecB2
        && bd::kBandAtanSecB2 < bd::kBandAtanSecB3);
}

} // namespace BandLeversTest
} // namespace aether_tests
