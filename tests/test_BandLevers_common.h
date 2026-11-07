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
#include <type_traits>
#include <random>
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
//  4. The compensated fold: a series that cancels by 2^30 over 2^19 terms.
//  The reference is an EXACT sum (a Shewchuk grow-expansion in `double`),
//  so the measured error is the fold's own, not the reference's.
// -------------------------------------------------------------------------

/// @brief Exactly add `x` into the non-overlapping expansion `e` (smallest first).
inline void growExpansion(std::vector<double>& e, double x)
{
    std::vector<double> h;
    double              q = x;
    for (double ei : e) {
        const double sum = q + ei, bb = sum - q, err = (q - (sum - bb)) + (ei - bb);
        if (err != 0.0)
            h.push_back(err);
        q = sum;
    }
    h.push_back(q);
    e.swap(h);
}

inline double expansionValue(const std::vector<double>& e)
{
    double v = 0.0;
    for (double c : e)
        v += c;
    return v;
}

struct CancelRun {
    double ulp53;      ///< |result - exact| in ULPs (53 bits) of the exact sum
    double err;        ///< |result - exact|
    double bound;      ///< 18*K^3*2^-95*sum(|acc_k|+|x_k|) + 2^-72*|exact|
};

/// @brief The accumulator's derived error bound (Accum.h) for renormalise period `K`.
inline double accumBound(double K, double sumAbs, double S)
{
    return 18.0 * K * K * K * std::ldexp(sumAbs, -95) + std::ldexp(std::fabs(S), -72);
}

/// @brief Fold the cancel-2^-30 class (pairs `a`, `-(a - a*2^-30*u)`) with `lanes` accumulators
/// (round-robin), merge them by a combine tree, and score the result against the exact sum.
inline CancelRun cancelClass(int n, int lanes)
{
    std::mt19937_64                        g(1234);
    std::uniform_real_distribution<double> u(1.0, 2.0);
    std::uniform_int_distribution<int>     ex(-40, 40);
    std::vector<double>                    v(static_cast<std::size_t>(n));
    std::vector<double>                    exact;
    double                                 sumAbs = 0.0;
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0)
            v[i] = std::ldexp(u(g), ex(g));
        else
            v[i] = -(v[i - 1] - v[i - 1] * std::ldexp(u(g) - 1.0, -30));
        growExpansion(exact, v[i]);
        sumAbs += std::fabs(v[i]);
    }
    std::vector<BandAccum> acc(static_cast<std::size_t>(lanes));
    for (int i = 0; i < n; ++i) {
        BandAccum& a = acc[static_cast<std::size_t>(i % lanes)];
        sumAbs += std::fabs(static_cast<double>(a.s.hi));
        a.addTerm(bandFromDouble(v[i]));
    }
    for (int stride = 1; stride < lanes; stride *= 2)
        for (int i = 0; i + stride < lanes; i += 2 * stride) {
            sumAbs += std::fabs(static_cast<double>(acc[i].s.hi)) + std::fabs(static_cast<double>(acc[i + stride].s.hi));
            acc[i].combine(acc[i + stride]);
        }
    const Band r = acc[0].normalized();
    std::vector<double> diff = exact;
    growExpansion(diff, -static_cast<double>(r.hi));
    growExpansion(diff, -static_cast<double>(r.lo));
    growExpansion(diff, -static_cast<double>(r.tail));
    const double S   = expansionValue(exact);
    const double err = std::fabs(expansionValue(diff));
    int          e   = 0;
    std::frexp(S, &e);
    return { err / std::ldexp(1.0, e - 53), err, accumBound(16, sumAbs, S) };
}

template<unsigned K>
inline CancelRun adversarialClass(int n)
{
    aether::banded::BandAccumK<K> a;
    std::vector<double>           exact;
    double                        sumAbs = 0.0;
    const Band                    one{ 1.0f, 0.0f, 0.0f };
    a.addTerm(one);
    growExpansion(exact, 1.0);
    for (int i = 1; i < n; ++i) {
        const Band y{ 1.2345678f, 0x1.3c6ef2p-26f, 0x1.b2a416p-51f };
        sumAbs += std::fabs(static_cast<double>(a.s.hi)) + std::fabs(static_cast<double>(y.hi));
        a.addTerm(y);
        growExpansion(exact, static_cast<double>(y.hi));
        growExpansion(exact, static_cast<double>(y.lo));
        growExpansion(exact, static_cast<double>(y.tail));
    }
    const Band r = a.normalized();
    std::vector<double> diff = exact;
    growExpansion(diff, -static_cast<double>(r.hi));
    growExpansion(diff, -static_cast<double>(r.lo));
    growExpansion(diff, -static_cast<double>(r.tail));
    const double S   = expansionValue(exact);
    const double err = std::fabs(expansionValue(diff));
    int          e   = 0;
    std::frexp(S, &e);
    return { err / std::ldexp(1.0, e - 53), err, accumBound(K, sumAbs, S) };
}

TEST_F(BandLevers, BandAccumCancelClassSequentialFoldWithinBound)
{
    const CancelRun r = cancelClass((1 << 19) - 1, 1);
    std::printf("[BandAccum/cancel-2^-30 sequential] ulp53=%.5f err=%.3e bound=%.3e\n", r.ulp53, r.err, r.bound);
    EXPECT_LE(r.ulp53, 0.01);
    EXPECT_LE(r.err, r.bound);
}

TEST_F(BandLevers, BandAccumAdversarialPlainSumWithinDerivedBound)
{
    const CancelRun r = adversarialClass<16>(1 << 19);
    std::printf("[BandAccum/adversarial K=16] ulp53=%.5f err=%.3e bound=%.3e\n", r.ulp53, r.err, r.bound);
    EXPECT_LE(r.ulp53, 0.25);
    EXPECT_LE(r.err, r.bound);
}

TEST_F(BandLevers, BandAccumAdversarialNeverRenormalisedExceedsTheBound)
{
    // The same corpus with the renormalise period pushed past the term count: the n-growth of the
    // error lane breaks the K = 16 bound, which is what the periodic fold prevents.
    const CancelRun bad  = adversarialClass<(1u << 20)>(1 << 19);
    const CancelRun good = adversarialClass<16>(1 << 19);
    std::printf("[BandAccum/adversarial K=2^20] ulp53=%.5f err=%.3e vs the K=16 bound %.3e\n", bad.ulp53,
        bad.err, good.bound);
    EXPECT_GT(bad.err, good.bound);
}

// -------------------------------------------------------------------------
//  Partials: the 16-byte scratch view of a BandAccum and the ordered block combine.
// -------------------------------------------------------------------------
static_assert(sizeof(aether::banded::BandPartial) == 16 && alignof(aether::banded::BandPartial) == 16);
static_assert(std::is_trivially_copyable_v<aether::banded::BandPartial>
    && std::is_standard_layout_v<aether::banded::BandPartial>);

TEST_F(BandLevers, BandPartialRoundTripFoldsIdentically)
{
    BandAccum a, zero;
    for (int i = 0; i < 100; ++i)
        a.addTerm(bandFromDouble(std::ldexp(1.0 + 0.01 * i, (i % 17) - 8) * ((i & 1) ? -1.0 : 1.0)));
    BandAccum b = BandAccum::fromPartial(a.partial());
    for (int i = 0; i < 100; ++i) {
        const Band t = bandFromDouble(std::ldexp(1.0 + 0.003 * i, (i % 13) - 6));
        a.addTerm(t);
        b.addTerm(t);
    }
    const Band na = a.normalized(), nb = b.normalized();
    EXPECT_EQ(std::memcmp(&na, &nb, sizeof(Band)), 0);
    // all-zero bytes are the zero partial
    aether::banded::BandPartial z;
    std::memset(&z, 0, sizeof z);
    EXPECT_TRUE(BandAccum::fromPartial(z) == zero);
    const aether::banded::BandPartial pz = zero.partial();
    EXPECT_EQ(std::memcmp(&pz, &z, sizeof z), 0);
}

/// @brief The block-order combine of per-block partials over a class: the carrier figure and the delivered-cell
/// figure against the exact sum, and the carrier's bits for the repeat check.
struct PartialsRun {
    double carrierUlp53, deliveredUlp53;
    Band   carrier;
};

inline PartialsRun partialsClass(bool cancel, int n, int blocks)
{
    std::mt19937_64                        g(cancel ? 1234 : 4321);
    std::uniform_real_distribution<double> u(1.0, 2.0);
    std::uniform_int_distribution<int>     ex(-40, 40);
    std::vector<double>                    v(static_cast<std::size_t>(n)), exact;
    for (int i = 0; i < n; ++i) {
        if (!cancel)
            v[i] = std::ldexp(u(g), ex(g)) * ((g() & 1) ? 1.0 : -1.0);
        else if (i % 2 == 0)
            v[i] = std::ldexp(u(g), ex(g));
        else
            v[i] = -(v[i - 1] - v[i - 1] * std::ldexp(u(g) - 1.0, -30));
        growExpansion(exact, v[i]);
    }
    std::vector<aether::banded::BandPartial> parts(static_cast<std::size_t>(blocks));
    // block b folds the terms i = b, b + blocks, ... (the strided assignment of a grid-stride loop), in order
    for (int b = 0; b < blocks; ++b) {
        BandAccum a;
        for (int i = b; i < n; i += blocks)
            a.addTerm(bandFromDouble(v[i]));
        parts[b] = a.partial();
    }
    BandAccum total;
    for (int b = 0; b < blocks; ++b)
        total.combine(BandAccum::fromPartial(parts[b]));
    const Band r = total.normalized();
    auto score = [&](double l0, double l1, double l2) {
        std::vector<double> diff = exact;
        growExpansion(diff, -l0);
        growExpansion(diff, -l1);
        growExpansion(diff, -l2);
        int e = 0;
        std::frexp(expansionValue(exact), &e);
        return std::fabs(expansionValue(diff)) / std::ldexp(1.0, e - 53);
    };
    const Band d = static_cast<Band>(BandedReal(r));
    return { score(r.hi, r.lo, r.tail), score(d.hi, d.lo, d.tail), r };
}

TEST_F(BandLevers, BandPartialsBlockOrderCombineIsAccurateAndRepeatable)
{
    for (bool cancel : { true, false }) {
        // the pinned corpus size (2^19 - 1 terms) over 64 blocks
        const int n = (1 << 19) - 1;
        const PartialsRun r1 = partialsClass(cancel, n, 64), r2 = partialsClass(cancel, n, 64),
                          r3 = partialsClass(cancel, n, 64);
        std::printf("[BandPartials/%s] carrier ulp53=%.5f delivered ulp53=%.5f\n", cancel ? "cancel-2^-30" : "uniform",
            r1.carrierUlp53, r1.deliveredUlp53);
        EXPECT_LE(r1.carrierUlp53, 0.01);
        EXPECT_LE(r1.deliveredUlp53, 0.5);
        EXPECT_EQ(std::memcmp(&r1.carrier, &r2.carrier, sizeof(Band)), 0);
        EXPECT_EQ(std::memcmp(&r1.carrier, &r3.carrier, sizeof(Band)), 0);
    }
}

// -------------------------------------------------------------------------
//  A warp-shaped tree combine of accumulators that never reached the renormalise period: 32 lanes of 8 terms each
//  (pairs `a`, `-(a - a * 2^-30 * u)` across 2^-40..2^40), merged by shuffle-down offsets 16, 8, 4, 2, 1. The lane
//  states are raw deferred folds, so the combine's renormalise sees limbs whose tail exceeds the residue of hi + lo.
// -------------------------------------------------------------------------

/// @brief The cancelling series of the warp rows: `n` terms from a fixed LCG (host and device twins share it).
inline std::vector<double> warpCancelSeries(int n)
{
    std::vector<double> v(static_cast<std::size_t>(n));
    unsigned            lcg  = 12345u;
    auto                next = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (lcg >> 8) / double(1 << 24);
    };
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            const double m = 1.0 + next();
            v[i]           = std::ldexp(m, static_cast<int>(next() * 80.0) - 40);
        } else {
            v[i] = -(v[i - 1] - v[i - 1] * std::ldexp(next(), -30));
        }
    }
    return v;
}

/// @brief ULPs at 53 bits of the three-limb result `r` against the exact sum of `v`.
inline double warpCancelUlp53(const std::vector<double>& v, const Band& r)
{
    std::vector<double> exact, diff;
    for (double x : v)
        growExpansion(exact, x);
    diff = exact;
    growExpansion(diff, -static_cast<double>(r.hi));
    growExpansion(diff, -static_cast<double>(r.lo));
    growExpansion(diff, -static_cast<double>(r.tail));
    int e = 0;
    std::frexp(expansionValue(exact), &e);
    return std::fabs(expansionValue(diff)) / std::ldexp(1.0, e - 53);
}

TEST_F(BandLevers, BandAccumWarpShapedTreeOfDeferredLanesIsAccurate)
{
    constexpr int lanes = 32, perLane = 8;
    const std::vector<double> v = warpCancelSeries(lanes * perLane);
    std::vector<BandAccum>    a(lanes);
    for (int l = 0; l < lanes; ++l)
        for (int k = 0; k < perLane; ++k)
            a[l].addTerm(bandFromDouble(v[l * perLane + k]));
    for (int off = lanes / 2; off > 0; off >>= 1) {
        std::vector<BandAccum> next = a;
        for (int i = 0; i < lanes; ++i) {
            BandAccum m = a[i];
            m.combine(i + off < lanes ? a[i + off] : a[i]);
            next[i] = m;
        }
        a = next;
    }
    const double ulp = warpCancelUlp53(v, a[0].normalized());
    std::printf("[BandAccum/warp tree of deferred lanes] ulp53=%.5f\n", ulp);
    EXPECT_LE(ulp, 0.01);
}

// A window whose terms cancel down to the tail level: after `period` terms the renormalise leaves the exact value
// in `lo` alone (hi == 0, not a carrier); every terminal repairs it and delivers the exact 2^-25.
TEST_F(BandLevers, BandAccumCancelledWindowIsExactAtEveryTerminal)
{
    BandAccum a;
    a.addTerm(Band{ 0x1p48f, 0x1p24f, -1.0f });
    a.addTerm(Band{ -0x1p48f, -0x1p24f, 0.0f });
    a.addTerm(Band{ 1.0f, 0x1p-25f, 0.0f });
    for (unsigned i = 3; i < BandAccum::period; ++i)
        a.addTerm(Band{ 0.0f, 0.0f, 0.0f });
    EXPECT_EQ(a.s.hi, 0.0f);
    EXPECT_EQ(a.s.lo, 0x1p-25f);
    EXPECT_EQ(a.s.tail, 0.0f);
    const Band r = a.normalized();
    EXPECT_EQ(r.hi, 0x1p-25f);
    EXPECT_EQ(r.lo, 0.0f);
    EXPECT_EQ(r.tail, 0.0f);
    const aether::banded::BandPartial p = a.partial();
    EXPECT_EQ(p.hi, 0x1p-25f);
    EXPECT_EQ(p.lo + p.tail + p.c, 0.0f);
    EXPECT_EQ(a.toBandedReal().toDouble(), 0x1p-25);
    BandAccum b;
    b.combine(a);
    EXPECT_EQ(b.toBandedReal().toDouble(), 0x1p-25);
}

TEST_F(BandLevers, BandAccumCancelClassCombineTreeWithinBound)
{
    const CancelRun r = cancelClass((1 << 19) - 1, 16384);
    std::printf("[BandAccum/cancel-2^-30 tree] ulp53=%.5f err=%.3e bound=%.3e\n", r.ulp53, r.err, r.bound);
    EXPECT_LE(r.ulp53, 0.01);
    EXPECT_LE(r.err, r.bound);
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
