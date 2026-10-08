// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandIngest_common.h
 * @brief `bandFromIEEE` limb-assembly conformance for `aether::banded`.
 *
 * ONE ROW DROPPED, NOT ADAPTED: `MaterializeFiftyThreeDeliversTheBandToIEEE
 * Value` names `det::materialize<53>` (a demand-typed terminal over
 * `BandedDemandT<53, SoftDouble>`) -- both the demand-ladder selection plane
 * and the `SoftDouble` type it selects between are outside aether's carried
 * surface. It stays `deferred-with-addon`; its device counterpart
 * (`MaterializeFiftyThreeCompilesAndAgreesOnDevice`, `.cu`) is dropped for
 * the same reason.
 *
 * Every other row here uses only `bandFromIEEE` and (where noted) the new
 * carrier-level `bandToIEEE` this port adds beside it in `aether/banded/
 * Band.h` -- see that file for why egress needed adding at all.
 */

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/backend/cuda/Launch.h>

#include "aether/banded/banded.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace aether_tests {
namespace BandIngestTest {

using aether::banded::Band;
namespace bd = aether::banded::detail;

// =========================================================================
//  The independent reference
// =========================================================================

inline std::uint32_t fbits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

/// @brief `bandFromIEEE`'s answer, derived rather than transcribed -- each
/// limb is an integer below 2^24 times a power of two, exact as a `double`
/// for every finite input, narrowed to `float` by ONE correctly-rounded
/// hardware operation.
inline Band referenceBandFromIEEE(std::uint32_t ieee_lo, std::uint32_t ieee_hi)
{
    const bool neg = (ieee_hi & 0x80000000u) != 0u;
    const int bexp = static_cast<int>((ieee_hi >> 20) & 0x7FFu);

    if (bexp == 0)
        return Band{ 0.0f, 0.0f, 0.0f };

    if (bexp == 2047) {
        const std::uint32_t mant_nz  = (ieee_hi & 0xFFFFFu) | ieee_lo;
        const std::uint32_t inf_bits = (ieee_hi & 0x80000000u)
            | (mant_nz ? 0x7FC00000u : 0x7F800000u);
        return Band{ bd::intAsFloat(static_cast<int>(inf_bits)), 0.0f, 0.0f };
    }

    const std::uint32_t top24
        = 0x800000u | ((ieee_hi & 0xFFFFFu) << 3) | (ieee_lo >> 29);
    const std::uint32_t lo24  = (ieee_lo >> 5) & 0xFFFFFFu;
    const std::uint32_t tail5 = ieee_lo & 0x1Fu;
    const int e                = bexp - 1023;

    const double vhi = std::ldexp(static_cast<double>(top24), e - 23);
    const double vlo = std::ldexp(static_cast<double>(lo24), e - 47);
    const double vtl = std::ldexp(static_cast<double>(tail5), e - 52);

    Band r;
    r.hi   = static_cast<float>(neg ? -vhi : vhi);
    r.lo   = (lo24 == 0u) ? 0.0f : static_cast<float>(neg ? -vlo : vlo);
    r.tail = (tail5 == 0u) ? 0.0f : static_cast<float>(neg ? -vtl : vtl);
    return r;
}

// =========================================================================
//  Rows
// =========================================================================

struct Pattern {
    std::uint32_t mantHi;
    std::uint32_t mantLo;
    const char* name;
};

inline const std::vector<Pattern>& patterns()
{
    static const std::vector<Pattern> p = {
        { 0x00000u, 0x00000000u, "1.0 -- lo and tail both absent" },
        { 0x80000u, 0x00000000u, "1.5 -- leading limb only" },
        { 0x00000u, 0x00000001u, "a single bit in the tail" },
        { 0x00000u, 0x0000001Fu, "a saturated tail, no lo" },
        { 0x00000u, 0x00000020u, "a single unit in the lo limb, no tail" },
        { 0x00000u, 0x1FFFFFE0u, "a saturated lo limb, no tail" },
        { 0xFFFFFu, 0xFFFFFFFFu, "all 52 mantissa bits set" },
        { 0x921FBu, 0x54442D18u, "pi -- an irregular real mantissa" },
        { 0xAAAAAu, 0xAAAAAAAAu, "alternating bits" },
        { 0x55555u, 0x55555555u, "alternating bits, opposite phase" },
    };
    return p;
}

inline void wordsFor(
    const Pattern& p, int e, bool neg, std::uint32_t& lo, std::uint32_t& hi)
{
    hi = (neg ? 0x80000000u : 0u)
        | (static_cast<std::uint32_t>(e + 1023) << 20) | p.mantHi;
    lo = p.mantLo;
}

inline constexpr int kWindowLow  = -126;
inline constexpr int kWindowHigh = 127;

inline const std::vector<int>& boundaryExponents()
{
    static const std::vector<int> e = { 100, 0, -40, -74, -75, -76, -78, -80,
        -88, -96, -97, -98, -101, -102, -103, -126 };
    return e;
}

// =========================================================================
//  Depth measurement
// =========================================================================

inline double survivingBits(long double got, long double want)
{
    if (!std::isfinite(static_cast<double>(got)))
        return 0.0;
    if (want == 0.0L)
        return (got == 0.0L) ? 53.0 : 0.0;
    if (got == want)
        return 53.0;
    const long double rel = std::fabs((got - want) / want);
    if (rel >= 1.0L)
        return 0.0;
    double b = -std::log2(static_cast<double>(rel));
    if (b > 53.0)
        b = 53.0;
    if (b < 0.0)
        b = 0.0;
    return b;
}

inline double doubleFromWords(std::uint32_t lo, std::uint32_t hi)
{
    const std::uint64_t u = (static_cast<std::uint64_t>(hi) << 32) | lo;
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

inline double bandRoundTrip(double x)
{
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    const Band b = bd::bandFromIEEE(
        static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32));
    std::uint32_t olo = 0, ohi = 0;
    bd::bandToIEEE(b, olo, ohi);
    return doubleFromWords(olo, ohi);
}

class BandIngest : public ::testing::Test {
};

// -------------------------------------------------------------------------
//  1. Every limb, against the reference, across the carrier's own window.
// -------------------------------------------------------------------------
TEST_F(BandIngest, LimbsMatchTheIndependentReferenceInTheCarrierWindow)
{
    std::uint64_t rows = 0, hiBad = 0, loBad = 0, tailBad = 0;
    int firstE = 1 << 30;
    for (int e = kWindowLow; e <= kWindowHigh; e++)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band got  = bd::bandFromIEEE(lo, hi);
                const Band want = referenceBandFromIEEE(lo, hi);
                rows++;
                bool bad = false;
                if (fbits(got.hi) != fbits(want.hi)) { hiBad++; bad = true; }
                if (fbits(got.lo) != fbits(want.lo)) { loBad++; bad = true; }
                if (fbits(got.tail) != fbits(want.tail)) { tailBad++; bad = true; }
                if (bad && e < firstE)
                    firstE = e;
            }
    std::printf("[BandIngest] window sweep: %llu rows, hi %llu bad, lo %llu "
                "bad, tail %llu bad\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(hiBad),
        static_cast<unsigned long long>(loBad),
        static_cast<unsigned long long>(tailBad));
    if (firstE != (1 << 30))
        std::printf("   highest exponent with any bad limb: 2^%d\n", firstE);
    EXPECT_EQ(hiBad, 0u) << "the leading limb disagrees with its exact value";
    EXPECT_EQ(loBad, 0u) << "the lo limb is not the correctly-rounded float "
                            "nearest lo24 * 2^(e-47)";
    EXPECT_EQ(tailBad, 0u) << "the tail limb is not the correctly-rounded "
                              "float nearest tail5 * 2^(e-52)";
    EXPECT_GT(rows, 4000u);
}

// -------------------------------------------------------------------------
//  2. The lower limbs, everywhere a finite double can reach.
// -------------------------------------------------------------------------
TEST_F(BandIngest, LowerLimbsMatchTheReferenceAcrossEveryFiniteDouble)
{
    std::uint64_t rows = 0, bad = 0;
    for (int e = -1022; e <= 1023; e += 7)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band got  = bd::bandFromIEEE(lo, hi);
                const Band want = referenceBandFromIEEE(lo, hi);
                rows++;
                if (fbits(got.lo) != fbits(want.lo)
                    || fbits(got.tail) != fbits(want.tail))
                    bad++;
            }
    std::printf("[BandIngest] full double range, lower limbs: %llu rows, "
                "%llu bad\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u)
        << "a lower limb is wrong somewhere in the double's range -- the "
           "construction must be total, not merely correct where it was tested";
    EXPECT_GT(rows, 4000u);
}

// -------------------------------------------------------------------------
//  3. Known answers, written out, at the exponents where behaviour turns.
// -------------------------------------------------------------------------
TEST_F(BandIngest, KnownAnswerVectorsAtTheBoundaryExponents)
{
    struct Anchor {
        int e;
        std::uint32_t mantLo;
        std::uint32_t wantTail;
        const char* why;
    };
    const Anchor anchors[] = {
        { -74, 0x1u, 0x00800000u, "2^-126, the smallest normal float" },
        { -76, 0x1u, 0x00200000u, "2^-128, subnormal and exact" },
        { -88, 0x1u, 0x00000200u, "2^-140, subnormal and exact" },
        { -96, 0x1u, 0x00000002u, "2^-148, two subnormal steps" },
        { -97, 0x1u, 0x00000001u, "2^-149, the last exactly representable" },
        { -98, 0x1u, 0x00000000u,
            "2^-150 is exactly half a step and ties to even, so zero" },
        { -101, 0x1Fu, 0x00000002u,
            "31 * 2^-153 is 1.9375 steps and rounds to two" },
        { -103, 0x1Fu, 0x00000000u,
            "31 * 2^-155 is under half a step and rounds away" },
    };
    for (const Anchor& a : anchors)
        for (int s = 0; s < 2; s++) {
            std::uint32_t lo = 0, hi = 0;
            const Pattern p{ 0x00000u, a.mantLo, "anchor" };
            wordsFor(p, a.e, s == 1, lo, hi);
            const Band got             = bd::bandFromIEEE(lo, hi);
            const std::uint32_t want = a.wantTail | (s == 1 ? 0x80000000u : 0u);
            EXPECT_EQ(fbits(got.tail), want)
                << "tail at 2^" << a.e << (s == 1 ? " negative" : " positive")
                << ": expected " << a.why;
        }

    struct LoAnchor {
        int e;
        std::uint32_t mantLo;
        std::uint32_t wantLo;
        const char* why;
    };
    const LoAnchor loAnchors[] = {
        { -102, 0x20u, 0x00000001u, "2^-149, the last exactly representable" },
        { -103, 0x20u, 0x00000000u, "2^-150 ties to even, so zero" },
        { -79, 0x20u, 0x00800000u, "2^-126, the smallest normal float" },
    };
    for (const LoAnchor& a : loAnchors) {
        std::uint32_t lo = 0, hi = 0;
        const Pattern p{ 0x00000u, a.mantLo, "anchor" };
        wordsFor(p, a.e, false, lo, hi);
        EXPECT_EQ(fbits(bd::bandFromIEEE(lo, hi).lo), a.wantLo)
            << "lo at 2^" << a.e << ": expected " << a.why;
    }

    std::uint32_t lo = 0, hi = 0;
    const Pattern one{ 0x00000u, 0x00000001u, "1 + 2^-52" };
    wordsFor(one, 0, false, lo, hi);
    const Band healthy = bd::bandFromIEEE(lo, hi);
    EXPECT_EQ(fbits(healthy.hi), 0x3F800000u);
    EXPECT_EQ(fbits(healthy.lo), 0x00000000u);
    EXPECT_EQ(fbits(healthy.tail), 0x25800000u) << "2^-52";
}

// -------------------------------------------------------------------------
//  4. Contract: nothing non-finite, nothing mis-signed, inside the window.
// -------------------------------------------------------------------------
TEST_F(BandIngest, EveryLimbIsFiniteAndCorrectlySignedInTheCarrierWindow)
{
    std::uint64_t rows = 0, nonFinite = 0, misSigned = 0, oversized = 0;
    for (int e = kWindowLow; e <= kWindowHigh; e++)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band b = bd::bandFromIEEE(lo, hi);
                rows++;
                if (!std::isfinite(b.hi) || !std::isfinite(b.lo)
                    || !std::isfinite(b.tail))
                    nonFinite++;
                const bool wrongSign
                    = (b.lo != 0.0f && ((b.lo < 0.0f) != (s == 1)))
                    || (b.tail != 0.0f && ((b.tail < 0.0f) != (s == 1)));
                if (wrongSign)
                    misSigned++;
                if (std::fabs(b.lo) > std::fabs(b.hi)
                    || std::fabs(b.tail) > std::fabs(b.hi))
                    oversized++;
            }
    std::printf("[BandIngest] contract: %llu rows, %llu non-finite, %llu "
                "mis-signed, %llu with a lower limb above the leading one\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(nonFinite),
        static_cast<unsigned long long>(misSigned),
        static_cast<unsigned long long>(oversized));
    EXPECT_EQ(nonFinite, 0u);
    EXPECT_EQ(misSigned, 0u);
    EXPECT_EQ(oversized, 0u)
        << "a lower limb larger than the leading limb is the signature of an "
           "exponent field that wrapped";
}

// -------------------------------------------------------------------------
//  5. The depth staircase, measured through a round trip.
// -------------------------------------------------------------------------
TEST_F(BandIngest, CertifiedDepthDegradesMonotonicallyThroughARoundTrip)
{
    std::printf("[BandIngest] depth vs exponent, round trip (min over %zu "
                "mantissa patterns):\n",
        patterns().size());
    std::vector<double> curve;
    std::vector<int> es;
    for (int e = 0; e >= -126; e -= 2) {
        double worst = 53.0;
        for (const Pattern& p : patterns()) {
            std::uint32_t lo = 0, hi = 0;
            wordsFor(p, e, false, lo, hi);
            const double x = doubleFromWords(lo, hi);
            const double b = survivingBits(
                static_cast<long double>(bandRoundTrip(x)),
                static_cast<long double>(x));
            if (b < worst)
                worst = b;
        }
        curve.push_back(worst);
        es.push_back(e);
    }
    for (std::size_t i = 0; i < curve.size(); i++)
        std::printf("   2^%-5d  %5.1f bits\n", es[i], curve[i]);

    std::uint64_t cliffs = 0, dead = 0;
    for (std::size_t i = 1; i < curve.size(); i++) {
        if (curve[i] > curve[i - 1] + 0.5)
            cliffs++;
        if (curve[i] <= 0.0)
            dead++;
    }
    EXPECT_EQ(dead, 0u)
        << "the delivered value became worthless at some exponent inside the "
           "float window -- that is corruption, not degradation";
    EXPECT_EQ(cliffs, 0u)
        << "depth rose again as the exponent fell, which no monotone loss "
           "mechanism can produce and a wrapped exponent field can";
    EXPECT_GE(curve.front(), 52.0) << "the healthy end is not healthy";
}

// -------------------------------------------------------------------------
//  6. The same staircase, seen through one chain operation.
// -------------------------------------------------------------------------
TEST_F(BandIngest, CertifiedDepthDegradesMonotonicallyThroughOneChainAdd)
{
    std::printf("[BandIngest] depth vs exponent, through one band add:\n");
    std::vector<double> curve;
    std::vector<int> es;
    for (int e = 0; e >= -126; e -= 2) {
        double worst = 53.0;
        for (const Pattern& p : patterns()) {
            std::uint32_t alo = 0, ahi = 0, blo = 0, bhi = 0;
            wordsFor(p, e, false, alo, ahi);
            const Pattern q{ p.mantHi ^ 0x3C3C3u, p.mantLo ^ 0x5A5A5A5Au,
                "partner" };
            wordsFor(q, e, false, blo, bhi);

            const double xa = doubleFromWords(alo, ahi);
            const double xb = doubleFromWords(blo, bhi);
            const Band sum  = bd::add(
                bd::bandFromIEEE(alo, ahi), bd::bandFromIEEE(blo, bhi));
            std::uint32_t olo = 0, ohi = 0;
            bd::bandToIEEE(sum, olo, ohi);

            const long double want = static_cast<long double>(xa)
                + static_cast<long double>(xb);
            const double b = survivingBits(
                static_cast<long double>(doubleFromWords(olo, ohi)), want);
            if (b < worst)
                worst = b;
        }
        curve.push_back(worst);
        es.push_back(e);
    }
    for (std::size_t i = 0; i < curve.size(); i++)
        std::printf("   2^%-5d  %5.1f bits\n", es[i], curve[i]);

    std::uint64_t cliffs = 0, dead = 0;
    for (std::size_t i = 1; i < curve.size(); i++) {
        if (curve[i] > curve[i - 1] + 0.5)
            cliffs++;
        if (curve[i] <= 0.0)
            dead++;
    }
    EXPECT_EQ(dead, 0u);
    EXPECT_EQ(cliffs, 0u);
    EXPECT_GE(curve.front(), 52.0);
}

// -------------------------------------------------------------------------
//  7. The whole admitted region, free of corruption.
// -------------------------------------------------------------------------
TEST_F(BandIngest, TheAdmittedRegionIsFreeOfCorruption)
{
    std::uint64_t rows = 0, mismatched = 0, nonFinite = 0;
    int worstE = 0;
    for (int e = -96; e <= 115; e++)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band got  = bd::bandFromIEEE(lo, hi);
                const Band want = referenceBandFromIEEE(lo, hi);
                rows++;
                if (!std::isfinite(got.hi) || !std::isfinite(got.lo)
                    || !std::isfinite(got.tail))
                    nonFinite++;
                if (fbits(got.hi) != fbits(want.hi)
                    || fbits(got.lo) != fbits(want.lo)
                    || fbits(got.tail) != fbits(want.tail)) {
                    mismatched++;
                    if (e < worstE || worstE == 0)
                        worstE = e;
                }
            }
    std::printf("[BandIngest] admitted region 2^-96..2^115: %llu rows, %llu "
                "mismatched, %llu non-finite\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(mismatched),
        static_cast<unsigned long long>(nonFinite));
    EXPECT_EQ(mismatched, 0u)
        << "the ingest is WRONG inside the region admission grants";
    EXPECT_EQ(nonFinite, 0u);
    EXPECT_GT(rows, 4000u);
}

// -------------------------------------------------------------------------
//  8. Non-vacuity: the sweep actually visited every regime it names.
// -------------------------------------------------------------------------
TEST_F(BandIngest, BoundaryCoverageIsNonVacuous)
{
    std::uint64_t normalTail = 0, subnormalTail = 0, zeroedTail = 0,
                  normalLo = 0, subnormalLo = 0, zeroedLo = 0;
    for (int e : boundaryExponents())
        for (const Pattern& p : patterns()) {
            std::uint32_t lo = 0, hi = 0;
            wordsFor(p, e, false, lo, hi);
            if ((p.mantLo & 0x1Fu) != 0u) {
                const float t = referenceBandFromIEEE(lo, hi).tail;
                const std::uint32_t x = (fbits(t) >> 23) & 0xFFu;
                if (t == 0.0f)
                    zeroedTail++;
                else if (x == 0u)
                    subnormalTail++;
                else
                    normalTail++;
            }
            if (((lo >> 5) & 0xFFFFFFu) != 0u) {
                const float l = referenceBandFromIEEE(lo, hi).lo;
                const std::uint32_t x = (fbits(l) >> 23) & 0xFFu;
                if (l == 0.0f)
                    zeroedLo++;
                else if (x == 0u)
                    subnormalLo++;
                else
                    normalLo++;
            }
        }
    std::printf("[BandIngest] coverage: tail normal %llu / subnormal %llu / "
                "flushed %llu; lo normal %llu / subnormal %llu / flushed %llu\n",
        static_cast<unsigned long long>(normalTail),
        static_cast<unsigned long long>(subnormalTail),
        static_cast<unsigned long long>(zeroedTail),
        static_cast<unsigned long long>(normalLo),
        static_cast<unsigned long long>(subnormalLo),
        static_cast<unsigned long long>(zeroedLo));
    EXPECT_GT(normalTail, 0u);
    EXPECT_GT(subnormalTail, 0u);
    EXPECT_GT(zeroedTail, 0u);
    EXPECT_GT(normalLo, 0u);
    EXPECT_GT(subnormalLo, 0u);
    EXPECT_GT(zeroedLo, 0u);
}

// =========================================================================
//  THE LEADING LIMB BELOW 2^-126
// =========================================================================

inline constexpr int kTinyTop    = -123;
inline constexpr int kTinyBottom = -152;

/// @brief The PRE-FIX leading-limb assembly, transcribed verbatim -- a
/// DEFECT ARM that exists to be wrong, sharing no path with the shipped body.
inline float preFixLeadingLimb(std::uint32_t ieee_lo, std::uint32_t ieee_hi)
{
    const std::uint32_t sign_bit = ieee_hi & 0x80000000u;
    const int bexp                = static_cast<int>((ieee_hi >> 20) & 0x7FFu);
    const std::uint32_t top24
        = 0x800000u | ((ieee_hi & 0xFFFFFu) << 3) | (ieee_lo >> 29);
    const int fexp_hi = bexp - 1023 + 127;
    const std::uint32_t hi_bits
        = sign_bit | (static_cast<std::uint32_t>(fexp_hi) << 23) | (top24 & 0x7FFFFFu);
    return bd::intAsFloat(static_cast<int>(hi_bits));
}

inline long double exactLimbSum(const Band& b)
{
    return static_cast<long double>(b.hi) + static_cast<long double>(b.lo)
        + static_cast<long double>(b.tail);
}

// -------------------------------------------------------------------------
//  9. The leading limb, densely, across and below FP32's normal floor.
// -------------------------------------------------------------------------
TEST_F(BandIngest, LeadingLimbIsCorrectlyRoundedBelowTheFp32NormalFloor)
{
    std::uint64_t rows = 0, hiBad = 0, loBad = 0, tailBad = 0;
    int firstBadE = 1 << 30;
    for (int e = kTinyTop; e >= kTinyBottom; e--)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band got  = bd::bandFromIEEE(lo, hi);
                const Band want = referenceBandFromIEEE(lo, hi);
                rows++;
                bool bad = false;
                if (fbits(got.hi) != fbits(want.hi)) { hiBad++; bad = true; }
                if (fbits(got.lo) != fbits(want.lo)) { loBad++; bad = true; }
                if (fbits(got.tail) != fbits(want.tail)) { tailBad++; bad = true; }
                if (bad && e < firstBadE)
                    firstBadE = e;
            }

    std::uint64_t deepRows = 0, deepBad = 0;
    for (int e = -153; e >= -1022; e -= 3)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const Band got  = bd::bandFromIEEE(lo, hi);
                const Band want = referenceBandFromIEEE(lo, hi);
                deepRows++;
                if (fbits(got.hi) != fbits(want.hi)
                    || fbits(got.lo) != fbits(want.lo)
                    || fbits(got.tail) != fbits(want.tail))
                    deepBad++;
            }

    std::printf("[BandIngest] sub-floor sweep 2^%d..2^%d: %llu rows, hi %llu "
                "bad, lo %llu bad, tail %llu bad; deep sweep to 2^-1022: %llu "
                "rows, %llu bad\n",
        kTinyTop, kTinyBottom, static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(hiBad),
        static_cast<unsigned long long>(loBad),
        static_cast<unsigned long long>(tailBad),
        static_cast<unsigned long long>(deepRows),
        static_cast<unsigned long long>(deepBad));
    if (firstBadE != (1 << 30))
        std::printf("   highest exponent with any bad limb: 2^%d\n", firstBadE);

    EXPECT_EQ(hiBad, 0u);
    EXPECT_EQ(loBad, 0u);
    EXPECT_EQ(tailBad, 0u);
    EXPECT_EQ(deepBad, 0u);
    EXPECT_GT(rows, 500u);
    EXPECT_GT(deepRows, 5000u);
}

// -------------------------------------------------------------------------
// 10. RED-FIRST, in the suite: the pre-fix assembly on the same rows.
// -------------------------------------------------------------------------
TEST_F(BandIngest, ThePreFixLeadingLimbIsGarbageOnTheSameCorpus)
{
    std::uint64_t okRows = 0, okDisagree = 0;
    std::uint64_t badRows = 0, badAgree = 0, badHuge = 0, badZero = 0, badInf = 0;
    std::uint64_t shippedWrong = 0;
    float worstPreFix           = 0.0f;
    for (int e = kTinyTop; e >= kTinyBottom; e--)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const std::uint32_t want = fbits(referenceBandFromIEEE(lo, hi).hi);
                const float old_          = preFixLeadingLimb(lo, hi);
                const bool agrees         = fbits(old_) == want;
                if (e >= kWindowLow) {
                    okRows++;
                    okDisagree += agrees ? 0u : 1u;
                } else {
                    badRows++;
                    badAgree += agrees ? 1u : 0u;
                    if (!std::isfinite(old_))
                        badInf++;
                    else if (std::fabs(old_) > 1.0f)
                        badHuge++;
                    else if (old_ == 0.0f)
                        badZero++;
                    if (std::isfinite(old_)
                        && std::fabs(old_) > std::fabs(worstPreFix))
                        worstPreFix = old_;
                }
                if (fbits(bd::bandFromIEEE(lo, hi).hi) != want)
                    shippedWrong++;
            }

    std::printf("[BandIngest] DEFECT ARM (pre-fix leading limb). Healthy "
                "overlap 2^%d..2^%d: %llu rows, %llu disagree (must be 0). "
                "Defect region 2^-127..2^%d: %llu rows, %llu agree (must be "
                "0) -- non-finite %llu, |x| > 1 %llu, flushed to zero %llu, "
                "worst finite magnitude %g. SHIPPED wrong on %llu of all.\n",
        kTinyTop, kWindowLow, static_cast<unsigned long long>(okRows),
        static_cast<unsigned long long>(okDisagree), kTinyBottom,
        static_cast<unsigned long long>(badRows),
        static_cast<unsigned long long>(badAgree),
        static_cast<unsigned long long>(badInf),
        static_cast<unsigned long long>(badHuge),
        static_cast<unsigned long long>(badZero),
        static_cast<double>(worstPreFix),
        static_cast<unsigned long long>(shippedWrong));

    EXPECT_GT(okRows, 50u);
    EXPECT_EQ(okDisagree, 0u);
    EXPECT_GT(badRows, 400u);
    EXPECT_EQ(badAgree, 0u);
    EXPECT_GT(badHuge + badInf, 100u);
    EXPECT_GE(badZero, 10u);
    EXPECT_EQ(shippedWrong, 0u);
}

// -------------------------------------------------------------------------
// 11. Known answers for the subnormal leading limb, written out by hand.
// -------------------------------------------------------------------------
TEST_F(BandIngest, KnownAnswerVectorsForTheSubnormalLeadingLimb)
{
    struct HeadAnchor {
        int e;
        std::uint32_t mantHi;
        std::uint32_t wantHi;
        const char* why;
    };
    const HeadAnchor anchors[] = {
        { -126, 0x00000u, 0x00800000u,
            "2^-126 is the smallest NORMAL float -- the last fast-path row" },
        { -127, 0x00000u, 0x00400000u,
            "2^-127 = 2^22 subnormal steps, exact (pre-fix: +0)" },
        { -128, 0x00000u, 0x00200000u, "2^-128, exact (pre-fix: -1.7e38)" },
        { -130, 0x00000u, 0x00080000u, "2^-130, exact" },
        { -127, 0x80000u, 0x00600000u, "1.5 * 2^-127, exact" },
        { -140, 0x00000u, 0x00000200u, "2^-140, exact" },
        { -149, 0x00000u, 0x00000001u,
            "2^-149, the smallest subnormal -- the last exact head" },
        { -149, 0x80000u, 0x00000002u,
            "1.5 * 2^-149 is 1.5 steps and ties to EVEN, so two" },
        { -150, 0x80000u, 0x00000001u, "0.75 of a step rounds to one" },
        { -150, 0x00000u, 0x00000000u,
            "2^-150 is exactly half a step and ties to even, so zero" },
        { -151, 0x00000u, 0x00000000u, "under half a step, rounds away" },
        { -1022, 0x00000u, 0x00000000u,
            "the smallest normal double -- far below anything FP32 holds" },
    };
    for (const HeadAnchor& a : anchors)
        for (int s = 0; s < 2; s++) {
            std::uint32_t lo = 0, hi = 0;
            const Pattern p{ a.mantHi, 0x00000000u, "head anchor" };
            wordsFor(p, a.e, s == 1, lo, hi);
            const std::uint32_t want = a.wantHi | (s == 1 ? 0x80000000u : 0u);
            EXPECT_EQ(fbits(bd::bandFromIEEE(lo, hi).hi), want)
                << "leading limb at 2^" << a.e << (s == 1 ? " negative" : "")
                << ": expected " << a.why;
        }
}

// -------------------------------------------------------------------------
// 12. The depth staircase, continued below the float floor.
// -------------------------------------------------------------------------
TEST_F(BandIngest, TheCertifiedDepthStaircaseContinuesBelowTheFloat32Floor)
{
    std::printf("[BandIngest] carrier depth vs exponent below the float "
                "floor (min over %zu mantissa patterns; predicted "
                "min(53, e+150)):\n",
        patterns().size());
    std::uint64_t belowPrediction = 0, nonMonotone = 0;
    double prev                    = 53.0;
    for (int e = -120; e >= kTinyBottom; e--) {
        double worst = 64.0;
        for (const Pattern& p : patterns()) {
            std::uint32_t lo = 0, hi = 0;
            wordsFor(p, e, false, lo, hi);
            const long double want
                = static_cast<long double>(doubleFromWords(lo, hi));
            const double b
                = survivingBits(exactLimbSum(bd::bandFromIEEE(lo, hi)), want);
            if (b < worst)
                worst = b;
        }
        const double predicted
            = std::min(53.0, static_cast<double>(e) + 150.0);
        std::printf("   2^%-6d  %5.1f bits   (predicted %5.1f)\n", e, worst,
            predicted);
        if (worst < predicted - 0.5)
            belowPrediction++;
        if (worst > prev + 0.5)
            nonMonotone++;
        prev = worst;
    }
    EXPECT_EQ(belowPrediction, 0u);
    EXPECT_EQ(nonMonotone, 0u);
}

// -------------------------------------------------------------------------
// 13. Non-vacuity for the sub-floor sweep: it visited all three regimes.
// -------------------------------------------------------------------------
TEST_F(BandIngest, SubFloorHeadCoverageIsNonVacuous)
{
    std::uint64_t normalHead = 0, subnormalHead = 0, zeroHead = 0, signedZero = 0;
    for (int e = kTinyTop; e >= kTinyBottom; e--)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                const float h            = bd::bandFromIEEE(lo, hi).hi;
                const std::uint32_t u    = fbits(h);
                const std::uint32_t x    = (u >> 23) & 0xFFu;
                if (h == 0.0f) {
                    zeroHead++;
                    if ((u & 0x80000000u) != 0u)
                        signedZero++;
                } else if (x == 0u)
                    subnormalHead++;
                else
                    normalHead++;
            }
    std::printf("[BandIngest] sub-floor head coverage: normal %llu / "
                "subnormal %llu / underflowed %llu (of which negative zero "
                "%llu)\n",
        static_cast<unsigned long long>(normalHead),
        static_cast<unsigned long long>(subnormalHead),
        static_cast<unsigned long long>(zeroHead),
        static_cast<unsigned long long>(signedZero));
    EXPECT_GT(normalHead, 0u);
    EXPECT_GT(subnormalHead, 0u);
    EXPECT_GT(zeroHead, 0u);
    EXPECT_GT(signedZero, 0u);
}

// NOTE: `MaterializeFiftyThreeDeliversTheBandToIEEEValue` and its
// device counterpart are not carried -- see this file's header comment.

} // namespace BandIngestTest
} // namespace aether_tests
