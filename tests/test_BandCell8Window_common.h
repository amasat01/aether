// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandCell8Window_common.h
 * @brief The tier-1 STORAGE WINDOW as a boundary between two egress terminals:
 *        which magnitudes `BandCell8` can hold, and what the working carrier's
 *        own `double` egress does with the ones it cannot.
 *
 * @section why Why this suite exists
 * `BandCell8Cert.OutOfWindowMagnitudesLoseTheLowerLimbs` already pins
 * `cell8BandIsStorable` on hand-picked rows. This suite pins the CONSEQUENCE
 * for a consumer, over the ENUMERATED corpora the ported working-carrier certs
 * actually draw from (`Ff1Cert`/`Ff2Cert`'s `AdversarialCornersF0`
 * deep-underflow arms), and it is the regression pin for the asymmetry
 * `tests/banded/carrier_egress.h` repairs:
 *
 *   - `Ff1::toDouble()` / `Ff2::toDouble()` are STORAGE terminals. Their
 *     precondition is `cell8BandIsStorable`. Below the window that precondition
 *     is false BY CONSTRUCTION, and the codec then aborts a debug build on its
 *     own assert (`BandCell8.h`, `cell8FromBand`) or, in a release build,
 *     silently encodes a value that is not the one it was handed —
 *     MEASURED: `1e-34` decodes back as `-7.884178e+35`. This is a shared
 *     contract of the format, not an aether-specific divergence.
 *   - `detail::bandToIEEE` (`Band.h`) is the CARRIER egress. It is TOTAL: it
 *     answers for every finite `Band`, in the window or below it, and that is
 *     what a deep-underflow CHARACTERISATION arm needs.
 *
 * ★ NON-VACUITY. Every row below enumerates its corpus (a fixed grid and every
 * power of two, no sampling and no seed) and asserts that BOTH sides of the
 * window edge are populated, so a row cannot go green on a predicate that has
 * become constant or on a corpus that has drifted off the boundary it exists
 * to straddle.
 *
 * ★ REVERT CHECK. Point `aether_tests::cert::carrierToDouble` at the storage
 * terminal and every row here aborts on `cell8FromBand`'s assert, together with
 * `Ff1Cert.AdversarialCornersF0` and `Ff2Cert.AdversarialCornersF0`.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"

#include "tests/banded/carrier_egress.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace aether_tests {
namespace BandCell8WindowTest {

namespace bd = aether::banded::detail;
using aether::banded::Band;
using aether::banded::BandedReal;
using aether_tests::cert::carrierToDouble;

/// @brief `double` -> `Band` through the UNCHECKED carrier ingest — the same
/// bridge `ff1FromDouble`/`ff2FromDouble` use, so this suite's corpus is the
/// cert arms' corpus and not a paraphrase of it.
[[nodiscard]] inline Band ingest(double v)
{
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bd::bandFromIEEE(
        static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32));
}

/// @brief The leading limb's unbiased exponent, or `kBelowFp32Normal` when the
/// limb is subnormal or zero (which is itself below every storable exponent).
inline constexpr int kBelowFp32Normal = -1000;
[[nodiscard]] inline int headExp(Band b)
{
    const std::uint32_t bh = static_cast<std::uint32_t>(bd::floatAsInt(b.hi));
    const int ebyte        = static_cast<int>((bh >> 23) & 0xFFu);
    return (ebyte == 0 || ebyte == 0xFF) ? kBelowFp32Normal : (ebyte - 127);
}

/// @brief The lowest exponent `cell8BandIsStorable` admits: the window floor
/// with the borrow's own binade of slack under it (@see `BandCell8.h`).
inline constexpr int kAdmittedFloorExp = aether::banded::detail::kBandCell8FloorExp - 1;

/// @brief The DERIVED absolute bound on the carrier round trip below the FP32
/// normal floor. `bandFromIEEE` splits the `double`'s 53 bits across three
/// float limbs (24 + 24 + 5); under the FP32 normal floor each limb is rounded
/// to the nearest multiple of the subnormal quantum `2^-149`, so each carries
/// at most HALF of it, and the three exact `double` widenings then sum with no
/// further error. One anchor (the quantum), three limbs — not a fitted number.
inline constexpr double kSubnormalGridBound = 3.0 * 0x1p-150;

// ─────────────────────────────────────────────────────────────────────────
//  The enumerated corpora. Fixed grids: no RNG, no seed, no sampling.
// ─────────────────────────────────────────────────────────────────────────

/// @brief The UNION of the two ported deep-underflow sweeps' supports —
/// `Ff1Cert`'s `[1e-44, 1e-39]` and `Ff2Cert`'s `[1e-40, 1e-28]` — on a fixed
/// 8193-point log grid, PLUS every power of two from the smallest FP32
/// subnormal up to one binade under the admitted floor. The grid straddles the
/// window edge (`1e-28` sits at `2^-93`, above the floor) deliberately: that is
/// what makes the boundary row discriminating rather than constant.
[[nodiscard]] inline const std::vector<double>& underflowCorpus()
{
    static const std::vector<double> rows = [] {
        std::vector<double> v;
        v.reserve(8250);
        for (int i = 0; i <= 8192; i++)
            v.push_back(std::pow(10.0, -44.0 + 16.0 * static_cast<double>(i) / 8192.0));
        for (int e = -149; e <= kAdmittedFloorExp - 1; e++)
            v.push_back(std::ldexp(1.0, e));
        // ★ The measured silent mis-encoder, named so the corpus cannot drift
        // off it: below the window this value decodes back as -7.884178e+35.
        v.push_back(1e-34);
        return v;
    }();
    return rows;
}

/// @brief The IN-WINDOW control: `[1e-27, 1e27]` on a fixed 20001-point log
/// grid, entirely inside tier 1 with headroom at both ends.
[[nodiscard]] inline const std::vector<double>& inWindowCorpus()
{
    static const std::vector<double> rows = [] {
        std::vector<double> v;
        v.reserve(20001);
        for (int i = 0; i <= 20000; i++)
            v.push_back(std::pow(10.0, -27.0 + 54.0 * static_cast<double>(i) / 20000.0));
        return v;
    }();
    return rows;
}

class BandCell8Window : public ::testing::Test { };

// -------------------------------------------------------------------------
//  1. The storage predicate tracks the window edge EXACTLY over the corpus the
//     deep-underflow cert arms draw from — so routing that arm through a
//     storage terminal is a precondition violation on most of it, and is not
//     a matter of taste.
// -------------------------------------------------------------------------
TEST_F(BandCell8Window, TheDeepUnderflowSweepLeavesTheStorageWindow)
{
    std::size_t below = 0, inside = 0, disagree = 0;
    double firstBad = 0.0;
    for (double v : underflowCorpus()) {
        const Band b       = ingest(v);
        const bool storable = bd::cell8BandIsStorable(b);
        const bool expected = headExp(b) >= kAdmittedFloorExp;
        if (storable != expected) {
            if (disagree == 0)
                firstBad = v;
            disagree++;
        }
        (storable ? inside : below)++;
    }
    std::printf("[Window] deep-underflow corpus n=%zu: %zu below the window, "
                "%zu inside, %zu predicate/exponent disagreements\n",
        underflowCorpus().size(), below, inside, disagree);

    EXPECT_EQ(disagree, 0u)
        << "cell8BandIsStorable must answer exactly `head exponent >= "
        << kAdmittedFloorExp << "` for a bandFromIEEE ingest (its truncated "
           "split never cancels, so the at-rest clause is always satisfied); "
           "first offender v=" << firstBad;
    EXPECT_GT(below, 3000u)
        << "NON-VACUITY: most of the ported sweep must be OUTSIDE the storage "
           "window, or this suite is not testing the boundary it exists for";
    EXPECT_GT(inside, 0u)
        << "NON-VACUITY: the corpus must also reach INSIDE the window, or the "
           "predicate could be constant-false and every row above would pass";
    EXPECT_FALSE(bd::cell8BandIsStorable(ingest(1e-34)))
        << "the named silent mis-encoder must stay outside the window";
}

// -------------------------------------------------------------------------
//  2. The CARRIER egress answers for the whole corpus, to a DERIVED bound —
//     the totality the storage terminal does not have and is not meant to.
// -------------------------------------------------------------------------
TEST_F(BandCell8Window, TheCarrierEgressIsTotalWhereStorageIsNot)
{
    double worstAbs = 0.0, worstAt = 0.0;
    std::size_t nonFinite = 0, wrongSign = 0, overBound = 0, inexact = 0;
    for (double v : underflowCorpus()) {
        const double rt = carrierToDouble(ingest(v));
        if (!std::isfinite(rt))
            nonFinite++;
        if (!(rt > 0.0))
            wrongSign++;
        const double err = std::fabs(rt - v);
        if (err > kSubnormalGridBound)
            overBound++;
        if (err > 0.0)
            inexact++;
        if (err > worstAbs) {
            worstAbs = err;
            worstAt  = v;
        }
    }
    std::printf("[Window] carrier egress over n=%zu: worst |rt-v|=%.6e at "
                "v=%.6e (derived bound %.6e), %zu non-finite, %zu sign flips, "
                "%zu over bound, %zu inexact\n",
        underflowCorpus().size(), worstAbs, worstAt, kSubnormalGridBound, nonFinite,
        wrongSign, overBound, inexact);

    EXPECT_EQ(nonFinite, 0u) << "the carrier egress is TOTAL: no finite Band may "
                                "leave it as an infinity or a NaN";
    EXPECT_EQ(wrongSign, 0u) << "every corpus row is positive and must come back "
                                "positive -- a sign flip is the storage codec's "
                                "below-window failure mode (1e-34 -> -7.88e+35)";
    EXPECT_EQ(overBound, 0u)
        << "carrier round trip must stay within 3 * 2^-150, half the FP32 "
           "subnormal quantum per limb across the 24+24+5 split";
    EXPECT_GT(inexact, 0u)
        << "NON-VACUITY: the corpus must actually REACH the subnormal grid, or "
           "the bound above is passing on exactness and gates nothing";
    EXPECT_GT(worstAbs, 0.0);
}

// -------------------------------------------------------------------------
//  3. Where storage IS applicable the two terminals are the SAME function —
//     so the egress swap changes nothing that legitimately measures the store
//     path (`Ff1Cert`/`Ff2Cert`'s `StoreRoundTrip` keep `toDouble()`).
// -------------------------------------------------------------------------
TEST_F(BandCell8Window, InsideTheWindowTheTwoEgressesAgreeBitExactly)
{
    std::size_t notStorable = 0, disagree = 0, notExact = 0;
    for (double v : inWindowCorpus()) {
        const Band b = ingest(v);
        if (!bd::cell8BandIsStorable(b)) {
            notStorable++;
            continue;
        }
        const double carrier = carrierToDouble(b);
        const double storage = BandedReal(b).toDouble();
        std::uint64_t cw, sw;
        std::memcpy(&cw, &carrier, sizeof(cw));
        std::memcpy(&sw, &storage, sizeof(sw));
        if (cw != sw)
            disagree++;
        if (carrier != v)
            notExact++;
    }
    std::printf("[Window] in-window corpus n=%zu: %zu not storable, %zu "
                "carrier/storage word disagreements, %zu inexact round trips\n",
        inWindowCorpus().size(), notStorable, disagree, notExact);

    EXPECT_EQ(notStorable, 0u) << "NON-VACUITY: the whole in-window control must "
                                  "be storable, or the comparison below runs on "
                                  "an empty set";
    EXPECT_EQ(disagree, 0u)
        << "inside the window the carrier egress and the storage egress must be "
           "the SAME function, bit for bit -- that is what makes swapping the "
           "deep-underflow arm onto the carrier egress a no-op everywhere the "
           "storage terminal was ever valid";
    EXPECT_EQ(notExact, 0u) << "bandFromIEEE's 24+24+5 split is exact while all "
                               "three limbs are normal floats, so the in-window "
                               "round trip must be exact";
}

} // namespace BandCell8WindowTest
} // namespace aether_tests
