// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandCell_common.h
 * @brief Conformance battery for `BandCell` — the 16-byte biased-`Band`
 *        storage cell.
 *
 * Corpus: mantissa patterns, magnitude sweep `2^-90..2^100`, edge/depth
 * exponent sets. Instruments: bitwise band/cell equality, the `long double`
 * independent reconstruction. Tolerances: every claim here is EXACT —
 * bit-for-bit or `long double`-exact, zero ULP budgets, zero CompDD/
 * SoftDouble coupling, across all 13 host rows and both device rows
 * (`test_BandCell.cu`).
 *
 * @section ingest Why the corpus is built from mantissa bits, never bandFromIEEE
 * Every band here is constructed DIRECTLY from an explicit 24+24+5 mantissa
 * split (`unitBand`), then moved to its target magnitude with the host's own
 * `std::ldexp` — never through `aether::banded::detail::bandFromCell8`/
 * `BandedReal::fromDouble`. This is deliberate: the exact value of every
 * row is known in closed form before any library code
 * runs, so the depth claim is checked against arithmetic this file did
 * itself, not against another primitive's own conditioning.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace aether_tests {
namespace BandCellTest {

using aether::banded::Band;
using aether::banded::BandCell;
namespace bd = aether::banded::detail;

// =========================================================================
//  Bit-level comparison helpers
// =========================================================================

inline std::uint32_t bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline bool sameBandBits(Band a, Band b)
{
    return bits(a.hi) == bits(b.hi) && bits(a.lo) == bits(b.lo) && bits(a.tail) == bits(b.tail);
}

inline bool sameCellBits(const BandCell& a, const BandCell& b)
{
    return bits(a.hi) == bits(b.hi) && bits(a.lo) == bits(b.lo) && bits(a.tail) == bits(b.tail)
        && a.bias == b.bias;
}

/// @brief True when a limb is one the carrier can hold without erosion:
/// either exactly zero, or at or above FP32's smallest normal magnitude
/// `2^-126`.
inline bool limbIsAtRest(float f)
{
    const std::uint32_t e = (bits(f) >> 23) & 0xFFu;
    return bits(f) == 0u || bits(f) == 0x80000000u || (e != 0u && e != 0xFFu);
}

// =========================================================================
//  Row construction -- from mantissa bits, never from an ingest
// =========================================================================

struct Mantissa {
    std::uint32_t top24;
    std::uint32_t lo24;
    std::uint32_t tail5;
    const char* name;
};

inline const std::vector<Mantissa>& mantissas()
{
    static const std::vector<Mantissa> m = {
        { 0x800000u, 0x000000u, 0u, "1.0 -- lo and tail both absent" },
        { 0xC00000u, 0x000000u, 0u, "1.5 -- leading limb only" },
        { 0x800000u, 0x000000u, 1u, "1 + 2^-52 -- a tail of one unit, no lo" },
        { 0x800000u, 0x000001u, 0u, "one unit in the lo limb, no tail" },
        { 0x800000u, 0x000000u, 31u, "saturated tail, no lo" },
        { 0xFFFFFFu, 0xFFFFFFu, 31u, "all 53 bits set" },
        { 0xC90FDAu, 0xA22168u, 0x0Cu, "pi/2 -- an irregular real mantissa" },
        { 0xAAAAAAu, 0x555555u, 21u, "alternating bits across all three limbs" },
        { 0x800001u, 0x800000u, 16u, "a power of two in each lower limb" },
    };
    return m;
}

inline Band unitBand(const Mantissa& m, bool negative)
{
    const float s = negative ? -1.0f : 1.0f;
    return Band{ s * static_cast<float>(m.top24) * 0x1p-23f, s * static_cast<float>(m.lo24) * 0x1p-47f,
        s * static_cast<float>(m.tail5) * 0x1p-52f };
}

inline Band scaledBand(const Mantissa& m, bool negative, int E)
{
    const Band b = unitBand(m, negative);
    return Band{ std::ldexp(b.hi, E), std::ldexp(b.lo, E), std::ldexp(b.tail, E) };
}

inline long double exactValue(const Mantissa& m, bool negative, int E)
{
    long double v = static_cast<long double>(m.top24) * 0x1p-23L
        + static_cast<long double>(m.lo24) * 0x1p-47L + static_cast<long double>(m.tail5) * 0x1p-52L;
    if (negative)
        v = -v;
    return std::ldexp(v, E);
}

inline long double bandValue(Band b)
{
    return static_cast<long double>(b.hi) + static_cast<long double>(b.lo)
        + static_cast<long double>(b.tail);
}

inline long double cellValue(const BandCell& c)
{
    const long double limbs = static_cast<long double>(c.hi) + static_cast<long double>(c.lo)
        + static_cast<long double>(c.tail);
    return std::ldexp(limbs, c.bias);
}

// =========================================================================
//  The sweep
// =========================================================================

inline constexpr int kSweepLow  = -90;
inline constexpr int kSweepHigh = 100;

struct Row {
    Band b;
    int E;
    bool negative;
    const Mantissa* m;
};

inline const std::vector<Row>& sweepRows()
{
    static const std::vector<Row> rows = [] {
        std::vector<Row> r;
        for (const Mantissa& m : mantissas())
            for (int E = kSweepLow; E <= kSweepHigh; E++)
                for (int s = 0; s < 2; s++)
                    r.push_back(Row{ scaledBand(m, s == 1, E), E, s == 1, &m });
        return r;
    }();
    return rows;
}

inline const std::vector<int>& edgeExponents()
{
    static const std::vector<int> e = [] {
        std::vector<int> v;
        for (int E = -126; E <= -118; E++)
            v.push_back(E);
        for (int E = 118; E <= 127; E++)
            v.push_back(E);
        return v;
    }();
    return e;
}

inline const std::vector<int>& depthExponents()
{
    static const std::vector<int> e = { -88, -92, -96, -100, -104, -110, -120 };
    return e;
}

// =========================================================================
//  Device-transportable row (a Band plus its bookkeeping, POD)
// =========================================================================

struct DeviceRow {
    float hi, lo, tail;
    std::int32_t E;
};

inline std::vector<DeviceRow> deviceRows()
{
    std::vector<DeviceRow> d;
    d.reserve(sweepRows().size());
    for (const Row& r : sweepRows())
        d.push_back(DeviceRow{ r.b.hi, r.b.lo, r.b.tail, r.E });
    return d;
}

class BandCellCert : public ::testing::Test { };

// -------------------------------------------------------------------------
//  1. The cell is one aligned 16-byte object, and nothing about it is fancy.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, LayoutIsOneSixteenByteCell)
{
    static_assert(sizeof(BandCell) == 16);
    static_assert(alignof(BandCell) == 16);
    static_assert(std::is_trivially_copyable_v<BandCell>);
    static_assert(sizeof(bd::BandCellRaw) == 16);
    static_assert(alignof(bd::BandCellRaw) == 16);

    EXPECT_EQ(offsetof(BandCell, hi), 0u);
    EXPECT_EQ(offsetof(BandCell, lo), 4u);
    EXPECT_EQ(offsetof(BandCell, tail), 8u);
    EXPECT_EQ(offsetof(BandCell, bias), 12u)
        << "the bias must occupy the fourth 32-bit slot, or the punned view and the cell "
           "disagree about which word is which";
}

// -------------------------------------------------------------------------
//  2. The headline inverse property, over the whole sweep.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, RoundTripIsBitExactOverTheMagnitudeSweep)
{
    std::uint64_t checked = 0, bad = 0;
    for (const Row& r : sweepRows()) {
        const Band back = bd::bandFromCell(bd::cellFromBand(r.b));
        checked++;
        if (!sameBandBits(back, r.b)) {
            bad++;
            if (bad == 1)
                ADD_FAILURE() << "band -> cell -> band is not the identity at 2^" << r.E
                              << (r.negative ? " (negative)" : " (positive)") << ", pattern '"
                              << r.m->name << "'";
        }
    }
    std::printf("[BandCell] round trip: %llu rows, %d binades, %llu bad\n",
        static_cast<unsigned long long>(checked), kSweepHigh - kSweepLow + 1,
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(checked, 3000u) << "the sweep collapsed -- a battery that runs no rows proves nothing";
}

// -------------------------------------------------------------------------
//  3. The rest-state invariant, on every cell the sweep produces.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, EveryProducedCellIsCentredOrZero)
{
    std::uint64_t centred = 0, zero = 0, bad = 0;
    for (const Row& r : sweepRows()) {
        const BandCell c = bd::cellFromBand(r.b);
        const float a    = std::fabs(c.hi);
        if (bits(c.hi) == 0u && bits(c.lo) == 0u && bits(c.tail) == 0u && c.bias == 0)
            zero++;
        else if (a >= 1.0f && a < 2.0f)
            centred++;
        else
            bad++;
    }
    std::printf("[BandCell] rest state: %llu centred, %llu zero, %llu neither\n",
        static_cast<unsigned long long>(centred), static_cast<unsigned long long>(zero),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(centred, 3000u);
}

// -------------------------------------------------------------------------
//  4. Storing a cell that was already stored changes nothing.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, CellFromBandIsIdempotent)
{
    std::uint64_t bad = 0;
    for (const Row& r : sweepRows()) {
        const BandCell c     = bd::cellFromBand(r.b);
        const BandCell again = bd::cellFromBand(bd::bandFromCell(c));
        if (!sameCellBits(c, again))
            bad++;
    }
    EXPECT_EQ(bad, 0u);
}

// -------------------------------------------------------------------------
//  5. Zero has exactly one representation.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, ZeroBandsProduceTheZeroCell)
{
    const Band zero{ 0.0f, 0.0f, 0.0f };
    const Band negZero{ -0.0f, -0.0f, -0.0f };
    const BandCell expected{ 0.0f, 0.0f, 0.0f, 0 };

    EXPECT_TRUE(sameCellBits(bd::cellFromBand(zero), expected));
    EXPECT_TRUE(sameCellBits(bd::cellFromBand(negZero), expected))
        << "a negative zero must still store as the one zero cell";
    EXPECT_TRUE(sameBandBits(bd::bandFromCell(expected), zero));
}

// -------------------------------------------------------------------------
//  6. The top and bottom of FP32's range, where a single multiplier dies.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, ExtremeExponentsSurviveTheTwoStepScale)
{
    std::uint64_t checked = 0, bad = 0;
    for (int E : edgeExponents())
        for (const Mantissa& m : mantissas())
            for (int s = 0; s < 2; s++) {
                const Band b      = scaledBand(m, s == 1, E);
                const BandCell c = bd::cellFromBand(b);
                const float a     = std::fabs(c.hi);
                checked++;
                const bool centred = (a >= 1.0f && a < 2.0f);
                const bool inverse = sameBandBits(bd::bandFromCell(c), b);
                if (!centred || !inverse || c.bias != E)
                    bad++;
            }
    std::printf("[BandCell] edge binades: %llu rows, %llu bad\n", static_cast<unsigned long long>(checked),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(checked, 300u);
}

// -------------------------------------------------------------------------
//  7. Known answer for the FTZ decision: what the shortcut would have done.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, SingleMultiplierWouldCollapseAtTheTopOfTheRange)
{
    const float atOneTwentySeven = bd::intAsFloat((127 - 127) << 23);
    const float atOneTwentySix   = bd::intAsFloat((127 - 126) << 23);

    EXPECT_EQ(bits(atOneTwentySeven), 0x00000000u)
        << "the single-multiplier shortcut is expected to encode +0.0f at e = 127";
    EXPECT_EQ(bits(atOneTwentySix), 0x00800000u)
        << "at e = 126 the shortcut lands on the smallest normal float";

    for (int E : { 126, 127 }) {
        const int k  = -E;
        const int k1 = k / 2;
        const int k2 = k - k1;
        EXPECT_GT(127 + k1, 0) << "first factor subnormal at e = " << E;
        EXPECT_LT(127 + k1, 255) << "first factor infinite at e = " << E;
        EXPECT_GT(127 + k2, 0) << "second factor subnormal at e = " << E;
        EXPECT_LT(127 + k2, 255) << "second factor infinite at e = " << E;
    }

    const Band top    = scaledBand(mantissas()[5], false, 127);
    const BandCell c = bd::cellFromBand(top);
    EXPECT_GE(std::fabs(c.hi), 1.0f);
    EXPECT_LT(std::fabs(c.hi), 2.0f);
    EXPECT_EQ(c.bias, 127);
    EXPECT_EQ(top.hi * atOneTwentySeven, 0.0f)
        << "the shortcut is supposed to annihilate this band";
}

// -------------------------------------------------------------------------
//  8. THE HEADLINE. Full 53-bit depth held at rest, at and below the old
//     admission floor.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, DepthAtRestSurvivesBelowTheBandCarrierFloor)
{
    const Mantissa& full = mantissas()[5];
    std::uint64_t checked = 0, inexact = 0, eroded = 0;

    for (int E : depthExponents())
        for (int s = 0; s < 2; s++) {
            const Band unit = unitBand(full, s == 1);
            const BandCell c{ unit.hi, unit.lo, unit.tail, E };
            checked++;

            if (cellValue(c) != exactValue(full, s == 1, E))
                inexact++;
            if (!limbIsAtRest(c.hi) || !limbIsAtRest(c.lo) || !limbIsAtRest(c.tail))
                eroded++;

            const BandCell atUnit = bd::cellFromBand(unit);
            EXPECT_EQ(atUnit.bias, 0);
            EXPECT_EQ(bits(atUnit.hi), bits(c.hi));
            EXPECT_EQ(bits(atUnit.lo), bits(c.lo));
            EXPECT_EQ(bits(atUnit.tail), bits(c.tail));
        }

    std::printf("[BandCell] depth at rest: %llu cells, %llu lost bits, %llu had an eroded limb\n",
        static_cast<unsigned long long>(checked), static_cast<unsigned long long>(inexact),
        static_cast<unsigned long long>(eroded));
    EXPECT_EQ(inexact, 0u);
    EXPECT_EQ(eroded, 0u);
    EXPECT_EQ(checked, depthExponents().size() * 2);
}

// -------------------------------------------------------------------------
//  9. The control that makes test 8 mean something.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, ARawBandLosesTheDepthTheCellKeeps)
{
    const Mantissa& full = mantissas()[5];
    int bandExactCount = 0, bandLossyCount = 0;

    std::printf("[BandCell] depth control, all 53 bits set:\n");
    const Band unit = unitBand(full, false);
    for (int E : depthExponents()) {
        const Band raw        = scaledBand(full, false, E);
        const long double ex = exactValue(full, false, E);
        const bool bandExact  = (bandValue(raw) == ex);
        const BandCell cell{ unit.hi, unit.lo, unit.tail, E };
        const bool cellExact = (cellValue(cell) == ex);

        std::printf("   2^%-5d  bare Band %-7s   BandCell %-7s\n", E, bandExact ? "exact" : "LOSSY",
            cellExact ? "exact" : "LOSSY");

        EXPECT_TRUE(cellExact) << "the cell must be exact at 2^" << E;
        if (bandExact)
            bandExactCount++;
        else
            bandLossyCount++;

        if (E >= -96) {
            EXPECT_TRUE(bandExact)
                << "a bare Band is expected to still hold this value at 2^" << E;
        }
        if (E <= -104) {
            EXPECT_FALSE(bandExact) << "a bare Band is expected to have LOST bits at 2^" << E;
        }
    }
    EXPECT_GT(bandLossyCount, 0)
        << "the control never reached a magnitude a bare Band cannot hold";
    EXPECT_GT(bandExactCount, 0) << "the control never reached a magnitude a bare Band CAN hold";
}

// -------------------------------------------------------------------------
//  10. Known answers: four cells whose every field was computed by hand.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, KnownAnswerCellFields)
{
    struct Anchor {
        Band in;
        BandCell want;
        const char* why;
    };

    const Anchor a1{ Band{ -3.0f, 0.0f, 0.0f }, BandCell{ -1.5f, 0.0f, 0.0f, 1 },
        "-3 centres to -1.5 with bias 1" };
    const Anchor a2{ Band{ 0x1p40f, 0.0f, 0x1p-12f }, BandCell{ 1.0f, 0.0f, 0x1p-52f, 40 },
        "a tail one unit wide survives a 40-binade re-centring" };
    const Anchor a3{ Band{ 0x1p-70f, 0x1p-100f, 0.0f }, BandCell{ 1.0f, 0x1p-30f, 0.0f, -70 },
        "an upward re-centring of 70 binades needs the two-step scale" };
    const Anchor a4{ Band{ -0x1.8p-33f, -0x1p-57f, -0x1p-85f },
        BandCell{ -1.5f, -0x1p-24f, -0x1p-52f, -33 }, "signs live in the limbs, all three of them" };

    for (const Anchor& a : { a1, a2, a3, a4 }) {
        const BandCell got = bd::cellFromBand(a.in);
        EXPECT_EQ(bits(got.hi), bits(a.want.hi)) << a.why << " (hi)";
        EXPECT_EQ(bits(got.lo), bits(a.want.lo)) << a.why << " (lo)";
        EXPECT_EQ(bits(got.tail), bits(a.want.tail)) << a.why << " (tail)";
        EXPECT_EQ(got.bias, a.want.bias) << a.why << " (bias)";
        EXPECT_TRUE(sameBandBits(bd::bandFromCell(a.want), a.in))
            << a.why << " (the hand-written cell must fold back to the band)";
    }

    const BandCell c = bd::cellFromBand(Band{ -3.0f, 0.0f, 0.0f });
    EXPECT_EQ(bits(c.hi), 0xBFC00000u);
    EXPECT_EQ(bits(c.lo), 0x00000000u);
    EXPECT_EQ(bits(c.tail), 0x00000000u);
    EXPECT_EQ(c.bias, 1);
}

// -------------------------------------------------------------------------
//  11. The punned access moves a cell without changing it.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, PunnedLoadStoreRoundTripsBitwise)
{
    std::vector<BandCell> src, dst;
    for (const Row& r : sweepRows())
        src.push_back(bd::cellFromBand(r.b));
    dst.assign(src.size(), BandCell{ 0.0f, 0.0f, 0.0f, 0 });

    for (std::size_t i = 0; i < src.size(); i++)
        bd::storeCell(&dst[i], bd::loadCell(&src[i]));

    std::uint64_t bad = 0;
    for (std::size_t i = 0; i < src.size(); i++)
        if (!sameCellBits(src[i], dst[i]))
            bad++;
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(src.size(), 3000u);
}

// -------------------------------------------------------------------------
//  12. What the out-of-contract inputs do, asserted rather than assumed.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, NonFiniteInputsProduceDocumentedFields)
{
    const float inf  = bd::intAsFloat(0x7F800000);
    const float nInf = bd::intAsFloat(static_cast<int>(0xFF800000u));
    const float nan  = bd::intAsFloat(0x7FC00000);

    for (float v : { inf, nInf, nan }) {
        const BandCell c = bd::cellFromBand(Band{ v, 0.0f, 0.0f });
        EXPECT_EQ(c.bias, 128)
            << "the bias is a bit read and must answer deterministically even for values outside "
               "the contract";
        EXPECT_FALSE(std::isfinite(c.hi))
            << "an infinity or NaN must stay one -- it must not be laundered into a finite number";
        const Band back = bd::bandFromCell(c);
        EXPECT_EQ(std::isnan(v), std::isnan(back.hi));
        if (!std::isnan(v)) {
            EXPECT_EQ(bits(back.hi), bits(v)) << "the round trip must be closed for infinities too";
        }
    }

    const float sub    = bd::intAsFloat(0x00000001);
    const BandCell z = bd::cellFromBand(Band{ sub, 0.0f, 0.0f });
    EXPECT_TRUE(sameCellBits(z, BandCell{ 0.0f, 0.0f, 0.0f, 0 }))
        << "a magnitude 30 binades below the point where the banded carrier stops delivering its "
           "bits stores as the zero cell";
}

// -------------------------------------------------------------------------
//  13. Non-vacuity: the sweep really did reach every regime it claims to.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, SweepCoverageIsNonVacuous)
{
    std::uint64_t upward = 0, downward = 0, unbiased = 0, negativeHi = 0, deepFloor = 0, allLimbs = 0,
                  emptyLower = 0;

    for (const Row& r : sweepRows()) {
        const BandCell c = bd::cellFromBand(r.b);
        if (c.bias > 0)
            downward++;
        else if (c.bias < 0)
            upward++;
        else
            unbiased++;
        if (c.hi < 0.0f)
            negativeHi++;
        if (c.bias <= -80)
            deepFloor++;
        if (c.lo != 0.0f && c.tail != 0.0f)
            allLimbs++;
        if (c.lo == 0.0f && c.tail == 0.0f)
            emptyLower++;
    }

    std::printf("[BandCell] coverage: %llu downward, %llu upward, %llu unbiased, %llu negative, "
                "%llu below 2^-80, %llu all-limbs, %llu leading-only\n",
        static_cast<unsigned long long>(downward), static_cast<unsigned long long>(upward),
        static_cast<unsigned long long>(unbiased), static_cast<unsigned long long>(negativeHi),
        static_cast<unsigned long long>(deepFloor), static_cast<unsigned long long>(allLimbs),
        static_cast<unsigned long long>(emptyLower));

    EXPECT_GT(downward, 0u);
    EXPECT_GT(upward, 0u);
    EXPECT_GT(unbiased, 0u);
    EXPECT_GT(negativeHi, 0u) << "no negative row reached the cell";
    EXPECT_GT(deepFloor, 0u) << "the sweep never went below 2^-80";
    EXPECT_GT(allLimbs, 0u) << "no row exercised all three limbs at once";
    EXPECT_GT(emptyLower, 0u) << "no row exercised a leading limb on its own";
}

} // namespace BandCellTest
} // namespace aether_tests
