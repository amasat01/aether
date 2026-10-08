// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandCell8_common.h
 * @brief Conformance battery for `BandCell8` -- the 8-byte `Band` storage cell
 *        (aether/banded/BandCell8.h).
 *
 * Every corpus, every tolerance, every enumerated pattern and every derived
 * bound below is enumerated, never sampled. Two sites are worth calling out:
 * the 16-byte-`BandCell` bridge row (aether carries no `BandCell`), and the
 * `double` delivery, which reads the EXACT limb sum (`detail::bandToDouble`)
 * rather than a store terminal that loses subnormal limbs below 2^-74,
 * twenty binades inside this format's window -- so the exact sum is the
 * tighter oracle rather than a weaker one.
 *
 * WHAT IS BEING CERTIFIED, AND AGAINST WHAT
 * -----------------------------------------
 * A `BandCell8` packs a `Band` into ONE 64-bit word whose high half is a
 * hardware `binary32` and whose low half is the next 32 significand bits: depth
 * 56, over a 220-binade window. A `Band` in mid-chain can hold about 72
 * significant bits and only 56 survive, so "is it the identity?" is the wrong
 * question for most of the surface. Six different claims are being made, and
 * they need six different kinds of oracle:
 *
 *  1. **Everything that FITS is stored exactly.** Any value of 56 significant
 *     bits or fewer, at any exponent in the window, comes back unchanged. The
 *     oracle is the input itself, compared bit for bit.
 *  2. **Everything that does not fit is rounded to nearest, ties to even.**
 *     A self-comparison says nothing about that, so the oracle is an exact
 *     128-bit fixed-point reconstruction of the input's true value, computed in
 *     this file from the limbs' own bit patterns. Nothing in the library is
 *     consulted to build it.
 *  3. **The window is what it says it is.** Every exponent from `2^-94` to
 *     `2^125` round-trips, the admission predicates reject magnitudes outside
 *     it -- tested by making them say no -- and the first exponent past each
 *     edge is pinned as a known answer so the cliff cannot move quietly.
 *  4. **A real `double` survives the whole journey.** Ingested with
 *     `bandFromIEEE`, stored, reloaded and delivered with `bandToDouble`, an
 *     IEEE-754 double comes back bit-identical wherever the BAND carrier can
 *     hold it.
 *  5. **The codec is the one that was measured.** Production `cell8FromBand` /
 *     `bandFromCell8` are compared BIT FOR BIT against a transcription of the
 *     storage round's probe codec (`f32cReference.h`), over the whole
 *     enumerated corpus and a million random in-window rows. That is the
 *     strongest available statement that the ratified depth, window and cost
 *     numbers describe the shipped body.
 *  6. **The second tier and the rebias do what their contracts say.** The
 *     escape tier round-trips every finite `double` at depth 45; the per-array
 *     rebias round-trips bit-exactly; and the two compose only inside the range
 *     the header derives, which is checked by exhausting the window at each
 *     rebias rather than by trusting the derivation.
 *
 * WHY THE ROUNDING ORACLE IS EXACT INTEGER ARITHMETIC AND NOT `long double`
 * ------------------------------------------------------------------------
 * The values under test carry up to 72 significant bits. `long double` on x86
 * carries 64. An oracle one bit short of the question cannot arbitrate a
 * tie-to-even decision, and ties are precisely the interesting rows. Every limb
 * is an integer below `2^24` times an exact power of two, so the whole value is
 * an integer in a fixed-point scale -- `__int128` holds it with room to spare
 * and the comparison is exact, not close.
 *
 * WHERE THE INPUTS COME FROM
 * --------------------------
 * The sweep rows are built DIRECTLY from explicit mantissa bits at a chosen
 * exponent -- never sampled and never filtered -- and the pattern list
 * ENUMERATES the structurally special encodings rather than hoping to meet
 * them: both signed zeros, exact powers of two with and without a cancelling
 * lower limb (the binade-borrow class), payloads one bit either side of the
 * stored depth, exact ties at the rounding boundary, the widest and narrowest
 * legal lower limbs, and both ends of the window. Random rows are added ON TOP
 * of that, never instead of it. The one test that routes through `bandFromIEEE`
 * is the `double` journey above, where the ingest is part of the subject.
 *
 * A corpus is only worth what it can catch, so `SeededDefectsAreCaughtByThisCorpus`
 * runs two deliberately broken codecs (`f32cReference.h`, `namespace
 * f32cdefect`) over the same rows and asserts that each one is CAUGHT. A zero
 * there would mean the rows are blind, and the fix would be the corpus rather
 * than the assertion.
 */

#include <gtest/gtest.h>

#include "aether/backend/cuda/Launch.h"
#include "aether/banded/banded.h"

#include "tests/banded/f32cReference.h"
#include "tests/banded/minimal_mode.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace aether_tests {
namespace BandCell8Test {

using aether::banded::Band;
using aether::banded::BandCell8;
using aether::banded::detail::EscapeParts;
namespace bd = aether::banded::detail;

/// @brief The format's own widths and edges, named once so the tests read as
/// claims about the format rather than as claims about magic numbers.
inline constexpr int kDepth   = bd::kBandCell8Depth;      // 56
inline constexpr int kFloorE  = bd::kBandCell8FloorExp;   // -94
inline constexpr int kCeilE   = bd::kBandCell8CeilingExp; // 125

// =========================================================================
//  Bit-level helpers
// =========================================================================

/// @brief The bit pattern of a float, so `-0.0f` and `+0.0f` are told apart.
inline uint32_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

/// @brief A float from a bit pattern.
inline float fromBits(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// @brief Bitwise band equality.
inline bool sameBandBits(Band a, Band b)
{
    return bits(a.hi) == bits(b.hi) && bits(a.lo) == bits(b.lo)
        && bits(a.tail) == bits(b.tail);
}

/// @brief Bitwise cell equality, bias included.
inline bool sameEscapeBits(const EscapeParts& a, const EscapeParts& b)
{
    return bits(a.hi) == bits(b.hi) && bits(a.lo) == bits(b.lo)
        && a.bias == b.bias;
}

/// @brief The stored word's leading limb, as a float.
inline float wordLeading(BandCell8 c) { return fromBits(static_cast<uint32_t>(c.w >> 32)); }

/// @brief The stored word's 32-bit continuation.
inline uint32_t wordTail(BandCell8 c) { return static_cast<uint32_t>(c.w); }

/// @brief The exponent the stored word's leading limb carries, or a sentinel
/// for the zero words (whose exponent field is `0x00`).
inline int wordExp(BandCell8 c)
{
    const int be = static_cast<int>((static_cast<uint32_t>(c.w >> 32) >> 23) & 0xFFu);
    return be - 127;
}

/// @brief The probe reference's carrier, built from a `Band`. The two structs
/// are field-for-field identical; keeping them distinct types is what stops a
/// production value being smuggled into the reference by accident.
inline f32c::Band3 asBand3(Band b) { return f32c::Band3{ b.hi, b.lo, b.tail }; }

// =========================================================================
//  The exact oracle
// =========================================================================

/// @brief The 128-bit integer the exact oracle counts in.
///
/// `__extension__` is not decoration: 128-bit integers are a compiler
/// extension rather than standard C++, and this suite compiles with
/// `-Wpedantic`, which would otherwise report every use of the type as a
/// portability warning. Naming it once, here, keeps that acknowledgement in one
/// place instead of scattering pragmas through the tests.
__extension__ typedef __int128 Exact;

/// @brief The scale the exact arithmetic is done at, RELATIVE to the row's own
/// exponent: `2^(e - 100)`. A value near `2^e` is near `2^100` there, the
/// deepest bit any row carries sits at `2^(e-71)`, and the whole thing stays
/// far inside a signed 128-bit integer. Working relative to the row's exponent
/// is what lets one scale serve a 220-binade window.
inline constexpr int kExactScale = 100;

/// @brief One float's TRUE value as an exact integer at scale `2^(e-100)`.
/// Reads the bit pattern directly and calls nothing.
///
/// SUBNORMALS ARE COUNTED, not dropped. The constructed sweep rows never carry
/// one -- `exactLimb` refuses to build a limb below FP32's smallest normal --
/// but a `Band` that came from `bandFromIEEE` near the bottom of the storage
/// window DOES: at `2^-94` a double's last five significand bits land in a
/// subnormal residue limb, and they are part of the value. An oracle that
/// treated them as zero would report the codec losing bits it actually stored.
inline Exact exactFloat(float f, int e)
{
    const uint32_t u    = bits(f);
    const int be        = static_cast<int>((u >> 23) & 0xFFu);
    const uint32_t frac = u & 0x7FFFFFu;
    if (be == 0 && frac == 0)
        return 0;
    // A normal float is `(0x800000|frac) * 2^(be-150)`; a subnormal one is
    // `frac * 2^-149`, with no implicit leading bit.
    const Exact m = static_cast<Exact>(be == 0 ? frac : (frac | 0x800000u));
    const int sh  = (be == 0 ? -49 : be - 50) - e;
    Exact v       = 0;
    if (sh >= 0)
        v = m << sh;
    else if (sh > -128)
        v = m >> (-sh);
    return (u & 0x80000000u) ? -v : v;
}

/// @brief A whole `Band`'s true value, at the same scale.
inline Exact exactBand(Band b, int e)
{
    return exactFloat(b.hi, e) + exactFloat(b.lo, e) + exactFloat(b.tail, e);
}

/// @brief What a STORED word denotes, at the same scale and relative to the
/// same exponent, so the two can be subtracted.
///
/// The word is `float(w1) + w0 * 2^(eW - 55)` with `eW` the leading limb's own
/// exponent -- which is not always the row's `e`, because a cancelling lower
/// limb makes the encoder step the leading limb one binade down.
inline Exact exactWord(BandCell8 c, int e)
{
    const uint32_t w1 = static_cast<uint32_t>(c.w >> 32);
    const uint32_t w0 = static_cast<uint32_t>(c.w);
    const int be      = static_cast<int>((w1 >> 23) & 0xFFu);
    if (be == 0)
        return 0; // a zero word: the continuation has no exponent to hang on
    const Exact m = static_cast<Exact>((w1 & 0x7FFFFFu) | 0x800000u);
    const int shH = be - 50 - e;  // the leading limb
    const int shT = be - 82 - e;  // the continuation, 32 bits further down
    Exact v       = 0;
    if (shH >= 0)
        v += m << shH;
    else if (shH > -128)
        v += m >> (-shH);
    if (shT >= 0)
        v += static_cast<Exact>(w0) << shT;
    else if (shT > -128)
        v += static_cast<Exact>(w0) >> (-shT);
    return (w1 & 0x80000000u) ? -v : v;
}

/// @brief One unit in the last place of the STORED word, at the same scale.
/// Zero would mean a comparison against it is meaningless, which the tests
/// check for rather than assume.
inline Exact exactWordUlp(BandCell8 c, int e)
{
    const int be = static_cast<int>((static_cast<uint32_t>(c.w >> 32) >> 23) & 0xFFu);
    if (be == 0)
        return 0;
    const int shT = be - 82 - e;
    return (shT >= 0) ? (Exact{ 1 } << shT) : Exact{ 0 };
}

/// @brief |x| for the oracle's integers.
inline Exact exactAbs(Exact x) { return x < 0 ? -x : x; }

/// @brief How many significant bits an exact value actually carries, counted
/// from its leading set bit down to its lowest. This is what decides which rows
/// MUST be stored exactly, and it is computed from the value rather than from
/// the pattern that produced it -- a cancelling row denotes a different number
/// from the one its three limbs spell out separately.
inline int exactWidth(Exact v)
{
    v = exactAbs(v);
    if (v == 0)
        return 0;
    int hiBit = 0, loBit = 0;
    for (int k = 126; k >= 0; k--)
        if ((v >> k) & 1) {
            hiBit = k;
            break;
        }
    for (int k = 0; k <= 126; k++)
        if ((v >> k) & 1) {
            loBit = k;
            break;
        }
    return hiBit - loBit + 1;
}

// =========================================================================
//  Row construction -- from mantissa bits, never from an ingest
// =========================================================================

/// @brief Build the float `m * 2^s` exactly, or `0` when that magnitude is
/// below FP32's smallest normal.
///
/// Rows are built limb by limb from integers, so every limb has to be an exact
/// float or the row does not mean what it says. Near the bottom of the window
/// the third limb of a 72-bit payload would land under `2^-126`; it is dropped
/// rather than allowed to become a subnormal, because a subnormal limb is
/// flushed to zero on the GPU and the host/device bit-identity claim would then
/// be comparing two different numbers. `SweepCoverageIsNonVacuous` counts the
/// drops so the shrinking payload is visible rather than silent.
inline float exactLimb(uint32_t m, int s, bool negative)
{
    if (m == 0)
        return 0.0f;
    int p = 0;
    for (int k = 23; k >= 0; k--)
        if (m & (1u << k)) {
            p = k;
            break;
        }
    const int e = s + p;
    if (e < -126)
        return 0.0f; // below FP32's smallest normal: this limb does not exist
    const uint32_t u = (negative ? 0x80000000u : 0u)
        | (static_cast<uint32_t>(e + 127) << 23)
        | ((m << (23 - p)) & 0x7FFFFFu);
    return fromBits(u);
}

/// @brief A rest-state significand, pre-split the way a normalized `Band`
/// splits one: 24 leading bits from `2^0`, 23 more from `2^-24`, 23 more from
/// `2^-48`.
///
/// Seventy-one bits in total, which is what a `Band` in mid-chain can actually
/// be carrying -- fifteen more than the cell stores. That is deliberate: a
/// battery whose rows all fit would never round anything. The two lower fields
/// stop one bit short of a full 24 so that the row satisfies
/// `banded::normalize`'s postcondition (`|lo| + |tail| <= half an ulp of `hi`),
/// which is the codec's stated precondition and the reason its 32-bit
/// continuation can never overflow.
struct Pattern {
    uint32_t top24;   ///< leading limb integer, in `[2^23, 2^24)`
    uint32_t mid23;   ///< second limb integer, in `[0, 2^23)`, scaled `2^-47`
    uint32_t low23;   ///< third limb integer, in `[0, 2^23)`, scaled `2^-71`
    const char* name; ///< what this pattern is for, in the failure message
};

/// @brief The patterns the sweep runs. ENUMERATED, not sampled: every
/// structurally special encoding the format has a case for appears here by
/// construction, and the rounding rows are named so a failure says which claim
/// broke.
///
/// A bit at `2^-k` in the third limb is `low23 = 2^(71-k)`, so `0x10000` is the
/// first DROPPED bit (`2^-56`, an exact tie), `0x20000` is the last STORED bit
/// (`2^-55`), and the middle limb's `0x400000` is exactly half an ulp of the
/// leading limb -- the widest lower limb the precondition allows.
inline const std::vector<Pattern>& patterns()
{
    static const std::vector<Pattern> p = {
        { 0x800000u, 0x000000u, 0x000000u, "1.0 -- an exact power of two, no "
                                           "lower limbs" },
        { 0xC00000u, 0x000000u, 0x000000u, "1.5 -- leading limb only" },
        { 0xFFFFFFu, 0x000000u, 0x000000u, "just under 2 -- the top of the "
                                           "binade" },
        { 0x800000u, 0x000001u, 0x000000u, "a power of two plus one unit of the "
                                           "middle limb" },
        { 0x800000u, 0x400000u, 0x000000u, "a power of two plus exactly half an "
                                           "ulp: the widest legal lower limb" },
        { 0x800000u, 0x000000u, 0x000001u, "one unit at 2^-71, far below the "
                                           "stored depth" },
        { 0x800000u, 0x000000u, 0x020000u, "one unit at 2^-55, the LAST stored "
                                           "bit" },
        { 0x800000u, 0x000000u, 0x010000u, "ROUNDING: exact tie at 2^-56, even "
                                           "-> down" },
        { 0x800000u, 0x000000u, 0x030000u, "ROUNDING: exact tie at 2^-56, odd "
                                           "-> up" },
        { 0x800000u, 0x000000u, 0x010800u, "ROUNDING: just above half -> up" },
        { 0x800000u, 0x000000u, 0x008000u, "ROUNDING: just below half -> down" },
        { 0xFFFFFFu, 0x3FFFFFu, 0x7FFFFFu, "all 71 bits set" },
        { 0xFFFFFFu, 0x3FFFFFu, 0x000000u, "the top 47 bits set, nothing below" },
        { 0xC90FDAu, 0x22168Cu, 0x234C4Cu, "pi/2 -- an irregular real value" },
        { 0xAAAAAAu, 0x555555u, 0x2AAAAAu, "alternating bits, all three limbs" },
        { 0x800001u, 0x400000u, 0x400000u, "a power of two in each lower limb" },
        { 0xFFFFFEu, 0x000001u, 0x000001u, "just under 2, lower limbs barely on" },
        { 0x9E3779u, 0x397F4Au, 0x7C15F3u, "the golden-ratio digits" },
        { 0xB504F3u, 0x33F9DEu, 0x648451u, "sqrt(2) digits" },
    };
    return p;
}

/// @brief The `Band` denoting `p` at `2^e`, built from the bits.
///
/// Every limb is exact: each integer is below `2^24` so it is a float exactly,
/// and each scale is an exact power of two.
///
/// @param cancelling give the two lower limbs the OPPOSITE sign to `hi`. A
/// `Band` is entitled to do that -- its limbs carry signed corrections -- and
/// it is the ONLY way to reach the encoder's borrow path, which is where the
/// binade-borrow fixup lives.
///
/// An ABSENT limb is a POSITIVE zero even on a negative row, because that is
/// what `bandFromIEEE` produces and what the whole family compares against.
inline Band restBand(const Pattern& p, bool negative, int e,
    bool cancelling = false)
{
    const bool lowerNeg = cancelling ? !negative : negative;
    return Band{ exactLimb(p.top24, e - 23, negative),
        exactLimb(p.mid23, e - 47, lowerNeg),
        exactLimb(p.low23, e - 71, lowerNeg) };
}

// =========================================================================
//  The sweep
// =========================================================================

// =========================================================================
//  Tier selection, recomputed test-side
// =========================================================================

/// @brief Which tier names `2^e`? Computed from the format's constants, not
/// from the encoder, so the two can be compared.
inline bool tier1Covers(int e) { return e >= kFloorE && e <= kCeilE; }
inline bool tier2Covers(int e)
{
    return (e >= bd::kBandCell8EscapeUpBase
               && e <= bd::kBandCell8EscapeMaxExp)
        || (e <= bd::kBandCell8EscapeDownBase
            && e >= bd::kBandCell8EscapeMinExp);
}

/// @brief What the escape encoder will do with this `double`, worked out
/// INDEPENDENTLY of the encoder: the 45-bit rounding, the carry, and the two
/// ways a row can fall outside the tier's domain afterwards.
///
/// The battery needs this because a row that the encoder is contracted to
/// REFUSE cannot be fed to it -- a debug build would abort. Recomputing the
/// rounding here rather than asking the encoder is what keeps the exclusion a
/// stated consequence of the layout instead of a filter on the answer.
struct EscapeRowFate {
    uint64_t m44;    ///< the rounded 44-bit stored fraction
    int e;           ///< the exponent AFTER any rounding carry
    bool encodable;  ///< false when the encoder is contracted to refuse it
    bool roundedOut; ///< refused because the carry walked it into tier 1
    bool stolen;     ///< refused because it lands on canonical zero's code point
};

inline EscapeRowFate escapeRowFate(int e, uint64_t mant, int sign)
{
    EscapeRowFate f{};
    f.m44               = mant >> 8;
    const uint64_t drop = mant & 0xFFu;
    if (drop > 0x80u || (drop == 0x80u && (f.m44 & 1u) != 0u))
        f.m44++;
    f.e = e;
    if (f.m44 == (uint64_t{ 1 } << 44)) { // the rounding carried out
        f.m44 = 0;
        f.e++;
    }
    f.roundedOut = f.e > bd::kBandCell8EscapeDownBase
        && f.e < bd::kBandCell8EscapeUpBase;
    f.stolen = !f.roundedOut && sign == 0
        && f.e == bd::kBandCell8EscapeDownBase && f.m44 == 0;
    f.encodable = !f.roundedOut && !f.stolen;
    return f;
}

/// @brief One sweep row and everything known about it in advance.
struct Row {
    Band b;
    int e;
    bool negative;
    bool cancelling;
    const Pattern* p;
};

/// @brief The whole sweep, built once: EVERY exponent in the window, every
/// pattern, both signs, and both limb polarities. 220 x 19 x 2 x 2 rows, with
/// no sampling and no filtering anywhere in the construction.
inline const std::vector<Row>& sweepRows()
{
    static const std::vector<Row> rows = [] {
        std::vector<Row> r;
        for (const Pattern& p : patterns())
            for (int e = kFloorE; e <= kCeilE; e++)
                for (int s = 0; s < 2; s++)
                    for (int k = 0; k < 2; k++)
                        r.push_back(Row{ restBand(p, s == 1, e, k == 1), e,
                            s == 1, k == 1, &p });
        return r;
    }();
    return rows;
}

class BandCell8Cert : public ::testing::Test {
};

// -------------------------------------------------------------------------
//  1. The cell is one aligned 64-bit word, and the fields tile it exactly.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, LayoutIsOneEightByteWord)
{
    static_assert(sizeof(BandCell8) == 8);
    static_assert(alignof(BandCell8) == 8);
    static_assert(std::is_trivially_copyable_v<BandCell8>);
    static_assert(sizeof(BandCell8) == sizeof(double),
        "the whole premise of this type is that it costs what a double costs");

    EXPECT_EQ(offsetof(BandCell8, w), 0u);

    // Sign, exponent and the three significand fields must tile the word with
    // nothing left over and nothing overlapping. A format that spent 63 or 65
    // bits would still compile and would still round-trip most values.
    EXPECT_EQ(1 + 8 + (bd::kBandCell8LeadingBits - 1) + bd::kBandCell8LoBits
            + bd::kBandCell8TailBits,
        64) << "the fields do not tile the 64-bit word";
    EXPECT_EQ(bd::kBandCell8LeadingBits + bd::kBandCell8LoBits
            + bd::kBandCell8TailBits,
        kDepth) << "the three limbs must account for the whole depth";
    EXPECT_EQ(kDepth, 56) << "depth 56 is what this layout was ratified for -- "
                             "three bits more than an IEEE double";
    EXPECT_EQ(1 + 8 + bd::kBandCell8EscapeExtBits + bd::kBandCell8EscapeSigHiBits
            + 32,
        64) << "the escape tier's fields must tile the word too";
    EXPECT_EQ(1 + bd::kBandCell8EscapeSigHiBits + 32, bd::kBandCell8EscapeDepth);

    // The high half of the word IS a float. That is the layout's whole claim,
    // and it is checkable in one line.
    const BandCell8 w = bd::cell8FromBand(Band{ -1.5f, 0.0f, 0.0f });
    EXPECT_EQ(bits(-1.5f), static_cast<uint32_t>(w.w >> 32))
        << "the leading 32 bits must be the leading limb's own IEEE pattern, "
           "unmodified -- otherwise the free FP32 read is not free";
    EXPECT_EQ(bits(bd::floatFromCell8(w)), bits(-1.5f));

    // The window edges are derived, not chosen. Both derivations are one line
    // and both are checked here so a change to either constant has to face the
    // mechanism that produced it.
    EXPECT_EQ(kFloorE, -126 + 32)
        << "the floor is where the tail's magic constant 2^(e-32) goes "
           "subnormal";
    EXPECT_EQ(kCeilE, 126 - 1)
        << "the ceiling is where the reciprocal 2^-e goes subnormal, less one "
           "binade for a leading limb that rounds up to a power of two";
    EXPECT_EQ(bd::kBandCell8RebiasLowSafe, kCeilE - 127);
    EXPECT_EQ(bd::kBandCell8RebiasHighSafe, kFloorE + 126);
}

// -------------------------------------------------------------------------
//  2. THE HEADLINE. Everything that fits is stored exactly, at every exponent
//     in the window.
//
//     Two claims of different strength, and the difference is not a hedge.
//     The VALUE is preserved for every row that fits, whatever shape it
//     arrived in. The BIT PATTERN is preserved for rows that arrived in the
//     format's canonical shape -- 24 bits, then 23, then 9 -- which is the
//     shape this cell's own unpacking produces. A row whose lower limbs cancel
//     denotes the same number written differently, and it comes back
//     rewritten in canonical form. That is a property of the format, not a
//     loss, and it is stated rather than papered over.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, ValuesThatFitAreStoredExactlyOverTheWholeWindow)
{
    uint64_t checked = 0, valueBad = 0, signBad = 0, reencoded = 0, bitBad = 0,
             offGrid = 0, offGridBad = 0;
    int lowest = 9999, highest = -9999;

    // The grid the codec writes on, at the oracle's scale. A value is stored
    // exactly when every one of its bits sits at or above the last
    // continuation bit, `2^(e-55)`, of the LEADING LIMB's binade -- which is
    // `2^45` at scale `2^(e-100)`.
    //
    // That qualifier is the whole content of the claim, and it is narrower than
    // "56 significant bits". A value just BELOW a power of two -- `2^e` minus
    // one unit of the halved grid -- carries 56 bits counted from its own
    // leading bit, and it is NOT on this grid: representing it would need the
    // encoder to decide the binade after rounding rather than before. The
    // encoder rounds first, so such a value is correctly rounded rather than
    // stored, and those rows are counted separately below instead of being
    // filtered away.
    const Exact grid = Exact{ 1 } << (kExactScale - (kDepth - 1));

    for (const Row& r : sweepRows()) {
        const Exact want   = exactBand(r.b, r.e);
        const BandCell8 w  = bd::cell8FromBand(r.b);
        const bool onGrid  = (exactAbs(want) % grid) == 0;
        if (!onGrid) {
            // Off the grid: the claim is correct ROUNDING, not exactness.
            offGrid++;
            const Exact ulp = exactWordUlp(w, r.e);
            const Exact d   = exactAbs(exactAbs(exactWord(w, r.e)) - exactAbs(want));
            if (ulp <= 0 || 2 * d > ulp) {
                offGridBad++;
                if (offGridBad == 1)
                    ADD_FAILURE()
                        << "a value one bit below the codec's grid at 2^" << r.e
                        << " (pattern '" << r.p->name
                        << "') was not correctly rounded";
            }
            continue;
        }
        checked++;
        lowest  = r.e < lowest ? r.e : lowest;
        highest = r.e > highest ? r.e : highest;

        if (exactWord(w, r.e) != want) {
            valueBad++;
            if (valueBad == 1)
                ADD_FAILURE()
                    << "a " << exactWidth(want) << "-bit value on the codec's "
                    << "own grid was not stored exactly at 2^" << r.e
                    << (r.cancelling ? " (cancelling limbs)" : "")
                    << ", pattern '" << r.p->name << "', word " << std::hex
                    << w.w << std::dec;
        }
        if ((w.w >> 63) != (want < 0 ? 1u : 0u))
            signBad++;

        // Re-decoding must land on the same value, and re-encoding the decoded
        // Band must land on the same word: the format's canonical form is a
        // fixed point.
        const Band back = bd::bandFromCell8(w);
        reencoded++;
        if (bd::cell8FromBand(back).w != w.w) {
            bitBad++;
            if (bitBad == 1)
                ADD_FAILURE()
                    << "word -> Band -> word moved at 2^" << r.e
                    << ", pattern '" << r.p->name << "': " << std::hex << w.w
                    << " -> " << bd::cell8FromBand(back).w << std::dec;
        }
    }

    std::printf("[BandCell8] exact storage: %llu rows on the grid over "
                "2^%d..2^%d (%d binades), %llu value errors, %llu sign errors; "
                "%llu re-encoded, %llu not a fixed point; %llu rows one bit "
                "below the grid, %llu of them misrounded\n",
        static_cast<unsigned long long>(checked), lowest, highest,
        highest - lowest + 1, static_cast<unsigned long long>(valueBad),
        static_cast<unsigned long long>(signBad),
        static_cast<unsigned long long>(reencoded),
        static_cast<unsigned long long>(bitBad),
        static_cast<unsigned long long>(offGrid),
        static_cast<unsigned long long>(offGridBad));

    EXPECT_EQ(valueBad, 0u);
    EXPECT_EQ(signBad, 0u);
    EXPECT_EQ(bitBad, 0u);
    EXPECT_EQ(offGridBad, 0u);
    EXPECT_GT(offGrid, 100u)
        << "not one row landed below the codec's grid, so the rounding half of "
           "this test never ran";
    EXPECT_EQ(lowest, kFloorE) << "the sweep never reached the bottom of the "
                                  "window";
    EXPECT_EQ(highest, kCeilE) << "the sweep never reached the top of the "
                                  "window";
    EXPECT_GT(checked, 3000u)
        << "the sweep collapsed -- a battery that runs no rows proves nothing";
}

// -------------------------------------------------------------------------
//  3. The postcondition on every word the sweep produces.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, EveryStoredWordCarriesANormalLeadingLimb)
{
    uint64_t normal = 0, bad = 0;

    for (const Row& r : sweepRows()) {
        const BandCell8 w = bd::cell8FromBand(r.b);
        const int be
            = static_cast<int>((static_cast<uint32_t>(w.w >> 32) >> 23) & 0xFFu);
        if (be >= 1 && be <= 0xFE) {
            normal++;
        } else {
            bad++;
            if (bad == 1)
                ADD_FAILURE()
                    << "the leading limb's exponent byte is " << be << " at 2^"
                    << r.e << " pattern '" << r.p->name
                    << "' -- 0x00 and 0xFF are the ESCAPE tags, and a tier-1 "
                       "word that carries one cannot be told apart from an "
                       "escape by any reader";
        }
    }
    std::printf("[BandCell8] tier-1 tags: %llu words with a normal leading "
                "limb, %llu carrying an escape tag\n",
        static_cast<unsigned long long>(normal),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(normal, 3000u);
}

// -------------------------------------------------------------------------
//  4. Known answers: the layout, the rounding and the borrow, by hand.
//
//     The words are spelled out in hexadecimal so that a change to the
//     encoding cannot pass by moving both sides of a comparison at once.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, KnownAnswerWordsAtTheRoundingBoundary)
{
    struct Anchor {
        Band in;
        uint64_t want;
        const char* why;
    };

    // Every one of these was derived by hand from the format definition. The
    // leading half is the leading limb's IEEE pattern; the trailing half counts
    // units of 2^(e-55), so at e = 0 one unit is 2^-55 and half an ulp of the
    // leading limb is 0x80000000 units.
    const Anchor anchors[] = {
        { Band{ 1.0f, 0.0f, 0.0f }, 0x3F80000000000000ull,
            "1.0: the leading limb is the whole value and nothing follows it" },
        { Band{ -1.5f, 0.0f, 0.0f }, 0xBFC0000000000000ull,
            "the sign lives in the leading limb's own sign bit" },
        { Band{ 1.0f, 0x1p-24f, 0.0f }, 0x3F80000080000000ull,
            "half an ulp above a power of two is 2^31 continuation units" },
        { Band{ 1.0f, -0x1p-24f, 0.0f }, 0x3F7FFFFF00000000ull,
            "THE BINADE BORROW: a cancelling limb at an exact power of two "
            "steps the leading limb one float down, and the remainder is "
            "re-expressed in the HALVED ulp of the binade below" },
        { Band{ 1.5f, -0x1p-24f, 0.0f }, 0x3FBFFFFF80000000ull,
            "a cancelling limb away from a power of two borrows without the "
            "halving" },
        { Band{ 1.0f, 0.0f, 0x1p-56f }, 0x3F80000000000000ull,
            "an exact tie at the last continuation bit, even -> down" },
        { Band{ 1.0f, 0x1.8p-55f, 0.0f }, 0x3F80000000000002ull,
            "an exact tie one unit up, odd -> up" },
        { Band{ 1.0f, 0x1.4p-54f, 0.0f }, 0x3F80000000000002ull,
            "2.5 units, an exact tie on an even neighbour -> down to 2" },
        { Band{ 1.0f, 0.0f, 0x1.1p-56f }, 0x3F80000000000001ull,
            "just above half -> up regardless of parity" },
        { Band{ 1.0f, 0.0f, 0x1p-57f }, 0x3F80000000000000ull,
            "just below half -> down" },
        { Band{ 0x1.FFFFFEp0f, 0x1p-24f, 0.0f }, 0x3FFFFFFF80000000ull,
            "the top of a binade: the continuation cannot carry out, because "
            "it is a fraction of the leading limb's own ulp" },
        { Band{ 1.0f, 0.0f, 0.0f }, 0x3F80000000000000ull, "1.0 again" },
    };

    for (const Anchor& a : anchors) {
        const BandCell8 got = bd::cell8FromBand(a.in);
        EXPECT_EQ(got.w, a.want) << a.why;
    }

    // The two ends of the window, with the exponent byte at its extremes.
    EXPECT_EQ(bd::cell8FromBand(Band{ 0x1p125f, 0.0f, 0.0f }).w,
        0x7E00000000000000ull)
        << "the top of the window";
    EXPECT_EQ(bd::cell8FromBand(Band{ 0x1p-94f, 0.0f, 0.0f }).w,
        0x1080000000000000ull)
        << "the bottom of the window";

    // Ties are decided on the MAGNITUDE, so the negative twin of each anchor
    // must be the same word with bit 63 flipped -- not a different rounding.
    for (const Anchor& a : anchors) {
        const Band n{ -a.in.hi, -a.in.lo, -a.in.tail };
        EXPECT_EQ(bd::cell8FromBand(n).w, a.want ^ (uint64_t{ 1 } << 63))
            << a.why << " (negated: only the sign bit may differ)";
    }

    // And every anchor decodes back to the value its word denotes.
    for (const Anchor& a : anchors) {
        const BandCell8 w = bd::cell8FromBand(a.in);
        EXPECT_EQ(exactWord(w, 0), exactBand(bd::bandFromCell8(w), 0))
            << a.why << " (decode disagrees with the word it was given)";
    }
}

// -------------------------------------------------------------------------
//  5. Rounding is nearest-ties-to-even where the format promises ONE rounding.
//
//     The encoder's middle-limb conversion is EXACT whenever that limb sits
//     within 32 binades of the leading one, which is every Band a double or a
//     certified chain produces. These rows are built to stay inside that
//     region, so the claim is the strict one: the stored word is the correctly
//     rounded 56-bit value, ties resolved to an even continuation.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, RoundingIsNearestTiesToEvenWhereOnlyTheTailRounds)
{
    std::mt19937_64 rng(20260806u);
    const int trials = isMinimalMode() ? 20000 : 200000;
    uint64_t rows = 0, bad = 0, ties = 0, tiesUp = 0;

    for (int i = 0; i < trials; i++) {
        const int e      = static_cast<int>(rng() % 200) - 80;
        const uint32_t t = 0x800000u | (static_cast<uint32_t>(rng()) & 0x7FFFFFu);
        const uint32_t m = static_cast<uint32_t>(rng()) & 0x3FFFFFu;
        // The third limb starts no lower than 2^-59 relative to the leading
        // limb, so both conversions inside the encoder see their whole input
        // and the ONLY rounding is the drop into 32 bits.
        const uint32_t l = (static_cast<uint32_t>(rng()) & 0xFFFu) << 11;
        const bool neg   = (rng() & 1) != 0;
        const Band b{ exactLimb(t, e - 23, neg), exactLimb(m, e - 47, neg),
            exactLimb(l, e - 71, neg) };

        const BandCell8 w = bd::cell8FromBand(b);
        const Exact want  = exactAbs(exactBand(b, e));
        const Exact got   = exactAbs(exactWord(w, e));
        const Exact ulp   = exactWordUlp(w, e);
        const Exact d     = exactAbs(got - want);
        rows++;

        if (ulp <= 0) {
            bad++;
            continue;
        }
        if (2 * d == ulp) {
            ties++;
            if ((w.w & 1u) != 0u) {
                bad++;
                if (bad == 1)
                    ADD_FAILURE() << "a tie rounded to an ODD continuation at "
                                     "2^" << e;
            } else if (got > want) {
                tiesUp++;
            }
        } else if (2 * d > ulp) {
            bad++;
            if (bad == 1)
                ADD_FAILURE()
                    << "the stored value is more than half an ulp from the "
                    << "exact value at 2^" << e << " -- the rounding is not "
                    << "nearest";
        }
    }
    std::printf("[BandCell8] strict RNE: %llu rows, %llu exact ties (%llu of "
                "them upward), %llu bad\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(ties),
        static_cast<unsigned long long>(tiesUp),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(ties, 100u) << "no exact tie occurred, so the ties-to-even half "
                             "of the claim was never exercised";
    EXPECT_GT(tiesUp, 10u) << "every tie rounded downward, so the sweep never "
                              "saw the odd-continuation case";
}

// -------------------------------------------------------------------------
//  6. The deep-limb case, where the format rounds TWICE and says so.
//
//     Both of the encoder's conversions are exact while the limb they see has
//     no bits below the last continuation bit. Push the limbs BELOW that line
//     and each conversion rounds on its own account, so the error can reach
//     half a unit twice over. Two roundings cannot promise half an ulp and this
//     test does not pretend otherwise: the honest bound is one ulp, and the
//     measured worst case is reported so the margin is visible rather than
//     merely asserted.
//
//     The generator is built to REACH that case rather than to stumble on it:
//     the middle limb is an ODD multiple of half a continuation unit, so its
//     conversion is an exact tie every time, and the residue limb sits below
//     it so its own conversion always discards something in the same
//     direction.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, DeepLowerLimbsStayInsideTheTwoRoundingBound)
{
    std::mt19937_64 rng(0x5EEDu);
    const int trials = isMinimalMode() ? 20000 : 200000;
    uint64_t rows = 0, bad = 0, beyondHalfUlp = 0;
    double worst = 0.0;

    for (int i = 0; i < trials; i++) {
        const int e      = static_cast<int>(rng() % 160) - 40;
        const uint32_t t = 0x800000u | (static_cast<uint32_t>(rng()) & 0x7FFFFFu);
        // 2m+1 units of 2^(e-56): half a continuation unit, always odd.
        const uint32_t m = (static_cast<uint32_t>(rng()) & 0x3FFFFFu) * 2u + 1u;
        const uint32_t l = 1u + (static_cast<uint32_t>(rng()) & 0xFFFFu);
        const bool neg   = (rng() & 1) != 0;
        const Band b{ exactLimb(t, e - 23, neg), exactLimb(m, e - 56, neg),
            exactLimb(l, e - 80, neg) };

        const BandCell8 w = bd::cell8FromBand(b);
        const Exact want  = exactAbs(exactBand(b, e));
        const Exact got   = exactAbs(exactWord(w, e));
        const Exact ulp   = exactWordUlp(w, e);
        const Exact d     = exactAbs(got - want);
        rows++;

        if (ulp <= 0 || d > ulp) {
            bad++;
            if (bad == 1)
                ADD_FAILURE()
                    << "a deep lower limb landed further than one ulp from the "
                    << "exact value at 2^" << e;
        }
        if (2 * d > ulp)
            beyondHalfUlp++;
        if (ulp > 0) {
            const double ratio
                = static_cast<double>(static_cast<int64_t>(2 * d))
                / static_cast<double>(static_cast<int64_t>(ulp));
            if (ratio > worst)
                worst = ratio;
        }
    }
    std::printf("[BandCell8] two-rounding bound: %llu rows, %llu bad, %llu "
                "beyond a strict half ulp, worst 2d/ulp = %.6f (allowed 2)\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(bad),
        static_cast<unsigned long long>(beyondHalfUlp), worst);
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(beyondHalfUlp, 0u)
        << "not one row exceeded a strict half ulp, so the second rounding "
           "this test exists to bound never happened -- the input generator "
           "probably stopped producing deep limbs";
    EXPECT_LE(worst, 2.0);
}

// -------------------------------------------------------------------------
//  7. Nothing is rounded when nothing needs to be.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, ValuesOfFiftySixBitsOrFewerAreStoredExactly)
{
    std::mt19937_64 rng(4242u);
    const int trials = isMinimalMode() ? 10000 : 100000;
    uint64_t rows = 0, inexact = 0;

    for (int i = 0; i < trials; i++) {
        const int e      = static_cast<int>(rng() % 220) - 94;
        const uint32_t t = 0x800000u | (static_cast<uint32_t>(rng()) & 0x7FFFFFu);
        const uint32_t m = static_cast<uint32_t>(rng()) & 0x3FFFFFu;
        // Eight bits at the bottom of the third limb takes the value to exactly
        // 2^-55: 56 significant bits, the last one the format stores.
        const uint32_t l = (static_cast<uint32_t>(rng()) & 0xFFu) << 16;
        const bool neg   = (rng() & 1) != 0;
        const Band b{ exactLimb(t, e - 23, neg), exactLimb(m, e - 47, neg),
            exactLimb(l, e - 71, neg) };

        const BandCell8 w = bd::cell8FromBand(b);
        rows++;
        if (exactWord(w, e) != exactBand(b, e))
            inexact++;
    }
    std::printf("[BandCell8] 56-bit inputs: %llu rows, %llu stored inexactly\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(inexact));
    EXPECT_EQ(inexact, 0u)
        << "a value that fits the stored depth was rounded anyway";
    EXPECT_GT(rows, 1000u);
}

// -------------------------------------------------------------------------
//  8. Storing a value that was already stored changes nothing.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, StoringAnAlreadyStoredValueIsIdempotent)
{
    uint64_t bad = 0, rows = 0;
    for (const Row& r : sweepRows()) {
        const BandCell8 w     = bd::cell8FromBand(r.b);
        const BandCell8 again = bd::cell8FromBand(bd::bandFromCell8(w));
        rows++;
        if (w.w != again.w) {
            bad++;
            if (bad == 1)
                ADD_FAILURE()
                    << "word -> Band -> word moved at 2^" << r.e << " pattern '"
                    << r.p->name << "': " << std::hex << w.w << " -> "
                    << again.w << std::dec;
        }
    }
    // aether carries no 16-byte `BandCell` biased-triple type today, so a
    // round trip through it has no subject here and is NOT silently
    // reported as passing. The word -> Band -> word fixed point below is
    // the whole of what this row claims.
    std::printf("[BandCell8] idempotence: %llu rows, %llu moved\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(rows, 3000u);
}

// -------------------------------------------------------------------------
//  9. A real IEEE-754 double, all the way there and back.
//
//     This is the only test that routes through `bandFromIEEE` /
//     `bandToDouble`,
//     and it does so because the journey IS the subject. The lower bound of
//     the exact region is MEASURED here, not assumed, and the measurement is
//     the evidence that the limit belongs to the Band carrier (whose lowest
//     limb goes subnormal) rather than to this cell.
//
//     Note what is NOT claimed: that the reloaded Band has the same LIMB
//     SPLIT as the ingested one. It does not. `bandFromIEEE` splits a double
//     24 + 24 + 5; the cell's canonical split is 24 + 23 + 9. The two spell
//     the same number with the bits distributed differently, and the value --
//     which is what a consumer of a storage format is owed -- is preserved bit
//     for bit.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, IeeeDoubleRoundTripIsBitExactWhereTheCarrierHoldsIt)
{
    std::mt19937_64 rng(1234567u);
    /// The exponent at and above which every 53-bit double survives. Below it
    /// the Band's `tail` limb sits under FP32's smallest normal, so the value
    /// is already incomplete before it ever reaches the cell.
    constexpr int kCarrierExact = -74;

    uint64_t rows = 0, doubleBad = 0, splitDiffers = 0;
    for (int e = kCarrierExact; e <= kCeilE; e++)
        for (int i = 0; i < 32; i++) {
            uint64_t mant = rng() & ((uint64_t{ 1 } << 52) - 1);
            if (i == 0)
                mant = 0; // an exact power of two
            if (i == 1)
                mant = (uint64_t{ 1 } << 52) - 1; // all 53 bits set
            if (i == 2)
                mant = 1; // the very last bit only
            if (i == 3)
                mant = uint64_t{ 1 } << 51;
            const uint64_t sgn = (i & 1) ? (uint64_t{ 1 } << 63) : 0;
            const uint64_t ub  = sgn
                | (static_cast<uint64_t>(e + 1023) << 52) | mant;

            const Band in = bd::bandFromIEEE(
                static_cast<uint32_t>(ub & 0xFFFFFFFFu),
                static_cast<uint32_t>(ub >> 32));
            const Band back = bd::bandFromCell8(bd::cell8FromBand(in));
            rows++;
            if (!sameBandBits(in, back))
                splitDiffers++;
            uint32_t olo = 0, ohi = 0;
            bd::cell8IeeeWords(bd::bandToDouble(back), olo, ohi);
            if (((static_cast<uint64_t>(ohi) << 32) | olo) != ub) {
                doubleBad++;
                if (doubleBad == 1)
                    ADD_FAILURE()
                        << "a double did not survive the journey at 2^" << e
                        << " -- the cell carries 56 bits and a double has 53, "
                           "so there is nothing here to round";
            }
        }

    // The instrument check: BELOW the window the journey must stop being
    // exact, or the bound above is not a bound at all and the "every double
    // survives" claim would be untested rather than true. In a DEBUG build the
    // encoder asserts on those magnitudes instead of encoding them, so there
    // the instrument is the predicate that assertion consults -- which has to
    // reject every one of the same rows.
    uint64_t deepRows = 0, deepBad = 0;
    for (int e = -140; e <= -96; e++)
        for (int i = 0; i < 32; i++) {
            const uint64_t mant = rng() & ((uint64_t{ 1 } << 52) - 1);
            const uint64_t ub
                = (static_cast<uint64_t>(e + 1023) << 52) | mant;
            const Band in = bd::bandFromIEEE(
                static_cast<uint32_t>(ub & 0xFFFFFFFFu),
                static_cast<uint32_t>(ub >> 32));
            deepRows++;
#ifdef AETHER_DEBUG_MODE
            if (!bd::cell8BandIsStorable(in))
                deepBad++;
#else
            const Band back = bd::bandFromCell8(bd::cell8FromBand(in));
            uint32_t olo = 0, ohi = 0;
            bd::cell8IeeeWords(bd::bandToDouble(back), olo, ohi);
            if (((static_cast<uint64_t>(ohi) << 32) | olo) != ub)
                deepBad++;
#endif
        }

    std::printf("[BandCell8] double journey: %llu rows over 2^%d..2^%d, %llu "
                "double mismatches, %llu with a different limb split; control "
                "below 2^-96: %llu of %llu already lossy\n",
        static_cast<unsigned long long>(rows), kCarrierExact, kCeilE,
        static_cast<unsigned long long>(doubleBad),
        static_cast<unsigned long long>(splitDiffers),
        static_cast<unsigned long long>(deepBad),
        static_cast<unsigned long long>(deepRows));

    EXPECT_EQ(doubleBad, 0u);
    EXPECT_GT(rows, 3000u);
    EXPECT_GT(deepBad, deepRows / 2)
        << "the control region is not lossy, so the exact region above it is "
           "not a boundary and this test is not measuring one";
}

// -------------------------------------------------------------------------
//  10. Both zeros round-trip, and only ONE of them is the canonical word.
//
//      This is a contract the layout CHANGED and it is pinned rather than
//      assumed. The leading limb carries the sign natively, so a negative zero
//      stores as the sign bit alone and comes back a negative zero -- one bit
//      better than the older layout, which collapsed both zeros onto one word
//      and lost the sign. The price is that an escape-tier reader, which
//      dispatches on the exponent byte, reads the sign-only word as an escape
//      whose value is -2^-95. An array that uses BOTH features canonicalizes
//      its zeros on the way in; `cell8CanonicalizeZero` is that one line, and
//      the hazard is tested here rather than left in prose.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, SignedZerosRoundTripAndOnlyPlusZeroIsCanonical)
{
    const Band pz{ 0.0f, 0.0f, 0.0f };
    const Band nz{ -0.0f, -0.0f, -0.0f };

    EXPECT_EQ(bd::cell8FromBand(pz).w, 0ull)
        << "positive zero must be the all-zero word: it is the canonical zero "
           "the escape tier's own decode agrees with";
    EXPECT_EQ(bd::cell8FromBand(nz).w, uint64_t{ 1 } << 63)
        << "negative zero stores as the sign bit and nothing else";

    EXPECT_TRUE(sameBandBits(bd::bandFromCell8(BandCell8{ 0 }), pz));
    // The sign of a zero survives the round trip -- which is what the
    // leading-limb layout buys for free, and one bit more than the older
    // layout kept. It survives in the LEADING limb only: the absent lower
    // limbs come back positive, as they do for every other value.
    const Band nzBack = bd::bandFromCell8(BandCell8{ uint64_t{ 1 } << 63 });
    EXPECT_EQ(bits(nzBack.hi), bits(-0.0f));
    EXPECT_EQ(bits(nzBack.lo), 0u);
    EXPECT_EQ(bits(nzBack.tail), 0u);

    // Canonical zero is what the escape decode agrees with, and it is the one
    // word both tiers read the same way.
    EXPECT_TRUE(bd::cell8IsEscape(BandCell8{ 0 }));
    const EscapeParts z = bd::escapePartsFromCell8(BandCell8{ 0 });
    EXPECT_EQ(bits(z.hi), 0u);
    EXPECT_EQ(bits(z.lo), 0u);
    EXPECT_EQ(z.bias, 0);

    // The hazard, stated as a measurement: the sign-only word is NOT read as
    // zero by the escape decode.
    const EscapeParts nzEsc = bd::escapePartsFromCell8(BandCell8{ uint64_t{ 1 } << 63 });
    EXPECT_EQ(nzEsc.bias, bd::kBandCell8EscapeDownBase)
        << "an escape-tier reader must see the negative-zero word as -2^-95, "
           "not as zero -- if that ever changes, the canonicalization advice "
           "in the header is wrong";
    EXPECT_EQ(bits(nzEsc.hi), bits(-1.0f));

    // ...and the one-line fix behaves.
    EXPECT_EQ(bd::cell8CanonicalizeZero(bd::cell8FromBand(nz)).w, 0ull);
    EXPECT_EQ(bd::cell8CanonicalizeZero(bd::cell8FromBand(pz)).w, 0ull);
    const BandCell8 ordinary = bd::cell8FromBand(Band{ -1.5f, 0.0f, 0.0f });
    EXPECT_EQ(bd::cell8CanonicalizeZero(ordinary).w, ordinary.w)
        << "canonicalization must touch nothing but a zero";
}

// -------------------------------------------------------------------------
//  11. An absent lower limb comes back as a POSITIVE zero, on both signs.
//
//      This is a deliberate reconciliation with `bandFromIEEE`, which has
//      always answered a positive zero for an absent limb and whose result the
//      store terminal compares against bit for bit. It costs nothing at all
//      here -- `x - x` is `+0` under round-to-nearest -- and it is pinned so
//      the convention cannot drift silently.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, AbsentLowerLimbsComeBackAsPositiveZero)
{
    for (int e : { kFloorE, -3, 0, 7, kCeilE }) {
        const Band in{ exactLimb(0xC00000u, e - 23, true), 0.0f, 0.0f };
        const Band back = bd::bandFromCell8(bd::cell8FromBand(in));
        EXPECT_EQ(bits(back.hi), bits(in.hi)) << "at 2^" << e;
        EXPECT_EQ(bits(back.lo), 0x00000000u)
            << "a negative value with no middle limb returned a NEGATIVE zero "
               "at 2^"
            << e
            << " -- that denotes the same number but does not compare equal to "
               "what bandFromIEEE produces";
        EXPECT_EQ(bits(back.tail), 0x00000000u) << "at 2^" << e;
    }
    // And a value WITH lower limbs carries the sign into all three.
    const Band full{ -1.5f, -0x1p-24f, -0x1p-48f };
    const Band back = bd::bandFromCell8(bd::cell8FromBand(full));
    EXPECT_LT(back.lo, 0.0f);
    EXPECT_LT(back.tail, 0.0f);
}

// -------------------------------------------------------------------------
//  12. The admission guard, tested by making it say NO.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, WindowAdmissionGuardRejectsOutOfWindowDeclarations)
{
    constexpr int margin = bd::kBandCell8AdmissionMargin;

    // The edges, exactly. With the default margin the guard admits up to
    // 125 - 8 = 117 at the top and down to -94 + 8 = -86 at the bottom.
    EXPECT_TRUE(bd::bandCell8CeilingAdmits(kCeilE - margin));
    EXPECT_FALSE(bd::bandCell8CeilingAdmits(kCeilE - margin + 1));
    EXPECT_TRUE(bd::bandCell8FloorAdmits(kFloorE + margin));
    EXPECT_FALSE(bd::bandCell8FloorAdmits(kFloorE + margin - 1));

    // Past the window itself, with the margin taken away, the answer is still
    // no -- a zero margin must not turn the guard off.
    EXPECT_TRUE(bd::bandCell8CeilingAdmits(kCeilE, 0, 0));
    EXPECT_FALSE(bd::bandCell8CeilingAdmits(kCeilE + 1, 0, 0));
    EXPECT_TRUE(bd::bandCell8FloorAdmits(kFloorE, 0, 0));
    EXPECT_FALSE(bd::bandCell8FloorAdmits(kFloorE - 1, 0, 0));

    // The combined predicate fails if EITHER leg fails, not only the ceiling.
    EXPECT_TRUE(bd::bandCell8Admits(40, -62));
    EXPECT_FALSE(bd::bandCell8Admits(130, -62)) << "the ceiling leg is asleep";
    EXPECT_FALSE(bd::bandCell8Admits(40, -130)) << "the floor leg is asleep";
    EXPECT_FALSE(bd::bandCell8Admits(130, -130));

    // It is `constexpr`, which is what lets a consumer put it in a
    // `static_assert` rather than in a runtime branch on the hot path.
    static_assert(bd::bandCell8Admits(40, -62));
    static_assert(!bd::bandCell8Admits(200, -62));

    // The shipped application's own envelope, in KILOMETRES: the largest real
    // stored scalar is the Sun's gravitational parameter at 2^37 and the
    // largest measured chain ceiling is 2^40, against a floor of 2^-62.
    EXPECT_TRUE(bd::bandCell8Admits(40, -62))
        << "the km-unit envelope of the shipped trajectory application must be "
           "admissible, or this format is not usable for what it was designed "
           "for";

    // ★ The same envelope in METRES. A cubed length gains 29.9 binades per unit
    // change, so |r|^3 at the outer domain moves from 2^98 to 2^128 -- which
    // cannot be carried by a Band at all (the leading limb is FP32 and reads
    // `inf`). `arrayBias` does NOT rescue it: the exactness question is
    // bias-independent -- only `bandCell8RebiasKeepsEscapeTagsClear` reads
    // `arrayBias` at all. An array that declares nothing is rejected, and so
    // is the identical array declaring any rebias.
    EXPECT_FALSE(bd::bandCell8Admits(128, 60))
        << "an array of cubed metres that declares no rebias must be "
           "rejected";
    EXPECT_FALSE(bd::bandCell8Admits(128, 60, 100))
        << "arrayBias must not move the exactness admission -- the window "
           "belongs to the value the chain holds, not to where the rebias "
           "parks the stored byte";
    EXPECT_FALSE(bd::bandCell8Admits(128, -62, 100))
        << "a rebias cannot rescue an array whose own elements span more than "
           "the window: 190 binades do not fit in 220 with margins at both "
           "ends";

    // ★ arrayBias must not move EITHER edge of the exactness window: pin both
    // sides of the boundary at several biases, including one well outside
    // bandCell8RebiasKeepsEscapeTagsClear's own safe range, to show the two
    // predicates now answer genuinely independent questions.
    for (int bias : { -3, 0, bd::kBandCell8RebiasLowSafe,
             bd::kBandCell8RebiasHighSafe, 32, 900 }) {
        EXPECT_TRUE(bd::bandCell8CeilingAdmits(kCeilE - margin, bias))
            << "bias " << bias;
        EXPECT_FALSE(bd::bandCell8CeilingAdmits(kCeilE - margin + 1, bias))
            << "bias " << bias;
        EXPECT_TRUE(bd::bandCell8FloorAdmits(kFloorE + margin, bias))
            << "bias " << bias;
        EXPECT_FALSE(bd::bandCell8FloorAdmits(kFloorE + margin - 1, bias))
            << "bias " << bias;
    }

    // ★★ THE COUNTEREXAMPLE (C-X). Before this fix,
    // maxAbsExp=127, minAbsExp=0, arrayBias=10 passed BOTH admission
    // predicates while the codec silently stored 24 bits where the format
    // promises 56 (worst relative error 2.98e-08 against the format's
    // 2^-56) -- the `- arrayBias` term let a nonzero bias admit a magnitude
    // the encoder cannot represent exactly. It must now be REJECTED.
    EXPECT_FALSE(bd::bandCell8Admits(127, 0, 10))
        << "the arrayBias=10 counterexample must be rejected by the fixed "
           "predicate, not silently admitted";
    static_assert(!bd::bandCell8Admits(127, 0, 10));
    // ...while the escape-tag question this predicate was never supposed to
    // be answering is untouched: bias 10 is well inside [-2, +32] and stays
    // clear, exactly as bandCell8RebiasKeepsEscapeTagsClear said before the
    // fix and says after it -- that predicate is correct and tight, and nothing
    // above touches it.
    EXPECT_TRUE(bd::bandCell8RebiasKeepsEscapeTagsClear(10));

    // The rebias composition rule, at the constants the header derives.
    EXPECT_TRUE(bd::bandCell8RebiasKeepsEscapeTagsClear(0));
    EXPECT_TRUE(bd::bandCell8RebiasKeepsEscapeTagsClear(bd::kBandCell8RebiasLowSafe));
    EXPECT_TRUE(bd::bandCell8RebiasKeepsEscapeTagsClear(bd::kBandCell8RebiasHighSafe));
    EXPECT_FALSE(bd::bandCell8RebiasKeepsEscapeTagsClear(bd::kBandCell8RebiasLowSafe - 1));
    EXPECT_FALSE(bd::bandCell8RebiasKeepsEscapeTagsClear(bd::kBandCell8RebiasHighSafe + 1));
    static_assert(!bd::bandCell8RebiasKeepsEscapeTagsClear(900));
}

// -------------------------------------------------------------------------
//  13. What a magnitude outside the window actually does.
//
//      In a debug build the encoder's assertion is the guard and the value
//      never gets encoded, so this test checks the predicate that assertion
//      consults. In a release build there is no assertion, and the documented
//      consequence is pinned as a known answer: past the ceiling the
//      reciprocal scale flushes and the lower limbs VANISH, leaving FP32
//      precision; past the floor the residue limb's magic constant goes
//      subnormal and the reconstruction is wrong by a factor of two. Both arms
//      assert something real.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, OutOfWindowMagnitudesLoseTheLowerLimbs)
{
    // Shared by both arms: the predicate the debug assertion consults must
    // accept exactly the window and reject exactly outside it.
    for (int e : { kFloorE, -50, 0, 50, kCeilE })
        EXPECT_TRUE(bd::cell8BandIsStorable(
            Band{ exactLimb(0x800000u, e - 23, false), 0.0f, 0.0f }))
            << "2^" << e << " is inside the window and must be storable";
    for (int e : { kFloorE - 2, kFloorE - 3, kCeilE + 1, kCeilE + 2 })
        EXPECT_FALSE(bd::cell8BandIsStorable(
            Band{ exactLimb(0x800000u, e - 23, false), 0.0f, 0.0f }))
            << "2^" << e << " is OUTSIDE the window and the guard must say so";
    // ...with exactly one binade of slack at the bottom, which is the borrow's
    // own: a value on the floor with cancelling limbs is STORED one binade
    // down, and reading it back and storing it again must not abort a debug
    // build. The consumer-facing floor is not relaxed by this.
    EXPECT_TRUE(bd::cell8BandIsStorable(
        Band{ exactLimb(0x800000u, kFloorE - 1 - 23, false), 0.0f, 0.0f }))
        << "the borrow's binade must stay storable, or store-load-store aborts "
           "at the bottom of the window";
    EXPECT_FALSE(bd::bandCell8FloorAdmits(kFloorE - 1, 0, 0))
        << "the slack above is an ENCODER tolerance, not a wider window -- the "
           "admission predicate must still refuse the same exponent";

    // A Band that is not at rest is equally unstorable, whatever its exponent:
    // a remainder the 32-bit continuation cannot hold is rejected, and at an
    // exact power of two with a CANCELLING remainder the bar is half as high,
    // because the borrow halves the unit the remainder is counted in.
    EXPECT_FALSE(bd::cell8BandIsStorable(Band{ 1.0f, -0x1p-23f, 0.0f }))
        << "a whole ulp cancelling at an exact power of two is more than the "
           "continuation can hold once the borrow halves its unit";
    EXPECT_TRUE(bd::cell8BandIsStorable(Band{ 1.0f, -0x1p-24f, 0.0f }))
        << "half an ulp cancelling at a power of two is exactly what fits";
    EXPECT_FALSE(bd::cell8BandIsStorable(Band{ 1.5f, 0x1p-22f, 0.0f }));
    EXPECT_FALSE(bd::cell8BandIsStorable(Band{ 1.5f, -0x1p-22f, 0.0f }));
    EXPECT_TRUE(bd::cell8BandIsStorable(Band{ 0.0f, 0.0f, 0.0f }))
        << "the zero Band is always storable";
    EXPECT_FALSE(bd::cell8BandIsStorable(
        Band{ std::numeric_limits<float>::infinity(), 0.0f, 0.0f }))
        << "infinities have no encoding in this format";

#ifdef AETHER_DEBUG_MODE
    std::printf("[BandCell8] out of window: DEBUG build -- the encoder asserts, "
                "so the predicate above is the whole guard here\n");
#else
    // Past the CEILING: the reciprocal 2^-e is subnormal or zero, so the lower
    // limbs are multiplied into nothing and silently dropped.
    {
        const int e = kCeilE + 2; // 127: the reciprocal's exponent field is 0
        const Band b{ exactLimb(0x800000u, e - 23, false),
            exactLimb(0x400000u, e - 47, false), 0.0f };
        const BandCell8 w = bd::cell8FromBand(b);
        EXPECT_EQ(wordTail(w), 0u)
            << "past the ceiling the continuation must come back EMPTY -- if "
               "it does not, the mechanism behind the ceiling constant has "
               "changed and the constant needs re-deriving";
        EXPECT_NE(exactWord(w, e), exactBand(b, e))
            << "the loss must be real, or this is not the boundary";
    }
    // Past the FLOOR: the residue limb's magic constant 2^(e-32) goes
    // subnormal, and the reconstruction jumps rather than degrading.
    {
        const int e = kFloorE - 1; // -95
        const BandCell8 w{ (static_cast<uint64_t>(bits(exactLimb(
                                0x800000u, e - 23, false)))
                               << 32)
            | 0x00000101u };
        const Band back = bd::bandFromCell8(w);
        EXPECT_NE(exactWord(w, e), exactBand(back, e))
            << "below the floor the decode must stop agreeing with the word it "
               "was handed -- that disagreement IS the cliff the admission "
               "predicate exists to keep consumers away from";
    }
    std::printf("[BandCell8] out of window: RELEASE build -- above the ceiling "
                "the continuation is dropped, below the floor the residue "
                "limb's constant is subnormal\n");
#endif
}

// -------------------------------------------------------------------------
//  14. Non-vacuity: the sweep really did reach every case it claims to.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, SweepCoverageIsNonVacuous)
{
    uint64_t inPlace = 0, borrowed = 0, negative = 0, deepest = 0, highest = 0,
             exactRows = 0, roundedDown = 0, roundedUp = 0, emptyTail = 0,
             droppedLimbs = 0, powerOfTwo = 0;

    for (const Row& r : sweepRows()) {
        const BandCell8 w = bd::cell8FromBand(r.b);
        if (wordExp(w) == r.e)
            inPlace++;
        else if (wordExp(w) == r.e - 1)
            borrowed++;
        if ((w.w >> 63) != 0)
            negative++;
        if (r.e <= kFloorE + 8)
            deepest++;
        if (r.e >= kCeilE - 8)
            highest++;
        if (wordTail(w) == 0u)
            emptyTail++;
        if ((bits(r.b.hi) & 0x7FFFFFu) == 0u)
            powerOfTwo++;
        if (r.p->low23 != 0 && bits(r.b.tail) == 0u)
            droppedLimbs++;

        const Exact want = exactAbs(exactBand(r.b, r.e));
        const Exact got  = exactAbs(exactWord(w, r.e));
        if (got == want)
            exactRows++;
        else if (got < want)
            roundedDown++;
        else
            roundedUp++;
    }

    std::printf("[BandCell8] coverage: %llu stored in place, %llu borrowed a "
                "binade, %llu negative, %llu within 8 of the floor, %llu "
                "within 8 of the ceiling, %llu exact, %llu rounded down, %llu "
                "rounded up, %llu with an empty continuation, %llu at an exact "
                "power of two, %llu with a limb dropped under FP32's floor\n",
        static_cast<unsigned long long>(inPlace),
        static_cast<unsigned long long>(borrowed),
        static_cast<unsigned long long>(negative),
        static_cast<unsigned long long>(deepest),
        static_cast<unsigned long long>(highest),
        static_cast<unsigned long long>(exactRows),
        static_cast<unsigned long long>(roundedDown),
        static_cast<unsigned long long>(roundedUp),
        static_cast<unsigned long long>(emptyTail),
        static_cast<unsigned long long>(powerOfTwo),
        static_cast<unsigned long long>(droppedLimbs));

    EXPECT_GT(inPlace, 1000u);
    EXPECT_GT(borrowed, 100u) << "no row made the encoder step the leading "
                                 "limb down, so the borrow path -- and the "
                                 "binade fixup inside it -- is untested";
    EXPECT_GT(negative, 1000u);
    EXPECT_GT(deepest, 100u) << "the sweep never went near the bottom binades";
    EXPECT_GT(highest, 100u) << "the sweep never went near the top binades";
    EXPECT_GT(exactRows, 1000u);
    EXPECT_GT(roundedDown, 100u);
    EXPECT_GT(roundedUp, 100u);
    EXPECT_GT(powerOfTwo, 100u) << "no row sat on an exact power of two, which "
                                   "is the only place the binade fixup fires";
}

// -------------------------------------------------------------------------
//  15. The punned access moves a word without changing it.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, PunnedLoadStoreRoundTripsBitwise)
{
    std::vector<BandCell8> src, dst;
    for (const Row& r : sweepRows())
        src.push_back(bd::cell8FromBand(r.b));
    dst.assign(src.size(), BandCell8{ 0xA5A5A5A5A5A5A5A5ull });

    for (std::size_t i = 0; i < src.size(); i++)
        bd::storeCell8(&dst[i], bd::loadCell8(&src[i]));

    uint64_t bad = 0;
    for (std::size_t i = 0; i < src.size(); i++)
        if (src[i].w != dst[i].w)
            bad++;
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(src.size(), 3000u);
}

// -------------------------------------------------------------------------
//  16. THE EQUIVALENCE GATE. The shipped codec is bit-for-bit the one that was
//      measured.
//
//      Every number the storage round ratified -- depth 56, the 220-binade
//      window, the instruction counts, the escape tier's price -- was measured
//      on the probe body transcribed into `f32cReference.h`. This test is the
//      link between those numbers and the code that ships: not "close", not
//      "equivalent in value", but the same 64 bits out of the same input, over
//      the whole enumerated corpus and a million random in-window rows.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, CodecIsBitIdenticalToTheProbeReference)
{
    uint64_t enumerated = 0, encBad = 0, decBad = 0, hiBad = 0, escBad = 0;

    for (const Row& r : sweepRows()) {
        enumerated++;
        const uint64_t mine = bd::cell8FromBand(r.b).w;
        const uint64_t theirs = f32c::encode(asBand3(r.b));
        if (mine != theirs) {
            encBad++;
            if (encBad == 1)
                ADD_FAILURE()
                    << "production and reference encoders disagree at 2^" << r.e
                    << " pattern '" << r.p->name << "': " << std::hex << mine
                    << " vs " << theirs << std::dec;
        }
        const Band md          = bd::bandFromCell8(BandCell8{ theirs });
        const f32c::Band3 td   = f32c::decode(theirs);
        if (bits(md.hi) != bits(td.hi) || bits(md.lo) != bits(td.lo)
            || bits(md.tail) != bits(td.tail)) {
            decBad++;
            if (decBad == 1)
                ADD_FAILURE()
                    << "production and reference decoders disagree at 2^"
                    << r.e << " pattern '" << r.p->name << "'";
        }
        if (bits(bd::floatFromCell8(BandCell8{ theirs }))
            != bits(f32c::decodeHiOnly(theirs)))
            hiBad++;
        if (bd::cell8IsEscape(BandCell8{ theirs })
            != f32c::isEscape(static_cast<uint32_t>(theirs >> 32)))
            escBad++;
    }

    // A million random in-window rows on top of the enumeration. Random rows
    // are added to the structural ones, never instead of them: they cover the
    // interior, the enumeration covers the edges.
    std::mt19937_64 rng(0xC0FFEEu);
    const int bulk = isMinimalMode() ? 200000 : 1000000;
    uint64_t bulkBad = 0;
    for (int i = 0; i < bulk; i++) {
        const int e      = kFloorE + static_cast<int>(rng() % 220);
        const uint32_t t = 0x800000u | (static_cast<uint32_t>(rng()) & 0x7FFFFFu);
        const uint32_t m = static_cast<uint32_t>(rng()) & 0x3FFFFFu;
        const uint32_t l = static_cast<uint32_t>(rng()) & 0x7FFFFFu;
        const bool neg   = (rng() & 1) != 0;
        const bool canc  = (rng() & 2) != 0;
        const Band b{ exactLimb(t, e - 23, neg),
            exactLimb(m, e - 47, canc ? !neg : neg),
            exactLimb(l, e - 71, canc ? !neg : neg) };
        if (bd::cell8FromBand(b).w != f32c::encode(asBand3(b)))
            bulkBad++;
    }

    // The rebias arm, at the archive's own sweep points.
    uint64_t rebiasRows = 0, rebiasBad = 0;
    for (int arrayBias : { -900, -400, -32, -2, 0, 7, 32, 400, 900 })
        for (const Row& r : sweepRows()) {
            rebiasRows++;
            const uint64_t mine
                = bd::cell8FromBandRebias(r.b, arrayBias).w;
            const uint64_t theirs = f32c::encodeRebias(asBand3(r.b), arrayBias);
            if (mine != theirs) {
                rebiasBad++;
                continue;
            }
            const Band md        = bd::bandFromCell8Rebias(BandCell8{ mine }, arrayBias);
            const f32c::Band3 td = f32c::decodeRebias(theirs, arrayBias);
            if (bits(md.hi) != bits(td.hi) || bits(md.lo) != bits(td.lo)
                || bits(md.tail) != bits(td.tail))
                rebiasBad++;
        }

    // The escape arm: the decode and the predicate have a reference; the
    // ENCODE does not, and deliberately differs (see the escape depth test).
    uint64_t escRows = 0, escDecBad = 0;
    const uint64_t escMants[] = { uint64_t{ 0 }, uint64_t{ 1 },
        (uint64_t{ 1 } << 52) - 1, uint64_t{ 1 } << 51,
        uint64_t{ 0x921FB54442D18ull } };
    for (int e = -1000; e <= 1000; e += 3)
        for (uint64_t mant : escMants)
            for (int s = 0; s < 2; s++) {
                const uint64_t u = (static_cast<uint64_t>(s) << 63)
                    | (static_cast<uint64_t>(e + 1023) << 52) | mant;
                // Words built by hand, so the DECODE is compared on words the
                // encoder may never produce as well as on the ones it does.
                const uint64_t w = (u << 7) ^ (u >> 11);
                escRows++;
                const EscapeParts mc  = bd::escapePartsFromCell8(BandCell8{ w });
                const f32c::BiasedBand tc = f32c::decodeEscaped(w);
                // `detail::EscapeParts` has no `tail` field at all -- the
                // escape decode's contract to leave that limb at `+0` is
                // expressed structurally, so the reference's half of that
                // claim is asserted directly instead of compared against a
                // field that cannot disagree.
                if (bits(mc.hi) != bits(tc.b.hi) || bits(mc.lo) != bits(tc.b.lo)
                    || bits(tc.b.tail) != bits(0.0f) || mc.bias != tc.bias)
                    escDecBad++;
            }

    std::printf("[BandCell8] equivalence: %llu enumerated rows (%llu encode, "
                "%llu decode, %llu float-tier, %llu predicate mismatches), %d "
                "random rows (%llu mismatches), %llu rebias rows (%llu), %llu "
                "escape-decode rows (%llu)\n",
        static_cast<unsigned long long>(enumerated),
        static_cast<unsigned long long>(encBad),
        static_cast<unsigned long long>(decBad),
        static_cast<unsigned long long>(hiBad),
        static_cast<unsigned long long>(escBad), bulk,
        static_cast<unsigned long long>(bulkBad),
        static_cast<unsigned long long>(rebiasRows),
        static_cast<unsigned long long>(rebiasBad),
        static_cast<unsigned long long>(escRows),
        static_cast<unsigned long long>(escDecBad));

    EXPECT_EQ(encBad, 0u);
    EXPECT_EQ(decBad, 0u);
    EXPECT_EQ(hiBad, 0u);
    EXPECT_EQ(escBad, 0u);
    EXPECT_EQ(bulkBad, 0u);
    EXPECT_EQ(rebiasBad, 0u);
    EXPECT_EQ(escDecBad, 0u);
    EXPECT_GT(enumerated, 3000u);
    EXPECT_GT(rebiasRows, 30000u);
    EXPECT_GT(escRows, 3000u);
}

// -------------------------------------------------------------------------
//  17. The FP32 tier costs one reinterpretation and answers the truncated head.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, TheFloatTierReadsTheLeadingLimbForFree)
{
    uint64_t rows = 0, bad = 0, differsFromValue = 0;
    for (const Row& r : sweepRows()) {
        const BandCell8 w = bd::cell8FromBand(r.b);
        const float f     = bd::floatFromCell8(w);
        rows++;
        // It IS the high half of the word, bit for bit. Nothing is computed.
        if (bits(f) != static_cast<uint32_t>(w.w >> 32))
            bad++;
        // And it is the correctly TRUNCATED head: the full value differs from
        // it by the continuation, which is non-negative and below one ulp.
        const Exact full = exactAbs(exactWord(w, r.e));
        const Exact head = exactAbs(exactFloat(f, r.e));
        if (full != head)
            differsFromValue++;
        EXPECT_LE(head, full) << "the FP32 tier must never overshoot: the "
                                 "continuation is an unsigned remainder";
        if (bad > 0)
            break;
    }
    std::printf("[BandCell8] float tier: %llu rows, %llu not the high half, "
                "%llu carrying a non-empty continuation\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(bad),
        static_cast<unsigned long long>(differsFromValue));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(differsFromValue, 1000u)
        << "every row's continuation was empty, so 'truncated head' was never "
           "actually tested";
}

// -------------------------------------------------------------------------
//  18. The escape tier carries every finite double at depth 45.
//
//      The tier-2 contract is a RELATIVE one -- 1 implicit bit plus 44 stored,
//      so a relative error no larger than 2^-45 -- and it is checked against
//      the exact 128-bit value of the input rather than against a float
//      recomputation, because the whole point is exponents no float can hold.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, EscapeTierRoundTripsEveryFiniteDoubleAtDepthFortyFive)
{
    // ENUMERATED mantissa payloads: the structurally special encodings, not a
    // sample. The last four are the rounding cases at the 8-bit drop.
    const uint64_t mants[] = {
        0,
        (uint64_t{ 1 } << 52) - 1,
        1,
        uint64_t{ 1 } << 51,
        0xAAAAAAAAAAAAAull,
        0x5555555555555ull,
        0x921FB54442D18ull,
        0xFFull,               // the whole dropped field set: worst truncation
        0x80ull,               // an exact tie, even -> down
        0x180ull,              // an exact tie, odd -> up
        0x7Full,               // just below the tie
        0xFFFFFFFFFFF80ull,    // a tie that carries all the way out
    };

    uint64_t rows = 0, overDepth = 0, carried = 0, tagBad = 0, stolen = 0,
             roundedOut = 0;
    long double worst = 0.0L;

    for (int e = -1000; e <= 1000; e++) {
        if (!tier2Covers(e))
            continue; // tier 1's territory: a different test owns those rows
        for (uint64_t mant : mants)
            for (int s = 0; s < 2; s++) {
                // Rows the encoder is contracted to REFUSE are excluded here --
                // never quietly: each is counted, each count is bounded below,
                // and each is pinned as a known answer by
                // `EscapeTagsAndTheTierBoundaryArePinned`. The fate is
                // recomputed from the layout rather than asked of the encoder,
                // so the exclusion is a stated consequence rather than a filter
                // on the property under test.
                const EscapeRowFate fate = escapeRowFate(e, mant, s);
                if (fate.roundedOut) {
                    roundedOut++;
                    continue;
                }
                if (fate.stolen) {
                    stolen++;
                    continue;
                }
                const uint64_t u = (static_cast<uint64_t>(s) << 63)
                    | (static_cast<uint64_t>(e + 1023) << 52) | mant;
                const BandCell8 w = bd::cell8EscapedFromIEEE(
                    static_cast<uint32_t>(u & 0xFFFFFFFFu),
                    static_cast<uint32_t>(u >> 32));
                rows++;
                if (!bd::cell8IsEscape(w))
                    tagBad++;

                const EscapeParts back = bd::escapePartsFromCell8(w);
                if (back.bias != e)
                    carried++; // a rounding carry lifted the exponent

                // Exact comparison at the common scale 2^(e-100): the input is
                // (2^52 | mant) * 2^(e-52), the output is the decoded
                // significand (a value in [1,2), read exactly off its limbs)
                // times 2^bias.
                const Exact want
                    = static_cast<Exact>((uint64_t{ 1 } << 52) | mant) << 48;
                Exact got = exactAbs(exactFloat(back.hi, 0))
                    + exactAbs(exactFloat(back.lo, 0));
                const int shift = back.bias - e;
                got = (shift >= 0) ? (got << shift) : (got >> (-shift));

                const Exact d = exactAbs(got - want);
                // depth 45: |d| / want <= 2^-45
                if ((d << bd::kBandCell8EscapeDepth) > want) {
                    overDepth++;
                    if (overDepth == 1)
                        ADD_FAILURE()
                            << "the escape tier lost more than 45 bits at 2^"
                            << e << " with mantissa " << std::hex << mant
                            << std::dec;
                }
                const long double ratio
                    = static_cast<long double>(static_cast<int64_t>(d))
                    / std::ldexp(static_cast<long double>(
                                      static_cast<int64_t>(want >> 40)),
                        40);
                if (ratio > worst)
                    worst = ratio;
            }
    }
    EXPECT_EQ(stolen, 6u)
        << "the corpus carries six payloads that round onto the code point "
           "canonical zero has taken -- four sitting on 2^-94 already, and two "
           "at 2^-95 whose significand carries up onto it. Every one of those "
           "magnitudes is inside TIER 1, which stores it exactly at 56 bits, "
           "so nothing is lost; if this count moves, either the escape "
           "rounding or the stolen code point has changed, and both are "
           "contracts";
    EXPECT_EQ(roundedOut, 4u)
        << "and four that the rounding walks out of the tier altogether, from "
           "2^-94 up into tier 1";

    std::printf("[BandCell8] escape depth: %llu rows over 2^-1000..2^1000, "
                "%llu past 2^-45, %llu lifted by a rounding carry, %llu not "
                "escape-tagged, %llu spent on canonical zero, %llu rounded out "
                "of the tier; worst relative error %.6Le (2^-45 = %.6Le)\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(overDepth),
        static_cast<unsigned long long>(carried),
        static_cast<unsigned long long>(tagBad),
        static_cast<unsigned long long>(stolen),
        static_cast<unsigned long long>(roundedOut), worst, 0x1p-45L);

    EXPECT_EQ(overDepth, 0u);
    EXPECT_EQ(tagBad, 0u) << "an escape word that is not escape-tagged would be "
                             "read as a tier-1 value and silently mean "
                             "something else";
    EXPECT_GT(rows, 3000u);
    EXPECT_GT(carried, 0u) << "no rounding carry occurred, so the case where "
                              "the significand rounds up out of its field is "
                              "untested";
}

// -------------------------------------------------------------------------
//  19. The tier boundary, the tags, and the two holes in the map.
//
//      Everything a mixed-tier READER depends on, pinned as known answers:
//      which words are escapes, which are not, what the boundary words decode
//      to, and the two magnitudes the ratified mapping does not name. The
//      holes are tested because an unpinned hole moves.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, EscapeTagsAndTheTierBoundaryArePinned)
{
    // Known-answer escape words, derived by hand from the layout.
    struct Anchor {
        uint32_t hi, lo;   ///< the IEEE double's two words
        uint64_t want;     ///< the escape word
        int bias;          ///< what the decode must hand back
        const char* why;
    };
    const Anchor anchors[] = {
        { 0x47D00000u, 0u, 0x7F80000000000000ull, 126,
            "2^126: the first exponent the upward tag names, ext = 0 -- and the "
            "binade immediately above tier 1's ceiling, which is what makes the "
            "two tiers contiguous" },
        { 0xC7D00000u, 0u, 0xFF80000000000000ull, 126,
            "-2^126: the sign rides in bit 63, as in tier 1" },
        { 0x47F00000u, 0u, 0x7F80200000000000ull, 128,
            "2^128: two binades up, ext = 2" },
        { 0xBA100000u, 0u, 0x8000000000000000ull, -94,
            "-2^-94: the first exponent the downward tag names -- and the same "
            "word tier 1 writes for a negative zero, which is why an "
            "escape-using array canonicalizes its zeros" },
        { 0xBA000000u, 0u, 0x8000100000000000ull, -95,
            "-2^-95: one binade below the tag's base, ext = 1" },
        { 0x7FE80000u, 0u, 0x7FB8180000000000ull, 1023,
            "1.5 x 2^1023, near the top of double's range" },
        { 0x00100000u, 0u, 0x003A000000000000ull, -1022,
            "2^-1022, the smallest normal double" },
    };
    for (const Anchor& a : anchors) {
        const BandCell8 w = bd::cell8EscapedFromIEEE(a.lo, a.hi);
        EXPECT_EQ(w.w, a.want) << a.why;
        EXPECT_TRUE(bd::cell8IsEscape(w)) << a.why;
        EXPECT_EQ(bd::escapePartsFromCell8(w).bias, a.bias) << a.why;
    }

    // The predicate, exhaustively over the exponent byte. Exactly two of the
    // 256 bytes are escapes, and they are the two the tier-1 encoder cannot
    // produce.
    int escapes = 0;
    for (int byte = 0; byte < 256; byte++) {
        const uint64_t w = static_cast<uint64_t>(static_cast<uint32_t>(byte)
                               << 23)
            << 32;
        if (bd::cell8IsEscape(BandCell8{ w }))
            escapes++;
        EXPECT_EQ(bd::cell8IsEscape(BandCell8{ w }), byte == 0 || byte == 0xFF)
            << "exponent byte " << byte;
        // ...and the sign bit must not change the answer.
        EXPECT_EQ(bd::cell8IsEscape(BandCell8{ w | (uint64_t{ 1 } << 63) }),
            bd::cell8IsEscape(BandCell8{ w }));
    }
    EXPECT_EQ(escapes, 2);

    // No tier-1 word anywhere in the window carries an escape tag, so a reader
    // dispatching on the predicate never mistakes one for the other.
    uint64_t collide = 0;
    for (const Row& r : sweepRows())
        if (bd::cell8IsEscape(bd::cell8FromBand(r.b)))
            collide++;
    EXPECT_EQ(collide, 0u)
        << "a tier-1 word was tagged as an escape, which makes the two tiers "
           "indistinguishable and the whole scheme unsound";

    // The checked read hands back the flag AND the tier-1 answer, and the flag
    // is the predicate's.
    bool esc = false;
    const Band b1 = bd::bandFromCell8Checked(bd::cell8FromBand(Band{ 1.5f, 0.0f, 0.0f }), esc);
    EXPECT_FALSE(esc);
    EXPECT_EQ(bits(b1.hi), bits(1.5f));
    (void)bd::bandFromCell8Checked(BandCell8{ 0x7F80000000000000ull }, esc);
    EXPECT_TRUE(esc);

    // ★ THE SEAM. The two bases are anchored on tier 1's measured window, and
    // the two consequences that follow are pinned here so that moving either
    // anchor is a deliberate act with a visible diff.
    //   (a) UPWARD: contiguous. The escape tier's first binade is the one
    //       immediately above tier 1's last, so no magnitude falls between
    //       them. (This retires the two-binade hole the probe's base of 128
    //       left behind.)
    EXPECT_EQ(bd::kBandCell8EscapeUpBase, kCeilE + 1)
        << "the upward tag must start exactly where tier 1 stops";
    EXPECT_TRUE(tier1Covers(kCeilE));
    EXPECT_FALSE(tier1Covers(kCeilE + 1));
    EXPECT_TRUE(tier2Covers(kCeilE + 1));
    EXPECT_FALSE(tier2Covers(kCeilE));
    //   (b) DOWNWARD: overlapping by exactly one binade, and that overlap is
    //       what pays for canonical zero. The all-zero word is the downward
    //       tag's first code point, so the magnitude it takes -- +2^-94 with an
    //       all-zero significand -- has to be one TIER 1 can hold, and it is.
    EXPECT_EQ(bd::kBandCell8EscapeDownBase, kFloorE)
        << "the downward tag must start at tier 1's floor, so the code point "
           "canonical zero takes is a magnitude tier 1 already stores";
    EXPECT_TRUE(tier1Covers(bd::kBandCell8EscapeDownBase));
    EXPECT_TRUE(tier2Covers(bd::kBandCell8EscapeDownBase));
    {
        // ...and tier 1 really does store it, exactly, at 56 bits.
        const Band b{ exactLimb(0x800000u, kFloorE - 23, false), 0.0f, 0.0f };
        const BandCell8 w = bd::cell8FromBand(b);
        EXPECT_EQ(exactWord(w, kFloorE), exactBand(b, kFloorE))
            << "the magnitude canonical zero's code point takes must be stored "
               "EXACTLY by tier 1, or the closure gave something away";
    }
    // (In a debug build the escape encoder ASSERTS on that input rather than
    // returning the word, so the known answer is pinned in release builds and
    // the assertion is the guard in debug ones -- the same division of labour
    // as the out-of-window test.)
#ifndef AETHER_DEBUG_MODE
    EXPECT_EQ(bd::cell8EscapedFromIEEE(0u, 0x3A100000u).w, 0ull)
        << "the one stolen code point, now a magnitude tier 1 covers";
#endif
    EXPECT_NE(bd::cell8EscapedFromIEEE(0u, 0x3A180000u).w, 0ull)
        << "any other significand at 2^-94 must encode normally";
    EXPECT_NE(bd::cell8EscapedFromIEEE(0u, 0xBA100000u).w, 0ull)
        << "the negative twin must encode normally";

    //   (c) THE RESIDUAL SEAM, and it is residual rather than closed: a value
    //       at the downward tag's base whose significand rounds UP walks into
    //       tier 1, where the escape encoder will not follow. Nothing is lost
    //       -- tier 1 holds those magnitudes at 56 bits rather than 45 -- and
    //       the fate is computed from the layout, so the encoder and this
    //       expectation cannot drift apart silently.
    {
        const EscapeRowFate f
            = escapeRowFate(bd::kBandCell8EscapeDownBase,
                (uint64_t{ 1 } << 52) - 1, 0);
        EXPECT_TRUE(f.roundedOut)
            << "an all-ones significand at the downward base must round up out "
               "of the tier";
        EXPECT_EQ(f.e, bd::kBandCell8EscapeDownBase + 1);
        EXPECT_TRUE(tier1Covers(f.e))
            << "...and land somewhere TIER 1 covers, which is why refusing it "
               "loses nothing";
    }

    // Zero and the subnormals go to canonical zero rather than to a tag.
    EXPECT_EQ(bd::cell8EscapedFromIEEE(0u, 0u).w, 0ull);
    EXPECT_EQ(bd::cell8EscapedFromIEEE(0u, 0x80000000u).w, 0ull)
        << "negative zero canonicalizes on the escape path";
    EXPECT_EQ(bd::cell8EscapedFromIEEE(1u, 0u).w, 0ull)
        << "a double subnormal is a thousand binades below the carrier's floor "
           "and flushes";
}

// -------------------------------------------------------------------------
//  20. CONTIGUITY. Between them the two tiers name every finite double, with
//      no magnitude falling between them.
//
//      This is the claim the two exponent bases exist to make, so it is
//      measured rather than argued, and it is measured at the SEAMS -- the
//      handful of binades where tier 1 stops and the escape tier starts -- as
//      well as across the whole of double's exponent range. Each seam exponent
//      is crossed with the enumerated mantissa payloads and both signs, and the
//      value is required to survive at the DEPTH OF WHICHEVER TIER NAMES IT:
//      56 bits through tier 1, 45 through the escape tier.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, TheTwoTiersCoverEveryFiniteDoubleWithoutAGap)
{
    // 1. Coverage, over every exponent a finite double can have -- normals
    //    down to 2^-1022 and subnormals normalized as far as 2^-1074.
    uint64_t uncovered = 0, byTier1 = 0, byTier2 = 0, byBoth = 0;
    for (int e = -1074; e <= 1023; e++) {
        const bool t1 = tier1Covers(e), t2 = tier2Covers(e);
        if (!t1 && !t2) {
            uncovered++;
            if (uncovered == 1)
                ADD_FAILURE() << "2^" << e << " is named by NEITHER tier";
        }
        byTier1 += t1 ? 1u : 0u;
        byTier2 += t2 ? 1u : 0u;
        byBoth += (t1 && t2) ? 1u : 0u;
    }
    EXPECT_EQ(uncovered, 0u);
    EXPECT_EQ(byBoth, 1u)
        << "the two tiers must overlap in exactly ONE binade -- the downward "
           "tag's base, which is where the code point spent on canonical zero "
           "lives";
    EXPECT_GT(byTier1, 200u);
    EXPECT_GT(byTier2, 1800u);

    // 2. The seams themselves, value by value. Three binades either side of
    //    each boundary, every enumerated payload, both signs.
    const uint64_t mants[] = {
        0,
        (uint64_t{ 1 } << 52) - 1,
        1,
        uint64_t{ 1 } << 51,
        0xAAAAAAAAAAAAAull,
        0x5555555555555ull,
        0x921FB54442D18ull,
        0xFFull,
        0x80ull,
    };
    uint64_t rows = 0, t1Rows = 0, t2Rows = 0, bad = 0, refused = 0;
    long double worstT1 = 0.0L, worstT2 = 0.0L;

    for (int e = kFloorE - 3; e <= kCeilE + 3; e++) {
        // Only the seams: the interior is the business of the other tests.
        if (e > kFloorE + 3 && e < kCeilE - 3)
            continue;
        for (uint64_t mant : mants)
            for (int s = 0; s < 2; s++) {
                const uint64_t u = (static_cast<uint64_t>(s) << 63)
                    | (static_cast<uint64_t>(e + 1023) << 52) | mant;
                const Exact want
                    = static_cast<Exact>((uint64_t{ 1 } << 52) | mant) << 48;
                rows++;
                ASSERT_TRUE(tier1Covers(e) || tier2Covers(e))
                    << "2^" << e << " is named by neither tier";

                if (tier1Covers(e)) {
                    // Tier 1 at depth 56, judged against the value the CARRIER
                    // handed the cell: the cell must lose nothing of its own.
                    const Band in = bd::bandFromIEEE(
                        static_cast<uint32_t>(u & 0xFFFFFFFFu),
                        static_cast<uint32_t>(u >> 32));
                    const BandCell8 w = bd::cell8FromBand(in);
                    const Exact got   = exactAbs(exactWord(w, e));
                    const Exact carried = exactAbs(exactBand(in, e));
                    const Exact ulp   = exactWordUlp(w, e);
                    t1Rows++;
                    if (ulp <= 0 || 2 * exactAbs(got - carried) > ulp) {
                        bad++;
                        if (bad == 1)
                            ADD_FAILURE()
                                << "tier 1 lost more than half an ulp at 2^"
                                << e << " (mantissa " << std::hex << mant
                                << std::dec << ")";
                    }
                    // ...and, for the record, how the whole journey from the
                    // double did. Below the carrier's own floor this is the
                    // CARRIER's loss, not the cell's, which is why it is
                    // reported rather than asserted.
                    const Exact d = exactAbs(got - want);
                    const long double ratio
                        = static_cast<long double>(static_cast<int64_t>(d))
                        / std::ldexp(static_cast<long double>(
                                          static_cast<int64_t>(want >> 40)),
                            40);
                    if (ratio > worstT1)
                        worstT1 = ratio;
                }
                if (tier2Covers(e)) {
                    const EscapeRowFate fate = escapeRowFate(e, mant, s);
                    if (!fate.encodable) {
                        refused++;
                        continue;
                    }
                    const BandCell8 w = bd::cell8EscapedFromIEEE(
                        static_cast<uint32_t>(u & 0xFFFFFFFFu),
                        static_cast<uint32_t>(u >> 32));
                    const EscapeParts back = bd::escapePartsFromCell8(w);
                    t2Rows++;
                    Exact got = exactAbs(exactFloat(back.hi, 0))
                        + exactAbs(exactFloat(back.lo, 0));
                    const int shift = back.bias - e;
                    got = (shift >= 0) ? (got << shift) : (got >> (-shift));
                    const Exact d = exactAbs(got - want);
                    if ((d << bd::kBandCell8EscapeDepth) > want) {
                        bad++;
                        if (bad == 1)
                            ADD_FAILURE()
                                << "the escape tier lost more than 45 bits at "
                                << "2^" << e;
                    }
                    const long double ratio
                        = static_cast<long double>(static_cast<int64_t>(d))
                        / std::ldexp(static_cast<long double>(
                                          static_cast<int64_t>(want >> 40)),
                            40);
                    if (ratio > worstT2)
                        worstT2 = ratio;
                }
            }
    }

    std::printf("[BandCell8] contiguity: %llu exponents over 2^-1074..2^1023, "
                "%llu uncovered, %llu tier-1, %llu tier-2, %llu in both; seam "
                "rows %llu (%llu through tier 1, %llu through tier 2, %llu "
                "refused at the seam), worst relative error tier 1 %.6Le "
                "(whole journey incl. the carrier), tier 2 %.6Le\n",
        2098ull, static_cast<unsigned long long>(uncovered),
        static_cast<unsigned long long>(byTier1),
        static_cast<unsigned long long>(byTier2),
        static_cast<unsigned long long>(byBoth),
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(t1Rows),
        static_cast<unsigned long long>(t2Rows),
        static_cast<unsigned long long>(refused), worstT1, worstT2);

    EXPECT_EQ(bad, 0u);
    EXPECT_GT(t1Rows, 50u);
    EXPECT_GT(t2Rows, 50u);
    EXPECT_GT(refused, 0u) << "the seam's refused rows are what makes this a "
                              "SEAM test rather than an interior one";

    // 3. What is NOT covered, pinned so it cannot be forgotten: a double
    //    SUBNORMAL has no encoding here. Its exponent is inside the escape
    //    tier's reach, but the encoder does not normalize -- it flushes, a
    //    thousand binades below where the banded carrier stopped delivering
    //    certified bits.
    EXPECT_TRUE(tier2Covers(-1074))
        << "the exponent is reachable...";
    EXPECT_EQ(bd::cell8EscapedFromIEEE(1u, 0u).w, 0ull)
        << "...but the ENCODER flushes subnormal doubles to canonical zero, "
           "which is a documented limit of the encoder rather than of the "
           "layout";
}

// -------------------------------------------------------------------------
//  21. The per-array rebias round-trips bit-exactly, anywhere in double's
//      range.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, PerArrayRebiasRoundTripsBitExactly)
{
    uint64_t rows = 0, bad = 0, movedWord = 0;
    for (int arrayBias : { -900, -400, -32, -2, 0, 7, 32, 400, 900 }) {
        for (const Row& r : sweepRows()) {
            rows++;
            const BandCell8 w = bd::cell8FromBandRebias(r.b, arrayBias);
            const Band back   = bd::bandFromCell8Rebias(w, arrayBias);
            const Band plain  = bd::bandFromCell8(bd::cell8FromBand(r.b));
            if (!sameBandBits(back, plain)) {
                bad++;
                if (bad == 1)
                    ADD_FAILURE()
                        << "the rebiased round trip did not agree with the "
                        << "plain one at arrayBias " << arrayBias << ", 2^"
                        << r.e << ", pattern '" << r.p->name << "'";
            }
            if (arrayBias != 0 && w.w != bd::cell8FromBand(r.b).w)
                movedWord++;
        }
    }
    std::printf("[BandCell8] rebias: %llu rows over 9 array biases, %llu "
                "mismatches, %llu words actually repositioned\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(bad),
        static_cast<unsigned long long>(movedWord));
    EXPECT_EQ(bad, 0u);
    EXPECT_GT(movedWord, 30000u)
        << "the rebias moved no word, so this test compared the plain codec "
           "against itself";
    EXPECT_GT(rows, 100000u);
}

// -------------------------------------------------------------------------
//  22. Rebias and the escape tier compose exactly where the header says.
//
//      The derivation in the header is arithmetic; this is the measurement. At
//      every array bias in a range that straddles both edges of the derived
//      interval, the whole window is swept and the stored words are tested for
//      an escape tag. Inside the interval there must be NONE; outside it there
//      must be SOME, or the predicate is refusing arrays it did not need to.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, RebiasComposesWithTheEscapeTierOnlyInsideItsRange)
{
    uint64_t insideCollisions = 0, outsideClean = 0, biases = 0;
    for (int arrayBias = bd::kBandCell8RebiasLowSafe - 6;
         arrayBias <= bd::kBandCell8RebiasHighSafe + 6; arrayBias++) {
        biases++;
        uint64_t collisions = 0;
        for (int e = kFloorE; e <= kCeilE; e++)
            for (int s = 0; s < 2; s++) {
                const Band b{ exactLimb(0xC90FDAu, e - 23, s == 1), 0.0f, 0.0f };
                if (bd::cell8IsEscape(bd::cell8FromBandRebias(b, arrayBias)))
                    collisions++;
            }
        const bool safe = bd::bandCell8RebiasKeepsEscapeTagsClear(arrayBias);
        if (safe && collisions > 0) {
            insideCollisions += collisions;
            ADD_FAILURE() << "arrayBias " << arrayBias << " is declared safe "
                          << "but produced " << collisions
                          << " words that look like escapes";
        }
        if (!safe && collisions == 0)
            outsideClean++;
    }

    // Tier-2 words are bias-ABSOLUTE: their extended exponent already carries
    // the whole magnitude, so a rebias must never be applied to them. The
    // decode proves it by not taking one.
    const BandCell8 esc = bd::cell8EscapedFromIEEE(0u, 0x7FE80000u);
    EXPECT_EQ(bd::escapePartsFromCell8(esc).bias, 1023)
        << "the escape decode takes no array bias -- that is what 'tier 2 is "
           "absolute' means, and it is why a mixed array's reader must test "
           "the STORED word before adding anything to it";

    std::printf("[BandCell8] rebias/escape composition: %llu array biases "
                "swept, %llu collisions inside the safe range, %llu biases "
                "outside it that were clean anyway\n",
        static_cast<unsigned long long>(biases),
        static_cast<unsigned long long>(insideCollisions),
        static_cast<unsigned long long>(outsideClean));

    EXPECT_EQ(insideCollisions, 0u);
    EXPECT_EQ(outsideClean, 0u)
        << "an array bias outside the derived range produced no collisions at "
           "all, so the range is more conservative than the mechanism requires "
           "-- re-derive it rather than widening it by hand";
}

// -------------------------------------------------------------------------
//  23. NON-VACUITY OF THE WHOLE BATTERY: two known defects, both caught.
//
//      A green suite means nothing until the corpus has been shown to be able
//      to go red. Two codecs with one contract-bearing line removed each
//      (`f32cReference.h`, namespace `f32cdefect`) are run over the SAME rows
//      and judged by the SAME oracles, and the number of rows that catch each
//      one is asserted to be non-zero and printed. A zero here would mean the
//      rows are blind and the corpus is what needs fixing.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, SeededDefectsAreCaughtByThisCorpus)
{
    // S1: the binade-borrow fixup removed. Caught by bit-identity against the
    // production encoder AND, independently, by the exact-value oracle.
    uint64_t s1Bits = 0, s1Value = 0, s1Rows = 0;
    for (const Row& r : sweepRows()) {
        s1Rows++;
        const uint64_t defect = f32cdefect::encodeNoBorrowFixup(asBand3(r.b));
        const BandCell8 good  = bd::cell8FromBand(r.b);
        if (defect != good.w)
            s1Bits++;
        if (exactWord(BandCell8{ defect }, r.e) != exactWord(good, r.e))
            s1Value++;
    }

    // S2: the canonical-zero case removed from the escape decode. Caught on
    // the one word the whole format agrees about.
    const f32c::BiasedBand z = f32cdefect::decodeEscapedNoZeroCase(0ull);
    const EscapeParts good      = bd::escapePartsFromCell8(BandCell8{ 0 });
    const uint64_t s2
        = (z.bias != good.bias || bits(z.b.hi) != bits(good.hi)) ? 1u : 0u;

    std::printf("[BandCell8] seeded defects: S1 (no borrow fixup) caught on "
                "%llu of %llu rows by bit identity and %llu by the exact "
                "oracle; S2 (no canonical zero) caught: %s\n",
        static_cast<unsigned long long>(s1Bits),
        static_cast<unsigned long long>(s1Rows),
        static_cast<unsigned long long>(s1Value), s2 ? "yes" : "NO");

    EXPECT_GT(s1Bits, 0u)
        << "no row in this corpus distinguishes a codec WITHOUT the binade "
           "borrow fixup from one with it -- the corpus is blind to the defect "
           "that the double round-trip gate originally caught on 484 rows, and "
           "the corpus is what needs fixing";
    EXPECT_GT(s1Value, 0u)
        << "the defect changes no VALUE by this oracle, so the oracle rather "
           "than the bit comparison is what is blind";
    EXPECT_EQ(s2, 1u)
        << "dropping the canonical-zero case changed nothing measurable, so "
           "the zero word is not actually being tested anywhere";
}

// -------------------------------------------------------------------------
//  16. THE BINADE-BORROW EDGE DEFECT. Do NOT soften this test to make it
//      pass -- a RED here, at the repro row, IS the deliverable this test
//      exists to pin.
//
//      `cell8BandIsStorable` admits ONE binade of slack at
//      `e == kFloorE - 1` (the borrow's own binade -- see the predicate's own
//      docstring). But `cell8FromBand`'s binade-borrow fixup steps the
//      leading limb's exponent field down by exactly one on EVERY borrow, so
//      a Band admitted at `e == kFloorE - 1` encodes with a leading exponent
//      field TWO below the true floor -- one further than
//      `bandFromCell8`'s continuation-splitting arithmetic (`eT = se -
//      (32 << 23)`) stays inside a valid exponent field. The low field wraps
//      into the reserved 0xFF (infinity) code point, and the decode computes
//      `-Infinity - -Infinity` for the tail limb, which is NaN -- out of a
//      Band the predicate just certified as storable.
//
//      This corpus does NOT fix the codec (that is out of this test's
//      scope); it enumerates the floor-exponent edge and PINS whichever
//      rows are broken, so the fix has a reproducible, mechanically-checked
//      red to point at.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, StorableImpliesFiniteDecode)
{
    enum Pattern { kAtRest = 0, kQuarterUlpRepro = 1, kHalfUlpBoundary = 2 };
    struct EdgeRow {
        Band b;
        int e;
        bool hiNeg;
        Pattern p;
    };
    std::vector<EdgeRow> rows;
    for (int e = kFloorE - 2; e <= kFloorE + 1; ++e) {
        for (bool hiNeg : { false, true }) {
            const float hi = exactLimb(0x800000u, e - 23, hiNeg);
            // The reducing lo must carry the OPPOSITE sign from hi, so `rest`
            // moves the magnitude DOWN -- the only path that borrows into the
            // leading limb (`cell8BandIsStorable`'s own `reducing` test).
            const bool loNeg           = !hiNeg;
            const float pats[3]        = {
                0.0f,                               // at rest: no borrow at all
                exactLimb(0x800000u, e - 48, loNeg), // quarter-ulp: the repro's own magnitude
                exactLimb(0x800000u, e - 47, loNeg), // half-ulp: the admitted boundary itself
            };
            for (int p = 0; p < 3; ++p)
                rows.push_back(
                    EdgeRow{ Band{ hi, pats[p], 0.0f }, e, hiNeg, static_cast<Pattern>(p) });
        }
    }
    ASSERT_EQ(rows.size(), 24u);

    // The literal repro for this defect must be IN this corpus, bit for
    // bit, or "incl. the repro" is not actually true.
    int reproIdx = -1;
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (rows[i].e == kFloorE - 1 && !rows[i].hiNeg && rows[i].p == kQuarterUlpRepro)
            reproIdx = static_cast<int>(i);
    ASSERT_GE(reproIdx, 0) << "the repro row was not found in the corpus";
    EXPECT_EQ(bits(rows[static_cast<std::size_t>(reproIdx)].b.hi), bits(2.5243549e-29f))
        << "the corpus row does not reproduce the defect repro's own hi literal";
    EXPECT_EQ(bits(rows[static_cast<std::size_t>(reproIdx)].b.lo), bits(-7.52316385e-37f))
        << "the corpus row does not reproduce the defect repro's own lo literal";

    // The FOUR rows this finding is about -- both signs, both
    // cancelling (non-`kAtRest`) patterns, at the slack binade `e ==
    // kFloorE - 1` -- are the ones `cell8BandIsStorable` must now REJECT: a
    // power-of-two `hi` there with a reducing `lo`/`tail` always borrows, and
    // being already on the slack binade, that borrow crosses ONE FIELD past
    // where `bandFromCell8`'s continuation split stays valid. Collected here
    // (not just the single repro row) so the predicate's fix is pinned on
    // every row the finding covers, not merely the one literal.
    std::vector<std::size_t> expectRejected;
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (rows[i].e == kFloorE - 1 && rows[i].p != kAtRest)
            expectRejected.push_back(i);
    ASSERT_EQ(expectRejected.size(), 4u)
        << "the corpus shape changed -- this is supposed to be exactly the 2 signs x 2 "
           "cancelling patterns at the slack binade";

    uint64_t storableN = 0, notFinite = 0, notRoundTrip = 0;
    bool reproStorable = false, reproFinite = false;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const EdgeRow& r    = rows[i];
        const bool storable = bd::cell8BandIsStorable(r.b);
        if (!storable)
            continue;
        storableN++;

        const BandCell8 c = bd::cell8FromBand(r.b);
        const Band back    = bd::bandFromCell8(c);
        const bool finite = std::isfinite(back.hi) && std::isfinite(back.lo)
            && std::isfinite(back.tail);
        if (static_cast<int>(i) == reproIdx) {
            reproStorable = true;
            reproFinite   = finite;
        }

        SCOPED_TRACE(testing::Message()
            << "row " << i << " e=" << r.e << " hiNeg=" << r.hiNeg << " pattern=" << r.p);
        if (!finite) {
            notFinite++;
            EXPECT_TRUE(finite)
                << "cell8BandIsStorable admitted this Band and the decode is NOT FINITE "
                   "-- a NaN or an infinity out of a value the predicate just certified";
            continue;
        }
        if (exactBand(r.b, r.e) != exactBand(back, r.e)) {
            notRoundTrip++;
            EXPECT_EQ(exactBand(r.b, r.e), exactBand(back, r.e))
                << "cell8BandIsStorable admitted this Band and store-then-load changed its "
                   "VALUE -- storable is supposed to mean exact";
        }
    }

    uint64_t rejectedOfFour = 0;
    for (std::size_t idx : expectRejected)
        if (!bd::cell8BandIsStorable(rows[idx].b))
            rejectedOfFour++;

    std::printf("[BandCell8] StorableImpliesFiniteDecode: %llu/%zu rows storable, %llu not "
                "finite, %llu did not round-trip; repro row %d storable=%s finite=%s; "
                "%llu/4 binade-borrow-past-the-slack rows now rejected\n",
        static_cast<unsigned long long>(storableN), rows.size(),
        static_cast<unsigned long long>(notFinite), static_cast<unsigned long long>(notRoundTrip),
        reproIdx, reproStorable ? "yes" : "no", reproFinite ? "yes" : "no",
        static_cast<unsigned long long>(rejectedOfFour));

    ASSERT_GT(storableN, 0u) << "no row in this corpus was admitted -- the predicate boundary "
                                 "moved and the corpus needs re-deriving";

    // The admission predicate now REJECTS the repro (and its three
    // siblings, checked below) rather than certifying a Band that decodes
    // to NaN -- the ruling is that aether fixes the codec rather than
    // reproducing the defect (the earlier oracle-comparison form of this row
    // was removed rather than kept passing).
    EXPECT_FALSE(reproStorable) << "the repro row must now be storable=false -- admission rejects "
                                    "the binade-borrow-past-the-slack class this row belongs to";
    EXPECT_EQ(rejectedOfFour, 4u)
        << "exactly the 4 binade-borrow-past-the-slack rows (2 signs x 2 cancelling patterns "
           "at e == kFloorE - 1) must be rejected -- fewer means the fix is too narrow, more "
           "means it is over-rejecting rows the predicate used to (and should still) admit";
}

} // namespace BandCell8Test
} // namespace aether_tests
