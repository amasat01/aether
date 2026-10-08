// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandCell8.h
 * @brief `BandCell8` — the 8-byte `Band` storage cell, and the codec between
 *        the two.
 *
 * One 64-bit word whose high half is a hardware IEEE-754 `binary32` and whose
 * low half is the next 32 significand bits. Logical value:
 * `float(w >> 32) + (w & 0xFFFFFFFF) * 2^(e - 55)`, with `e` the leading float's
 * exponent and the sign the leading float's. Depth 56 — three more than an IEEE
 * `double` carries — over a 220-binade window.
 *
 * @section why_bandcell8 Why the storage face is a different type from the carrier
 * A `Band` is three bare FP32 limbs with no memory form; a chain holds about 72
 * significant bits in one and only 56 survive a store. So "is the codec the
 * identity?" is the wrong question for most of the surface, and the codec makes
 * six separable claims, each needing its own check — see
 * `tests/test_BandCell8_common.h`, which is where they are certified:
 *  1. everything that fits is stored exactly (checked against the input itself,
 *     bitwise);
 *  2. everything that does not is rounded to nearest, ties to even (checked
 *     against an exact 128-bit fixed-point reconstruction, sharing no code with
 *     the codec);
 *  3. the window is what it says it is, and the first exponent past each edge is
 *     pinned as a known answer so the cliff cannot move quietly;
 *  4. a real `double` survives ingest → store → reload → delivery bit-identically
 *     wherever the carrier can hold it;
 *  5. the second (escape) tier round-trips every finite `double` at depth 45;
 *  6. the per-array rebias round-trips bit-exactly, and composes with the escape
 *     tier only inside the range this header derives.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "aether/banded/Band.h"
#include "aether/err/Error.h"
#include "aether/macros.h"

#ifdef AETHER_DEBUG_MODE
#include <cassert>
#endif

namespace aether {
namespace banded {

// =====================================================================
//  BandCell8 -- the 8-byte storage face of the banded carrier
// =====================================================================

/// @brief 8-byte storage cell: one 64-bit word whose high half IS an IEEE-754
/// `binary32` and whose low half is the next 32 significand bits.
///
/// The single member is deliberately a bare word rather than a bitfield pair:
/// bitfield layout is implementation-defined, and every operation this type
/// supports is a shift and a mask anyway.
struct alignas(8) BandCell8 {
    std::uint64_t w; ///< the packed word: `[ binary32 leading limb : 32 more bits ]`

#ifndef AETHER_CPP_MODE
    /**
     * @brief Reassemble a cell from a fetched `int2` texel. The entire
     *        fetch-side decode — two shifts and an OR, no arithmetic.
     *
     * The word order here is not the one the field names suggest: `x` is the
     * low word and `y` the high one, because an `int2` texel over eight
     * little-endian bytes maps `x` -> bytes 0-3, and a `uint64_t`'s low half
     * lives there. So `x` carries the unsigned 32-bit continuation and `y`
     * the leading `binary32` with the sign and exponent. Reading them the
     * other way round does not perturb a value — it reinterprets a
     * significand field as an exponent, which is wrong by an arbitrary power
     * of two and frequently by a sign, and which symmetric test data cannot
     * see. Checked by a value round-trip over asymmetric rows with a
     * deliberately swapped arm shown to fail
     * (`BandCell8Cert.PunnedLoadStoreRoundTripsBitwise`'s texel sibling).
     */
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 fromTexel(int2 t)
    {
        return BandCell8{ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(t.y)) << 32)
            | static_cast<std::uint64_t>(static_cast<std::uint32_t>(t.x)) };
    }
#endif

    /**
     * @brief Storage identity — the packed words are equal.
     *
     * A bit comparison, deliberately not a numeric one. The tier-1 codec is
     * canonical (one word per value), so on cells produced by `cell8FromBand`
     * the two coincide. They part company on the words the format keeps distinct
     * on purpose: `+0` (all-zero) versus `-0` (the sign bit alone) are unequal
     * here, which is the right answer for "is this buffer still the fill value"
     * and the wrong one for arithmetic. Compare decoded carriers, or
     * `BandedReal`s, when a numeric answer is wanted.
     */
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator==(
        BandCell8 a, BandCell8 b)
    {
        return a.w == b.w;
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator!=(
        BandCell8 a, BandCell8 b)
    {
        return a.w != b.w;
    }

    /**
     * @brief → `Band`. The stored word's three limbs, about six instructions.
     *
     * A thin spelling of `detail::bandFromCell8` (body supplied out-of-line at
     * the bottom of this file, once that function is complete). Exact on the
     * value: the word holds 56 significand bits and this redistributes them into
     * the format's canonical 24 + 23 + 9 split without rounding. The limb
     * boundaries are the format's, not those of whatever `Band` produced the
     * word.
     */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band band() const;
};

static_assert(sizeof(BandCell8) == 8,
    "BandCell8 must be exactly one 64-bit memory transaction — the whole point "
    "of the type is that it costs what a double costs");
static_assert(alignof(BandCell8) == 8,
    "BandCell8 must be naturally aligned for a single 64-bit access");
static_assert(std::is_trivially_copyable_v<BandCell8>,
    "BandCell8 must stay trivially copyable (SoA buffers, cudaMemcpy, bit-cast "
    "punning)");

namespace detail {

// =====================================================================
//  Format constants
// =====================================================================

/// @brief Significand bits the cell carries, leading bit INCLUDED. The leading
/// bit is IMPLICIT in the stored word exactly as IEEE-754 leaves it implicit:
/// 24 in the leading float (1 implicit + 23 stored) plus 32 in the continuation.
inline constexpr int kBandCell8Depth = 56;

/// @brief Significand bits carried by the leading limb — a `binary32`'s own.
inline constexpr int kBandCell8LeadingBits = 24;

/// @brief Continuation bits routed to the reloaded `Band`'s middle limb.
inline constexpr int kBandCell8LoBits = 23;

/// @brief Continuation bits routed to the reloaded `Band`'s residue limb.
inline constexpr int kBandCell8TailBits = 9;

/// @brief Smallest exponent the tier-1 codec is exact at.
inline constexpr int kBandCell8FloorExp = -94;

/// @brief Largest exponent the tier-1 codec is exact at.
inline constexpr int kBandCell8CeilingExp = 125;

/// @brief Default admission margin, in binades. Matches the carrier's own so a
/// consumer declaring one envelope satisfies both.
inline constexpr int kBandCell8AdmissionMargin = 8;

/// @brief Exponent bits the escape tier spends on its extended exponent.
inline constexpr int kBandCell8EscapeExtBits = 11;

/// @brief Significand bits the escape tier stores in the high word, below its
/// extended exponent.
inline constexpr int kBandCell8EscapeSigHiBits = 12;

/// @brief Escape-tier depth: 1 implicit + 12 + 32.
inline constexpr int kBandCell8EscapeDepth = 45;

/// @brief The exponent the `0xFF` escape tag counts UPWARD from.
inline constexpr int kBandCell8EscapeUpBase = kBandCell8CeilingExp + 1;

/// @brief The exponent the `0x00` escape tag counts DOWNWARD from.
inline constexpr int kBandCell8EscapeDownBase = kBandCell8FloorExp;

/// @brief Largest value the 11-bit extended exponent can hold.
inline constexpr int kBandCell8EscapeExtMax = 2047;

/// @brief Highest and lowest exponents the escape tier can name.
inline constexpr int kBandCell8EscapeMaxExp
    = kBandCell8EscapeUpBase + kBandCell8EscapeExtMax;
inline constexpr int kBandCell8EscapeMinExp
    = kBandCell8EscapeDownBase - kBandCell8EscapeExtMax;

/// @brief Lowest `arrayBias` at which no rebiased tier-1 word can be mistaken
/// for an escape tag.
inline constexpr int kBandCell8RebiasLowSafe = kBandCell8CeilingExp - 127;

/// @brief Highest such `arrayBias`.
inline constexpr int kBandCell8RebiasHighSafe = kBandCell8FloorExp + 126;

// =====================================================================
//  The reserved specials region
// =====================================================================

/// @brief The extended-exponent value the reserved code points sit at: the TOP
/// of the upward tag's range, out of reach of every finite `double`.
inline constexpr int kBandCell8SpecialExt = kBandCell8EscapeExtMax;

/// @brief The largest exponent a finite `double` can present to the escape tier.
inline constexpr int kBandCell8MaxIngestExp = 1024;

static_assert(
    kBandCell8SpecialExt > kBandCell8MaxIngestExp - kBandCell8EscapeUpBase,
    "the reserved special code points must be OUT of reach of every finite "
    "double: the upward tag's exponent is ext + EscapeUpBase, so an ext at or "
    "below MaxIngestExp - EscapeUpBase is a word an ordinary ingest can write");
static_assert(kBandCell8SpecialExt <= kBandCell8EscapeExtMax,
    "the reserved ext must still fit the escape tier's 11-bit field");

/// @brief The bits of the high word that identify the reserved region — sign
/// EXCLUDED, so both infinities share it.
inline constexpr std::uint32_t kBandCell8SpecialMask = 0x7FFFF000u;

/// @brief The value those bits take in the reserved region.
inline constexpr std::uint32_t kBandCell8SpecialTag = 0x7F800000u
    | ((static_cast<std::uint32_t>(kBandCell8SpecialExt) & 0x7FFu) << 12);

static_assert(kBandCell8SpecialTag == kBandCell8SpecialMask,
    "the reserved region is the TOP of the upward tag's exponent range, so the "
    "identifying bits and the mask coincide — if they stop coinciding, "
    "kBandCell8SpecialExt moved and cell8IsSpecial needs re-deriving");

/// @brief The three canonical special words. The NaN takes the top significand
/// bit inside the reserved region, so there is exactly ONE NaN code point.
inline constexpr std::uint64_t kBandCell8PosInfWord
    = static_cast<std::uint64_t>(kBandCell8SpecialTag) << 32;
inline constexpr std::uint64_t kBandCell8NegInfWord
    = static_cast<std::uint64_t>(kBandCell8SpecialTag | 0x80000000u) << 32;
inline constexpr std::uint64_t kBandCell8NanWord
    = static_cast<std::uint64_t>(kBandCell8SpecialTag | 0x800u) << 32;

/// @brief Does this word hold an infinity or a NaN? One mask and one compare.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool cell8IsSpecial(BandCell8 c)
{
    return (static_cast<std::uint32_t>(c.w >> 32) & kBandCell8SpecialMask)
        == kBandCell8SpecialTag;
}

/// @brief The special a reserved word denotes, as a canonical `Band`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromCell8Special(
    BandCell8 c)
{
    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32);
    const bool isNan
        = ((w1 & 0xFFFu) | static_cast<std::uint32_t>(c.w)) != 0u;
    const std::uint32_t bits
        = isNan ? kBandCanonicalNanBits : ((w1 & 0x80000000u) | kBandFp32ExpMask);
    return Band{ intAsFloat(static_cast<int>(bits)), 0.0f, 0.0f };
}

/// @brief The reserved word for the special an FP32 leading limb holds.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8SpecialFromFloatBits(
    std::uint32_t hiBits)
{
    const bool isNan = (hiBits & 0x7FFFFFu) != 0u;
    if (isNan)
        return BandCell8{ kBandCell8NanWord };
    return BandCell8{ static_cast<std::uint64_t>(
                          kBandCell8SpecialTag | (hiBits & 0x80000000u))
        << 32 };
}

/// @brief The reserved word for the special the HIGH IEEE word of a `double`
/// holds. EVERY IEEE NaN — signalling included — maps onto the one NaN code
/// point, which is why the limits specialization says `has_signaling_NaN` is
/// false: there is no second encoding to signal with.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8SpecialFromIEEE(
    std::uint32_t ieee_lo, std::uint32_t ieee_hi)
{
    const std::uint32_t mant_nz = (ieee_hi & 0x000FFFFFu) | ieee_lo;
    if (mant_nz != 0u)
        return BandCell8{ kBandCell8NanWord };
    return BandCell8{ static_cast<std::uint64_t>(
                          kBandCell8SpecialTag | (ieee_hi & 0x80000000u))
        << 32 };
}

// =====================================================================
//  Memory movement
// =====================================================================

/// @brief Read one cell from memory as a single 64-bit transaction.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 loadCell8(
    const BandCell8* p)
{
    return *p;
}

/// @brief Write one cell to memory as a single 64-bit transaction.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void storeCell8(BandCell8* p, BandCell8 c)
{
    *p = c;
}

// =====================================================================
//  Storability
// =====================================================================

/// @brief True when a `Band` is one the tier-1 codec is contracted to accept:
/// normalized, and inside the storage window where the encoder's own scaling
/// constants are still normal floats.
///
/// The `powerOfTwo && reducing` arm is the binade-borrow class: at an exact
/// power of two the lower limbs may only claim half an ulp before the encode has
/// to borrow into the leading limb's exponent, and the borrow is only defined
/// when they reduce the magnitude.
///
/// @section d24y The one-binade slack, and where it stops
/// `e == kBandCell8FloorExp - 1` is admitted as slack only when the row does
/// not itself take the binade-borrow branch. Derivation, read off
/// `cell8FromBand`'s own borrow handling: a `powerOfTwo && reducing` row
/// always borrows (the cancelling remainder rounds to a negative fixed-point
/// residue) and, being a power of two, that borrow always crosses — the
/// leading limb's exponent field steps down by exactly one from the row's own
/// `e`. `bandFromCell8` then splits the continuation at `se - (32 << 23)`,
/// which is only a valid (normal) exponent field when the stored exponent is
/// `>= kBandCell8FloorExp`: one field below that (`kBandCell8FloorExp - 1`,
/// the slack binade itself, reachable only by a row that started there and
/// crossed again) the field underflows past zero and wraps into the reserved
/// `0xFF` code point, and the decode reads `-Infinity - -Infinity` for the
/// tail limb — NaN, out of a `Band` this predicate had just certified
/// storable. So a crossing row needs its own `e` at `kBandCell8FloorExp` or
/// above, landing the one step the fixup takes at the slack binade, never past
/// it; a non-crossing row (not power-of-two, or not reducing) never moves the
/// stored exponent off its own `e`, so the existing `e >= kBandCell8FloorExp -
/// 1` admission is already correct for it.
///
/// The case that motivated this: `hi=2.5243549e-29f`
/// (`= 2^-95 = 2^(kBandCell8FloorExp-1)`, exact power of two),
/// `lo=-7.52316385e-37f` (a quarter-ulp cancelling remainder), which without
/// this check would report storable=true while decoding to NaN. @see
/// `BandCell8Cert.StorableImpliesFiniteDecode` (the enumerated corpus).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool cell8BandIsStorable(Band b)
{
    const std::uint32_t bh = static_cast<std::uint32_t>(floatAsInt(b.hi));
    const int ebyte        = static_cast<int>((bh >> 23) & 0xFFu);

    if (absf(b.hi) == 0.0f)
        return absf(b.lo) == 0.0f && absf(b.tail) == 0.0f;
    if (ebyte == 0 || ebyte == 0xFF)
        return false;

    const int e = ebyte - 127;
    if (e < kBandCell8FloorExp - 1 || e > kBandCell8CeilingExp)
        return false;

    const bool powerOfTwo = (bh & 0x7FFFFFu) == 0u;
    const float ulp       = intAsFloat((127 + e - 23) << 23);
    const float mag       = absf(b.lo) + absf(b.tail);

    const float rest    = b.lo + b.tail;
    const bool reducing = ((bh & 0x80000000u) != 0u) ? (rest > 0.0f) : (rest < 0.0f);

    if (!(powerOfTwo && reducing))
        return mag <= ulp;

    // The binade-borrow branch: this row will cross one binade down, so its
    // own `e` must not already be sitting on the one-binade slack -- crossing
    // from there lands one field below where the continuation split stays
    // valid (see the derivation above).
    if (e == kBandCell8FloorExp - 1)
        return false;
    return mag <= 0.5f * ulp;
}

// =====================================================================
//  The tier-1 codec
// =====================================================================

/// @brief `Band` -> `BandCell8`: pack three limbs into one word.
///
/// The two lower limbs are scaled by the RECIPROCAL of the leading limb's
/// exponent (an exact power of two assembled as an integer field, so the scaling
/// is exact) into `[-2^-23, 2^-23]`, then rounded to a 55-bit-scaled integer.
/// A NEGATIVE remainder borrows one from the leading limb's word — and at an
/// exact power of two that borrow crosses a binade, where the continuation's
/// worth doubles, which is what the `cross` shift corrects.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8FromBand(Band b)
{
    const std::uint32_t bh0 = static_cast<std::uint32_t>(floatAsInt(b.hi));

    if ((bh0 & kBandFp32ExpMask) == kBandFp32ExpMask)
        return cell8SpecialFromFloatBits(bh0);

#ifdef AETHER_DEBUG_MODE
    assert(cell8BandIsStorable(b)
        && "BandCell8: the Band is not normalized and inside the storage window "
           "-- either it did not come from a certified op, or its magnitude is "
           "outside [2^-94, 2^126) where the codec's own scaling constants stop "
           "being normal floats");
#endif
    const std::uint32_t bh  = bh0;
    const std::uint32_t sgn = bh & 0x80000000u;
    const std::uint32_t ex  = bh & kBandFp32ExpMask;

    const float sig = intAsFloat(static_cast<int>((0x7F000000u - ex) | sgn));
    const float scale55 = intAsFloat((127 + 55) << 23);

    const float ul = b.lo * sig;   // in [-2^-23, 2^-23]
    const float ut = b.tail * sig; // far below that

    std::int64_t rem = roundToInt64(ul * scale55)
        + static_cast<std::int64_t>(roundToInt32(ut * scale55));

    const std::uint32_t borrow = static_cast<std::uint32_t>(rem < 0);
    const std::uint32_t cross
        = borrow & static_cast<std::uint32_t>((bh & 0x7FFFFFu) == 0u);
    rem <<= cross; // the binade-borrow fixup

    const std::uint32_t w1 = bh - borrow;
    return BandCell8{ (static_cast<std::uint64_t>(w1) << 32)
        | static_cast<std::uint32_t>(rem) };
}

/// @brief `BandCell8` -> `Band`: unpack one word into three limbs.
///
/// The continuation is split at the FORMAT's boundary (23 + 9), not at whatever
/// boundary the producing `Band` happened to have. Each limb is materialised as
/// `(exponent field | payload) - (exponent field)`, which is exact: the
/// subtraction removes the implicit leading bit the assembly introduced.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromCell8(BandCell8 c)
{
    if (cell8IsSpecial(c))
        return bandFromCell8Special(c);

    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32);
    const std::uint32_t w0 = static_cast<std::uint32_t>(c.w);

    const std::uint32_t se = w1 & 0xFF800000u;
    const std::uint32_t eL = se - (23u << 23);
    const std::uint32_t eT = se - (32u << 23);

    const std::uint32_t L = w0 >> kBandCell8TailBits;
    const std::uint32_t T = w0 & ((1u << kBandCell8TailBits) - 1u);

    Band b;
    b.hi   = intAsFloat(static_cast<int>(w1));
    b.lo   = intAsFloat(static_cast<int>(eL | L)) - intAsFloat(static_cast<int>(eL));
    b.tail = intAsFloat(static_cast<int>(eT | T)) - intAsFloat(static_cast<int>(eT));
    return b;
}

/// @brief The FP32 tier: the leading limb, for a consumer that does not want the
/// other 32 bits. Free — it is a field read, not a decode.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float floatFromCell8(BandCell8 c)
{
    if (cell8IsSpecial(c))
        return bandFromCell8Special(c).hi;
    return intAsFloat(static_cast<int>(static_cast<std::uint32_t>(c.w >> 32)));
}

// =====================================================================
//  Per-array rebias
// =====================================================================

/// @brief Forward declaration — the rebias pair has to ask the escape/specials
/// tag space whether this bias keeps it unambiguous.
[[nodiscard]] AETHER_DEVICEHOST()
    AETHER_FORCEINLINE() constexpr bool bandCell8RebiasKeepsEscapeTagsClear(int arrayBias);

/// @brief `Band` -> `BandCell8` with the stored exponent LOWERED by `arrayBias`
/// — a pure integer shift of the finished word, so the encode is unchanged.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8FromBandRebias(
    Band b, std::int32_t arrayBias)
{
    const BandCell8 c = cell8FromBand(b);
    if (cell8IsSpecial(c)) {
#ifdef AETHER_DEBUG_MODE
        assert(bandCell8RebiasKeepsEscapeTagsClear(arrayBias)
            && "BandCell8: this array's arrayBias is outside the range where the "
               "tag space is unambiguous, so it is tier-1 ONLY and its reader "
               "cannot tell a special apart from a rebiased finite word");
#endif
        return c;
    }
    const std::uint64_t shift
        = static_cast<std::uint64_t>(static_cast<std::uint32_t>(arrayBias) << 23) << 32;
    return BandCell8{ c.w - shift };
}

/// @brief `BandCell8` -> `Band` with `arrayBias` added back to the stored
/// exponent. @see cell8FromBandRebias.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromCell8Rebias(
    BandCell8 c, std::int32_t arrayBias)
{
    if (bandCell8RebiasKeepsEscapeTagsClear(arrayBias) && cell8IsSpecial(c))
        return bandFromCell8Special(c);

    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32)
        + (static_cast<std::uint32_t>(arrayBias) << 23);
    return bandFromCell8(BandCell8{ (static_cast<std::uint64_t>(w1) << 32)
        | static_cast<std::uint32_t>(c.w) });
}

// =====================================================================
//  The escape tier
// =====================================================================

/// @brief Does this word carry an escape tag (leading-limb exponent field all
/// zeros or all ones)? One subtract and one unsigned compare.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool cell8IsEscape(BandCell8 c)
{
    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32);
    return ((w1 & kBandFp32ExpMask) - 0x00800000u) >= 0x7F000000u;
}

/// @brief Decode a word and report whether it was an escaped one, so a consumer
/// that only contracted for tier 1 can notice rather than silently believe the
/// tier-1 reading of an escape word.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromCell8Checked(
    BandCell8 c, bool& escaped)
{
    escaped = cell8IsEscape(c);
    return bandFromCell8(c);
}

/**
 * @brief The parts an escaped word decodes to: two re-centred limbs plus the
 *        exponent they share.
 *
 * A local type rather than the separate 16-byte `BandCell` (whose other
 * consumers are the coefficient-table family, `BandTable`/`BandPair`),
 * since nothing else here needs that type.
 */
struct EscapeParts {
    float hi;  ///< leading limb, re-centred into `[1,2)` (signed)
    float lo;  ///< second limb at the same centring (signed)
    int bias;  ///< the exponent both limbs share
};

/// @brief `BandCell8` -> `EscapeParts` for an escaped (or reserved) word.
///
/// An escaped magnitude is one three FP32 limbs cannot represent — that is WHY
/// it escaped — so the decode hands back limbs plus a separate exponent and lets
/// the caller fold the exponent on where there is room for it.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() EscapeParts escapePartsFromCell8(
    BandCell8 c)
{
    if (cell8IsSpecial(c)) {
        EscapeParts s;
        s.hi   = bandFromCell8Special(c).hi;
        s.lo   = 0.0f;
        s.bias = 0;
        return s;
    }

    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32);
    const std::uint32_t w0 = static_cast<std::uint32_t>(c.w);

    const std::uint32_t tag    = (w1 >> 23) & 0xFFu;
    const std::int32_t extExp  = static_cast<std::int32_t>((w1 >> 12) & 0x7FFu);
    const std::uint32_t sigHi  = w1 & 0xFFFu;

    EscapeParts r;
    r.bias = (tag == 0xFFu) ? (extExp + kBandCell8EscapeUpBase)
                            : (-extExp + kBandCell8EscapeDownBase);

    const std::uint32_t m0 = 0x800000u | (sigHi << 11) | (w0 >> 21); // 24 bits
    const std::uint32_t m1 = (w0 & 0x1FFFFFu) << 2;                  // 21 bits
    r.hi = intAsFloat(static_cast<int>((127u << 23) | (m0 & 0x7FFFFFu)));
    r.lo = intAsFloat(static_cast<int>(((127u - 23u) << 23) | (m1 & 0x7FFFFFu)))
        - intAsFloat(static_cast<int>((127u - 23u) << 23));

    if ((w1 | w0) == 0u) { // canonical zero
        r.hi   = 0.0f;
        r.lo   = 0.0f;
        r.bias = 0;
    }
    if ((w1 & 0x80000000u) != 0u) {
        r.hi = -r.hi;
        r.lo = -r.lo;
    }
    return r;
}

/// @brief IEEE double halves -> an ESCAPED `BandCell8` at depth 45: the 52-bit
/// significand rounded to 44 stored bits, round-to-nearest ties-to-even, with a
/// carry out of the rounding bumping the exponent.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8EscapedFromIEEE(
    std::uint32_t ieee_lo, std::uint32_t ieee_hi)
{
    const std::uint32_t sign = ieee_hi & 0x80000000u;
    const int biasedExp      = static_cast<int>((ieee_hi >> 20) & 0x7FFu);

#ifdef AETHER_DEBUG_MODE
    assert(biasedExp != 0x7FF
        && "BandCell8: infinities and NaNs have no encoding in this tier");
#endif

    if (biasedExp == 0) // zero and subnormal
        return BandCell8{ 0 };

    const std::uint64_t m52
        = (static_cast<std::uint64_t>(ieee_hi & 0x000FFFFFu) << 32) | ieee_lo;

    std::uint64_t m44        = m52 >> 8;
    const std::uint64_t drop = m52 & 0xFFu;
    if (drop > 0x80u || (drop == 0x80u && (m44 & 1u) != 0u))
        m44++;
    int e = biasedExp - 1023;
    if (m44 == (std::uint64_t{ 1 } << 44)) { // the rounding carried out
        m44 = 0;
        e++;
    }

    const bool up = e >= kBandCell8EscapeUpBase;
    const std::int32_t ext
        = up ? (e - kBandCell8EscapeUpBase) : (kBandCell8EscapeDownBase - e);

#ifdef AETHER_DEBUG_MODE
    assert(ext >= 0 && ext <= kBandCell8EscapeExtMax
        && "BandCell8: this exponent is not in the escape tier's domain -- it "
           "belongs to tier 1, which holds it at 56 bits rather than 45");
    assert(!(sign == 0u && ext == 0 && !up && m44 == 0)
        && "BandCell8: +2^-94 with an all-zero significand is the code point "
           "canonical zero has taken; tier 1 stores that magnitude exactly");
#endif

    const std::uint32_t tag = up ? 0x7F800000u : 0x00000000u;
    const std::uint32_t w1  = sign | tag
        | ((static_cast<std::uint32_t>(ext) & 0x7FFu) << 12)
        | static_cast<std::uint32_t>(m44 >> 32);
    return BandCell8{ (static_cast<std::uint64_t>(w1) << 32)
        | static_cast<std::uint32_t>(m44) };
}

/// @brief Fold `-0` onto the canonical `+0` word. Only escape-enabled arrays
/// need it: the escape tier's own zero code point is the all-zero word, so a
/// signed zero arriving through it would collide with a legal escaped value.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8CanonicalizeZero(
    BandCell8 c)
{
    return BandCell8{ ((c.w << 1) == 0u) ? std::uint64_t{ 0 } : c.w };
}

/// @brief IEEE double halves -> `BandCell8`, the DEVICE-CALLABLE encode.
///
/// The `double` split itself is host-only (device code has no `double` to be
/// handed), so this takes the two halves; `cell8FromDouble` below is the host
/// FACE of this one body, not a sibling implementation of it.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell8 cell8FromIEEE(
    std::uint32_t ieee_lo, std::uint32_t ieee_hi, std::int32_t arrayBias = 0,
    bool useEscape = false)
{
    const int bexp = static_cast<int>((ieee_hi >> 20) & 0x7FFu);

    if (bexp == 0x7FF)
        return cell8SpecialFromIEEE(ieee_lo, ieee_hi);

    if (bexp == 0) // zero and every double subnormal
        return BandCell8{ 0 };

    const int e      = bexp - 1023;
    const bool tier1 = (e >= kBandCell8FloorExp && e <= kBandCell8CeilingExp);

    if (!tier1 && useEscape) {
        // A significand that CARRIES at the escape tier's rounding can land back
        // inside tier 1, one binade up. Re-route it there rather than escaping a
        // value tier 1 holds exactly.
        const std::uint64_t m52
            = (static_cast<std::uint64_t>(ieee_hi & 0x000FFFFFu) << 32) | ieee_lo;
        std::uint64_t m44        = m52 >> 8;
        const std::uint64_t drop = m52 & 0xFFu;
        if (drop > 0x80u || (drop == 0x80u && (m44 & 1u) != 0u))
            m44++;
        if (m44 == (std::uint64_t{ 1 } << 44)) {
            const int eAfter = e + 1;
            if (eAfter >= kBandCell8FloorExp && eAfter <= kBandCell8CeilingExp) {
                const std::uint32_t rhi = (ieee_hi & 0x80000000u)
                    | (static_cast<std::uint32_t>(eAfter + 1023) << 20);
                return cell8CanonicalizeZero(
                    cell8FromBandRebias(bandFromIEEE(0u, rhi), arrayBias));
            }
        }
    }

    if (tier1) {
        const BandCell8 c
            = cell8FromBandRebias(bandFromIEEE(ieee_lo, ieee_hi), arrayBias);
        return useEscape ? cell8CanonicalizeZero(c) : c;
    }

#ifdef AETHER_DEBUG_MODE
    assert(useEscape
        && "BandCell8 ingest: this value is outside tier 1's exact window "
           "[2^-94, 2^126) and this array did not opt into the escape tier");
#endif
    return cell8EscapedFromIEEE(ieee_lo, ieee_hi);
}

// =====================================================================
//  Admission predicates
// =====================================================================

/// @brief Does a consumer's declared largest magnitude fit under the codec's
/// tier-1 ceiling, with margin?
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandCell8CeilingAdmits(
    int maxAbsExp, [[maybe_unused]] int arrayBias = 0,
    int margin = kBandCell8AdmissionMargin)
{
    return maxAbsExp + margin <= kBandCell8CeilingExp;
}

/// @brief …and above the floor?
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandCell8FloorAdmits(
    int minAbsExp, [[maybe_unused]] int arrayBias = 0,
    int margin = kBandCell8AdmissionMargin)
{
    return minAbsExp - margin >= kBandCell8FloorExp;
}

/// @brief Both ends at once — the predicate a consumer declares its envelope
/// with, and the one `numeric_limits<BandedReal>` derives its range from.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandCell8Admits(
    int maxAbsExp, int minAbsExp, int arrayBias = 0,
    int margin = kBandCell8AdmissionMargin)
{
    return bandCell8CeilingAdmits(maxAbsExp, arrayBias, margin)
        && bandCell8FloorAdmits(minAbsExp, arrayBias, margin);
}

/// @brief Is the tag space unambiguous at this `arrayBias`? Outside this range a
/// rebiased tier-1 word can present an escape tag, so such an array is tier-1
/// ONLY.
[[nodiscard]] AETHER_DEVICEHOST()
    AETHER_FORCEINLINE() constexpr bool bandCell8RebiasKeepsEscapeTagsClear(int arrayBias)
{
    return arrayBias >= kBandCell8RebiasLowSafe
        && arrayBias <= kBandCell8RebiasHighSafe;
}

// =====================================================================
//  Host-only `double` terminals
// =====================================================================
//
// `double` must not appear in device code (the whole 0-FP64 portability
// warrant), so these are plain `inline` HOST functions and say so.

/// @brief The two 32-bit halves of a `double`'s bit pattern. Host only.
inline void cell8IeeeWords(double x, std::uint32_t& lo, std::uint32_t& hi)
{
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    lo = static_cast<std::uint32_t>(u);
    hi = static_cast<std::uint32_t>(u >> 32);
}

/// @brief …and back. Host only.
[[nodiscard]] inline double cell8DoubleFromWords(std::uint32_t lo, std::uint32_t hi)
{
    const std::uint64_t u = (static_cast<std::uint64_t>(hi) << 32) | lo;
    double x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

/// @brief The unbiased base-2 exponent of a finite nonzero `double`; `0` for a
/// zero or a subnormal (both of which take the zero path anyway). Host only.
[[nodiscard]] inline int cell8ExponentOf(double x)
{
    std::uint32_t lo = 0, hi = 0;
    cell8IeeeWords(x, lo, hi);
    const int bexp = static_cast<int>((hi >> 20) & 0x7FFu);
    return (bexp == 0) ? 0 : bexp - 1023;
}

/// @brief Is this value one TIER 1 holds exactly? Host only.
///
/// The window test is on the value's OWN exponent, not the rebiased one, and
/// that is not a slip: the encoder builds the reciprocal `2^-e` of the leading
/// limb, which has to be a normal float, so the window is a property of the
/// value the CHAIN holds. The bias moves where the finished word sits, not what
/// the encoder can do.
[[nodiscard]] inline bool cell8Tier1Holds(double x)
{
    if (x == 0.0)
        return true;
    const int e = cell8ExponentOf(x);
    return e >= kBandCell8FloorExp && e <= kBandCell8CeilingExp;
}

/**
 * @brief `double` -> `BandCell8`. The ingest terminal, one element. Host only.
 *
 * The encode itself is `cell8FromIEEE` above, because device code needs it too
 * and cannot be handed a `double`. What is here is what the device cannot do:
 * the `double` split and the fail-loud window check. There is no second copy of
 * the encode to drift.
 *
 * @throws aether::Error when the value is finite and outside tier 1's
 *         exact window on an array that did not opt into the escape tier — the
 *         only way to reach it is to have declared an envelope the data then
 *         left. Infinities and NaNs do not throw: the format reserves code
 *         points for them, and refusing to store the state a diverged sample
 *         actually reached is not a service to the caller.
 */
[[nodiscard]] inline BandCell8 cell8FromDouble(
    double x, std::int32_t arrayBias = 0, bool useEscape = false)
{
    std::uint32_t lo = 0, hi = 0;
    cell8IeeeWords(x, lo, hi);
    const int bexp = static_cast<int>((hi >> 20) & 0x7FFu);

    if (bexp != 0 && bexp != 0x7FF && !cell8Tier1Holds(x) && !useEscape) {
        err::fail("BandCell8 ingest", "host", sizeof(BandCell8),
            "value " + std::to_string(x)
                + " is outside tier 1's exact window [2^-94, 2^126) and this "
                  "array did not opt into the escape tier: declare the "
                  "envelope with bandCell8Admits and keep the data inside it, "
                  "choose an arrayBias that brings it inside, or enable the "
                  "escape tier and accept 45 bits on the outliers");
    }

    return cell8FromIEEE(lo, hi, arrayBias, useEscape);
}

/**
 * @brief `BandCell8` -> `double`. The egress terminal, one element. Host only.
 *
 * The exact inverse of `cell8FromDouble` at the same bias and escape setting on
 * every value tier 1 holds; on an escaped value it returns the 45-bit value that
 * was stored.
 *
 * The tier-1 branch sums the limbs directly rather than composing a chain
 * store terminal, and that is exact rather than merely better: each limb is a
 * `float`, hence exact as a `double`; `hi + lo` needs at most 47 significand
 * bits and is therefore exact; adding `tail` brings it to at most 56, which
 * rounds once, to nearest, and does not round at all for any value that arrived
 * as a `double`.
 *
 * The escaped branch does not go through the limb sum, and must not: an
 * escaped magnitude is one three FP32 limbs cannot represent, so the exponent is
 * folded on here, in `double` arithmetic, where there is room for it. The two
 * escaped limbs are disjoint (24 + 21 bits), so their sum is exact and the
 * scaling by an exact power of two is too.
 */
[[nodiscard]] inline double doubleFromCell8(
    BandCell8 c, std::int32_t arrayBias = 0, bool useEscape = false)
{
    if (useEscape && cell8IsEscape(c)) {
        const EscapeParts p = escapePartsFromCell8(c);
        // The two limbs are re-centred into `[1,2)` with the magnitude in
        // `bias`; both share it, so ONE scaling of their sum is the value. The
        // sum is exact (24 + 21 disjoint bits) and `std::ldexp` is exact.
        // HOST-only code, so `<cmath>` here costs the device arm nothing.
        const double m = static_cast<double>(p.hi) + static_cast<double>(p.lo);
        return std::ldexp(m, p.bias);
    }
    const Band b = bandFromCell8Rebias(c, arrayBias);
    return (static_cast<double>(b.hi) + static_cast<double>(b.lo))
        + static_cast<double>(b.tail);
}

/// @brief `Band` -> `double` at bias 0, the exact limb sum. Host only.
///
/// Provided so a TEST or a host diagnostic can read a carrier without going
/// through the codec. Exact for the same reason `doubleFromCell8`'s tier-1
/// branch is; a carrier whose magnitude is outside `double`'s range is outside
/// the carrier's declared envelope anyway.
[[nodiscard]] inline double bandToDouble(Band b)
{
    return (static_cast<double>(b.hi) + static_cast<double>(b.lo))
        + static_cast<double>(b.tail);
}

} // namespace detail

// Defined here, after `detail::bandFromCell8` is complete. An ordinary member
// function (not a friend), so an out-of-line definition carries none of the
// "unhiding" hazard the hidden-friend operators in Band.h are written to avoid.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band BandCell8::band() const
{
    return detail::bandFromCell8(*this);
}

} // namespace banded
} // namespace aether
