// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandedLimits.h
 * @brief `std::numeric_limits` specialization for `aether::banded::BandedReal`.
 *
 * @section contract_bandedlimits Design
 * Every limit is a `static constexpr` member, so the `AETHER_DEVICEHOST()`
 * accessor is a literal read: no host-constexpr call appears in a device
 * body, and `--expt-relaxed-constexpr` is never required. `is_iec559` is
 * `false`: the representation is well defined, but the banded ops are
 * certified to <= 1 ULP rather than correctly rounded, and the range is
 * FP32's rather than IEEE double's.
 *
 * @section words Constant assembly
 * The encode chain is not constexpr — the bit reinterprets go through
 * `std::memcpy` and the roundings through `__builtin_lrintf`/`llrintf` —
 * so `max() = cell8FromDouble(...)` cannot appear in a constant
 * expression. Every word below is instead assembled by `bandedRealWord()`
 * from the `kBandCell8*` names, with the same derivation restated in
 * `tests/test_BandedReal_common.h` so the two are checked to agree.
 *
 * @section envelope The range members describe the admissible envelope
 * The codec's tier-1 window is `[2^-94, 2^126)`, but that window is not
 * an admissible declaration: `bandCell8Admits` applies an 8-binade margin
 * at each end, so a consumer declaring `maxAbsExp = 125` is refused.
 * Returning a value from `max()` that the ingest guard would itself
 * reject would be a trap, so `max()` reports the margin-respecting
 * admissible top rather than the codec's raw top.
 *
 * `min()`, `min_exponent`, `max_exponent` and the two decimal exponents
 * follow `max()` to the same envelope, so a bound derived from one member
 * and compared against another stays consistent. The codec's raw window
 * keeps its own names below (`kBandedRealCodec*`) and its own test row.
 */

#include <cstdint>
#include <limits>

#include "aether/banded/BandCell8.h"
#include "aether/banded/BandedReal.h"
#include "aether/macros.h"

namespace aether {
namespace banded {

// =====================================================================
//  Word assembly -- the one derivation every constant below goes through
// =====================================================================

/**
 * @brief The `BandCell8` word for `+/- (1 + mant23/2^23) * 2^e + cont * 2^(e-55)`.
 *
 * `BandCell8`'s own layout written as a constant expression: the high
 * half is an IEEE `binary32` and the low half is an unsigned 32-bit
 * continuation worth `2^(e-55)` per unit.
 *
 * @param e       the leading limb's unbiased base-2 exponent
 * @param mant23  the leading limb's 23 stored significand bits
 * @param cont    the 32-bit continuation
 * @param neg     the sign (the leading limb carries it for the whole value)
 */
[[nodiscard]] AETHER_DEVICEHOST() constexpr std::uint64_t bandedRealWord(
    int e, std::uint32_t mant23, std::uint32_t cont, bool neg)
{
    const std::uint32_t w1 = (neg ? 0x80000000u : 0u)
        | (static_cast<std::uint32_t>(e + 127) << 23) | (mant23 & 0x7FFFFFu);
    return (static_cast<std::uint64_t>(w1) << 32) | static_cast<std::uint64_t>(cont);
}

// =====================================================================
//  The envelope, derived
// =====================================================================

/// @brief Significand bits the carrier holds, leading bit included.
inline constexpr int kBandedRealDigits = detail::kBandCell8Depth; // 56

/// @brief The codec's raw tier-1 window, kept named so nothing is hidden by
/// the margin-respecting envelope below. Not an admissible declaration.
inline constexpr int kBandedRealCodecFloorExp = detail::kBandCell8FloorExp; // -94
inline constexpr int kBandedRealCodecCeilingExp = detail::kBandCell8CeilingExp; // 125

/// @brief The admissible envelope: the raw window pulled in by
/// `kBandCell8AdmissionMargin` at each end, exactly the region
/// `bandCell8Admits` accepts at its default margin.
inline constexpr int kBandedRealMinExp
    = kBandedRealCodecFloorExp + detail::kBandCell8AdmissionMargin; // -86
inline constexpr int kBandedRealMaxExp
    = kBandedRealCodecCeilingExp - detail::kBandCell8AdmissionMargin; // 117

/* The envelope matches what the admission guard accepts: the endpoints are
 * admitted, and one binade outside either of them is refused. If
 * `kBandCell8AdmissionMargin` ever moves, these four lines move with it or
 * the build stops. */
static_assert(detail::bandCell8Admits(kBandedRealMaxExp, kBandedRealMinExp),
    "the declared BandedReal envelope must be admissible under the SHIPPED "
    "bandCell8Admits at its default margin");
static_assert(!detail::bandCell8Admits(kBandedRealMaxExp + 1, kBandedRealMinExp),
    "one binade above the declared top must be REFUSED, or the top is not the "
    "top and this envelope claims nothing");
static_assert(!detail::bandCell8Admits(kBandedRealMaxExp, kBandedRealMinExp - 1),
    "one binade below the declared floor must be REFUSED");
static_assert(!detail::bandCell8Admits(
                  kBandedRealCodecCeilingExp, kBandedRealCodecFloorExp),
    "★ the CODEC's own tier-1 window is NOT admissible at the default margin -- "
    "that is this file's whole subject, and if it ever became admissible the "
    "margin-respecting envelope above would be pointless");

/// @brief `max()`'s word: the largest magnitude inside the admissible
/// envelope — the largest `binary32` at `kBandedRealMaxExp` plus a full
/// continuation.
///
/// The continuation is `0xFFFFFFFF` because the decode reads it as an
/// unsigned field; the signedness in the encoder's remainder is a borrow,
/// not a property of the stored field, so the largest stored word is the
/// all-ones one.
inline constexpr std::uint64_t kBandedRealMaxWord
    = bandedRealWord(kBandedRealMaxExp, 0x7FFFFFu, 0xFFFFFFFFu, false);
inline constexpr std::uint64_t kBandedRealLowestWord
    = bandedRealWord(kBandedRealMaxExp, 0x7FFFFFu, 0xFFFFFFFFu, true);

/// @brief `min()`'s word: the smallest positive magnitude inside the envelope.
inline constexpr std::uint64_t kBandedRealMinWord
    = bandedRealWord(kBandedRealMinExp, 0u, 0u, false);

/// @brief `epsilon()`'s word: `2^(1 - digits)`, derived from the codec
/// depth and nothing else.
inline constexpr std::uint64_t kBandedRealEpsilonWord
    = bandedRealWord(1 - kBandedRealDigits, 0u, 0u, false);

/// @brief `round_error()`'s word: one half.
inline constexpr std::uint64_t kBandedRealRoundErrorWord
    = bandedRealWord(-1, 0u, 0u, false);

/// @brief The codec top, named but never returned by `max()`, kept so the
/// distinction drawn above has a spelling and a test row.
inline constexpr std::uint64_t kBandedRealCodecTopWord
    = bandedRealWord(kBandedRealCodecCeilingExp, 0x7FFFFFu, 0xFFFFFFFFu, false);

/* Decimal exponent members, computed in integer arithmetic only: 643/2136
 * is the rational approximation of log10(2) that <float.h> uses for
 * FLT_DIG; keeping it integer avoids an FP64 constant expression in a
 * device TU. */
inline constexpr int kBandedRealLog10Num = 643;
inline constexpr int kBandedRealLog10Den = 2136;

} // namespace banded
} // namespace aether

/// \cond AETHER_DOXYGEN_STD_SPECIALIZATION_SCOPE_WORKAROUND
template<>
struct std::numeric_limits<aether::banded::BandedReal> {
private:
    using BR = aether::banded::BandedReal;

    static constexpr BR kMin_        = BR::fromBits(aether::banded::kBandedRealMinWord);
    static constexpr BR kMax_        = BR::fromBits(aether::banded::kBandedRealMaxWord);
    static constexpr BR kLowest_     = BR::fromBits(aether::banded::kBandedRealLowestWord);
    static constexpr BR kEpsilon_    = BR::fromBits(aether::banded::kBandedRealEpsilonWord);
    static constexpr BR kRoundError_ = BR::fromBits(aether::banded::kBandedRealRoundErrorWord);
    static constexpr BR kInfinity_
        = BR::fromBits(aether::banded::detail::kBandCell8PosInfWord);
    static constexpr BR kNegInfinity_
        = BR::fromBits(aether::banded::detail::kBandCell8NegInfWord);
    static constexpr BR kQuietNaN_ = BR::fromBits(aether::banded::detail::kBandCell8NanWord);

public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed      = true;
    static constexpr bool is_integer     = false;
    static constexpr bool is_exact       = false;

    static constexpr bool has_infinity  = true;
    static constexpr bool has_quiet_NaN = true;
    /* False: the format reserves exactly one NaN code point.
     * `cell8SpecialFromIEEE` maps every IEEE NaN, signalling included,
     * onto `kBandCell8NanWord`, so there is no second encoding to signal
     * with. */
    static constexpr bool has_signaling_NaN = false;

    /* Every `double` subnormal encodes to canonical zero, and the flush is
     * sign-destroying: a negative subnormal comes back `+0`, not `-0`.
     * Checked by
     * `BandedRealCert.SubnormalDoublesFlushToCanonicalZeroSignDestroying`. */
    static constexpr std::float_denorm_style has_denorm = std::denorm_absent;
    static constexpr bool has_denorm_loss                = false;

    /* Representation is well defined but the arithmetic is not IEC 559:
     * the banded ops are certified to <= 1 ULP rather than correctly
     * rounded, and the range is FP32's rather than IEEE double's. */
    static constexpr bool is_iec559 = false;

    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo  = false;

    static constexpr int digits = aether::banded::kBandedRealDigits; // 56
    static constexpr int radix  = 2;
    /* digits10 = floor((digits - 1) * log10 2); max_digits10 =
     * ceil(digits * log10 2) + 1. Integer arithmetic throughout. */
    static constexpr int digits10
        = (digits - 1) * aether::banded::kBandedRealLog10Num / aether::banded::kBandedRealLog10Den; // 16
    static constexpr int max_digits10
        = (digits * aether::banded::kBandedRealLog10Num + (aether::banded::kBandedRealLog10Den - 1))
            / aether::banded::kBandedRealLog10Den
        + 1; // 18

    /* The admissible envelope, not the codec window (see the file
     * header). `min_exponent` is the standard's "one more than the
     * smallest e with radix^(e-1) normalized"; `max_exponent` is its
     * mirror. */
    static constexpr int min_exponent = aether::banded::kBandedRealMinExp + 1; // -85
    static constexpr int max_exponent = aether::banded::kBandedRealMaxExp + 1; // 118
    static constexpr int min_exponent10
        = -((-min_exponent * aether::banded::kBandedRealLog10Num) / aether::banded::kBandedRealLog10Den); // -25
    static constexpr int max_exponent10
        = max_exponent * aether::banded::kBandedRealLog10Num / aether::banded::kBandedRealLog10Den; // 35

    static constexpr bool traps           = false;
    static constexpr bool tinyness_before = false;
    /* Tier-1 ingest is exact (nothing is rounded), the escape tier rounds
     * to nearest with ties to even, and the chain arithmetic is certified
     * to <= 1 ULP rather than either. `round_to_nearest` is the single
     * answer that fits the tier a consumer of this type actually lives
     * in. */
    static constexpr std::float_round_style round_style = std::round_to_nearest;

    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR min() noexcept { return kMin_; }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR max() noexcept { return kMax_; }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR lowest() noexcept
    {
        return kLowest_;
    }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR epsilon() noexcept
    {
        return kEpsilon_;
    }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR round_error() noexcept
    {
        return kRoundError_;
    }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR infinity() noexcept
    {
        return kInfinity_;
    }
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR quiet_NaN() noexcept
    {
        return kQuietNaN_;
    }
    /* `has_signaling_NaN` is false, so this returns the canonical quiet
     * NaN: there is no other NaN encoding in the format. Provided rather
     * than omitted so generic code that names it still compiles. */
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR signaling_NaN() noexcept
    {
        return kQuietNaN_;
    }
    /* denorm_absent: the smallest positive value is `min()`. */
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR denorm_min() noexcept
    {
        return kMin_;
    }
    /* Not a standard member: `-infinity()`, provided because the format
     * has a distinct reserved code point for it and a consumer would
     * otherwise have to negate through the carrier to reach it. */
    [[nodiscard]] static AETHER_DEVICEHOST() constexpr BR neg_infinity() noexcept
    {
        return kNegInfinity_;
    }
};
/// \endcond
