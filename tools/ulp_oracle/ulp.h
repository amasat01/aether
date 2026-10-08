// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/ulp_oracle/ulp.h — the ULP checker the Band math test suites include.
//
// Header-only, host+device-agnostic C++ (plain bit tricks over <cstdint>/
// <cstring>/<cmath>; no MPFR, no aether headers, no exceptions, no dynamic
// allocation). Deliberately dependency-free: this is the ONE artifact the
// golden headers minted by ulp_oracle.cpp and the Band math test suites
// both need, and neither of those should have to pull MPFR along for the
// ride.
//
// The definition of "ULP distance" used here is the standard MONOTONIC
// ORDERED-INTEGER one (Bruce Dawson's "Comparing Floating Point Numbers"
// trick): map every double bit pattern to a signed integer key that
// increases exactly when the real value increases, across the zero crossing
// and across every subnormal/normal binade boundary, then take the plain
// integer difference of the two keys. That gives, for FREE and without
// special-casing:
//   * signed zeros:      key(-0.0) == key(+0.0)              -> distance 0
//   * subnormal/normal:  the boundary is just another integer step
//   * binade crossings:  ditto — no per-exponent renormalization needed
// NaN and infinity are NOT points on that grid (there is no well-ordered
// notion of "how many doubles between 3.0 and NaN"), so they are handled
// by CLASS EQUALITY instead: two NaNs of any payload/sign compare equal
// (distance 0); two same-signed infinities compare equal (distance 0); any
// other pairing that involves a NaN, or opposite-signed/mixed infinities,
// reports the sentinel kMismatch rather than a numeric distance.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>

namespace aether_tools {
namespace ulp {

/// @brief IEEE-754 double exponent class, in the same 0..4 encoding the
///        minted golden headers store in each row's `refClass` field —
///        see tools/ulp_oracle/README.md.
enum class Class : int {
    kZero      = 0,
    kSubnormal = 1,
    kNormal    = 2,
    kInf       = 3,
    kNaN       = 4,
};

/// @brief Classify a double's raw bit pattern. Pure bit inspection — no
///        libm calls, so it is safe on both the host and device arms.
inline Class classifyBits(std::uint64_t bits)
{
    const std::uint64_t exp  = (bits >> 52) & 0x7FFULL;
    const std::uint64_t frac = bits & 0xFFFFFFFFFFFFFULL;
    if (exp == 0x7FFULL) return frac ? Class::kNaN : Class::kInf;
    if (exp == 0ULL) return frac ? Class::kSubnormal : Class::kZero;
    return Class::kNormal;
}

/// @brief Classify a double directly.
inline Class classify(double v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return classifyBits(bits);
}

namespace detail {

// Monotonic ordered-integer key for a double bit pattern. -0.0 is
// canonicalized to +0.0 FIRST so the two map to the identical key (the
// "signed zeros" requirement above) — everything downstream just sees one
// zero. For a genuine negative value the key is the NEGATED unsigned
// magnitude field, which is why the ordering stays monotonic across the
// zero crossing (key(-tiny) = -1, key(+0) = 0, key(+tiny) = +1, ...).
inline std::int64_t orderKey(std::uint64_t bits)
{
    if (bits == 0x8000000000000000ULL) bits = 0ULL; // canonicalize -0.0
    const bool neg              = (bits >> 63) != 0ULL;
    const std::uint64_t mag     = bits & 0x7FFFFFFFFFFFFFFFULL;
    return neg ? -static_cast<std::int64_t>(mag) : static_cast<std::int64_t>(mag);
}

// (b - a) saturated to the int64_t range, WITHOUT ever forming an
// intermediate that overflows int64_t (no reliance on __int128 or wraparound
// UB, so this stays legal on a device compiler too). orderKey() values are
// each individually bounded to +-0x7FF0000000000000, which is inside
// [INT64_MIN, INT64_MAX] with room to spare for either operand alone, but
// NOT for their difference when they sit at opposite extremes (e.g. -DBL_MAX
// vs +DBL_MAX) — that is the one case this function exists to saturate
// rather than silently wrap.
inline std::int64_t saturatingDiff(std::int64_t a, std::int64_t b)
{
    constexpr std::int64_t kMax = (std::numeric_limits<std::int64_t>::max)();
    constexpr std::int64_t kMin = (std::numeric_limits<std::int64_t>::min)();
    if (a < 0 && b >= 0) {
        // b - a is the large-positive direction; overflow iff b > kMax + a
        // (kMax + a cannot itself overflow: a < 0 so kMax + a <= kMax - 1).
        if (b > kMax + a) return kMax;
        return b - a;
    }
    if (a >= 0 && b < 0) {
        // b - a is the large-negative direction; underflow iff b < kMin + a
        // (kMin + a cannot itself overflow: a >= 0 so kMin + a >= kMin).
        if (b < kMin + a) return kMin;
        return b - a;
    }
    // Same sign: magnitudes only shrink under subtraction, no overflow risk.
    return b - a;
}

} // namespace detail

/// @brief Sentinel returned for a class mismatch (NaN vs non-NaN, or
///        differently-signed/mixed infinities) — not a real distance, an
///        out-of-band "these are not comparable on the ULP grid" flag.
///        Always strictly larger than any distance a finite pair can produce.
inline constexpr std::int64_t kMismatch = (std::numeric_limits<std::int64_t>::max)();

/// @brief Signed ULP distance from `ref` (the oracle value) to `got` (the
///        candidate). Positive means `got` rounds ABOVE `ref` on the ordered
///        grid, negative means below; the magnitude is the number of
///        representable doubles strictly between them (0 for bit-identical
///        or numerically-equal signed zeros).
///
/// Correct across binade boundaries, signed zeros and subnormals by
/// construction (see orderKey() above). NaN and infinity are class-equality:
///   * both NaN                              -> 0
///   * both infinite, same sign              -> 0
///   * exactly one of {ref, got} is NaN       -> kMismatch
///   * exactly one of {ref, got} is infinite, or both infinite with
///     different sign                        -> kMismatch
inline std::int64_t ulpDistance(double ref, double got)
{
    std::uint64_t rb = 0, gb = 0;
    std::memcpy(&rb, &ref, sizeof(rb));
    std::memcpy(&gb, &got, sizeof(gb));

    const Class rc = classifyBits(rb);
    const Class gc = classifyBits(gb);

    if (rc == Class::kNaN || gc == Class::kNaN) {
        return (rc == Class::kNaN && gc == Class::kNaN) ? 0 : kMismatch;
    }
    if (rc == Class::kInf || gc == Class::kInf) {
        const bool bothInf    = (rc == Class::kInf && gc == Class::kInf);
        const bool sameSign   = std::signbit(ref) == std::signbit(got);
        return (bothInf && sameSign) ? 0 : kMismatch;
    }
    return detail::saturatingDiff(detail::orderKey(rb), detail::orderKey(gb));
}

/// @brief |ulpDistance|, saturating at kMismatch rather than overflowing on
///        the one input for which negation would (ulpDistance(...) == the
///        int64_t minimum only when both are class-comparable and
///        maximally far apart, which given orderKey's bounds never actually
///        reaches INT64_MIN — this guard exists purely so callers never have
///        to reason about std::abs(INT64_MIN) UB).
inline std::int64_t ulpDistanceAbs(double ref, double got)
{
    const std::int64_t d = ulpDistance(ref, got);
    if (d == kMismatch) return kMismatch;
    if (d == (std::numeric_limits<std::int64_t>::min)()) return kMismatch;
    return d < 0 ? -d : d;
}

} // namespace ulp
} // namespace aether_tools
