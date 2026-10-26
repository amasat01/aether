// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Backend.h
 * @brief `aether::random::detail` — the three counter-based core primitives
 *        every distribution leaf is built on: `randomBits64`, `uniform01`,
 *        `standardNormal`.
 *
 * No host/device split: aether runs the same `detail::PhiloxStream` code on
 * both arms (`detail/Philox.h`), so a seed names one stream everywhere. The
 * invariant is unconditional cross-mode bit-identity of the bits and
 * uniform streams.
 *
 * Positioning: every `(seed, global, counter)` owns ONE Philox block,
 * `philox4x32_10({lo(counter), hi(counter), lo(global), hi(global)}, seed)` —
 * cuRAND's block at (subsequence = global, word offset = 4 * counter). A draw
 * reads only the words of its own block, so neighbouring counters share no
 * word and the draws are independent. The raw block words equal cuRAND's;
 * the uniform conversion is aether's own open-interval one (below).
 *
 * No SoftDouble arm: aether carries no SoftDouble/Banded scalar backend in
 * this module yet — that is not yet implemented.
 *
 * Keying: the SplitMix64 finalizer, the `streamKey` mixing,
 * `Generator::substream`'s salt, the per-distribution salts and
 * `INDEX_FREE_DOMAIN` are fixed constants, never re-derived.
 *
 * Transforms:
 *  - `uniform01<double>` = `w * 2^-32 + 2^-33` from one Philox word, in the
 *    OPEN interval `(0,1)` (never 0, never 1; exact).
 *  - `uniform01<float>` = `(2*(w>>9) + 1) * 2^-24`, also `(0,1)` (exact; the
 *    cuRAND float form rounds to 1.0f for the top words).
 *    Open on both sides, so `log(u)` and `log(1-u)` are always finite.
 *    (The double Box-Muller below uses cuRAND's two-word `_curand_uniform_double_hq`.)
 *  - `standardNormal` mirrors cuRAND's Box-Muller structure
 *    (`sqrt(-2 log u)`, angle from the second uniform) but routes the
 *    transcendentals through `aether::math::{log,sqrt,sincos}` — the FP64
 *    `sincos` being the zero-stack `detail::cwSinCos` path
 *    (`aether/math/detail/BoundedTrig.h`), not cuRAND's `sincospi`, which
 *    aether does not have and will not take a CUDA dependency for. The
 *    argument is bounded by construction (`v * pi` with `v` in `(0,2)`), so
 *    it always takes that path's fast reduction. Consequence, documented
 *    rather than hidden: against the reference cuRAND device stream the
 *    normal stream matches to a tolerance (abs+rel 1e-12 double, 1e-5
 *    float), not bit-for-bit — the bits and uniform streams remain
 *    bit-for-bit. The float leg's tolerance is the looser one because the
 *    reference reaches `__sincosf`, the fast hardware approximation, where
 *    aether calls `sincosf`.
 */

#include <cstdint>
#include <type_traits>

#include "aether/index/Offset.h"
#include "aether/macros.h"
#include "aether/math/math.h"
#include "aether/random/detail/Philox.h"

/// @cond INTERNAL

namespace aether {
namespace random {
namespace detail {

/**
 * @brief SplitMix64 finalizer — a fast, well-mixed 64-bit bijection.
 *
 * The exact mixing constants below are fixed: a `Generator` built from a
 * given seed must always carry the same key, so `substream` chains and
 * per-distribution salting land on the same streams.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/**
 * @brief Mix a `(seed, global sample id, sub-counter)` triple into one
 *        64-bit stream key.
 *
 * Nothing in this library calls it today — aether has no host RNG arm that
 * needs a seeded `std::mt19937_64` (see the file docstring). It is kept as
 * part of the fixed keying surface: any later aether path that needs a
 * single mixed 64-bit key from that triple must use THIS mixing, not invent
 * another.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr std::uint64_t streamKey(std::uint64_t seed, offset_t global, std::uint64_t sub)
{
    const std::uint64_t g = splitmix64(static_cast<std::uint64_t>(global) * 0x100000001B3ULL + sub);
    return splitmix64(seed ^ g);
}

// ---------------------------------------------------------------------------
// cuRAND's word -> real conversions, reproduced exactly.
//
// Each is "exact scaling + one rounding" and therefore contraction-proof:
// no `-ffp-contract`/`-fmad` setting can change the result, which is what
// makes a bit-for-bit comparison against the real cuRAND device stream
// decidable.
// ---------------------------------------------------------------------------

/** @brief `2^-32` as a double (cuRAND's `CURAND_2POW32_INV_DOUBLE`). */
inline constexpr double kTwoPow32InvDouble = 2.3283064365386963e-10;
/** @brief `2^-53` as a double (cuRAND's `CURAND_2POW53_INV_DOUBLE`). */
inline constexpr double kTwoPow53InvDouble = 1.1102230246251565e-16;
/** @brief pi to double precision (cuRAND's `CURAND_PI_DOUBLE`). */
inline constexpr double kPiDouble = 3.1415926535897932;
/** @brief `2^-32` as a float (cuRAND's `CURAND_2POW32_INV`; the decimal
 *         literal rounds to exactly `2^-32`). */
inline constexpr float kTwoPow32InvFloat = 2.3283064e-10f;
/** @brief `2^-32 * 2pi` in float (cuRAND's `CURAND_2POW32_INV_2PI`), folded
 *         at compile time exactly as the macro is. */
inline constexpr float kTwoPow32Inv2PiFloat = 2.3283064e-10f * 6.2831855f;

/** @brief One word to a double in the OPEN interval `(0,1)`: `w * 2^-32 + 2^-33`
 *         (exact; spans `[2^-33, 1 - 2^-33]`, so neither 0 nor 1 is reachable). */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() double uniformDoubleFromWord(std::uint32_t w)
{
    return w * kTwoPow32InvDouble + (kTwoPow32InvDouble / 2.0);
}

/** @brief One word to a float in the OPEN interval `(0,1)`: `(2*(w>>9) + 1) * 2^-24`
 *         (23 bits of resolution, exact; spans `[2^-24, 1 - 2^-24]`). The plain
 *         `w * 2^-32 + 2^-33` of cuRAND rounds to exactly 1.0f for the top words. */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() float uniformFloatFromWord(std::uint32_t w)
{
    return static_cast<float>(((w >> 9) << 1) | 1u) * 5.9604645e-8f; // 2^-24
}

/**
 * @brief The Philox block owned by `(seed, global, counter)`.
 *
 * Counter words `{lo(counter), hi(counter), lo(global), hi(global)}` under key
 * `seed` — the block cuRAND reaches at `(subsequence = global, offset = 4 *
 * counter)`. Every draw reads only the four words of its own block, so
 * neighbouring counters never share a word.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Uint4 philoxBlock(std::uint64_t seed, offset_t global, std::uint64_t counter)
{
    const std::uint64_t g = static_cast<std::uint64_t>(global);
    return philox4x32_10(Uint4{ static_cast<std::uint32_t>(counter), static_cast<std::uint32_t>(counter >> 32),
                             static_cast<std::uint32_t>(g), static_cast<std::uint32_t>(g >> 32) },
        static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32));
}

/** @brief cuRAND's `_curand_uniform_double_hq(x, y)`: 53 bits spliced from two
 *         words, scaled to `(0,1)`. Used ONLY by the double Box-Muller below
 *         (the path `curand_normal_double` reaches on the device). */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() double uniformDoubleHq(std::uint32_t x, std::uint32_t y)
{
    const std::uint64_t z = static_cast<std::uint64_t>(x) ^ (static_cast<std::uint64_t>(y) << (53 - 32));
    return static_cast<double>(z) * kTwoPow53InvDouble + (kTwoPow53InvDouble / 2.0);
}

/**
 * @brief Draw 64 raw uniform bits.
 *
 * Pure function of `(seed, global, counter)`. Two Philox words, low word
 * first, matching cuRAND's device spelling exactly (`lo = curand(&st); hi =
 * curand(&st); (hi << 32) | lo`). This is the INTEGER-domain primitive:
 * FP-free by construction, which is what keeps `uniformInt` legal inside
 * emulated (zero-FP64) kernels once they exist.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::uint64_t randomBits64(std::uint64_t seed, offset_t global, std::uint64_t counter)
{
    const Uint4 x          = philoxBlock(seed, global, counter);
    const std::uint32_t lo = x.x;
    const std::uint32_t hi = x.y;
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

/**
 * @brief Draw a uniform value in the open interval `(0, 1)`.
 *
 * Pure function of `(seed, global, counter)`: word 0 of the `(seed, global,
 * counter)` Philox block — identical on both arms.
 */
template<typename Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real uniform01(std::uint64_t seed, offset_t global, std::uint64_t counter)
{
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
        "aether::random: Real must be float or double (SoftDouble support is not yet implemented)");
    const Uint4 x = philoxBlock(seed, global, counter);
    if constexpr (std::is_same_v<Real, double>) {
        return uniformDoubleFromWord(x.x);
    } else {
        return uniformFloatFromWord(x.x);
    }
}

/**
 * @brief Draw a standard normal `N(0,1)` value.
 *
 * Pure function of `(seed, global, counter)`. Box-Muller, mirroring cuRAND's
 * `_curand_box_muller_double` / `_curand_box_muller` structure and word
 * consumption (all four words of the block for `double`, two for
 * `float`), with the transcendentals routed through `aether::math` — see the
 * file docstring for why, and for the TIER-TOL consequence. Only the FIRST
 * of Box-Muller's two normals is returned; cuRAND caches the second in its
 * state, but a counter-based generator that re-positions on every draw can
 * never observe it, so it is simply discarded.
 */
template<typename Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real standardNormal(std::uint64_t seed, offset_t global, std::uint64_t counter)
{
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
        "aether::random: Real must be float or double (SoftDouble support is not yet implemented)");
    const Uint4 x = philoxBlock(seed, global, counter);
    if constexpr (std::is_same_v<Real, double>) {
        const double u = uniformDoubleHq(x.x, x.y);
        // cuRAND's angle variable lives in (0,2) and is fed to sincospi();
        // `v * kPiDouble` is the same angle for a sincos() that takes radians.
        const std::uint64_t zy = static_cast<std::uint64_t>(x.z) ^ (static_cast<std::uint64_t>(x.w) << (53 - 32));
        const double v = static_cast<double>(zy) * (kTwoPow53InvDouble * 2.0) + kTwoPow53InvDouble;
        const double s = math::sqrt(-2.0 * math::log(u));
        double sinv = 0.0;
        double cosv = 0.0;
        math::sincos(v * kPiDouble, &sinv, &cosv);
        return s * sinv;
    } else {
        const std::uint32_t yw = x.y;
        const float u = uniformFloatFromWord(x.x);
        const float v = yw * kTwoPow32Inv2PiFloat + (kTwoPow32Inv2PiFloat / 2.0f);
        const float s = math::sqrt(-2.0f * math::log(u));
        float sinv = 0.0f;
        float cosv = 0.0f;
        math::sincos(v, &sinv, &cosv);
        return s * sinv;
    }
}

} // namespace detail
} // namespace random
} // namespace aether

/// @endcond
