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
 * uniform streams; and because the shared code reproduces cuRAND's
 * positioning exactly, the device stream matches the real cuRAND device
 * stream bit-for-bit.
 *
 * No SoftDouble arm: aether carries no SoftDouble/Banded scalar backend in
 * this module yet — that is not yet implemented.
 *
 * Keying: the SplitMix64 finalizer, the `streamKey` mixing,
 * `Generator::substream`'s salt, the per-distribution salts and
 * `INDEX_FREE_DOMAIN` are fixed constants, never re-derived.
 *
 * Transforms:
 *  - `uniform01<double>` = `(w + 1) * 2^-32` from one Philox word — cuRAND's
 *    `_curand_uniform_double(unsigned int)`, which is what
 *    `curand_uniform_double(curandStatePhilox4_32_10_t*)` calls. (cuRAND's
 *    higher-quality two-word `_curand_uniform_double_hq` conversion is used
 *    here too — inside the double Box-Muller below, which reaches it
 *    through `curand_normal_double`.) Range `(0, 1]`, so `log(u)` is always
 *    safe and no host-only lower-endpoint guard is needed on either arm.
 *  - `uniform01<float>` = `w * 2^-32f + 2^-33f` — cuRAND's `_curand_uniform`.
 *    Also `(0, 1]`.
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

/** @brief cuRAND's `_curand_uniform_double(unsigned int)`: `(w+1) * 2^-32`, in `(0,1]`. */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() double uniformDoubleFromWord(std::uint32_t w)
{
    return w * kTwoPow32InvDouble + kTwoPow32InvDouble;
}

/** @brief cuRAND's `_curand_uniform(unsigned int)`: `w * 2^-32 + 2^-33`, in `(0,1]`. */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() float uniformFloatFromWord(std::uint32_t w)
{
    return w * kTwoPow32InvFloat + (kTwoPow32InvFloat / 2.0f);
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
    PhiloxStream st(seed, static_cast<std::uint64_t>(global), counter);
    const std::uint32_t lo = st.next();
    const std::uint32_t hi = st.next();
    return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

/**
 * @brief Draw a uniform value in `(0, 1]`.
 *
 * Pure function of `(seed, global, counter)`: the Philox stream is positioned
 * at `(seed, subsequence = global, offset = counter)` and ONE word is
 * consumed — identical on both arms.
 */
template<typename Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real uniform01(std::uint64_t seed, offset_t global, std::uint64_t counter)
{
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
        "aether::random: Real must be float or double (SoftDouble support is not yet implemented)");
    PhiloxStream st(seed, static_cast<std::uint64_t>(global), counter);
    if constexpr (std::is_same_v<Real, double>) {
        return uniformDoubleFromWord(st.next());
    } else {
        return uniformFloatFromWord(st.next());
    }
}

/**
 * @brief Draw a standard normal `N(0,1)` value.
 *
 * Pure function of `(seed, global, counter)`. Box-Muller, mirroring cuRAND's
 * `_curand_box_muller_double` / `_curand_box_muller` structure and word
 * consumption (four words for `double` via one `next4()`, two words for
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
    PhiloxStream st(seed, static_cast<std::uint64_t>(global), counter);
    if constexpr (std::is_same_v<Real, double>) {
        const Uint4 x = st.next4();
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
        const std::uint32_t xw = st.next();
        const std::uint32_t yw = st.next();
        const float u = xw * kTwoPow32InvFloat + (kTwoPow32InvFloat / 2.0f);
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
