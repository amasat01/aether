// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file AtanSector.h
 * @brief `AtanSector` / `atanSector`: a branchless sector lookup used by
 *        the `atan2` angle reduction.
 *
 * @section scope Scope
 * `atanSector` picks one of five pre-computed `(tan(phi_i), phi_i)` `Band`
 * pairs from the leading limbs of its two operands, branchlessly. It is a
 * standalone lookup table: each stored `t` is `tan` of its paired stored
 * `phi` (checked by `Atan2SectorConstantsAreAMatchedPairOnHost`), and the
 * table does not itself perform a division, a Newton iteration, or any
 * other transcendental computation. The reduction that consumes this
 * table (`atan2`'s division-once argument, `atan`, `asin`, `acos`) lives
 * elsewhere; this file is the lookup only.
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

/// @brief The four sector boundaries, each `tan` of a boundary angle
/// (5/15/25/35 degrees) rounded to one float. Only the branch taken
/// depends on them, never the returned value.
inline constexpr float kBandAtanSecB0 = 8.7488666177e-02f;
inline constexpr float kBandAtanSecB1 = 2.6794919372e-01f;
inline constexpr float kBandAtanSecB2 = 4.6630766988e-01f;
inline constexpr float kBandAtanSecB3 = 7.0020753145e-01f;

/// @brief One sector: the tangent the reduction rotates by, and the base
/// angle that is `atan` of that stored tangent (a matched pair, not a
/// nominal-angle pairing).
struct AtanSector {
    Band t;   ///< the rotation's tangent: ONE float (`lo == tail == 0`), the float nearest `tan(phi_i_nominal)`
    Band phi; ///< `atan(t)` of that one-float `t`, three limbs, to `2^-78` (any `t` is a valid pair; a one-float `t` makes the rotation's products cheaper)
};

/// @brief Pick the sector from the leading limbs alone, branchlessly.
/// Sector 0 carries `t = phi = 0` as the value of the table (not a
/// special case): the unrotated head. Comparisons use the leading limbs
/// only; `b > a * k_i` decides only which identity a downstream reduction
/// runs, never the reduction's own answer.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() AtanSector atanSector(float b, float a)
{
    const bool s1 = b > a * kBandAtanSecB0;
    const bool s2 = b > a * kBandAtanSecB1;
    const bool s3 = b > a * kBandAtanSecB2;
    const bool s4 = b > a * kBandAtanSecB3;
    const Band t{ s4 ? 8.3909964561e-01f : (s3 ? 5.7735025883e-01f : (s2 ? 3.6397022009e-01f : (s1 ? 1.7632697523e-01f : 0.0f))),
        0.0f, 0.0f };
    const Band phi{ s4 ? 6.9813168049e-01f
                       : (s3 ? 5.2359879017e-01f : (s2 ? 3.4906584024e-01f : (s1 ? 1.7453292012e-01f : 0.0f))),
        s4 ? 2.8781325767e-08f
           : (s3 ? -2.2342275585e-08f
                 : (s2 ? -2.3642485747e-09f : (s1 ? -2.3947191452e-10f : 0.0f))),
        s4 ? 8.2247232698e-16f
           : (s3 ? -2.5420676771e-18f
                 : (s2 ? 6.2237142080e-17f : (s1 ? -1.3189705887e-17f : 0.0f))) };
    return AtanSector{ t, phi };
}

} // namespace detail
} // namespace banded
} // namespace aether
