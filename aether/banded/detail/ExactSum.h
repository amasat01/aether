// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ExactSum.h
 * @brief Order-free exact-sum renormalization cascades (`exactSum3`,
 *        `exactSum4`) shared across the transcendental families:
 *        `BandExpLog.h`'s `log` reconstruction, and `BandRound.h`'s
 *        `round`/`floor` integer-part cascades.
 *
 * Both consumers land in the same translation unit via
 * `aether/banded/banded.h`'s umbrella, so each cascade is defined exactly
 * once, here, and included from both rather than duplicated (which would
 * be a duplicate-definition compile error) — the same pattern `FmaRaw.h`
 * uses for its own shared primitives.
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

/// @brief Renormalize an EXACT three-term sum into a carrier, order-free.
/// Every step is a `twoSum`, so no operand ordering is assumed and no bit is
/// dropped: `rh + m + t == x + y + z` exactly. Used by the integer-part
/// cascades (`floor`'s `a.tail` branch, `round`'s third-limb branch).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band exactSum3(
    float x, float y, float z)
{
    float s, e;
    twoSum(y, z, s, e);
    float rh, rl;
    twoSum(x, s, rh, rl);
    float m, t;
    twoSum(rl, e, m, t);
    if (bandHiIsSpecial(rh))
        return bandSpecialCarrier(rh);
    return Band{ rh, m, t };
}

/// @brief Renormalize an EXACT four-term sum into a carrier, order-free:
/// every step is a `twoSum`, so `s3 + u2 + (v2+v1) == x+y+z+w` exactly and
/// no operand ordering is assumed. Used by `log`'s reconstruction
/// (`BandExpLog.h`) and `round`'s own fourth-limb branch (`BandRound.h`).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band exactSum4(
    float x, float y, float z, float w)
{
    float s1, e1;
    twoSum(z, w, s1, e1);
    float s2, e2;
    twoSum(y, s1, s2, e2);
    float s3, e3;
    twoSum(x, s2, s3, e3);
    float u1, v1;
    twoSum(e3, e2, u1, v1);
    float u2, v2;
    twoSum(u1, e1, u2, v2);
    if (bandHiIsSpecial(s3))
        return bandSpecialCarrier(s3);
    return Band{ s3, u2, addRN(v2, v1) };
}

} // namespace detail
} // namespace banded
} // namespace aether
