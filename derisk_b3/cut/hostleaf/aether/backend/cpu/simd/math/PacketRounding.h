// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketRounding.h
 * @brief Vectorised `floor`, `ceil`, `trunc`, `round` (half away from
 *        zero, as `std::round`) and `rint` (nearest-even) for
 *        `aether::simd::Packet<double, W>`.
 *
 * One branchless body for every width: |x| is rounded to nearest-even with
 * the 2^52 magic constant (exact for |x| < 2^52; larger values are already
 * integers and pass through), the direction-specific correction is a
 * compare-and-adjust, and the sign is re-applied last so that -0 and
 * negative results that round to zero keep their sign. Exact (0 ULP)
 * against `std::` for every finite input; inf/NaN pass through.
 */

#include "aether/backend/cpu/simd/math/PacketMathCore.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

template<class D>
AETHER_PM_INLINE D truncAbs(D ax)
{
    const D r = rintNonNeg(ax);
    return select(gt(r, ax), r - 1.0, r);
}

template<class D>
AETHER_PM_INLINE D ceilAbs(D ax)
{
    const D r = rintNonNeg(ax);
    return select(lt(r, ax), r + 1.0, r);
}

template<class D>
AETHER_PM_INLINE D vtrunc(D x)
{
    return copysign(truncAbs(abs(x)), x);
}

template<class D>
AETHER_PM_INLINE D vfloor(D x)
{
    const D ax = abs(x);
    return copysign(select(signbit(x), ceilAbs(ax), truncAbs(ax)), x);
}

template<class D>
AETHER_PM_INLINE D vceil(D x)
{
    const D ax = abs(x);
    return copysign(select(signbit(x), truncAbs(ax), ceilAbs(ax)), x);
}

template<class D>
AETHER_PM_INLINE D vround(D x)
{
    const D ax = abs(x);
    const D t = truncAbs(ax);
    return copysign(select(ge(ax - t, bc<D>(0.5)), t + 1.0, t), x);
}

template<class D>
AETHER_PM_INLINE D vrint(D x)
{
    return copysign(rintNonNeg(abs(x)), x);
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
