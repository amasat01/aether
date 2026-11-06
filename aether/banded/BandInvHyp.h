// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandInvHyp.h
 * @brief `asinh` / `acosh` / `atanh` over `Band`: the classical identities
 *        over certified `log`, `log1p`, `sqrt_` and `div`, with every
 *        cancellation moved into an exact subtraction.
 *
 *  - `asinh`: `|x| < 2` -> `log1p(a + a^2/(1 + sqrt(1 + a^2)))`; `[2, 2^28)` ->
 *    `log(2a + 1/(a + sqrt(a^2 + 1)))`; `>= 2^28` -> `log(a) + ln2` (`a^2`
 *    would leave the FP32 envelope). Sign by `copysign`.
 *  - `acosh`: `x < 1` -> NaN; `[1, 2)` -> `t = x - 1` (exact),
 *    `log1p(t + sqrt(2t + t^2))`; `[2, 2^28)` -> `log(x + sqrt(x^2 - 1))`;
 *    `>= 2^28` -> `log(x) + ln2`; `acosh(1) = +0`.
 *  - `atanh`: `|x| > 1` -> NaN; `+-1` -> `+-inf`; `|x| < 1/2` ->
 *    `log1p(2a + 2a^2/(1 - a))/2`; else `log1p(2a/(1 - a))/2`, `1 - a` exact.
 * Certified bound: 2 ULP at 53 bits.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandExpLog.h"
#include "aether/banded/RsqrtCore.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

/// @brief `asinh(x)`: `+-0`, `+-inf` and NaN pass through.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band asinh(Band x)
{
    if (x.hi != x.hi || x.hi == 0.0f || fabsf(x.hi) == intAsFloat(0x7F800000))
        return x;
    const Band one{ 1.0f, 0.0f, 0.0f };
    const Band a = abs(x);
    Band       r;
    if (a.hi >= 0x1p28f) {
        r = add(log(a), kBandLn2);
    } else {
        // ONE formula over [0, 2^28): asinh = log1p(a + a^2/(1 + sqrt(a^2 + 1))). The [2, 2^28) leg's own
        // formula, log(2a + 1/(a + sqrt(a^2+1))), is the same value; range legs that both execute on divergent lanes cost their sum.
        const Band a2 = mul(a, a);
        r = log1p(add(a, div(a2, add(one, sqrt_(add(a2, one))))));
    }
    return copysign(r, x);
}

/// @brief `acosh(x)`: NaN for `x < 1`, `acosh(1) = +0`, `acosh(+inf) = +inf`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band acosh(Band x)
{
    if (!(x.hi >= 1.0f))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    if (x.hi == intAsFloat(0x7F800000))
        return x;
    const Band one{ 1.0f, 0.0f, 0.0f };
    if (x.hi >= 0x1p28f)
        return add(log(x), kBandLn2);
    // ONE formula over [1, 2^28): t = x - 1 (exact on [1, 2], 2^-72-relative above), acosh = log1p(t + sqrt(t(t+2))).
    const Band t = sub(x, one);
    return log1p(add(t, sqrt_(add(add(t, t), mul(t, t)))));
}

/// @brief `atanh(x)`: NaN for `|x| > 1`, `+-1` gives `+-inf`, `+-0` passes through.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band atanh(Band x)
{
    if (!(fabsf(x.hi) <= 1.0f))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    if (x.hi == 0.0f)
        return x;
    const Band a = abs(x);
    if (a.hi == 1.0f && a.lo == 0.0f && a.tail == 0.0f)
        return copysign(Band{ intAsFloat(0x7F800000), 0.0f, 0.0f }, x);
    const Band one{ 1.0f, 0.0f, 0.0f };
    const Band om = sub(one, a); // exact for a >= 0.5
    // t = 2a/(1 - a) for every a (om's 2^-72-relative rounding below 0.5 is a relative error of t, not an absolute one).
    const Band t  = div(add(a, a), om);
    return copysign(scalePow2f(log1p(t), 0.5f), x);
}

} // namespace detail
} // namespace banded
} // namespace aether
