// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandHyp.h
 * @brief `tanh` / `sinh` / `cosh` over `Band`, as compositions of `expm1`
 *        (`BandExpLog.h`): no cancellation at small `|x|`, no overflow of an
 *        intermediate before the result itself overflows.
 *
 * With `e = expm1(|x|)` and `E = e + 1 = e^|x|`:
 *  - `tanh(x) = sign(x) * e2/(e2 + 2)`, `e2 = expm1(2|x|)`; `|x| >= 22` is
 *    exactly `+-1` to 53 bits (`1 - tanh < 2^-62`).
 *  - `sinh(x) = sign(x) * (e + e/E)/2`.
 *  - `cosh(x) = (E + 1/E)/2`.
 *  From `|x| >= 22` the `sinh`/`cosh` correction terms are under `2^-62`
 *  relative and the result is `exp(|x| - ln2)`, which reaches the FP32
 *  envelope edge (`|x| > kBandExpSatHigh + ln2`) before overflowing. Certified bound: 2 ULP at
 * 53 bits (`expm1`'s half ULP, one division or reciprocal, one sum, one
 * scaling; observed far below).
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandExpLog.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

/// @brief `|x| >= 22`: `tanh` is `+-1`, and `sinh`/`cosh` are `e^|x|/2`, each
/// to well beyond 53 bits (the dropped terms are under `2^-62`).
inline constexpr float kBandHypSat = 22.0f;

/// @brief `tanh(x)`: `+-0` and NaN pass through, `+-inf` and `|x| >= 22` give `+-1`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band tanh(Band x)
{
    if (x.hi != x.hi || x.hi == 0.0f)
        return x;
    const Band a = abs(x);
    if (a.hi >= kBandHypSat)
        return copysign(Band{ 1.0f, 0.0f, 0.0f }, x);
    const Band e = expm1(scalePow2f(a, 2.0f)); // 2|x| is exact
    return copysign(div(e, add(e, Band{ 2.0f, 0.0f, 0.0f })), x);
}

/// @brief `sinh(x)`: `+-0` and NaN pass through, `+-inf` gives `+-inf`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sinh(Band x)
{
    if (x.hi != x.hi || x.hi == 0.0f)
        return x;
    const Band a = abs(x);
    if (a.hi >= kBandHypSat) // e^|x|/2 = exp(|x| - ln2): finite up to the last binade
        return copysign(exp(sub(a, kBandLn2)), x);
    const Band e = expm1(a);
    return copysign(scalePow2f(add(e, div(e, add(e, Band{ 1.0f, 0.0f, 0.0f }))), 0.5f), x);
}

/// @brief `cosh(x)`: NaN passes through, `+-inf` gives `+inf`, `cosh(+-0) = 1`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band cosh(Band x)
{
    if (x.hi != x.hi)
        return x;
    const Band a = abs(x);
    if (a.hi >= kBandHypSat)
        return exp(sub(a, kBandLn2));
    const Band E = exp(a); // exp directly: the same value as `expm1(a) + 1`, one add and its rounding less
    return scalePow2f(add(E, recip(E)), 0.5f);
}

} // namespace detail
} // namespace banded
} // namespace aether
