// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file explog.h
 * @brief `aether::math` — exponential and logarithm family: `exp`, `exp2`,
 *        `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`. Device compiles route to the CUDA
 *        math intrinsic (`expf`/`::%exp`, ...); host compiles route to
 *        `std::`. `exp10`/`::%exp10` is a device-only intrinsic; the host
 *        leg deliberately does not call a bare `::%exp10` (a non-standard
 *        GNU libm extension, not guaranteed portable) — it uses
 *        `std::pow(T(10), x)` instead.
 *
 * Uses the same two-leg (`float`/`double`, device/host) scaffold
 * `aether/math/detail/MathDispatch.h` already generates for `math.h`'s
 * `abs`/`floor`/`fmax`/`fmin`/`pow`/`sqrt` — same macros, same shape, no
 * new dispatch machinery. `pow` already lives in `math.h` and is not
 * repeated here.
 *
 * `exp`/`log` also accept `aether::banded::{Band, BandedReal}` and forward
 * through `aether::banded::detail::BandedFacade<T>::{exp,log}`
 * (`BandedRealOps.h`) to the bodies in `aether/banded/BandExpLog.h`,
 * exactly as `abs`/`copysign`/`fmax`/`fmin`/`pow` already do in `math.h`.
 * `exp2`/`exp10`/`expm1`/`log2`/`log10`/`log1p` stay `float`/`double`-only.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief Base `e` exponential function. */
AETHER_MATH_UNARY(exp, expf, ::exp, AETHER_MATH_HOST_ROUTE(exp, x))
/** @brief …on a banded operand: degree-13 Taylor on a three-rung demand
 *         ladder over a Cody-Waite reduction, certified over
 *         `banded::detail::bandExpAdmits`. */
AETHER_MATH_UNARY_BAND(exp, exp)

/** @brief Base `2` exponential function. */
AETHER_MATH_UNARY(exp2, exp2f, ::exp2, AETHER_MATH_HOST_ROUTE(exp2, x))

/** @brief Base `10` exponential function. */
AETHER_MATH_UNARY(exp10, exp10f, ::exp10, AETHER_MATH_HOST_ROUTE_OR(exp10, std::pow(T(10), x), x))

/** @brief `exp(x) - 1`, accurate for small `|x|`. */
AETHER_MATH_UNARY(expm1, expm1f, ::expm1, AETHER_MATH_HOST_ROUTE(expm1, x))

/** @brief Base `e` logarithm of the argument. */
AETHER_MATH_UNARY(log, logf, ::log, AETHER_MATH_HOST_ROUTE(log, x))
/** @brief …on a banded operand: 16-entry table reduction + degree-12 series
 *         on the same three-rung demand ladder, certified over
 *         `banded::detail::bandLogAdmits`. */
AETHER_MATH_UNARY_BAND(log, log)

/** @brief Base `2` logarithm of the argument. */
AETHER_MATH_UNARY(log2, log2f, ::log2, AETHER_MATH_HOST_ROUTE(log2, x))

/** @brief Base `10` logarithm of the argument. */
AETHER_MATH_UNARY(log10, log10f, ::log10, AETHER_MATH_HOST_ROUTE(log10, x))

/** @brief `log(1 + x)`, accurate for small `|x|`. */
AETHER_MATH_UNARY(log1p, log1pf, ::log1p, AETHER_MATH_HOST_ROUTE(log1p, x))

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
