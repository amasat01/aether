// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file invtrig.h
 * @brief `aether::math` — the inverse-trig family: `atan`/`atan2`/`asin`/
 *        `acos` gain 3-way (`float`/`double`/`Band`) dispatch entries.
 *
 * Own independent include/define/undef cycle over `MathDispatch.h`, like
 * `root.h`'s (and `explog.h`'s) — `math.h` includes this file after its
 * own `MathDispatchUndef.h` pass, so the macros this file (re-)defines
 * never collide with `math.h`'s or `root.h`'s own invocations.
 *
 * All four ops have standard `std::`/CUDA-intrinsic spellings
 * (`std::atan`/`atanf`/`::%atan`, `std::atan2`/`atan2f`/`::%atan2`,
 * `std::asin`/`asinf`/`::%asin`, `std::acos`/`acosf`/`::%acos`), so the
 * scalar (non-`Band`) legs use the same `AETHER_MATH_UNARY`/
 * `AETHER_MATH_BINARY` scaffold `math.h`'s own `abs`/`fmax`/... and
 * `root.h`'s own `cbrt`/`hypot` already do — no hand-written template is
 * needed here (unlike `root.h`'s `rsqrt`/`rsqrtCube`, which have no
 * standard spelling to route through).
 *
 * Every Band-typed entry point routes through `BandedFacade<T>`
 * (`aether/banded/BandedRealOps.h`), so this file names no
 * `aether::banded::` symbol directly — same header-diet discipline
 * `math.h`'s own docstring describes.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief Arctangent of `x`. */
AETHER_MATH_UNARY(atan, atanf, ::atan, AETHER_MATH_HOST_ROUTE(atan, x))
/** @brief …`atan` on a banded operand: forwards to
 *         `BandInvTrig.h`'s `atan` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(atan, atan)

/** @brief Arctangent of `a/b`, using the sign of both to determine the
 *         correct quadrant (IEEE-754/C99 `atan2(y, x)` convention). */
AETHER_MATH_BINARY(atan2, atan2f, ::atan2, AETHER_MATH_HOST_ROUTE(atan2, a, b))
/** @brief …`atan2` on a banded operand: forwards to
 *         `BandInvTrig.h`'s `atan2` via `BandedFacade`. */
AETHER_MATH_BINARY_BAND(atan2, atan2)

/** @brief Arcsine of `x`, `x` in `[-1, 1]`. */
AETHER_MATH_UNARY(asin, asinf, ::asin, AETHER_MATH_HOST_ROUTE(asin, x))
/** @brief …`asin` on a banded operand: forwards to
 *         `BandInvTrig.h`'s `asin` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(asin, asin)

/** @brief Arccosine of `x`, `x` in `[-1, 1]`. */
AETHER_MATH_UNARY(acos, acosf, ::acos, AETHER_MATH_HOST_ROUTE(acos, x))
/** @brief …`acos` on a banded operand: forwards to
 *         `BandInvTrig.h`'s `acos` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(acos, acos)

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
