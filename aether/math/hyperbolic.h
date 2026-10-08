// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file hyperbolic.h
 * @brief `aether::math` — hyperbolic and inverse hyperbolic functions:
 *        `sinh`, `cosh`, `asinh`, `acosh`, `atanh` (`tanh` lives in
 *        `trig.h`).
 *
 * Device compiles route to the CUDA math library (`sinhf`/`::%sinh`, ...);
 * host compiles route to `std::`, or — `double` under
 * `AETHER_HOST_VECTOR_MATH` — to the faithfully rounded packet functions
 * (`detail/HostVectorMath.h`). Real-only (`float`/`double`): a banded
 * operand hits the scaffold's `static_assert`.
 *
 * Own independent include/define/undef cycle over `MathDispatch.h`, like
 * `explog.h`/`root.h`; `math.h` includes this after its own
 * `MathDispatchUndef.h` pass.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief Hyperbolic sine of `x`. */
AETHER_MATH_UNARY(sinh, sinhf, ::sinh, AETHER_MATH_HOST_ROUTE(sinh, x))

/** @brief Hyperbolic cosine of `x`. */
AETHER_MATH_UNARY(cosh, coshf, ::cosh, AETHER_MATH_HOST_ROUTE(cosh, x))

/** @brief Inverse hyperbolic sine of `x`. */
AETHER_MATH_UNARY(asinh, asinhf, ::asinh, AETHER_MATH_HOST_ROUTE(asinh, x))

/** @brief Inverse hyperbolic cosine of `x` (NaN for `x < 1`). */
AETHER_MATH_UNARY(acosh, acoshf, ::acosh, AETHER_MATH_HOST_ROUTE(acosh, x))

/** @brief Inverse hyperbolic tangent of `x` (`+-inf` at `+-1`, NaN beyond). */
AETHER_MATH_UNARY(atanh, atanhf, ::atanh, AETHER_MATH_HOST_ROUTE(atanh, x))

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
