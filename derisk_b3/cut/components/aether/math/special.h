// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file special.h
 * @brief `aether::math` — special functions: `erf` and `erfc`.
 *
 * Device compiles route to the CUDA math library (`erff`/`::%erf`, ...);
 * host compiles route to `std::` (scalar on the host: no packet version,
 * so `AETHER_HOST_VECTOR_MATH` leaves them on `std::`). The standard
 * normal CDF is `0.5 * erfc(-x / sqrt(2))`.
 *
 * Not provided yet: `lgamma`/`tgamma`/`digamma` and the Bessel functions.
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

/** @brief Error function, `2/sqrt(pi) * integral_0^x exp(-t^2) dt`. */
AETHER_MATH_UNARY(erf, erff, ::erf, std::erf(x))

/** @brief Complementary error function, `1 - erf(x)` without the
 *         cancellation for large `x`. */
AETHER_MATH_UNARY(erfc, erfcf, ::erfc, std::erfc(x))

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
