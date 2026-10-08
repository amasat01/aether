// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file trig.h
 * @brief `aether::math` — the trig family: `sin`/`cos` gain a banded
 *        companion overload, and `sincos` gains a Band-constrained
 *        overload alongside `math.h`'s own `float`/`double` template.
 *
 * Own independent include/define/undef cycle over `MathDispatch.h`, like
 * `explog.h`/`root.h` — `math.h` includes this after its own
 * `MathDispatchUndef.h` pass.
 *
 * `sincos` has no `AETHER_MATH_*_BAND` macro shape (both macros return a
 * single `Band`; `sincos` is `void`, two out-params), so its Band overload
 * is hand-written below, mirroring `rsqrt`/`rsqrtCube` in `root.h`. Its
 * parameter shape is `(T, T*, T*)` — identical to `math.h`'s own base
 * template, constrained only by `requires IsBandedFamily<T>`, so it wins
 * over the unconstrained base template under C++20 partial ordering
 * without any edit to `math.h`'s own `sin`/`cos`.
 *
 * `trigReduce` is not dispatched here: it is an internal reduction
 * primitive with no `aether::math` unary/binary shape, staying
 * `aether::banded::detail`-only (see `BandTrig.h`).
 *
 * `tan`/`tanh` use the plain `AETHER_MATH_UNARY` scaffold directly, the
 * same as `atan`/`asin`/`acos` in `invtrig.h`. Real-only (`float`/
 * `double`): no `AETHER_MATH_UNARY_BAND` companion is defined for either
 * name yet, so a `Band`/`BandedReal` argument hits the scaffold's own
 * `static_assert` and fails to compile with that message.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief Tangent of `x`. Real-only: no banded overload yet. */
AETHER_MATH_UNARY(tan, tanf, ::tan, AETHER_MATH_HOST_ROUTE(tan, x))

/** @brief Hyperbolic tangent of `x`. Real-only: no banded overload yet. */
AETHER_MATH_UNARY(tanh, tanhf, ::tanh, AETHER_MATH_HOST_ROUTE(tanh, x))

/** @brief …`sin` on a banded operand: forwards to the certified
 *         `BandTrig.h::sin` (coefficient-swap core) via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(sin, sin)

/** @brief …`cos` on a banded operand: forwards to the certified
 *         `BandTrig.h::cos` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(cos, cos)

/**
 * @brief Sine and cosine of a banded `x`, sharing one reduction. Forwards
 *        to `BandTrig.h::sincos` via `BandedFacade` — not the same bits
 *        as calling this file's `sin`/`cos` separately (@see
 *        `BandTrig.h`'s `trigSelectCore` doc comment: `sincos` and the
 *        lone `sin`/`cos` are different bodies, both inside the certified
 *        bound).
 */
template<class T>
    requires aether::banded::IsBandedFamily<T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void sincos(T x, T* sinOut, T* cosOut)
{
    // `BandedWorkingOfT<T>`, not `aether::banded::Band`, for the local
    // temporaries: `Band` is a non-dependent name, and declaring a local
    // variable of a non-dependent incomplete type is checked at parse
    // time (not instantiation), which would break every non-banded TU
    // that reaches this header through `math.h`. `BandedFwd.h`'s own doc
    // comment names this same trap for the companion macros' `static_cast`.
    aether::banded::detail::BandedWorkingOfT<T> s, c;
    aether::banded::detail::BandedFacade<T>::sincos(
        static_cast<aether::banded::detail::BandedWorkingOfT<T>>(x), s, c);
    *sinOut = s;
    *cosOut = c;
}

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
