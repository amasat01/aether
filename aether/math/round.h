// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file round.h
 * @brief `aether::math` — the round family: `floor` gains its banded
 *        companion, and `ceil`/`round`/`trunc`/`fdim`/`fmod` are 3-way
 *        (`float`/`double`/`Band`) dispatch entries; `rint` and the
 *        numpy-semantics `remainder` are `float`/`double` only. `fma` gains its
 *        banded overload (hand-written, ternary — no
 *        `AETHER_MATH_*_BAND` macro exists for arity 3).
 *
 * Own independent include/define/undef cycle over `MathDispatch.h`, the
 * same pattern `root.h`/`explog.h` already establish — `math.h` includes
 * this after its own `MathDispatchUndef.h` pass (and after `root.h`'s),
 * so the macros this file (re-)defines never collide with the earlier
 * invocations, and this file's `AETHER_MATH_UNARY_BAND(floor, floor)` is
 * simply a second overload of `aether::math::floor` (Band-constrained;
 * `math.h`'s own `floor` entry stays float/double-only and untouched).
 *
 * `ceil`/`round`/`trunc` (unary) and `fdim`/`fmod` (binary) have standard
 * `std::`/CUDA spellings, so they use the same `AETHER_MATH_UNARY`/
 * `AETHER_MATH_BINARY` scaffold `math.h`'s own `floor`/`fmax`/… do — no
 * partial-ordering edit is needed for these five: the unary scaffold
 * already wins cleanly against a constrained Band companion, and
 * `AETHER_MATH_BINARY` already carries the `requires(!IsBandedFamily<T>)`
 * guard that makes the same true for the binary two.
 *
 * @section fmaBand `fma`'s Band overload
 * `fma` is hand-written in `math.h` (no `std::`/CUDA-intrinsic-uniform
 * ternary dispatch macro exists — same reason `root.h`'s `rsqrt`/
 * `rsqrtCube` are hand-written), so there is no `AETHER_MATH_TERNARY_BAND`
 * companion macro to reuse either. Its scalar template was
 * `template<class T> T fma(T,T,T)` with no constraint excluding the
 * banded family, and a third templated argument makes
 * `fma(Band,Band,Band)` more specialized than a three-independent-type
 * companion, so the existing unconstrained scalar template would keep
 * winning even with a Band overload added here. `math.h`'s scalar `fma`
 * therefore gained one line, `requires(!%aether::banded::IsBandedFamily<T>)`,
 * mirroring `AETHER_MATH_BINARY`'s own macro-generated guard.
 *
 * Every Band-typed entry point routes through `BandedFacade<T>`
 * (`aether/banded/BandedRealOps.h`), so this file names no
 * `%aether::banded::` symbol directly.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief …`floor` on a banded operand: forwards to the ported
 *         `BandRound.h::floor` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(floor, floor)

/** @brief Least integer >= `x`. */
AETHER_MATH_UNARY(ceil, ceilf, ::ceil, AETHER_MATH_HOST_ROUTE(ceil, x))
/** @brief …`ceil` on a banded operand: `-floor(-x)`, sign-bit flips only. */
AETHER_MATH_UNARY_BAND(ceil, ceil)

/** @brief Nearest integer, half away from zero (C's `round`, not
 *         `rint`'s half-to-even; numpy's `round` is `rint`). */
AETHER_MATH_UNARY(round, roundf, ::round, AETHER_MATH_HOST_ROUTE(round, x))
/** @brief …`round` on a banded operand: Shewchuk grow-expansion +
 *         early-stop floor cascade, ported verbatim. */
AETHER_MATH_UNARY_BAND(round, round)

/** @brief Nearest integer in the current rounding mode (half to even by
 *         default; numpy's `rint`/`round`). */
AETHER_MATH_UNARY(rint, rintf, ::rint, AETHER_MATH_HOST_ROUTE(rint, x))

/** @brief Round toward zero. */
AETHER_MATH_UNARY(trunc, truncf, ::trunc, AETHER_MATH_HOST_ROUTE(trunc, x))
/** @brief …`trunc` on a banded operand: `copysign(floor(|x|), x)`. */
AETHER_MATH_UNARY_BAND(trunc, trunc)

/** @brief Positive difference: `a > b ? a - b : +0`. */
AETHER_MATH_BINARY(fdim, fdimf, ::fdim, AETHER_MATH_HOST_ROUTE(fdim, a, b))
/** @brief …`fdim` on a banded operand: forwards to `Band.h`'s own `fdim`. */
AETHER_MATH_BINARY_BAND(fdim, fdim)

/** @brief Floating-point remainder of `a/b`, sign of the dividend. */
AETHER_MATH_BINARY(fmod, fmodf, ::fmod, std::fmod(a, b))
/** @brief …`fmod` on a banded operand: forwards to `BandRound.h::fmod` —
 *         exact on span <= 72 (`bandFmodAdmits`), ported verbatim. */
AETHER_MATH_BINARY_BAND(fmod, fmod)

/**
 * @brief Floored remainder of `a/b` with numpy's `remainder` semantics:
 *        the result has the sign of the DIVISOR `b` (`a - floor(a/b)*b`,
 *        computed exactly from `fmod`), a zero result is `copysign(0, b)`,
 *        and `remainder(-1, inf) == inf` as in numpy. NOT C's IEEE
 *        `remainder` (round-to-nearest quotient); use `fmod` for the
 *        truncated (sign of the dividend) form. Scalar on the host.
 */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T remainder(T a, T b)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "aether::math::remainder: T must be float or double");
    T m = fmod(a, b);
    if (m != T(0)) {
        if ((b < T(0)) != (m < T(0)))
            m += b;
    } else {
        m = copysign(T(0), b);
    }
    return m;
}

/**
 * @brief …`fma` on a banded operand: hand-written ternary companion — no
 *         `AETHER_MATH_TERNARY_BAND` macro exists (arity 3 has no
 *         scaffold, @see this file's "fmaBand" section). Same `auto`-
 *         return / `BandedWorkingOfT`-cast shape the macro-generated
 *         unary/binary companions use, spelled out by hand for three
 *         arguments.
 */
template<class A, class B, class C>
    requires(aether::banded::IsBandedFamily<A> && aether::banded::IsBandedFamily<B>
        && aether::banded::IsBandedFamily<C>)
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto fma(A a, B b, C c)
{
    return aether::banded::detail::BandedFacade<A>::fma(
        static_cast<aether::banded::detail::BandedWorkingOfT<A>>(a),
        static_cast<aether::banded::detail::BandedWorkingOfT<B>>(b),
        static_cast<aether::banded::detail::BandedWorkingOfT<C>>(c));
}

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
