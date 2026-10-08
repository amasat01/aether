// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandedRealOps.h
 * @brief The namespace-scope arithmetic operator set for `BandedReal`, and
 *        the definition of the `detail::BandedFacade` the `aether::math`
 *        facade routes through.
 *
 * @section why_bandedrealops Why these cannot be `Band`'s operators, and cannot be members
 *
 * `Band`'s `+ - * /` are hidden friends (`Band.h`): ADL finds them only when an
 * operand already is a `Band`. A conversion cannot reach a hidden friend, so an
 * implicit `operator Band()` on the storage type does **not** make
 * `BandedReal / BandedReal` resolve. Namespace-scope templates do, and that is
 * the shape reproduced here.
 *
 * Members are excluded by design rather than by mechanics: no arithmetic
 * operators on the storage type itself. What that rule protects is that a chain
 * must not pay a pack per op — and these operators preserve exactly that, since
 * every one of them returns `Band`:
 *
 *      BandedReal e = (a - b) * c / d;      // three ops, one pack
 *
 * packs once, at the assignment terminal (`BandedReal(Band)`), and never in
 * between.
 *
 * @section shape_bandedrealops The constraint shape
 * Each operator requires both operands in the banded family and at least one to
 * be the storage leaf. `Band op Band` is therefore left entirely to `Band`'s own
 * hidden friends: nothing here is even a candidate for it, so there is no
 * overload-resolution question to get wrong. (A non-template hidden friend would
 * beat these templates anyway, but "would win" and "is never considered" are
 * different amounts of safety for the same zero cost.)
 *
 * @section wall The operator wall
 * `BandedReal` is deliberately kept out of `BandWallFamily`: it is a sanctioned
 * region-entry demote source, not a chain-resident carrier. Mixing it with a
 * bare `float`/`double` still hits the wall's deleted overloads through the
 * `Band` conversion — the wall is not weakened by anything here, and
 * `tests/compile_fail/check_bandedreal_typing_rejected.sh` has an arm that says
 * so.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandedReal.h"
#include "aether/banded/BandExpLog.h"
#include "aether/banded/BandRoot.h"
#include "aether/banded/BandTrig.h"
#include "aether/banded/BandRound.h"
#include "aether/banded/BandInvTrig.h"
#include "aether/banded/RsqrtCore.h"
#include "aether/macros.h"
#include "aether/math/detail/BandedFwd.h"

namespace aether {
namespace banded {

/// @brief Both operands banded, at least one of them the storage leaf.
template<typename A, typename B>
concept BandedRealOperands = IsBandedFamily<A> && IsBandedFamily<B>
    && (IsBandedStorage<A> || IsBandedStorage<B>);

/// @brief Banded add. Returns the working carrier — the chain stays unpacked.
template<typename A, typename B>
    requires BandedRealOperands<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator+(A a, B b)
{
    return detail::add(static_cast<Band>(a), static_cast<Band>(b));
}

/// @brief Banded sub. @see operator+.
template<typename A, typename B>
    requires BandedRealOperands<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(A a, B b)
{
    return detail::sub(static_cast<Band>(a), static_cast<Band>(b));
}

/// @brief Banded mul. @see operator+.
template<typename A, typename B>
    requires BandedRealOperands<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator*(A a, B b)
{
    return detail::mul(static_cast<Band>(a), static_cast<Band>(b));
}

/// @brief Banded div — the one operator a vector expression algebra bottoms out
/// in (an error-controller's `(error / desired).maxNorm()` fails at exactly this
/// point without it). @see operator+.
template<typename A, typename B>
    requires BandedRealOperands<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator/(A a, B b)
{
    return detail::div(static_cast<Band>(a), static_cast<Band>(b));
}

/// @brief Unary negate on the storage leaf. (`Band`'s own hidden friend covers
/// the working carrier.)
template<typename A>
    requires IsBandedStorage<A>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(A a)
{
    return detail::neg(static_cast<Band>(a));
}

/* ── Compound assignment: the one place a pack is intended ────────────────
 *
 * `a += b` names its own destination, so there is no chain to keep unpacked and
 * the pack is the point. Written out rather than left to the generic
 * `operator=` terminal so the cost is visible at the call site: each of these is
 * one decode, one certified op, one encode.
 *
 * There is deliberately no `operator/=`: `Band` has none, and giving the storage
 * face a spelling its own working carrier refuses would be worse than the
 * omission. */
template<typename B>
    requires IsBandedFamily<B>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal& operator+=(BandedReal& a, B b)
{
    a = static_cast<Band>(a) + b;
    return a;
}
template<typename B>
    requires IsBandedFamily<B>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal& operator-=(BandedReal& a, B b)
{
    a = static_cast<Band>(a) - b;
    return a;
}
template<typename B>
    requires IsBandedFamily<B>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal& operator*=(BandedReal& a, B b)
{
    a = static_cast<Band>(a) * b;
    return a;
}

namespace detail {

/**
 * @brief The certified entry points
 *        `aether::math::abs/fmax/fmin/copysign/pow/exp/log` reach for a
 *        banded operand, re-exposed as members of a class template.
 *
 * Declared in `aether/math/detail/BandedFwd.h`, defined here once `Band` and
 * the certified `detail::` bodies are complete. @see BandedFwd.h for why the
 * indirection exists at all (the short version: a companion body that names a
 * `Band`-returning free function directly is a non-dependent call, which
 * nvcc's front end resolves while parsing the template and rejects because
 * the return type is incomplete there).
 *
 * Every member is a one-line forward to the `detail::` body; the parameter is
 * already the working carrier, and the storage-to-working conversion happens
 * in the companion's own `static_cast` at the call site. `T` is unused in
 * every body on purpose — it exists to make the name dependent, nothing
 * else.
 */
template<typename T>
struct BandedFacade {
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band abs(Band a)
    {
        return banded::detail::abs(a);
    }
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band copysign(
        Band a, Band b)
    {
        return banded::detail::copysign(a, b);
    }
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmax(Band a, Band b)
    {
        return banded::detail::fmax(a, b);
    }
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmin(Band a, Band b)
    {
        return banded::detail::fmin(a, b);
    }
    /**
     * @brief Base `e` exponential. One-line forward to `banded::detail::exp`
     *        (`BandExpLog.h`), same shape as the four members above.
     */
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band exp(Band a)
    {
        return banded::detail::exp(a);
    }
    /**
     * @brief Base `e` logarithm. One-line forward to `banded::detail::log`
     *        (`BandExpLog.h`).
     */
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log(Band a)
    {
        return banded::detail::log(a);
    }
    /**
     * @brief `a ** b`. `pow = exp(b * log(a))` over `BandExpLog.h`'s
     *        `exp`/`log` — see that file for the algorithm and the certified
     *        domain (`bandPowAdmits`). One-line forward, same shape as
     *        `abs`/`copysign`/`fmax`/`fmin` above.
     */
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band pow(Band a, Band b)
    {
        return banded::detail::pow(a, b);
    }

    // Reciprocal/root family: BandedFacade is the sole routing point the
    // aether::math dispatch companions (aether/math/detail/MathDispatch.h)
    // can reach for sqrt/rsqrt/rsqrtCube/cbrt/hypot.

    /// @brief `sqrt(x)`. Forwards to `RsqrtCore.h`'s `sqrt_` — named `sqrt_`
    /// there only to avoid shadowing `::%sqrt`; no such collision exists on a
    /// facade member.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sqrt(Band a)
    {
        return banded::detail::sqrt_(a);
    }
    /// @brief `x^(-1/2)`. Forwards to `BandRoot.h`'s `rsqrtIeee`, which
    /// applies the `rsqrt(+Inf)=+0` fix ahead of `RsqrtCore.h`'s own `rsqrt`
    /// rather than inside it. @see `BandRoot.h`'s own file-header section.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band rsqrt(Band a)
    {
        return banded::detail::rsqrtIeee(a);
    }
    /// @brief `x^(-3/2)`, fused. Forwards to `RsqrtCore.h`'s `rsqrtCube`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band rsqrtCube(Band a)
    {
        return banded::detail::rsqrtCube(a);
    }
    /// @brief `x^(1/3)`. Forwards to `BandRoot.h`'s `cbrt`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band cbrt(Band a)
    {
        return banded::detail::cbrt(a);
    }
    /// @brief `sqrt(x^2+y^2)`, overflow-/underflow-safe. Forwards to
    /// `BandRoot.h`'s `hypot`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band hypot(Band a, Band b)
    {
        return banded::detail::hypot(a, b);
    }

    // Trig family: BandedFacade is the sole routing point the aether::math
    // dispatch entries in aether/math/trig.h can reach.

    /// @brief `sin(x)`. Forwards to `BandTrig.h`'s `sin`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sin(Band a)
    {
        return banded::detail::sin(a);
    }
    /// @brief `cos(x)`. Forwards to `BandTrig.h`'s `cos`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band cos(Band a)
    {
        return banded::detail::cos(a);
    }
    /// @brief `sin(x)` and `cos(x)` together, sharing one reduction. Forwards
    /// to `BandTrig.h`'s `sincos` — @see that file's own doc comment for why
    /// this and the individual `sin`/`cos` paths can deliver different bits.
    static AETHER_DEVICEHOST() AETHER_FORCEINLINE() void sincos(Band a, Band& s, Band& c)
    {
        banded::detail::sincos(a, s, c);
    }

    // Rounding family: BandedFacade is the sole routing point the
    // aether::math dispatch companions (aether/math/round.h) can reach.

    /// @brief Greatest integer <= `a`. Forwards to `BandRound.h`'s `floor`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band floor(Band a)
    {
        return banded::detail::floor(a);
    }
    /// @brief Least integer >= `a`. Forwards to `BandRound.h`'s `ceil`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band ceil(Band a)
    {
        return banded::detail::ceil(a);
    }
    /// @brief Nearest integer, half away from zero. Forwards to
    /// `BandRound.h`'s `round`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band round(Band a)
    {
        return banded::detail::round(a);
    }
    /// @brief Round toward zero. Forwards to `BandRound.h`'s `trunc`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band trunc(Band a)
    {
        return banded::detail::trunc(a);
    }
    /// @brief `a > b ? a - b : +0`. Forwards to `Band.h`'s own `fdim`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fdim(Band a, Band b)
    {
        return banded::detail::fdim(a, b);
    }
    /// @brief IEEE remainder, truncated quotient, sign of the dividend.
    /// Forwards to `BandRound.h`'s `fmod`, exact on span <= 72
    /// (`bandFmodAdmits`).
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmod(Band a, Band b)
    {
        return banded::detail::fmod(a, b);
    }
    /// @brief `a*b + c`, single rounding barrier. Forwards to
    /// `BandRound.h`'s `fma`, built over the shared `fmaRaw` family
    /// (`aether/banded/detail/FmaRaw.h`).
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fma(Band a, Band b, Band c)
    {
        return banded::detail::fma(a, b, c);
    }

    // Inverse-trig family: BandedFacade is the sole routing point the
    // aether::math companions can reach.

    /// @brief `atan(x)`. Forwards to `BandInvTrig.h`'s `atan` (`atan2(x, 1)`
    /// internally).
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band atan(Band a)
    {
        return banded::detail::atan(a);
    }
    /// @brief `atan2(y, x)`, the full plane. Forwards to `BandInvTrig.h`'s
    /// `atan2`, built on `AtanSector.h`'s table.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band atan2(Band a, Band b)
    {
        return banded::detail::atan2(a, b);
    }
    /// @brief `asin(x)` on `[-1, 1]`. Forwards to `BandInvTrig.h`'s `asin`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band asin(Band a)
    {
        return banded::detail::asin(a);
    }
    /// @brief `acos(x)` on `[-1, 1]`. Forwards to `BandInvTrig.h`'s `acos`.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band acos(Band a)
    {
        return banded::detail::acos(a);
    }
};

} // namespace detail
} // namespace banded
} // namespace aether
