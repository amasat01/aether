// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Elementwise.h
 * @brief `aether::detail::CWiseMin<E>` / `CWiseMax<E>` / `CWiseAbs<E>` /
 *        `CWiseClamp<E>` / `CWiseSign<E>` — the `cwiseMin`/`cwiseMax`/
 *        `cwiseAbs`/`clamp`/`cwiseSign` free functions, rank-generic
 *        component-wise ops.
 *
 * Mirrors `CWiseScale`'s shape (`aether/expr/nodes/Arithmetic.h`): one
 * expression operand (the `isLeaf`-conditional `const E&`/`const E`
 * capture every node in this library uses) plus a scalar bound (or two, for
 * `clamp`) captured by value and broadcast against every component at
 * `eval()` time — no second expression operand, same as `CWiseScale`'s
 * single `factor_`. `CWiseAbs`/`CWiseSign` carry no scalar at all (pure
 * unary — same shape minus the broadcast field). Reached via free functions
 * (this file's own bottom section), same call-site shape as `CWiseScale`'s
 * free `operator*`/`operator/` (`aether/expr/Operations.h`) — not
 * `Expression` member functions (unlike `cwiseMul`/`transpose`/...,
 * `aether/expr/nodes/Product.h`'s convention).
 *
 * Dispatches through `aether::math::{fmin,fmax,abs,sign}` per component.
 *
 * Also carries, further down this file:
 *  - rank>=1 unary transcendentals — `sqrt`, `rsqrt`, `exp`, `log`, `sin`,
 *    `cos`, `asin`, `acos`, `atan`, `tan`, `tanh` (`abs` already has
 *    `CWiseAbs` above). Same shape as `CWiseAbs`: one operand, no scalar,
 *    dispatched through `aether::math::<name>`.
 *  - two-tensor (both operands full expressions, not a broadcast scalar)
 *    `cwiseDiv`/`cwisePow`/`cwiseMin`/`cwiseMax` — `cwiseMin`/`cwiseMax`
 *    overload the existing scalar-bound free functions above (a plain C++
 *    overload set, resolved by whether the second argument satisfies
 *    `aether_expression` — the `aether_expression` concept requires an
 *    `element_type` nested typedef etc., so a bare scalar argument like
 *    `1.0` cleanly fails that constraint and only the scalar overload above
 *    stays viable, with no ambiguity in either direction); `cwiseDiv`/
 *    `cwisePow` are new names (mirrors `CWiseScale`'s `L op R` shape from
 *    `aether/expr/nodes/Arithmetic.h`'s `Sum`, computing the per-component
 *    result through `/` / `aether::math::pow` instead of `+`/`-`).
 *    `cwisePow` also gets a scalar-exponent overload (`cwisePow(e, n)`,
 *    mirroring `CWiseMin`/`CWiseMax`'s own scalar-bound shape).
 *  - `aether::select(cond, a, b)`: a three-operand node, same capture/eval
 *    protocol as every node above, just one more operand — see its own doc
 *    comment below for why this needs no new evaluation machinery.
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace detail {

/** @brief `min(expr, bound)`, component-wise — mirrors `CWiseScale`'s shape. */
template<aether_expression E>
class CWiseMin : public Expression<CWiseMin<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    element_type bound_;
    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr CWiseMin(const E& expr, const element_type& bound)
        : bound_(bound)
        , expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::fmin(evalW<Is...>(expr_, i), toWorkingValue(bound_));
    }
};

/** @brief `max(expr, bound)`, component-wise — mirrors `CWiseScale`'s shape. */
template<aether_expression E>
class CWiseMax : public Expression<CWiseMax<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    element_type bound_;
    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr CWiseMax(const E& expr, const element_type& bound)
        : bound_(bound)
        , expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::fmax(evalW<Is...>(expr_, i), toWorkingValue(bound_));
    }
};

/** @brief `abs(expr)`, component-wise — pure unary, no scalar operand. */
template<aether_expression E>
class CWiseAbs : public Expression<CWiseAbs<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseAbs(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::abs(evalW<Is...>(expr_, i));
    }
};

/**
 * @brief `clamp(expr, lo, hi)`, component-wise — two scalar bounds, same
 *        shape as `CWiseScale` with a second broadcast field.
 */
template<aether_expression E>
class CWiseClamp : public Expression<CWiseClamp<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    element_type lo_;
    element_type hi_;
    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr CWiseClamp(const E& expr, const element_type& lo, const element_type& hi)
        : lo_(lo)
        , hi_(hi)
        , expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::fmin(math::fmax(evalW<Is...>(expr_, i), toWorkingValue(lo_)), toWorkingValue(hi_));
    }
};

/**
 * @brief `sign(expr)`, component-wise — pure unary, no
 *        scalar operand. Dispatches through `aether::math::sign`
 *        (`aether/math/math.h`).
 */
template<aether_expression E>
class CWiseSign : public Expression<CWiseSign<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseSign(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::sign(evalW<Is...>(expr_, i));
    }
};

} // namespace detail

// ---------------------------------------------------------------------------
// cwiseMin/cwiseMax/cwiseAbs/clamp/cwiseSign — free functions, natural
// call-site syntax (mirrors CWiseScale's free operator*/operator/,
// aether/expr/Operations.h).
// ---------------------------------------------------------------------------

/** @brief `min(expr, bound)`, component-wise; scalar `bound` broadcast across every component. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseMin<E> cwiseMin(const E& e, const typename E::element_type& bound)
{
    return detail::CWiseMin<E>(e, bound);
}

/** @brief `max(expr, bound)`, component-wise; scalar `bound` broadcast across every component. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseMax<E> cwiseMax(const E& e, const typename E::element_type& bound)
{
    return detail::CWiseMax<E>(e, bound);
}

/** @brief `abs(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseAbs<E> cwiseAbs(const E& e)
{
    return detail::CWiseAbs<E>(e);
}

/** @brief `clamp(expr, lo, hi)`, component-wise; scalar bounds `[lo, hi]` broadcast across every component. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseClamp<E> clamp(const E& e, const typename E::element_type& lo, const typename E::element_type& hi)
{
    return detail::CWiseClamp<E>(e, lo, hi);
}

/** @brief `sign(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseSign<E> cwiseSign(const E& e)
{
    return detail::CWiseSign<E>(e);
}

// =============================================================================
// Rank>=1 unary transcendentals. Each is
// the IDENTICAL shape to `CWiseAbs` above (one operand, no scalar, dispatch
// through the matching `aether::math` name); `tan`/`tanh` route to
// `aether/math/trig.h`'s facade entries.
// =============================================================================

namespace detail {

/** @brief `sqrt(expr)`, component-wise. */
template<aether_expression E>
class CWiseSqrt : public Expression<CWiseSqrt<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseSqrt(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::sqrt(evalW<Is...>(expr_, i));
    }
};

/** @brief `rsqrt(expr)`, component-wise. */
template<aether_expression E>
class CWiseRsqrt : public Expression<CWiseRsqrt<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseRsqrt(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::rsqrt(evalW<Is...>(expr_, i));
    }
};

/** @brief `exp(expr)`, component-wise. */
template<aether_expression E>
class CWiseExp : public Expression<CWiseExp<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseExp(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::exp(evalW<Is...>(expr_, i));
    }
};

/** @brief `log(expr)`, component-wise. */
template<aether_expression E>
class CWiseLog : public Expression<CWiseLog<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseLog(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::log(evalW<Is...>(expr_, i));
    }
};

/** @brief `sin(expr)`, component-wise. */
template<aether_expression E>
class CWiseSin : public Expression<CWiseSin<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseSin(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::sin(evalW<Is...>(expr_, i));
    }
};

/** @brief `cos(expr)`, component-wise. */
template<aether_expression E>
class CWiseCos : public Expression<CWiseCos<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseCos(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::cos(evalW<Is...>(expr_, i));
    }
};

/** @brief `asin(expr)`, component-wise. */
template<aether_expression E>
class CWiseAsin : public Expression<CWiseAsin<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseAsin(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::asin(evalW<Is...>(expr_, i));
    }
};

/** @brief `acos(expr)`, component-wise. */
template<aether_expression E>
class CWiseAcos : public Expression<CWiseAcos<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseAcos(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::acos(evalW<Is...>(expr_, i));
    }
};

/** @brief `atan(expr)`, component-wise. */
template<aether_expression E>
class CWiseAtan : public Expression<CWiseAtan<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseAtan(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::atan(evalW<Is...>(expr_, i));
    }
};

/** @brief `tan(expr)`, component-wise (routes to `aether::math::tan`). */
template<aether_expression E>
class CWiseTan : public Expression<CWiseTan<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseTan(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::tan(evalW<Is...>(expr_, i));
    }
};

/** @brief `tanh(expr)`, component-wise (routes to `aether::math::tanh`). */
template<aether_expression E>
class CWiseTanh : public Expression<CWiseTanh<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit CWiseTanh(const E& expr)
        : expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::tanh(evalW<Is...>(expr_, i));
    }
};

} // namespace detail

/** @brief `sqrt(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseSqrt<E> cwiseSqrt(const E& e)
{
    return detail::CWiseSqrt<E>(e);
}

/** @brief `rsqrt(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseRsqrt<E> cwiseRsqrt(const E& e)
{
    return detail::CWiseRsqrt<E>(e);
}

/** @brief `exp(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseExp<E> cwiseExp(const E& e)
{
    return detail::CWiseExp<E>(e);
}

/** @brief `log(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseLog<E> cwiseLog(const E& e)
{
    return detail::CWiseLog<E>(e);
}

/** @brief `sin(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseSin<E> cwiseSin(const E& e)
{
    return detail::CWiseSin<E>(e);
}

/** @brief `cos(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseCos<E> cwiseCos(const E& e)
{
    return detail::CWiseCos<E>(e);
}

/** @brief `asin(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseAsin<E> cwiseAsin(const E& e)
{
    return detail::CWiseAsin<E>(e);
}

/** @brief `acos(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseAcos<E> cwiseAcos(const E& e)
{
    return detail::CWiseAcos<E>(e);
}

/** @brief `atan(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseAtan<E> cwiseAtan(const E& e)
{
    return detail::CWiseAtan<E>(e);
}

/** @brief `tan(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseTan<E> cwiseTan(const E& e)
{
    return detail::CWiseTan<E>(e);
}

/** @brief `tanh(expr)`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseTanh<E> cwiseTanh(const E& e)
{
    return detail::CWiseTanh<E>(e);
}

// =============================================================================
// Two-tensor div/pow/min/max, plus a
// scalar-exponent `cwisePow` mirroring `CWiseMin`/`CWiseMax`'s own shape.
// The two-operand nodes mirror `Sum<L,R>`'s capture protocol
// (`aether/expr/nodes/Arithmetic.h`) exactly — TWO independent expression
// operands, each `isLeaf`-conditionally captured — computing a per-component
// combine instead of `+`/`-`.
// =============================================================================

namespace detail {

/** @brief `l / r`, component-wise, both operands full expressions (unlike
 *         `CWiseScale`'s `expr / scalar` reciprocal-multiply, reached via
 *         `aether/expr/Operations.h`'s `operator/`). */
template<aether_expression L, aether_expression R>
class CWiseDiv : public Expression<CWiseDiv<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "CWiseDiv: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "CWiseDiv: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr CWiseDiv(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<Is...>(l_, i) / evalW<Is...>(r_, i);
    }
};

/** @brief `pow(l, r)`, component-wise, both operands full expressions. */
template<aether_expression L, aether_expression R>
class CWisePowExpr : public Expression<CWisePowExpr<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "CWisePowExpr: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "CWisePowExpr: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr CWisePowExpr(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::pow(evalW<Is...>(l_, i), evalW<Is...>(r_, i));
    }
};

/** @brief `min(l, r)`, component-wise, both operands full expressions
 *         (the SCALAR-bound `CWiseMin<E>` above stays untouched — this is a
 *         separate class, reached via a `cwiseMin` FREE-FUNCTION overload
 *         below; a class template cannot itself be overloaded by arity). */
template<aether_expression L, aether_expression R>
class CWiseMinExpr : public Expression<CWiseMinExpr<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "CWiseMinExpr: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "CWiseMinExpr: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr CWiseMinExpr(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::fmin(evalW<Is...>(l_, i), evalW<Is...>(r_, i));
    }
};

/** @brief `max(l, r)`, component-wise, both operands full expressions. @see `CWiseMinExpr`. */
template<aether_expression L, aether_expression R>
class CWiseMaxExpr : public Expression<CWiseMaxExpr<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "CWiseMaxExpr: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "CWiseMaxExpr: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr CWiseMaxExpr(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::fmax(evalW<Is...>(l_, i), evalW<Is...>(r_, i));
    }
};

/** @brief `pow(expr, exponent)`, component-wise; scalar `exponent` broadcast
 *         across every component — mirrors `CWiseMin<E>`'s own shape above
 *         (one expression operand plus one scalar bound). */
template<aether_expression E>
class CWisePow : public Expression<CWisePow<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    element_type exponent_;
    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr CWisePow(const E& expr, const element_type& exponent)
        : exponent_(exponent)
        , expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return math::pow(evalW<Is...>(expr_, i), toWorkingValue(exponent_));
    }
};

} // namespace detail

/** @brief `div(l, r)`, component-wise; both operands full expressions of matching shape. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::CWiseDiv<L, R> cwiseDiv(const L& l, const R& r)
{
    return detail::CWiseDiv<L, R>(l, r);
}

/** @brief `pow(expr, exponent)`, component-wise; scalar `exponent` broadcast across every component. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWisePow<E> cwisePow(const E& e, const typename E::element_type& exponent)
{
    return detail::CWisePow<E>(e, exponent);
}

/** @brief `pow(l, r)`, component-wise; both operands full expressions of matching shape. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::CWisePowExpr<L, R> cwisePow(const L& l, const R& r)
{
    return detail::CWisePowExpr<L, R>(l, r);
}

/** @brief `min(l, r)`, component-wise; both operands full expressions of matching shape — OVERLOADS `cwiseMin(e, bound)` above. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::CWiseMinExpr<L, R> cwiseMin(const L& l, const R& r)
{
    return detail::CWiseMinExpr<L, R>(l, r);
}

/** @brief `max(l, r)`, component-wise; both operands full expressions of matching shape — OVERLOADS `cwiseMax(e, bound)` above. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::CWiseMaxExpr<L, R> cwiseMax(const L& l, const R& r)
{
    return detail::CWiseMaxExpr<L, R>(l, r);
}

// =============================================================================
// `aether::select(cond, a, b)`: elementwise
// choose between two expressions by a per-element or per-sample boolean mask.
// =============================================================================

namespace detail {

/**
 * @brief `cond ? a : b`, component-wise. `Cond`'s
 *        `element_type` must be `bool`; its `element_extents` is either
 *        RANK-0 (a single boolean broadcast to every component — the
 *        "per-SAMPLE" case) or IDENTICAL to `A`/`B`'s own `element_extents`
 *        (a per-component mask — the "per-ELEMENT" case), selected via
 *        `if constexpr` in `eval()` so there is no per-call runtime branch
 *        and no third instantiation shape.
 *
 * Same three-operand capture protocol every node in this file already uses
 * (`isLeaf`-conditional `const X&`/`const X`), just with THREE operands
 * instead of `Sum`'s two. This is what resolves HAWK's own `_select` doc
 * comment (`hawk/emit/aether.py`): that comment describes wanting an
 * aether "materialise this expression at a SampleIndex" primitive because
 * HAWK's STRING renderer, as written, would otherwise emit a raw C++
 * ternary `(c ? aText : bText)` directly in a kernel body — and `A`/`B` are,
 * in general, DIFFERENT aether expression C++ TYPES (e.g. a `View` vs. a
 * `Sum<...>`), so that ternary has no common type and is ill-formed. A
 * `Select` NODE sidesteps the problem entirely rather than needing that
 * primitive: `A` and `B` are never compared as C++ types at all — each is
 * read independently through the shared `evalW` helper, which converts
 * BOTH into the SAME `working_type` before this class's OWN internal
 * ternary chooses between them, so the two branches of THIS ternary always
 * share one type regardless of what `A`/`B` themselves are. No expression is
 * ever materialised at an arbitrary `SampleIndex` — the whole point of
 * laziness — so no new evaluation machinery is needed.
 */
template<aether_expression Cond, aether_expression A, aether_expression B>
class Select : public Expression<Select<Cond, A, B>, typename A::element_type> {
    static_assert(std::is_same_v<typename A::element_type, typename B::element_type>,
        "select: a/b element_type must match");
    static_assert(std::is_same_v<typename A::element_extents, typename B::element_extents>,
        "select: a/b element_extents must match");
    static_assert(std::is_same_v<typename Cond::element_type, bool>, "select: cond element_type must be bool");
    static_assert(Cond::element_extents::Rank == 0 || std::is_same_v<typename Cond::element_extents, typename A::element_extents>,
        "select: cond element_extents must be rank-0 (per-sample) or match a/b (per-element)");

public:
    using element_type = typename A::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename A::element_extents;
    static constexpr bool isLeaf = false;

    SelectOperand<Cond> cond_;
    SelectOperand<A> a_;
    SelectOperand<B> b_;

    AETHER_DEVICEHOST() constexpr Select(const Cond& cond, const A& a, const B& b)
        : cond_{ cond }
        , a_{ a }
        , b_{ b }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        if constexpr (Cond::element_extents::Rank == 0) {
            return evalW<>(cond_, i) ? evalW<Is...>(a_, i) : evalW<Is...>(b_, i);
        } else {
            return evalW<Is...>(cond_, i) ? evalW<Is...>(a_, i) : evalW<Is...>(b_, i);
        }
    }
};

} // namespace detail

/** @brief `select(cond, a, b)`, component-wise: `cond ? a : b` per element. @see `detail::Select`'s own doc comment. */
template<aether_expression Cond, aether_expression A, aether_expression B>
AETHER_DEVICEHOST() constexpr detail::Select<Cond, A, B> select(const Cond& cond, const A& a, const B& b)
{
    return detail::Select<Cond, A, B>(cond, a, b);
}

} // namespace aether
