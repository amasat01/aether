// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Operations.h
 * @brief Expression-template operators: `a + b`, `a - b`, `s * e`,
 *        `e * s`, `e / s` (`s` = a `T` scalar matching `E::element_type`).
 *
 * One rank-generic operator set: no vector/matrix family split, no
 * separate gating per rank — every operator is gated only by the
 * `aether_expression` concept (plus, for the scalar overloads, the scalar
 * operand's type matching `E::element_type`). `element_extents`/`element_type`
 * agreement between two expression operands is enforced by the node
 * constructors themselves (`Sum`'s `static_assert`s), not re-checked here.
 */

#include "aether/expr/Expression.h"
#include "aether/expr/nodes/Arithmetic.h"
#include "aether/macros.h"

namespace aether {

/** @brief Component-wise `l + r`. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::Sum<L, R, false> operator+(const L& l, const R& r)
{
    return detail::Sum<L, R, false>(l, r);
}

/** @brief Component-wise `l - r`. */
template<aether_expression L, aether_expression R>
AETHER_DEVICEHOST() constexpr detail::Sum<L, R, true> operator-(const L& l, const R& r)
{
    return detail::Sum<L, R, true>(l, r);
}

/** @brief `scalar * expr`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseScale<E> operator*(const typename E::element_type& s, const E& e)
{
    return detail::CWiseScale<E>(e, s);
}

/** @brief `expr * scalar`, component-wise. */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseScale<E> operator*(const E& e, const typename E::element_type& s)
{
    return detail::CWiseScale<E>(e, s);
}

/**
 * @brief `expr / scalar`, component-wise. The reciprocal is computed once
 *        here and handed to the same `CWiseScale` node `operator*` uses —
 *        there is no separate "divide" node type.
 */
template<aether_expression E>
AETHER_DEVICEHOST() constexpr detail::CWiseScale<E> operator/(const E& e, const typename E::element_type& s)
{
    return detail::CWiseScale<E>(e, typename E::element_type{ 1 } / s);
}

} // namespace aether
