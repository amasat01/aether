// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Geometric.h
 * @brief `aether::detail::Cross<L,R>` / `aether::detail::UnitVector<E>` —
 *        back `Expression::cross()`/`unitVector()` (declared + defined
 *        inline in `aether/expr/Expression.h`; see that file's docstring
 *        for why the whole member surface lives there).
 *
 * `Cross`'s three components each independently recompute l0..l2/r0..r2
 * from the operands, matching `Cross`'s own per-component recompute
 * convention used throughout `expr/nodes/`. `UnitVector` is the node
 * behind the `unitVector()` accessor.
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/expr/Reduce.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/** @brief Cross product of two 3-vector expressions (element_extents<3> only). */
template<class L, class R>
class Cross : public Expression<Cross<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_extents, extents<3>>,
        "Cross: L must be a 3-vector (element_extents<3>)");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "Cross: operand element_extents must match");
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "Cross: operand element_type must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr Cross(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        const working_type l0 = evalW<0>(l_, i);
        const working_type l1 = evalW<1>(l_, i);
        const working_type l2 = evalW<2>(l_, i);
        const working_type r0 = evalW<0>(r_, i);
        const working_type r1 = evalW<1>(r_, i);
        const working_type r2 = evalW<2>(r_, i);
        if constexpr (I == 0) {
            return l1 * r2 - l2 * r1;
        } else if constexpr (I == 1) {
            return l2 * r0 - l0 * r2;
        } else {
            return l0 * r1 - l1 * r0;
        }
    }
};

/**
 * @brief Unit-vector (normalized) view of a rank-1 expression: `eval<I>(i) =
 *        e.eval<I>(i) * e.rNorm()`. `.rNorm()` is evaluated
 *        SAMPLE-FREE — the intended usage is always on an already-materialized
 *        (Item-level) `E`, mirroring `dot`/`norm`/etc.'s own convention
 *        (`Expression.h`'s docstring).
 */
template<class E>
class UnitVector : public Expression<UnitVector<E>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 1, "UnitVector: E must be a rank-1 expression");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit UnitVector(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<I>(expr_, i) * toWorkingValue(expr_.rNorm());
    }
};

} // namespace detail
} // namespace aether
