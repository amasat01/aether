// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Arithmetic.h
 * @brief Component-wise arithmetic expression nodes: `Sum<L,R,subtract>`
 *        (`Diff = Sum<L,R,true>`) and `CWiseScale<E>` (scalar-by-value).
 *
 * Operand capture is `std::conditional_t<isLeaf, const E&, const E>` — a
 * leaf operand (`Item`/`View`) is captured by const reference (its storage
 * outlives the expression, which is a same-statement temporary); a
 * composite operand (another node) is captured by value (it is itself a
 * temporary chain of references — copying the (typically small) node keeps
 * every leaf reference inside it valid without extending any lifetime).
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/**
 * @brief `L op R`, component-wise: `sub == false` is `+`, `sub == true` is
 *        `-` (`Diff` below). L3 LOCKED shape: `Sum<L, R, bool subtract=false>`.
 */
template<aether_expression L, aether_expression R, bool subtract = false>
class Sum : public Expression<Sum<L, R, subtract>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "Sum: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "Sum: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr Sum(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    /**
     * @brief Returns the working carrier, not the storage scalar. The two
     *        operand reads go through `detail::evalW`, which converts a
     *        leaf read once and is the identity on a value already in the
     *        carrier. `working_type == element_type` for every native
     *        dtype.
     */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        if constexpr (subtract) {
            return evalW<Is...>(l_, i) - evalW<Is...>(r_, i);
        } else {
            return evalW<Is...>(l_, i) + evalW<Is...>(r_, i);
        }
    }
};

/** @brief `L - R`, component-wise: `Diff = Sum<L, R, true>`. */
template<aether_expression L, aether_expression R>
using Diff = Sum<L, R, true>;

/**
 * @brief `expr * scalar` / `scalar * expr` / `expr / scalar`, component-wise.
 *        `CWiseScale<E>` stores the scalar by value — no separate
 *        reciprocal-scale node; `operator/` in `Operations.h` pre-computes
 *        the reciprocal once and constructs a `CWiseScale` with it.
 */
template<aether_expression E>
class CWiseScale : public Expression<CWiseScale<E>, typename E::element_type> {
public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    /* The broadcast scalar stays the storage type: it is what the call site
     * hands over (`aether/expr/Operations.h`'s `operator*`/`operator/` take
     * `const typename E::element_type&`), and it is converted once per node
     * construction rather than once per component. */
    element_type factor_;
    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr CWiseScale(const E& expr, const element_type& factor)
        : factor_(factor)
        , expr_{ expr }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return toWorkingValue(factor_) * evalW<Is...>(expr_, i);
    }
};

} // namespace detail
} // namespace aether
