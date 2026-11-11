// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Inverse.h
 * @brief `aether::detail::Inverse<E>`: the closed-form matrix inverse node
 *        backing `Expression::inverse()` (declared + defined inline in
 *        `aether/expr/Expression.h`; see that file's docstring for why
 *        the geometric/structural member surface lives there while the
 *        node definitions live in `expr/nodes/`).
 *
 * Square rank-2, 2x2 or 3x3 only (class-body `static_assert`s below) —
 * general LU/solve is explicitly out of scope; a non-2x2/3x3 shape
 * hard-errors with a message pointing at the deferral. Original aether
 * design, same footing as the elementwise ops
 * (`aether/expr/nodes/Elementwise.h`).
 *
 * `eval<R,C>(i)` independently recomputes the full source matrix (all 4 or 9
 * components) and its determinant from `expr_` at the `i` it is actually
 * called with — matches `Cross`'s own per-component recompute convention
 * (`aether/expr/nodes/Geometric.h`'s docstring: "each independently
 * recomputing ... from the operands"), which is what makes `Inverse` safe to
 * assign into a batched destination (a per-sample `View`), unlike
 * `Expression::trace()`/`det()` (eager, hardcoded `SampleIndex::make(0)`)
 * — `Inverse` deliberately does not reuse those two members internally
 * for exactly this reason.
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/**
 * @brief Closed-form inverse of a square 2x2 or 3x3 rank-2 expression —
 *        `eval<R,C>(i) = adjugate(expr)(R,C) / det(expr)`, both re-derived
 *        from `expr_`'s own components at `i` on every call. `adjugate`
 *        here is the standard cofactor-transpose (verified by direct
 *        cofactor-expansion derivation, not copied from any external
 *        source): for 2x2, `[d -b; -c a] / det`; for 3x3, the nine
 *        `C_{jk}^T / det` cofactor terms below.
 */
template<class E>
class Inverse : public Expression<Inverse<E>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 2, "inverse: E must be a rank-2 (matrix) expression");
    static_assert(E::element_extents::static_extent(0) == E::element_extents::static_extent(1), "inverse: E must be square");
    static_assert(E::element_extents::static_extent(0) == 2 || E::element_extents::static_extent(0) == 3,
        "inverse: only 2x2 and 3x3 closed forms are implemented (general N would need LU/solve, not implemented here)");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;
    static constexpr std::size_t N = element_extents::static_extent(0);

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Inverse(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t R, std::size_t C>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        static_assert(R < N && C < N, "Inverse::eval: index out of range");
        if constexpr (N == 2) {
            const working_type a = evalW<0, 0>(expr_, i);
            const working_type b = evalW<0, 1>(expr_, i);
            const working_type c = evalW<1, 0>(expr_, i);
            const working_type d = evalW<1, 1>(expr_, i);
            const working_type det = a * d - b * c;
            if constexpr (R == 0 && C == 0) {
                return d / det;
            } else if constexpr (R == 0 && C == 1) {
                return -b / det;
            } else if constexpr (R == 1 && C == 0) {
                return -c / det;
            } else {
                return a / det;
            }
        } else {
            const working_type a = evalW<0, 0>(expr_, i);
            const working_type b = evalW<0, 1>(expr_, i);
            const working_type c = evalW<0, 2>(expr_, i);
            const working_type d = evalW<1, 0>(expr_, i);
            const working_type e = evalW<1, 1>(expr_, i);
            const working_type f = evalW<1, 2>(expr_, i);
            const working_type g = evalW<2, 0>(expr_, i);
            const working_type h = evalW<2, 1>(expr_, i);
            const working_type k = evalW<2, 2>(expr_, i);
            const working_type det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
            if constexpr (R == 0 && C == 0) {
                return (e * k - f * h) / det;
            } else if constexpr (R == 0 && C == 1) {
                return (c * h - b * k) / det;
            } else if constexpr (R == 0 && C == 2) {
                return (b * f - c * e) / det;
            } else if constexpr (R == 1 && C == 0) {
                return (f * g - d * k) / det;
            } else if constexpr (R == 1 && C == 1) {
                return (a * k - c * g) / det;
            } else if constexpr (R == 1 && C == 2) {
                return (c * d - a * f) / det;
            } else if constexpr (R == 2 && C == 0) {
                return (d * h - e * g) / det;
            } else if constexpr (R == 2 && C == 1) {
                return (b * g - a * h) / det;
            } else {
                return (a * e - b * d) / det;
            }
        }
    }
};

} // namespace detail
} // namespace aether
