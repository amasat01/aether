// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Structural.h
 * @brief `aether::detail::Segment<E,Off,Len>`: a lazy, read-only
 *        contiguous sub-range view over a rank-1 expression — backs
 *        `Expression::segment<Off,Len>()`/`head<N>()`/`tail<N>()` (declared
 *        + defined inline in `aether/expr/Expression.h`; see that file's
 *        docstring for why the whole member surface lives there).
 *
 * Read-only, deliberately: `Segment` is a plain lazy expression node
 * (`isLeaf = false`), same shape as `Cross`/`UnitVector`/`QuatMul` — no
 * write-back path. A future addition may add a writable counterpart; this
 * type is not it.
 *
 * Also carries the matrix structural counterparts — `Transpose<E>`
 * (rank-2 -> rank-2), `Row<E,R>`/`Col<E,C>` (rank-2 -> rank-1), `Block<E,
 * R0,C0,BR,BC>` (rank-2 -> rank-2) — backing `Expression::transpose()`/
 * `row<R>`/`col<C>`/`block<R0,C0,BR,BC>`. Same read-only convention
 * as `Segment`, with the same swapped-index/offset-index access shape,
 * over aether's unified rank-generic `eval<Is...>(SampleIndex)` protocol
 * (`element_extents`'s own rank carries the vector-vs-matrix distinction,
 * so one `Expression<Derived,T>` base needs no separate split).
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
 * @brief Read-only view of `Len` consecutive elements of rank-1 expression
 *        `E`, starting at static offset `Off`: `eval<I>(i) = expr.eval<Off+I>(i)`.
 */
template<class E, std::size_t Off, std::size_t Len>
class Segment : public Expression<Segment<E, Off, Len>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 1, "Segment: E must be a rank-1 expression");
    static_assert(Off + Len <= E::element_extents::static_extent(0), "Segment: [Off, Off+Len) out of range");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<Len>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Segment(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<Off + I>(expr_, i);
    }
};

/**
 * @brief Read-only transpose of a rank-2 expression —
 *        `eval<R,C>(i) = expr.eval<C,R>(i)`.
 */
template<class E>
class Transpose : public Expression<Transpose<E>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 2, "Transpose: E must be a rank-2 (matrix) expression");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<E::element_extents::static_extent(1), E::element_extents::static_extent(0)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Transpose(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t R, std::size_t C>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<C, R>(expr_, i);
    }
};

/**
 * @brief Read-only view of row `R` of a rank-2 expression — a
 *        rank-1 (vector) expression result so it composes with the whole
 *        vector algebra (`A.row<0>.dot(x)`).
 */
template<class E, std::size_t R>
class Row : public Expression<Row<E, R>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 2, "Row: E must be a rank-2 (matrix) expression");
    static_assert(R < E::element_extents::static_extent(0), "Row: row index out of range");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<E::element_extents::static_extent(1)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Row(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t C>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<R, C>(expr_, i);
    }
};

/**
 * @brief Read-only view of column `C` of a rank-2 expression —
 *        a rank-1 (vector) expression result.
 */
template<class E, std::size_t C>
class Col : public Expression<Col<E, C>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 2, "Col: E must be a rank-2 (matrix) expression");
    static_assert(C < E::element_extents::static_extent(1), "Col: column index out of range");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<E::element_extents::static_extent(0)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Col(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t R>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<R, C>(expr_, i);
    }
};

/**
 * @brief Read-only view of a `BR x BC` block of a rank-2
 *        expression starting at `(R0, C0)` — rank-2 expression result.
 */
template<class E, std::size_t R0, std::size_t C0, std::size_t BR, std::size_t BC>
class Block : public Expression<Block<E, R0, C0, BR, BC>, typename E::element_type> {
    static_assert(E::element_extents::Rank == 2, "Block: E must be a rank-2 (matrix) expression");
    static_assert(R0 + BR <= E::element_extents::static_extent(0), "Block: row range out of bounds");
    static_assert(C0 + BC <= E::element_extents::static_extent(1), "Block: column range out of bounds");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<BR, BC>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit Block(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t R, std::size_t C>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<R0 + R, C0 + C>(expr_, i);
    }
};

} // namespace detail
} // namespace aether
