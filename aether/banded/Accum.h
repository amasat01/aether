// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Accum.h
 * @brief `BandAccum` / `BandAccumVec<N>` — running-sum fold accumulators
 *        over the banded carrier.
 *
 * State is a single `Band`; no other members, no dispatch, no heap.
 * `addTerm`/`subTerm` fold through the certified `detail::add`/
 * `detail::sub` (each of which already normalizes at its own exit), so
 * the invariant "the accumulator's state is a certified carrier" holds
 * after construction and after every mutator.
 *
 * @section egress_accum Two egress shapes
 * `toDouble()` is host-only, via `detail::bandToDouble` (`BandCell8.h`),
 * the exact limb sum `(hi+lo)+tail`; it carries no device qualifier
 * because a `double` may not appear in device code at all. `toBandedReal()`
 * is `AETHER_DEVICEHOST()`, packing (round-to-nearest-even) into a
 * `BandedReal`, the same assignment terminal a `Band` chain pays once, at
 * the pack — the egress a device-resident accumulator (a kernel folding
 * partial sums into an SoA plane) reaches for.
 *
 * `normalized()` re-normalizes `s` through `detail::normalizeSafe`; since
 * `s` is already a certified `Band` after every mutator, this is a value
 * no-op that costs nothing observable.
 *
 * @section vec BandAccumVec<N> — the vector-expression overload
 * `BandAccumVec<N>::%addTerm(Expression)`/`%subTerm(Expression)` unroll the
 * per-lane fold, strictly in lane order, via a private
 * member-function-template fold over `std::index_sequence` (called with
 * `std::make_index_sequence<N>`) rather than a device lambda, since this
 * tree's CUDA flags carry no `--expt-extended-lambda`. The comma-fold
 * sequences left to right, matching a hand-rolled per-lane loop.
 *
 * The vector-expression protocol threads `aether::Expression<E,Band>`
 * (`aether/expr/Expression.h`) and `.template eval<d>(SampleIndex)`,
 * aether's own canonical accessor; `aether::Item<Band, L>`
 * (`aether/view/Item.h`) is the element type the accumulator's vector
 * slice constructs against.
 */

#include <cstddef>
#include <utility>

#include "aether/banded/Band.h"
#include "aether/banded/BandCell8.h"
#include "aether/banded/BandedReal.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

namespace aether {
namespace banded {

/**
 * @brief Band running-sum accumulator lane.
 *
 * Chains terms through the order-free banded add (cancellation-safe at the
 * terminal). State is a single `Band` — no other members, no dispatch, no
 * heap.
 */
struct BandAccum {
    Band s{ 0.0f, 0.0f, 0.0f };

    BandAccum() = default;

    /// @brief Add a Band term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(Band t) { s = detail::add(s, t); }
    /// @brief Subtract a Band term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(Band t) { s = detail::sub(s, t); }

    /// @brief Terminal: the accumulated value (still Band).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalized() const
    {
        return detail::normalizeSafe(s);
    }

    /**
     * @brief Host-only terminal: the accumulated value as a `double`, via
     * the exact limb sum (`detail::bandToDouble`, @see BandCell8.h). No
     * `double` reaches device code, so this function carries no device
     * qualifier, matching `bandToDouble` itself and `BandedReal::toDouble()`.
     *
     * `s` is already a certified carrier on entry (every mutator normalizes
     * at its own exit), so summing the limbs needs no re-normalization
     * first.
     */
    [[nodiscard]] inline double toDouble() const { return detail::bandToDouble(s); }

    /**
     * @brief The accumulated value packed into the storage value type: one
     * `BandedReal` pack (round-to-nearest-even), the carrier's own
     * assignment terminal. This is the egress a device-resident accumulator
     * (a kernel folding partial sums into an SoA plane) reaches for, since
     * `BandedReal(Band)` is `AETHER_DEVICEHOST()`.
     */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return BandedReal(s);
    }

    /**
     * @brief Storage identity — the three limbs hold the same bits. A bit
     * comparison, deliberately not a numeric one (`+0.0f != -0.0f` here, a
     * NaN limb equals itself): the right answer for "is this buffer still the
     * fill value", the wrong one for arithmetic. @see BandCell8.h's
     * `operator==` for the same reasoning.
     */
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator==(
        const BandAccum& a, const BandAccum& b)
    {
        return detail::floatAsInt(a.s.hi) == detail::floatAsInt(b.s.hi)
            && detail::floatAsInt(a.s.lo) == detail::floatAsInt(b.s.lo)
            && detail::floatAsInt(a.s.tail) == detail::floatAsInt(b.s.tail);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator!=(
        const BandAccum& a, const BandAccum& b)
    {
        return !(a == b);
    }
};

/**
 * @brief N independent `BandAccum` lanes.
 *
 * @see the file docstring's "vec" section for why `addTerm`/`subTerm`'s
 * per-lane unroll uses a `std::index_sequence` fold, strictly in lane
 * order.
 */
template<int N>
struct BandAccumVec {
    BandAccum lanes[N];

    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandAccum& lane(int i) { return lanes[i]; }
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() const BandAccum& lane(int i) const
    {
        return lanes[i];
    }

private:
    template<class E, std::size_t... Is>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTermUnroll_(
        const E& e, std::index_sequence<Is...>)
    {
        ((lane(static_cast<int>(Is)).addTerm(e.template eval<Is>(SampleIndex::make(0)))), ...);
    }
    template<class E, std::size_t... Is>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTermUnroll_(
        const E& e, std::index_sequence<Is...>)
    {
        ((lane(static_cast<int>(Is)).subTerm(e.template eval<Is>(SampleIndex::make(0)))), ...);
    }

public:
    /// @brief Add a vector expression term across all N lanes, in lane order.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(const Expression<E, Band>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "BandAccumVec<N>::addTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "BandAccumVec<N>::addTerm: term vector dimension must match N lanes");
        addTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }

    /// @brief Subtract a vector expression term across all N lanes, in lane order.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(const Expression<E, Band>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "BandAccumVec<N>::subTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "BandAccumVec<N>::subTerm: term vector dimension must match N lanes");
        subTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }
};

} // namespace banded
} // namespace aether
