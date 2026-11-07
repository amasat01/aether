// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Accum.h
 * @brief `BandAccum` / `BandAccumVec<N>` — running-sum fold accumulators
 *        over the banded carrier.
 *
 * State is a `Band`, one `float` error lane `c` and a term
 * counter; no dispatch, no heap. `addTerm`/`subTerm` fold with the
 * compensated add: the carrier's `addRaw` rounds three times on its residue
 * chain, and here those three roundings are made exact (`twoSum`) with their
 * errors gathered into `c`. The fold itself does not normalise: every `K`
 * terms (`renormFold_`) the error lane is folded into the tail limb exactly
 * and the carrier is normalised, which restarts `c` at the size of a single
 * rounding error. Between those points the limbs only grow (every `twoSum`
 * is exact), so after `k <= K` deferred terms
 * `|s.lo| <= (1+k) 2^-23 M`, `|s.tail| <= ~2k^2 2^-46 M` (M = the largest
 * `|acc| + |x|` seen) and the only roundings are the three adds into `c`:
 *
 *   `|err| <= 18 * K^3 * 2^-95 * sum_k(|acc_k| + |x_k|) + 2^-72 * |result|`
 *
 * which is linear in the term count n; with `K = 16` a plain sum stays
 * within a quarter ULP at 53 bits to n = 2^22 and within one ULP to
 * n = 2^24. The worst-case bound is that linear-in-n one. The 2^30-cancellation
 * figure (<= 0.01 ULP over 2^19 terms spanning 80 binades, for the sequential
 * and the tree fold) is a MEASURED result on a pinned corpus, asserted by
 * `BandLevers.BandAccumCancelClass*` (`tests/test_BandLevers_common.h`); it
 * is not a theorem. `K = 32` measured 0.03 ULP on it. This is THE banded
 * fold (no plain variant). Every compensated figure here is a figure of the
 * CARRIER (`normalized()`); a value delivered to a cell (`toBandedReal()`)
 * adds one more rounding of at most 0.5 ULP at 53 bits.
 * `s` is a carrier (normalised) after construction and after every
 * terminal (`combine`, `partial()`, `normalized()`, `toBandedReal()`), each
 * of which renormalises first. The renormalise itself is exact but is one
 * pass of order-free `twoSum`s: a window whose terms cancel down to the
 * tail level can leave its value in `lo` alone (`hi == 0`), which the
 * terminal's own pass repairs (`BandLevers.BandAccumCancelledWindow*`).
 * Read `s` through a terminal, never raw. `combine` merges two
 * accumulators (the shuffle/tree step), `c` lanes included.
 *
 * The gate `if (--n_ == 0)` costs an add and a branch per term. A fold loop
 * unrolled by `period` makes the counter a compile-time constant and the
 * gate disappears (measured: the fold then runs at the plain fold's wall);
 * emitters of fold-only loops key their unroll on `period`.
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
 * `normalized()` runs the renormalise-and-fold first, then adds `c`.
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
 * @brief Band running-sum accumulator lane: the carrier `s`, the float
 * error lane `c` of the compensated fold and the term counter (see the file
 * docstring). `K` is the renormalise period; the shipped
 * `BandAccum` is `K = 16`. Its worst-case error bound is linear in the term
 * count; the 2^30-cancellation figure is measured on a pinned corpus, not proven.
 */
/**
 * @brief The 16-byte scratch view of a `BandAccum`: the normalised carrier limbs plus the error lane. All-zero bytes
 * are the zero partial, so a zeroed plane of them is a plane of empty sums. A partial carries no term counter: the
 * accumulator rebuilt from it starts a fresh renormalise cadence. Combining partials in a fixed order with
 * `BandAccumK::combine` gives the same bits on every run.
 */
struct alignas(16) BandPartial {
    float hi;
    float lo;
    float tail;
    float c;
};

template<unsigned K>
struct BandAccumK {
    static_assert(K >= 1, "BandAccumK: the renormalise period must be positive");

    /// The renormalise period: the unroll factor at which a fold loop's gate folds away (file docstring).
    static constexpr unsigned period = K;

    Band  s{ 0.0f, 0.0f, 0.0f };
    float c = 0.0f;

    BandAccumK() = default;

private:
    unsigned n_ = K; // terms left before the next renormalise: a decrement whose flags the branch reads

    /// Fold `y` into `s` with the three residue roundings made exact, without normalising; returns their summed errors.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() float foldBand_(Band y)
    {
        float s_hi, e_hi;
        detail::twoSum(s.hi, y.hi, s_hi, e_hi);
        float s_lo, e_lo;
        detail::twoSum(s.lo, y.lo, s_lo, e_lo);
        float s_lo2, e_lo2;
        detail::twoSum(s_lo, e_hi, s_lo2, e_lo2);
        float r1, q1;
        detail::twoSum(e_lo, e_lo2, r1, q1);
        float r2, q2;
        detail::twoSum(r1, s.tail, r2, q2);
        float r3, q3;
        detail::twoSum(r2, y.tail, r3, q3);
        s = Band{ s_hi, s_lo2, r3 };
        return (q1 + q2) + q3;
    }

    /// Fold `c` into the tail exactly (the twoSum error is the new `c`) and normalise the carrier. The limbs are a
    /// deferred fold's, so the tail may exceed the residue of `hi + lo` and `hi` may have cancelled: every step is a
    /// twoSum (order-free, exact), where `normalizeSafe`'s `fast2Sum(l1, tail)` assumes an ordered tail.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void renormFold_()
    {
        float t, e;
        detail::twoSum(s.tail, c, t, e);
        c  = e;
        n_ = K;
        if (detail::bandHiIsSpecial(s.hi)) {
            s = detail::bandSpecialCarrier(s.hi);
            return;
        }
        float h1, l1, l2, t2, h, l4, m, tl;
        detail::twoSum(s.hi, s.lo, h1, l1);
        detail::twoSum(l1, t, l2, t2);
        detail::twoSum(h1, l2, h, l4);
        detail::twoSum(l4, t2, m, tl);
        s = Band{ h, m, tl };
    }

public:
    /// @brief Add a Band term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(Band t)
    {
        c += foldBand_(t);
        if (--n_ == 0)
            renormFold_();
    }
    /// @brief Subtract a Band term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(Band t) { addTerm(detail::neg(t)); }
    /// @brief Merge another accumulator into this one (the tree/shuffle step), error lanes included.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void combine(const BandAccumK& o)
    {
        BandAccumK t = o;
        renormFold_();
        t.renormFold_();
        c += t.c;
        addTerm(t.s);
    }

    /// @brief The scratch view: the carrier renormalised, then the four floats.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandPartial partial() const
    {
        BandAccumK t = *this;
        t.renormFold_();
        return BandPartial{ t.s.hi, t.s.lo, t.s.tail, t.c };
    }
    /// @brief The accumulator a partial describes (fresh cadence); `fromPartial(a.partial())` folds as `a` does.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandAccumK fromPartial(const BandPartial& p)
    {
        BandAccumK a;
        a.s = Band{ p.hi, p.lo, p.tail };
        a.c = p.c;
        return a;
    }

    /// @brief Terminal: the accumulated value (still Band), the error lane folded in.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalized() const
    {
        BandAccumK t = *this;
        t.renormFold_();
        return detail::add(t.s, Band{ t.c, 0.0f, 0.0f });
    }

    /**
     * @brief Host-only terminal: the accumulated value as a `double`, via
     * the exact limb sum (`detail::bandToDouble`, @see BandCell8.h). No
     * `double` reaches device code, so this function carries no device
     * qualifier, matching `bandToDouble` itself and `BandedReal::toDouble()`.
     *
     * `normalized()` folds the error lane in first.
     */
#ifndef __CUDACC_RTC__ // host-only: double interop
    [[nodiscard]] inline double toDouble() const { return detail::bandToDouble(normalized()); }
#endif // __CUDACC_RTC__

    /**
     * @brief The accumulated value packed into the storage value type: one
     * `BandedReal` pack (round-to-nearest-even), the carrier's own
     * assignment terminal. This is the egress a device-resident accumulator
     * (a kernel folding partial sums into an SoA plane) reaches for, since
     * `BandedReal(Band)` is `AETHER_DEVICEHOST()`.
     */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return BandedReal(normalized());
    }

    /**
     * @brief Storage identity — the three limbs and the error lane hold the same bits. A bit
     * comparison, deliberately not a numeric one (`+0.0f != -0.0f` here, a
     * NaN limb equals itself): the right answer for "is this buffer still the
     * fill value", the wrong one for arithmetic. @see BandCell8.h's
     * `operator==` for the same reasoning.
     */
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator==(
        const BandAccumK& a, const BandAccumK& b)
    {
        return detail::floatAsInt(a.s.hi) == detail::floatAsInt(b.s.hi)
            && detail::floatAsInt(a.s.lo) == detail::floatAsInt(b.s.lo)
            && detail::floatAsInt(a.s.tail) == detail::floatAsInt(b.s.tail)
            && detail::floatAsInt(a.c) == detail::floatAsInt(b.c);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator!=(
        const BandAccumK& a, const BandAccumK& b)
    {
        return !(a == b);
    }
};

/// @brief The shipped accumulator lane: renormalise period 16.
using BandAccum = BandAccumK<16>;

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
