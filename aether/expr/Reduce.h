// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Reduce.h
 * @brief Compile-time-unrolled reductions: the `detail::` recursive-reduce
 *        machinery + functors that back `Expression`'s `dot`/`norm`/
 *        `squaredNorm`/`cubedNorm`/`rNorm`/`rSquaredNorm`/`rCubedNorm`/
 *        `maxNorm`/`sum` member functions (declared+defined inline in
 *        `aether/expr/Expression.h` — see that file's docstring for why
 *        the whole member-function surface stays inline), plus the free
 *        `aether::isFinite(expr)`.
 *
 * The recursion walks the component index descending (dim = N-1 down to 0,
 * terminating at 0) — i.e. for N=3 a sum reduce unrolls to
 * `f(get<2>) + (f(get<1>) + f(get<0>))`, right-associated. This
 * associativity is deliberate and fixed (not just the same set of terms),
 * since floating point addition is not associative and a different fold
 * order is a different, merely-close, answer.
 *
 * The fold runs in the working carrier (`aether/dtype/WorkingType.h`) and
 * its callers (`Expression.h`'s norm/dot/sum surface) pay the single encode
 * back to `element_type` at their own return. Each leaf read is funnelled
 * through `detail::toWorkingValue`, which is the identity for a value
 * already in the carrier and for every native dtype - so the associativity
 * pinned above is untouched and so is `double`/`float` codegen.
 *
 * Depends on `aether/math/math.h`'s dispatch header for `sqrt`/`isfinite`
 * — not on `aether/expr/Expression.h` — so `Expression.h` can safely
 * `#include` this header with no cycle (`aether/math/` has no dependency
 * back on `expr/`); its inline member function bodies call straight into
 * `detail::GenericReduce`/`GenericBinaryReduce`.
 */

#include <cstddef>

#include "aether/dtype/WorkingType.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace detail {

// ---------------------------------------------------------------------------
// Reduce/transform functors (unary transform + binary reduce operators).
// ---------------------------------------------------------------------------

template<class T>
struct SumOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a, T b) const { return a + b; }
};
template<class T>
struct TimesOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a, T b) const { return a * b; }
};
/**
 * @brief `max(a, b)`. The ORDER RELATION is not universal.
 *
 * The reduce now folds in the WORKING carrier, and a working carrier need not
 * carry comparison operators at all - `aether::banded::Band` deliberately does
 * not (its sign lives in the VALUE of three signed limbs, and a lexicographic
 * limb compare is WRONG; see `banded::detail::fmax`'s own counterexample). The
 * native arm below is the previous expression, character for character, so
 * `double`/`float` codegen cannot move; a carrier without `>` routes to the
 * CERTIFIED facade entry instead, which is exactly the entry that exists for
 * this call site ("an error-controller's `(error/desired).maxNorm()` fails at
 * exactly this point without it", `BandedRealOps.h`).
 */
template<class T>
struct MaxOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a, T b) const
    {
        if constexpr (requires(T x, T y) { x > y ? x : y; }) {
            return a > b ? a : b;
        } else {
            return math::fmax(a, b);
        }
    }
};
struct LogicalAndOp {
    AETHER_DEVICEHOST() constexpr bool operator()(bool a, bool b) const { return a && b; }
};

template<class T>
struct SelfOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a) const { return a; }
};
template<class T>
struct SquareOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a) const { return a * a; }
};
template<class T>
struct AbsOp {
    /* `T(-a)`, not `-a`. For `float`/`double` the two are the
     * same expression and the same code. For an EMULATED scalar whose unary
     * minus returns the WORKING carrier rather than the storage type (which is
     * the whole "no pack per op" doctrine — `aether::banded::BandedReal`'s
     * `operator-` returns `Band`), the two arms of the conditional then have
     * DIFFERENT types that convert both ways, and the conditional is ambiguous:
     *     the conditional below: operands to '?:' have different types
     *     'aether::banded::Band' and 'aether::banded::BandedReal'
     * Naming the result type once collapses that. This is the SMALLEST change
     * that lets `maxNorm()` instantiate on a storage/working split scalar at
     * all, and it keeps the one-element_type-per-tree protocol exactly as it
     * was — the pack it forces at this node is a different concern, not
     * this line's.
     *
     * And `T{}`, not `T{ 0 }`. Value-initialisation is zero for every arithmetic
     * type, so nothing about `float`/`double` moves — but `T{ 0 }` on an
     * emulated scalar means "construct from the literal 0", and a carrier whose
     * ingest from a native scalar is deliberately HOST-ONLY (as `BandedReal`'s
     * is, so no `double` can reach device code by accident) then makes this
     * `AETHER_DEVICEHOST()` body illegal on the device pass:
     *     the body below: calling a __host__ function
     *     ("aether::banded::BandedReal::BandedReal(double)") from a __device__
     *     function ("AbsOp<BandedReal>::operator ") is not allowed
     * `T{}` asks for the type's own zero instead of for a conversion, which is
     * what this comparison actually wants and what every scalar can answer on
     * both arms. */
    AETHER_DEVICEHOST() constexpr T operator()(T a) const
    {
        if constexpr (requires(T x) { x < T{} ? T(-x) : x; }) {
            return a < T{} ? T(-a) : a;
        } else {
            /* `T` is now the WORKING carrier, which need not order or
             * value-initialise (see `MaxOp` above). The certified facade entry
             * is the answer for exactly the carriers that land here. */
            return math::abs(a);
        }
    }
};
template<class T>
struct IsFiniteOp {
    AETHER_DEVICEHOST() constexpr bool operator()(T a) const { return math::isfinite(a); }
};

// ---------------------------------------------------------------------------
// Unary recursive reduce: fold Transf(e.eval<K>(i)) over K = Dim..0 with
// Reduce, DESCENDING (see the file docstring for the associativity rationale).
// ---------------------------------------------------------------------------

template<std::size_t Dim, class E, class ReduceOp, class TransfOp, class OutT>
struct RecursiveReduce {
    AETHER_DEVICEHOST() static constexpr OutT eval(const E& e, const SampleIndex& i)
    {
        return ReduceOp{}(TransfOp{}(toWorkingValue(e.template eval<Dim>(i))),
            RecursiveReduce<Dim - 1, E, ReduceOp, TransfOp, OutT>::eval(e, i));
    }
};
template<class E, class ReduceOp, class TransfOp, class OutT>
struct RecursiveReduce<0, E, ReduceOp, TransfOp, OutT> {
    AETHER_DEVICEHOST() static constexpr OutT eval(const E& e, const SampleIndex& i)
    {
        return TransfOp{}(toWorkingValue(e.template eval<0>(i)));
    }
};

/** @brief `N`-component unary reduce entry point (`N = element_extents::static_extent(0)`). */
template<std::size_t N, class E, class ReduceOp, class TransfOp, class OutT>
struct GenericReduce {
    AETHER_DEVICEHOST() static constexpr OutT eval(const E& e, const SampleIndex& i)
    {
        return RecursiveReduce<N - 1, E, ReduceOp, TransfOp, OutT>::eval(e, i);
    }
};

// ---------------------------------------------------------------------------
// Binary recursive reduce (dot product): fold Transf(a.eval<K>(i), b.eval<K>(i))
// over K = Dim..0 with Reduce, DESCENDING.
// ---------------------------------------------------------------------------

template<std::size_t Dim, class L, class R, class ReduceOp, class TransfOp, class OutT>
struct BinaryRecursiveReduce {
    AETHER_DEVICEHOST() static constexpr OutT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return ReduceOp{}(TransfOp{}(toWorkingValue(a.template eval<Dim>(i)), toWorkingValue(b.template eval<Dim>(i))),
            BinaryRecursiveReduce<Dim - 1, L, R, ReduceOp, TransfOp, OutT>::eval(a, b, i));
    }
};
template<class L, class R, class ReduceOp, class TransfOp, class OutT>
struct BinaryRecursiveReduce<0, L, R, ReduceOp, TransfOp, OutT> {
    AETHER_DEVICEHOST() static constexpr OutT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return TransfOp{}(toWorkingValue(a.template eval<0>(i)), toWorkingValue(b.template eval<0>(i)));
    }
};

template<std::size_t N, class L, class R, class ReduceOp, class TransfOp, class OutT>
struct GenericBinaryReduce {
    AETHER_DEVICEHOST() static constexpr OutT eval(const L& a, const R& b, const SampleIndex& i)
    {
        return BinaryRecursiveReduce<N - 1, L, R, ReduceOp, TransfOp, OutT>::eval(a, b, i);
    }
};

} // namespace detail


// Public functor names (EAGLE port): the reduction operators are part of
// aether's consumer surface — the call shape is the functor call `SumOp<T>{}(a, b)`.
using detail::SumOp;
using detail::TimesOp;
using detail::MaxOp;
using detail::LogicalAndOp;

/**
 * @brief `true` iff every component of `expr` is finite (rank-1). Free
 *        function (not a member — one deliberate asymmetry vs. the
 *        rest of the reduction surface). Sample-free (`SampleIndex::make(0)`),
 *        same convention as the member reductions in `Expression.h`.
 */
template<class E>
AETHER_DEVICEHOST() constexpr bool isFinite(const E& expr)
{
    static_assert(E::element_extents::Rank == 1, "isFinite: only rank-1 expressions are supported");
    return detail::GenericReduce<E::element_extents::static_extent(0), E, detail::LogicalAndOp,
        detail::IsFiniteOp<working_type_t<typename E::element_type>>, bool>::eval(expr, SampleIndex::make(0));
}

} // namespace aether
