// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Rt.h
 * @brief `aether::rt`: rolled-loop twins of the static reductions and
 *        contractions, for one sample inside a kernel or a host loop.
 *
 * The static forms (`a[i].get().dot(b[i].get())`, `y[i] = M * v`) unroll every
 * element, so a long fold keeps about n values live: on sm_61 a 144-long dot
 * takes 32 registers and a 16x144 `M * v` spills. The twins walk the element
 * index in a loop kept rolled (`AETHER_NO_UNROLL()`) and read each element when
 * needed: 17 and 21 registers on the same shapes, no spill.
 *
 * Each twin returns the same bits as its static form: same functors, same
 * order (`acc = T(x0)`, then `acc = Op(T(x_k), acc)` for ascending `k`;
 * products `acc = a_k * b_k + acc`), same working carrier, one encode per
 * result. FMA contraction is left to the TU flags. The k = 1 step is peeled
 * and written as the static fold writes it, so a contracting compiler fuses
 * the same product: nvcc sm_61 fuses `a0*b0` over registers (`Item`, the fused
 * `=` caches) and `a1*b1` over views (`MatVecRowDot`, `+=`/`-=` `MatMatDot`);
 * each twin mirrors its own static path. This holds where the compiler
 * contracts per statement (nvcc). GCC -O3 with FMA may vectorise the static
 * fold and fuse differently, so compare on the host with `-ffp-contract=off`.
 *
 * Operands are the per-sample proxies `view[i]`. An `Item` is accepted as a
 * host convenience; on the device a runtime-indexed `Item` lives in local
 * memory, so pass the proxy. Shapes are static types (a mismatch is a compile
 * error). Outputs must not alias the operands. Unlike `eval/Runtime.h`, there
 * are no descriptors, launch or checks.
 */

#include <cstddef>
#include <type_traits>

#include "aether/dtype/WorkingType.h"
#include "aether/expr/Assign.h"
#include "aether/expr/Reduce.h"
#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace rt {
namespace detail {

using aether::detail::AssignOp;
using aether::detail::toWorkingValue;

template<class X>
using elem_t = std::remove_cv_t<typename std::remove_cvref_t<X>::element_type>;
template<class X>
using ext_t = typename std::remove_cvref_t<X>::element_extents;

/** @brief Element `ks...` of one sample: `x.at(ks...)` on a proxy, `x(ks...)` on an `Item`. */
template<class X, class... Ks>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr decltype(auto) at(X&& x, Ks... ks)
{
    if constexpr (requires { x.at(ks...); }) {
        return x.at(ks...);
    } else {
        return x(ks...);
    }
}

/** @brief `acc = T(x0); acc = R(T(x_k), acc)`, k ascending: `GenericReduce`'s order. */
template<class R, class T, class Out, class A>
AETHER_DEVICEHOST() constexpr Out reduce(const A& a)
{
    static_assert(ext_t<A>::Rank == 1, "rt: only rank-1 operands are supported");
    constexpr std::size_t N = ext_t<A>::static_extent(0);
    Out acc = T{}(toWorkingValue(at(a, std::size_t{ 0 })));
    if constexpr (N > 1) {
        acc = R{}(T{}(toWorkingValue(at(a, std::size_t{ 1 }))), acc);
    }
    AETHER_NO_UNROLL()
    for (std::size_t k = 2; k < N; ++k) {
        acc = R{}(T{}(toWorkingValue(at(a, k))), acc);
    }
    return acc;
}

template<class A>
AETHER_DEVICEHOST() constexpr working_type_t<elem_t<A>> squaredNormW(const A& a)
{
    using W = working_type_t<elem_t<A>>;
    return reduce<aether::detail::SumOp<W>, aether::detail::SquareOp<W>, W>(a);
}

/**
 * @brief `out (=|+=|-=) m * v`, rows ascending, `acc = m(r,0)*v(0); acc = m(r,k)*v(k) + acc`.
 *        `Registers` picks the contraction of the first two terms the static form takes
 *        for its operands: over views `m(r,1)*v(1) + m(r,0)*v(0)`, over `Item`s (register
 *        caches) `m(r,1)*v(1) + acc` with `acc = m(r,0)*v(0)` rounded first, as `matMat` does.
 */
template<AssignOp Op, bool Registers = false, class Out, class M, class V>
AETHER_DEVICEHOST() constexpr void matVec(Out&& out, const M& m, const V& v)
{
    static_assert(ext_t<M>::Rank == 2 && ext_t<V>::Rank == 1, "rt::matVec: m must be rank-2, v rank-1");
    static_assert(ext_t<M>::static_extent(1) == ext_t<V>::static_extent(0), "rt::matVec: m's column count must equal v's length");
    static_assert(std::is_same_v<ext_t<Out>, extents<ext_t<M>::static_extent(0)>>, "rt::matVec: out must have m's row count");
    static_assert(std::is_same_v<elem_t<M>, elem_t<V>> && std::is_same_v<elem_t<M>, elem_t<Out>>, "rt::matVec: element_type must match");
    using W                    = working_type_t<elem_t<M>>;
    constexpr std::size_t Rows = ext_t<M>::static_extent(0);
    constexpr std::size_t K    = ext_t<M>::static_extent(1);
    AETHER_NO_UNROLL()
    for (std::size_t r = 0; r < Rows; ++r) {
        W acc = toWorkingValue(at(m, r, std::size_t{ 0 })) * toWorkingValue(at(v, std::size_t{ 0 }));
        if constexpr (K > 1 && Registers) {
            acc = toWorkingValue(at(m, r, std::size_t{ 1 })) * toWorkingValue(at(v, std::size_t{ 1 })) + acc;
        } else if constexpr (K > 1) {
            acc = toWorkingValue(at(m, r, std::size_t{ 1 })) * toWorkingValue(at(v, std::size_t{ 1 }))
                + toWorkingValue(at(m, r, std::size_t{ 0 })) * toWorkingValue(at(v, std::size_t{ 0 }));
        }
        AETHER_NO_UNROLL()
        for (std::size_t k = 2; k < K; ++k) {
            acc = toWorkingValue(at(m, r, k)) * toWorkingValue(at(v, k)) + acc;
        }
        aether::detail::storeWorking<Op>(at(out, r), acc);
    }
}

/** @brief `out (=|+=|-=) l * r`, (row, col) row-major, `acc = l(i,0)*r(0,j); acc = l(i,k)*r(k,j) + acc`. */
template<AssignOp Op, class Out, class L, class R>
AETHER_DEVICEHOST() constexpr void matMat(Out&& out, const L& l, const R& r)
{
    static_assert(ext_t<L>::Rank == 2 && ext_t<R>::Rank == 2, "rt::matMat: operands must be rank-2");
    static_assert(ext_t<L>::static_extent(1) == ext_t<R>::static_extent(0), "rt::matMat: l's column count must equal r's row count");
    static_assert(std::is_same_v<ext_t<Out>, extents<ext_t<L>::static_extent(0), ext_t<R>::static_extent(1)>>,
        "rt::matMat: out must be l's rows x r's columns");
    static_assert(std::is_same_v<elem_t<L>, elem_t<R>> && std::is_same_v<elem_t<L>, elem_t<Out>>, "rt::matMat: element_type must match");
    using W                    = working_type_t<elem_t<L>>;
    constexpr std::size_t Rows = ext_t<L>::static_extent(0);
    constexpr std::size_t K    = ext_t<L>::static_extent(1);
    constexpr std::size_t Cols = ext_t<R>::static_extent(1);
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < Rows; ++i) {
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < Cols; ++j) {
            W acc = toWorkingValue(at(l, i, std::size_t{ 0 })) * toWorkingValue(at(r, std::size_t{ 0 }, j));
            if constexpr (K > 1 && Op == AssignOp::Set) {
                // The static `=` folds register caches of l and r (Product.h's fused path).
                acc = toWorkingValue(at(l, i, std::size_t{ 1 })) * toWorkingValue(at(r, std::size_t{ 1 }, j)) + acc;
            } else if constexpr (K > 1) {
                // The static `+=`/`-=` folds the views themselves (`MatMatDot`, one expression).
                acc = toWorkingValue(at(l, i, std::size_t{ 1 })) * toWorkingValue(at(r, std::size_t{ 1 }, j))
                    + toWorkingValue(at(l, i, std::size_t{ 0 })) * toWorkingValue(at(r, std::size_t{ 0 }, j));
            }
            AETHER_NO_UNROLL()
            for (std::size_t k = 2; k < K; ++k) {
                acc = toWorkingValue(at(l, i, k)) * toWorkingValue(at(r, k, j)) + acc;
            }
            aether::detail::storeWorking<Op>(at(out, i, j), acc);
        }
    }
}

} // namespace detail

/**
 * @brief A rank-2 operand read transposed: `at(r, c)` is `x.at(c, r)`. Holds `x` by
 *        reference, so it lives within the statement that names it.
 */
template<class X>
class Transposed {
public:
    using element_type    = typename std::remove_cvref_t<X>::element_type;
    using element_extents = extents<detail::ext_t<X>::static_extent(1), detail::ext_t<X>::static_extent(0)>;

    AETHER_DEVICEHOST() constexpr explicit Transposed(const X& x)
        : x_(x)
    {
        static_assert(detail::ext_t<X>::Rank == 2, "rt::transposed: only rank-2 operands are supported");
    }

    AETHER_DEVICEHOST() constexpr decltype(auto) at(std::size_t r, std::size_t c) const
    {
        return detail::at(x_, c, r);
    }

private:
    const X& x_;
};

/** @brief `x` read as its transpose, for an operand of `matVec`/`matMat`: the twin of `x.transpose()`. */
template<class X>
AETHER_DEVICEHOST() constexpr Transposed<X> transposed(const X& x)
{
    return Transposed<X>(x);
}

/**
 * @brief `out = x` element by element, rolled: twin of `view[i] = x` for an operand `x` (an `Item`)
 *        of the same extents. The static assignment unrolls one load and one store per element,
 *        and a wide `x` keeps a register for each; the loop keeps one.
 */
template<class Out, class X>
AETHER_DEVICEHOST() constexpr void store(Out&& out, const X& x)
{
    static_assert(detail::ext_t<X>::Rank == 1 || detail::ext_t<X>::Rank == 2, "rt::store: only rank-1 and rank-2 operands are supported");
    static_assert(std::is_same_v<detail::ext_t<Out>, detail::ext_t<X>>, "rt::store: operand element_extents must match");
    static_assert(std::is_same_v<detail::elem_t<Out>, detail::elem_t<X>>, "rt::store: operand element_type must match");
    using detail::at;
    if constexpr (detail::ext_t<X>::Rank == 1) {
        constexpr std::size_t N = detail::ext_t<X>::static_extent(0);
        AETHER_NO_UNROLL()
        for (std::size_t k = 0; k < N; ++k) {
            at(out, k) = at(x, k);
        }
    } else {
        constexpr std::size_t R = detail::ext_t<X>::static_extent(0);
        constexpr std::size_t C = detail::ext_t<X>::static_extent(1);
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < R; ++i) {
            AETHER_NO_UNROLL()
            for (std::size_t j = 0; j < C; ++j) {
                at(out, i, j) = at(x, i, j);
            }
        }
    }
}

/** @brief Dot product of two rank-1 operands; twin of `Expression::dot`. */
template<class A, class B>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> dot(const A& a, const B& b)
{
    using E = detail::elem_t<A>;
    using W = working_type_t<E>;
    static_assert(detail::ext_t<A>::Rank == 1, "rt::dot: only rank-1 operands are supported");
    static_assert(std::is_same_v<detail::ext_t<A>, detail::ext_t<B>>, "rt::dot: operand element_extents must match");
    static_assert(std::is_same_v<E, detail::elem_t<B>>, "rt::dot: operand element_type must match");
    using aether::detail::SumOp;
    using aether::detail::TimesOp;
    using detail::at;
    using detail::toWorkingValue;
    constexpr std::size_t N = detail::ext_t<A>::static_extent(0);
    W acc = TimesOp<W>{}(toWorkingValue(at(a, std::size_t{ 0 })), toWorkingValue(at(b, std::size_t{ 0 })));
    if constexpr (N > 1) {
        acc = SumOp<W>{}(TimesOp<W>{}(toWorkingValue(at(a, std::size_t{ 1 })), toWorkingValue(at(b, std::size_t{ 1 }))), acc);
    }
    AETHER_NO_UNROLL()
    for (std::size_t k = 2; k < N; ++k) {
        acc = SumOp<W>{}(TimesOp<W>{}(toWorkingValue(at(a, k)), toWorkingValue(at(b, k))), acc);
    }
    return aether::detail::fromWorkingValue<E>(acc);
}

/** @brief Sum of the components; twin of `Expression::sum`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> sum(const A& a)
{
    using W = working_type_t<detail::elem_t<A>>;
    return aether::detail::fromWorkingValue<detail::elem_t<A>>(
        detail::reduce<aether::detail::SumOp<W>, aether::detail::SelfOp<W>, W>(a));
}

/** @brief Squared L2 norm; twin of `Expression::squaredNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> squaredNorm(const A& a)
{
    return aether::detail::fromWorkingValue<detail::elem_t<A>>(detail::squaredNormW(a));
}

/** @brief L2 norm; twin of `Expression::norm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> norm(const A& a)
{
    return aether::detail::fromWorkingValue<detail::elem_t<A>>(math::sqrt(detail::squaredNormW(a)));
}

/** @brief `1 / norm` (division, not rsqrt); twin of `Expression::rNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> rNorm(const A& a)
{
    using E = detail::elem_t<A>;
    return aether::detail::fromWorkingValue<E>(E{ 1 } / math::sqrt(detail::squaredNormW(a)));
}

/** @brief `1 / squaredNorm`; twin of `Expression::rSquaredNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> rSquaredNorm(const A& a)
{
    using E = detail::elem_t<A>;
    return aether::detail::fromWorkingValue<E>(E{ 1 } / detail::squaredNormW(a));
}

/** @brief `norm^3`; twin of `Expression::cubedNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> cubedNorm(const A& a)
{
    const auto s = detail::squaredNormW(a);
    return aether::detail::fromWorkingValue<detail::elem_t<A>>(s * math::sqrt(s));
}

/** @brief `1 / norm^3`; twin of `Expression::rCubedNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> rCubedNorm(const A& a)
{
    using E      = detail::elem_t<A>;
    const auto s = detail::squaredNormW(a);
    return aether::detail::fromWorkingValue<E>(E{ 1 } / (s * math::sqrt(s)));
}

/** @brief Max-abs norm (same NaN behaviour, `MaxOp` operand order kept); twin of `Expression::maxNorm`. */
template<class A>
AETHER_DEVICEHOST() constexpr detail::elem_t<A> maxNorm(const A& a)
{
    using W = working_type_t<detail::elem_t<A>>;
    return aether::detail::fromWorkingValue<detail::elem_t<A>>(
        detail::reduce<aether::detail::MaxOp<W>, aether::detail::AbsOp<W>, W>(a));
}

/** @brief `true` iff every component is finite; twin of `aether::isFinite`. */
template<class A>
AETHER_DEVICEHOST() constexpr bool isFinite(const A& a)
{
    return detail::reduce<aether::detail::LogicalAndOp, aether::detail::IsFiniteOp<working_type_t<detail::elem_t<A>>>, bool>(a);
}

/** @brief Frobenius inner product of two rank-2 operands, row-major; twin of `Expression::matDot`. */
template<class L, class R>
AETHER_DEVICEHOST() constexpr detail::elem_t<L> matDot(const L& l, const R& r)
{
    using E = detail::elem_t<L>;
    using W = working_type_t<E>;
    static_assert(detail::ext_t<L>::Rank == 2, "rt::matDot: only rank-2 operands are supported");
    static_assert(std::is_same_v<detail::ext_t<L>, detail::ext_t<R>>, "rt::matDot: operand element_extents must match");
    static_assert(std::is_same_v<E, detail::elem_t<R>>, "rt::matDot: operand element_type must match");
    using detail::at;
    using detail::toWorkingValue;
    constexpr std::size_t Cols = detail::ext_t<L>::static_extent(1);
    constexpr std::size_t N    = detail::ext_t<L>::static_extent(0) * Cols;
    W acc = toWorkingValue(at(l, std::size_t{ 0 }, std::size_t{ 0 })) * toWorkingValue(at(r, std::size_t{ 0 }, std::size_t{ 0 }));
    if constexpr (N > 1) {
        acc = toWorkingValue(at(l, std::size_t{ 1 } / Cols, std::size_t{ 1 } % Cols))
                * toWorkingValue(at(r, std::size_t{ 1 } / Cols, std::size_t{ 1 } % Cols))
            + acc;
    }
    AETHER_NO_UNROLL()
    for (std::size_t k = 2; k < N; ++k) {
        acc = toWorkingValue(at(l, k / Cols, k % Cols)) * toWorkingValue(at(r, k / Cols, k % Cols)) + acc;
    }
    return aether::detail::fromWorkingValue<E>(acc);
}

/** @brief `out = m * v`; twin of `view[i] = M * v`. */
template<class Out, class M, class V>
AETHER_DEVICEHOST() constexpr void matVec(Out&& out, const M& m, const V& v)
{
    detail::matVec<detail::AssignOp::Set>(out, m, v);
}
/** @brief `out = m * v` of register operands; twin of `view[i] = m.get() * v.get()` (the contraction differs, see `detail::matVec`). */
template<class Out, class M, class V>
AETHER_DEVICEHOST() constexpr void matVecItems(Out&& out, const M& m, const V& v)
{
    detail::matVec<detail::AssignOp::Set, true>(out, m, v);
}
/** @brief `out += m * v`. */
template<class Out, class M, class V>
AETHER_DEVICEHOST() constexpr void matVecAdd(Out&& out, const M& m, const V& v)
{
    detail::matVec<detail::AssignOp::Add>(out, m, v);
}
/** @brief `out -= m * v`. */
template<class Out, class M, class V>
AETHER_DEVICEHOST() constexpr void matVecSub(Out&& out, const M& m, const V& v)
{
    detail::matVec<detail::AssignOp::Sub>(out, m, v);
}

/** @brief `out = l * r`; twin of `view[i] = L * R`. */
template<class Out, class L, class R>
AETHER_DEVICEHOST() constexpr void matMat(Out&& out, const L& l, const R& r)
{
    detail::matMat<detail::AssignOp::Set>(out, l, r);
}
/** @brief `out += l * r`. */
template<class Out, class L, class R>
AETHER_DEVICEHOST() constexpr void matMatAdd(Out&& out, const L& l, const R& r)
{
    detail::matMat<detail::AssignOp::Add>(out, l, r);
}
/** @brief `out -= l * r`. */
template<class Out, class L, class R>
AETHER_DEVICEHOST() constexpr void matMatSub(Out&& out, const L& l, const R& r)
{
    detail::matMat<detail::AssignOp::Sub>(out, l, r);
}

} // namespace rt
} // namespace aether
