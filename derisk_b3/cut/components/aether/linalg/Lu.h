// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Lu.h
 * @brief `aether::linalg::Lu`: `P A = L U` with partial pivoting, and the
 *        solves, determinant and inverse built on it.
 *
 * Column-by-column (Crout) elimination on a dense row-major matrix: each entry
 * is reduced by one dot product of already-final factors, which is the order
 * LAPACK's `getrf` kernels use, so an exactly singular matrix such as
 * `[[1,2,3],[4,5,6],[7,8,9]]` hits the same zero pivot. The pivot is the
 * first entry of largest magnitude in the column (LAPACK `idamax`, ties to the
 * lowest row). A column whose pivot is exactly zero leaves `info` at the first
 * such column, 1-based as LAPACK `getrf` numbers it; the elimination carries
 * on, so `a` holds the partial factorisation. Every consumer of a failed
 * factor writes zeros (`det` returns 0): a singular system has no answer to
 * hand back.
 *
 * Matrices and right-hand sides are row-major pointers or per-sample operands
 * (`view[i]`, `Item`; see `detail/Common.h`). A right-hand side is solved one
 * column at a time through a length-`N` local, so a lane holds the factor and
 * one column, never a copy of the whole plane. Every loop stays rolled
 * (`AETHER_NO_UNROLL()`): the matrix lives in local memory and a long fold
 * does not turn into registers.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/linalg/detail/Common.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {

/** @brief Result of `lu`: packed factors, pivot rows and status. */
template<class T, std::size_t N>
struct Lu {
    T a[N * N];            ///< Row-major: `U` on and above the diagonal, `L`'s multipliers (unit diagonal implied) below.
    std::int32_t piv[N];   ///< Row exchanged with row `k` at step `k` (0-based).
    std::int32_t info;     ///< 0 on success; `k > 0` when column `k` (1-based) had a zero pivot.
};

/**
 * @brief Factor `a` as `P A = L U` into `f`, in place: the factor is built where it
 *        lives, never returned or copied by value.
 *
 * @tparam N  Dimension.
 * @param f   The factorisation, overwritten.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class T, class A>
    requires(detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICEHOST() void lu(Lu<T, N>& f, const A& a)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    f.info = 0;
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < N; ++j) {
            f.a[i * N + j] = detail::elemAt<N>(a, i, j);
        }
    }
    AETHER_NO_UNROLL()
    for (std::size_t j = 0; j < N; ++j) {
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t m = i < j ? i : j;
            T acc = T(0);
            AETHER_NO_UNROLL()
            for (std::size_t k = 0; k < m; ++k) {
                acc += f.a[i * N + k] * f.a[k * N + j];
            }
            f.a[i * N + j] -= acc;
        }
        std::size_t p = j;
        T big = math::abs(f.a[j * N + j]);
        AETHER_NO_UNROLL()
        for (std::size_t i = j + 1; i < N; ++i) {
            const T m = math::abs(f.a[i * N + j]);
            if (m > big) {
                big = m;
                p = i;
            }
        }
        f.piv[j] = static_cast<std::int32_t>(p);
        if (f.a[p * N + j] != T(0)) {
            if (p != j) {
                AETHER_NO_UNROLL()
                for (std::size_t c = 0; c < N; ++c) {
                    const T t = f.a[j * N + c];
                    f.a[j * N + c] = f.a[p * N + c];
                    f.a[p * N + c] = t;
                }
            }
            const T r = T(1) / f.a[j * N + j];
            AETHER_NO_UNROLL()
            for (std::size_t i = j + 1; i < N; ++i) {
                f.a[i * N + j] *= r;
            }
        } else if (f.info == 0) {
            f.info = static_cast<std::int32_t>(j + 1);
        }
    }
}

/**
 * @brief Factor `a` as `P A = L U`, returned by value.
 *
 * @tparam N  Dimension.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class A>
    requires detail::operand<A>
AETHER_DEVICEHOST() Lu<detail::value_t<A>, N> lu(const A& a)
{
    Lu<detail::value_t<A>, N> f;
    lu<N>(f, a);
    return f;
}

namespace detail {

/** @brief One right-hand-side column of `luSolve`, in place: the row exchanges, `L y = P b`, `U x = y`. */
template<class T, std::size_t N>
AETHER_DEVICEHOST() void luSolveColumn(const Lu<T, N>& f, T* x)
{
    AETHER_NO_UNROLL()
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t p = static_cast<std::size_t>(f.piv[k]);
        if (p != k && p < N) {
            const T t = x[k];
            x[k] = x[p];
            x[p] = t;
        }
    }
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < i; ++j) {
            s -= f.a[i * N + j] * x[j];
        }
        x[i] = s;
    }
    AETHER_NO_UNROLL()
    for (std::size_t r = 0; r < N; ++r) {
        const std::size_t i = N - 1 - r;
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = i + 1; j < N; ++j) {
            s -= f.a[i * N + j] * x[j];
        }
        x[i] = s / f.a[i * N + i];
    }
}

/** @brief One right-hand-side column of `luSolveT`, in place: `U^T y = b`, `L^T z = y`, the row exchanges undone. */
template<class T, std::size_t N>
AETHER_DEVICEHOST() void luSolveTColumn(const Lu<T, N>& f, T* x)
{
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < i; ++j) {
            s -= f.a[j * N + i] * x[j];
        }
        x[i] = s / f.a[i * N + i];
    }
    AETHER_NO_UNROLL()
    for (std::size_t r = 0; r < N; ++r) {
        const std::size_t i = N - 1 - r;
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = i + 1; j < N; ++j) {
            s -= f.a[j * N + i] * x[j];
        }
        x[i] = s;
    }
    AETHER_NO_UNROLL()
    for (std::size_t r = 0; r < N; ++r) {
        const std::size_t k = N - 1 - r;
        const std::size_t p = static_cast<std::size_t>(f.piv[k]);
        if (p != k && p < N) {
            const T t = x[k];
            x[k] = x[p];
            x[p] = t;
        }
    }
}

/**
 * @brief Solve the `K` columns of `b` against `A` (`A^T` when `Transposed`), writing `x`:
 *        all zeros when `f` failed. `b` and `x` may be the same operand.
 */
template<std::size_t K, bool Transposed, class T, std::size_t N, class B, class X>
AETHER_DEVICEHOST() void solveColumns(const Lu<T, N>& f, const B& b, X&& x)
{
    if (f.info == 0) {
        AETHER_NO_UNROLL()
        for (std::size_t c = 0; c < K; ++c) {
            T col[N];
            AETHER_NO_UNROLL()
            for (std::size_t i = 0; i < N; ++i) {
                col[i] = elemAt<K>(b, i, c);
            }
            if constexpr (Transposed) {
                luSolveTColumn(f, col);
            } else {
                luSolveColumn(f, col);
            }
            AETHER_NO_UNROLL()
            for (std::size_t i = 0; i < N; ++i) {
                elemAt<K>(x, i, c) = col[i];
            }
        }
    } else {
        AETHER_NO_UNROLL()
        for (std::size_t c = 0; c < K; ++c) {
            AETHER_NO_UNROLL()
            for (std::size_t i = 0; i < N; ++i) {
                elemAt<K>(x, i, c) = T(0);
            }
        }
    }
}

} // namespace detail

/**
 * @brief Solve `A X = B`: apply the row exchanges, `L Y = P B`, `U X = Y`.
 *        All zeros when the factorisation failed.
 *
 * @tparam K  Number of right-hand sides (1 for a vector).
 * @param f   Factorisation of `A`.
 * @param b   `N x K`, row-major.
 * @param x   `N x K`, row-major. May be `b` (in-place).
 */
template<std::size_t K = 1, class T, std::size_t N, class B, class X>
    requires(detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() void luSolve(const Lu<T, N>& f, const B& b, X&& x)
{
    detail::solveColumns<K, false>(f, b, x);
}

/**
 * @brief Solve `A^T X = B` from the same factor: `U^T Y = B`, `L^T Z = Y`,
 *        then the row exchanges undone. All zeros when the factorisation
 *        failed.
 *
 * @tparam K  Number of right-hand sides.
 * @param f   Factorisation of `A`.
 * @param b   `N x K`, row-major.
 * @param x   `N x K`, row-major. May be `b` (in-place).
 */
template<std::size_t K = 1, class T, std::size_t N, class B, class X>
    requires(detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() void luSolveT(const Lu<T, N>& f, const B& b, X&& x)
{
    detail::solveColumns<K, true>(f, b, x);
}

/** @brief `det(A)` from its factor: the product of `U`'s diagonal, one sign flip per row exchange. 0 when the factorisation failed. */
template<class T, std::size_t N>
AETHER_DEVICEHOST() T luDet(const Lu<T, N>& f)
{
    T d = T(0);
    if (f.info == 0) {
        d = T(1);
        AETHER_NO_UNROLL()
        for (std::size_t k = 0; k < N; ++k) {
            d *= f.a[k * N + k];
            if (static_cast<std::size_t>(f.piv[k]) != k) {
                d = -d;
            }
        }
    }
    return d;
}

/**
 * @brief `A^{-1}` from its factor into `out` (`N x N`), one unit column at a
 *        time: no `N x N` identity is built. All zeros when the factorisation
 *        failed.
 */
template<class T, std::size_t N, class X>
    requires detail::operand<X>
AETHER_DEVICEHOST() void luInverse(const Lu<T, N>& f, X&& out)
{
    AETHER_NO_UNROLL()
    for (std::size_t c = 0; c < N; ++c) {
        T col[N];
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            col[i] = (i == c) ? T(1) : T(0);
        }
        if (f.info == 0) {
            detail::luSolveColumn(f, col);
        } else {
            detail::zeroFill(col, N);
        }
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            detail::elemAt<N>(out, i, c) = col[i];
        }
    }
}

/**
 * @brief Solve `A x = b` (`K` right-hand sides). Returns `info`; `x` is zero when it is not.
 *
 * @tparam N  Dimension.
 * @tparam K  Number of right-hand sides.
 */
template<std::size_t N, std::size_t K = 1, class A, class B, class X>
    requires(detail::operand<A> && detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() std::int32_t solve(const A& a, const B& b, X&& x)
{
    const auto f = lu<N>(a);
    luSolve<K>(f, b, x);
    return f.info;
}

/** @brief `A^{-1}` into `ainv` (`N x N`). Returns `info`; `ainv` is zero when it is not. */
template<std::size_t N, class A, class X>
    requires(detail::operand<A> && detail::operand<X>)
AETHER_DEVICEHOST() std::int32_t inv(const A& a, X&& ainv)
{
    const auto f = lu<N>(a);
    luInverse(f, ainv);
    return f.info;
}

/** @brief `det(A)`; 0 for a singular matrix. */
template<std::size_t N, class A>
    requires detail::operand<A>
AETHER_DEVICEHOST() detail::value_t<A> det(const A& a)
{
    return luDet(lu<N>(a));
}

} // namespace linalg
} // namespace aether
