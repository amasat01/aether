// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Triangular.h
 * @brief `aether::linalg::triLower` / `triUpperT`: multi-right-hand-side
 *        solves against a lower-triangular factor and its transpose.
 *
 * The two halves of a Cholesky solve, and the building blocks of the Cholesky
 * factor's derivative. Matrices are dense row-major; `L` is read on and below
 * its diagonal only. Every row is a forward-ordered dot product
 * (`s -= l * x`, ascending index) divided by the diagonal, so a result depends
 * only on the operands and the order, never on the target.
 */

#include <cstddef>

#include "aether/linalg/detail/Common.h"

namespace aether {
namespace linalg {

namespace detail {

/** @brief One column of `triLower`, in place: `L y = x` by forward substitution. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void triLowerColumn(const T* l, T* x)
{
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < i; ++j) {
            s -= l[i * N + j] * x[j];
        }
        x[i] = s / l[i * N + i];
    }
}

/** @brief One column of `triUpperT`, in place: `L^T y = x` by back substitution. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void triUpperTColumn(const T* l, T* x)
{
    AETHER_NO_UNROLL()
    for (std::size_t r = 0; r < N; ++r) {
        const std::size_t i = N - 1 - r;
        T s = x[i];
        AETHER_NO_UNROLL()
        for (std::size_t j = i + 1; j < N; ++j) {
            s -= l[j * N + i] * x[j];
        }
        x[i] = s / l[i * N + i];
    }
}

/** @brief Run `Lower`-then-`UpperT` column solvers over the `K` columns of `b`, writing `x` (`b` may be `x`). */
template<std::size_t N, std::size_t K, bool Lower, bool UpperT, class T, class B, class X>
AETHER_DEVICEHOST() void triColumns(const T* l, const B& b, X&& x)
{
    AETHER_NO_UNROLL()
    for (std::size_t c = 0; c < K; ++c) {
        T col[N];
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            col[i] = elemAt<K>(b, i, c);
        }
        if constexpr (Lower) {
            triLowerColumn<N>(l, col);
        }
        if constexpr (UpperT) {
            triUpperTColumn<N>(l, col);
        }
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            elemAt<K>(x, i, c) = col[i];
        }
    }
}

} // namespace detail

/**
 * @brief Solve `L X = B` for `X` by forward substitution.
 *
 * @tparam N  Dimension of `L`.
 * @tparam K  Number of right-hand sides (columns of `B` and `X`).
 * @param l   Lower-triangular `N x N`, row-major (strict upper part unread).
 * @param b   Right-hand side, `N x K` row-major (pointer or per-sample operand).
 * @param x   Solution, `N x K` row-major. May be `b` (in-place).
 */
template<std::size_t N, std::size_t K = 1, class T, class B, class X>
    requires(detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() void triLower(const T* l, const B& b, X&& x)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    detail::triColumns<N, K, true, false>(l, b, x);
}

/**
 * @brief Solve `L^T X = B` for `X` by back substitution.
 *
 * @tparam N  Dimension of `L`.
 * @tparam K  Number of right-hand sides.
 * @param l   Lower-triangular `N x N`, row-major (strict upper part unread).
 * @param b   Right-hand side, `N x K` row-major (pointer or per-sample operand).
 * @param x   Solution, `N x K` row-major. May be `b` (in-place).
 */
template<std::size_t N, std::size_t K = 1, class T, class B, class X>
    requires(detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() void triUpperT(const T* l, const B& b, X&& x)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    detail::triColumns<N, K, false, true>(l, b, x);
}

} // namespace linalg
} // namespace aether
