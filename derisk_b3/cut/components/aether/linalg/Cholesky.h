// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Cholesky.h
 * @brief `aether::linalg::Cholesky`: `S = L L^T` of the symmetrised input,
 *        and the solve against it.
 *
 * The factorisation reads `S = (A + A^T) / 2`, so a non-symmetric `A` gives
 * the factor of its symmetric part, bit for bit what `cholesky` returns for
 * that symmetric part. A column whose pivot is not positive (`!(d > 0)`, so a
 * NaN fails too) stops the factorisation: `info` is that column, 1-based as
 * LAPACK `potrf` numbers it, the columns before it are complete and the rest
 * of `l` is zero. `choleskyFrom` wraps a factor the caller built.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/linalg/Triangular.h"
#include "aether/linalg/detail/Common.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {

/** @brief Result of `cholesky`: the lower factor (upper part zero) and its status. */
template<class T, std::size_t N>
struct Cholesky {
    T l[N * N];         ///< Lower factor, row-major; partial when `info > 0`.
    std::int32_t info;  ///< 0 on success; `k > 0` when column `k` (1-based) was not positive.
};

/**
 * @brief Factor the symmetric part of `a` as `L L^T` into `f`, in place: the factor
 *        is built where it lives, never returned or copied by value.
 *
 * @tparam N  Dimension.
 * @param f   The factorisation, overwritten.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class T, class A>
    requires(detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICEHOST() void cholesky(Cholesky<T, N>& f, const A& a)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    f.info = 0;
    detail::zeroFill(f.l, N * N);
    AETHER_NO_UNROLL()
    for (std::size_t j = 0; j < N; ++j) {
        if (f.info == 0) {
            T d = detail::symAt<N>(a, j, j);
            AETHER_NO_UNROLL()
            for (std::size_t k = 0; k < j; ++k) {
                d -= f.l[j * N + k] * f.l[j * N + k];
            }
            if (d > T(0)) {
                const T ljj = math::sqrt(d);
                f.l[j * N + j] = ljj;
                AETHER_NO_UNROLL()
                for (std::size_t i = j + 1; i < N; ++i) {
                    T s = detail::symAt<N>(a, i, j);
                    AETHER_NO_UNROLL()
                    for (std::size_t k = 0; k < j; ++k) {
                        s -= f.l[i * N + k] * f.l[j * N + k];
                    }
                    f.l[i * N + j] = s / ljj;
                }
            } else {
                f.info = static_cast<std::int32_t>(j + 1);
            }
        }
    }
}

/**
 * @brief Factor the symmetric part of `a` as `L L^T`, returned by value.
 *
 * @tparam N  Dimension.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class A>
    requires detail::operand<A>
AETHER_DEVICEHOST() Cholesky<detail::value_t<A>, N> cholesky(const A& a)
{
    Cholesky<detail::value_t<A>, N> f;
    cholesky<N>(f, a);
    return f;
}

/**
 * @brief The factorisation of a lower triangular `l` taken as the factor itself
 *        (`S = L L^T`), into `f`: `f.l` is the lower triangle of `l`, the part above
 *        the diagonal is never read.
 *
 * The status is read from the diagonal alone: `info` is the first column (1-based)
 * whose entry is not positive and finite (`!(d > 0)`, so zero, negative and NaN
 * fail, and so does infinity). A failed factor is all zeros, so every solve against
 * it returns zeros. The entries below the diagonal are not checked.
 *
 * @tparam N  Dimension.
 * @param f   The factorisation, overwritten.
 * @param l   `N x N`, row-major.
 */
template<std::size_t N, class T, class A>
    requires(detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICEHOST() void choleskyFrom(Cholesky<T, N>& f, const A& l)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    f.info = 0;
    AETHER_NO_UNROLL()
    for (std::size_t j = 0; j < N; ++j) {
        const T d = detail::elemAt<N>(l, j, j);
        if (f.info == 0 && !(d > T(0) && math::isfinite(d))) {
            f.info = static_cast<std::int32_t>(j + 1);
        }
    }
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < N; ++j) {
            f.l[i * N + j] = (f.info == 0 && j <= i) ? detail::elemAt<N>(l, i, j) : T(0);
        }
    }
}

/**
 * @brief The factor as a user-visible matrix: `f.l`, or all zeros when the
 *        factorisation failed.
 *
 * @param f    Factorisation.
 * @param out  `N x N`, row-major.
 */
template<class T, std::size_t N, class X>
    requires detail::operand<X>
AETHER_DEVICEHOST() void choleskyFactor(const Cholesky<T, N>& f, X&& out)
{
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < N; ++i) {
        AETHER_NO_UNROLL()
        for (std::size_t j = 0; j < N; ++j) {
            detail::elemAt<N>(out, i, j) = (f.info == 0) ? f.l[i * N + j] : T(0);
        }
    }
}

/**
 * @brief Solve `S X = B` from the factor: `L Y = B`, then `L^T X = Y`. All
 *        zeros when the factorisation failed.
 *
 * @tparam K  Number of right-hand sides (1 for a vector).
 * @param f   Factorisation of `S`.
 * @param b   `N x K`, row-major.
 * @param x   `N x K`, row-major. May be `b` (in-place).
 */
template<std::size_t K = 1, class T, std::size_t N, class B, class X>
    requires(detail::operand<B> && detail::operand<X>)
AETHER_DEVICEHOST() void choSolve(const Cholesky<T, N>& f, const B& b, X&& x)
{
    if (f.info == 0) {
        detail::triColumns<N, K, true, true>(f.l, b, x);
    } else {
        AETHER_NO_UNROLL()
        for (std::size_t c = 0; c < K; ++c) {
            AETHER_NO_UNROLL()
            for (std::size_t i = 0; i < N; ++i) {
                detail::elemAt<K>(x, i, c) = T(0);
            }
        }
    }
}

} // namespace linalg
} // namespace aether
