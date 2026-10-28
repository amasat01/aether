// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Cholesky.h
 * @brief `aether::linalg::team::cholesky`: `S = L L^T` of the symmetrised input by
 *        a team, and the solve against it.
 *
 * Same arithmetic as the one-thread `linalg::cholesky`, entry for entry: one
 * thread owns each `L(i, j)` and runs its dot product serially in index order,
 * so the factor agrees with the one-thread factor to a few ulp (a compiler may
 * contract the two loops differently) and is bitwise independent of the team size. `info` is LAPACK `potrf` numbering; columns from the
 * failing one on are zero, and every solve against a failed factor writes zeros.
 *
 * Scratch: `l` is `N*N` elements, `info` 1.
 */

#include "aether/linalg/team/Triangular.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {
namespace team {

/** @brief Views into the scratch of one `N x N` Cholesky factor. */
template<class T, std::size_t N>
struct TeamCholesky {
    T* l;                 ///< `N*N`, row-major lower factor (upper part zero).
    std::int32_t* info;   ///< 1: 0 on success; `k > 0` when column `k` (1-based) was not positive.
};

namespace detail {

/** @brief Zero the strict upper part and, when `*info > 0`, every column from the failing one on. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void cleanFactor(const Team& tm, const TeamCholesky<T, N>& f)
{
    const std::size_t bad = *f.info > 0 ? static_cast<std::size_t>(*f.info - 1) : N;
    for (std::size_t e = static_cast<std::size_t>(tm.rank); e < N * N; e += static_cast<std::size_t>(tm.size)) {
        const std::size_t i = e / N;
        const std::size_t j = e % N;
        if (j > i || j >= bad) {
            f.l[e] = T(0);
        }
    }
    tm.sync();
}

} // namespace detail

/**
 * @brief Factor the symmetric part of `a_in` (`N x N`, row-major) into `f`. `a_in` may be `f.l`:
 *        the factor is then formed in place, which is how a Gram matrix is factored.
 *
 * Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void cholesky(const Team& tm, const TeamCholesky<T, N>& f, const T* a_in)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::team: T must be float or double and 1 <= N <= 4096");
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    T* l = f.l;
    if (rank == 0) {
        *f.info = 0;
    }
    tm.sync();
    for (std::size_t j = 0; j < N; ++j) {
        if (rank == 0 && *f.info == 0) {
            T d = T(0.5) * a_in[j * N + j] + T(0.5) * a_in[j * N + j];
            for (std::size_t k = 0; k < j; ++k) {
                d -= l[j * N + k] * l[j * N + k];
            }
            if (d > T(0)) {
                l[j * N + j] = math::sqrt(d);
            } else {
                *f.info = static_cast<std::int32_t>(j + 1);
            }
        }
        tm.sync();
        if (*f.info == 0) {
            const T ljj = l[j * N + j];
            for (std::size_t i = j + 1 + rank; i < N; i += size) {
                T s = T(0.5) * a_in[i * N + j] + T(0.5) * a_in[j * N + i];
                for (std::size_t k = 0; k < j; ++k) {
                    s -= l[i * N + k] * l[j * N + k];
                }
                l[i * N + j] = s / ljj;
            }
        }
        tm.sync();
    }
    detail::cleanFactor(tm, f);
}

/**
 * @brief The factorisation of a lower triangular `l_in` taken as the factor itself, into `f`
 *        (`l_in` may be `f.l`). `info` is the first column whose diagonal is not positive and finite.
 *
 * Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void choleskyFrom(const Team& tm, const TeamCholesky<T, N>& f, const T* l_in)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::team: T must be float or double and 1 <= N <= 4096");
    if (tm.rank == 0) {
        *f.info = 0;
        for (std::size_t j = 0; j < N; ++j) {
            const T d = l_in[j * N + j];
            if (*f.info == 0 && !(d > T(0) && math::isfinite(d))) {
                *f.info = static_cast<std::int32_t>(j + 1);
            }
        }
    }
    tm.sync();
    detail::copyIn(tm, f.l, l_in, N * N);
    tm.sync();
    if (*f.info != 0) {
        detail::zeroOut(tm, f.l, N * N);
        tm.sync();
    } else {
        detail::cleanFactor(tm, f);
    }
}

/** @brief Solve `L X = B`, `B` and `X` `N x k` row-major, `X` may be `B`. Collective; the factor is untouched. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void triLower(const Team& tm, const TeamCholesky<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::copyIn(tm, x, b, N * k);
    tm.sync();
    detail::substitute<N, true, false, false>(tm, f.l, x, k);
}

/** @brief Solve `L^T X = B`; same shapes and rules as `triLower`. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void triUpperT(const Team& tm, const TeamCholesky<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::copyIn(tm, x, b, N * k);
    tm.sync();
    detail::substitute<N, false, true, false>(tm, f.l, x, k);
}

/**
 * @brief Solve `S X = B` from the factor: `L Y = B`, then `L^T X = Y`. Zeros when the
 *        factorisation failed. Shapes and rules as `triLower`. Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void choSolve(const Team& tm, const TeamCholesky<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::copyIn(tm, x, b, N * k);
    tm.sync();
    if (*f.info != 0) {
        detail::zeroOut(tm, x, N * k);
        tm.sync();
        return;
    }
    detail::substitute<N, true, false, false>(tm, f.l, x, k);
    detail::substitute<N, false, true, false>(tm, f.l, x, k);
}

} // namespace team
} // namespace linalg
} // namespace aether
