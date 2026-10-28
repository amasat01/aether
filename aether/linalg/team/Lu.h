// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Lu.h
 * @brief `aether::linalg::team::lu`: `P A = L U` of one matrix by a team, and
 *        the solves, determinant and inverse built on it.
 *
 * Right-looking elimination with partial pivoting (the one-thread `linalg::lu`
 * is Crout, so the two round differently: they agree to a ULP contract, not
 * bit for bit). The pivot is the first entry of largest magnitude in the
 * column, ties to the lowest row, found by one thread; every other step
 * updates independent entries, so the factor and the solutions are bitwise
 * the same for any team size. `info` is LAPACK `getrf` numbering; a failed
 * factor makes every solve write zeros.
 *
 * Scratch (caller-provided, global): `a` is `N*N` elements, `piv` `N`, `info` 1.
 */

#include "aether/linalg/team/Triangular.h"

namespace aether {
namespace linalg {
namespace team {

/** @brief Views into the scratch of one `N x N` factorisation. */
template<class T, std::size_t N>
struct TeamLu {
    T* a;                 ///< `N*N`, row-major: `U` on and above the diagonal, `L`'s multipliers below.
    std::int32_t* piv;    ///< `N`: row exchanged with row `k` at step `k` (0-based).
    std::int32_t* info;   ///< 1: 0 on success; `k > 0` when column `k` (1-based) had a zero pivot.
};

/**
 * @brief Factor `a_in` (`N x N`, row-major) into `f`. `a_in` may be `f.a`.
 *
 * Collective. On return the factor, pivots and `*f.info` are visible to the team.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void lu(const Team& tm, const TeamLu<T, N>& f, const T* a_in)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::team: T must be float or double and 1 <= N <= 4096");
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    T* a = f.a;
    detail::copyIn(tm, a, a_in, N * N);
    if (rank == 0) {
        *f.info = 0;
    }
    tm.sync();
    for (std::size_t j = 0; j < N; ++j) {
        if (rank == 0) {
            std::size_t p = j;
            T big = a[j * N + j] < T(0) ? -a[j * N + j] : a[j * N + j];
            for (std::size_t i = j + 1; i < N; ++i) {
                const T m = a[i * N + j] < T(0) ? -a[i * N + j] : a[i * N + j];
                if (m > big) {
                    big = m;
                    p = i;
                }
            }
            f.piv[j] = static_cast<std::int32_t>(p);
            if (a[p * N + j] == T(0) && *f.info == 0) {
                *f.info = static_cast<std::int32_t>(j + 1);
            }
        }
        tm.sync();
        const std::size_t p = static_cast<std::size_t>(f.piv[j]);
        const bool ok = a[p * N + j] != T(0);
        tm.sync();
        if (ok) {
            if (p != j) {
                for (std::size_t c = rank; c < N; c += size) {
                    const T t = a[j * N + c];
                    a[j * N + c] = a[p * N + c];
                    a[p * N + c] = t;
                }
            }
            tm.sync();
            const T r = T(1) / a[j * N + j];
            for (std::size_t i = j + 1 + rank; i < N; i += size) {
                a[i * N + j] *= r;
            }
            tm.sync();
            const std::size_t w = N - j - 1;
            for (std::size_t e = rank; e < w * w; e += size) {
                const std::size_t i = j + 1 + e / w;
                const std::size_t k = j + 1 + e % w;
                a[i * N + k] -= a[i * N + j] * a[j * N + k];
            }
        }
        tm.sync();
    }
}

namespace detail {

/** @brief Row exchanges of `P` applied to the `K` columns of `x`, one thread per column. */
template<class T, std::size_t N>
AETHER_DEVICEHOST() void permute(const Team& tm, const TeamLu<T, N>& f, T* x, std::size_t k, bool reverse)
{
    for (std::size_t c = static_cast<std::size_t>(tm.rank); c < k; c += static_cast<std::size_t>(tm.size)) {
        for (std::size_t s = 0; s < N; ++s) {
            const std::size_t i = reverse ? N - 1 - s : s;
            const std::size_t p = static_cast<std::size_t>(f.piv[i]);
            if (p != i && p < N) {
                const T t = x[i * k + c];
                x[i * k + c] = x[p * k + c];
                x[p * k + c] = t;
            }
        }
    }
}

template<bool Transposed, std::size_t N, class T>
AETHER_DEVICEHOST() void luSolveImpl(const Team& tm, const TeamLu<T, N>& f, const T* b, T* x, std::size_t k)
{
    copyIn(tm, x, b, N * k);
    tm.sync();
    if (*f.info != 0) {
        zeroOut(tm, x, N * k);
        tm.sync();
        return;
    }
    if constexpr (Transposed) {
        substitute<N, true, true, false>(tm, f.a, x, k);    // U^T y = b
        substitute<N, false, true, true>(tm, f.a, x, k);    // L^T z = y
        permute(tm, f, x, k, true);
    } else {
        permute(tm, f, x, k, false);
        tm.sync();  // the substitution reads rows other threads exchanged
        substitute<N, true, false, true>(tm, f.a, x, k);    // L y = P b
        substitute<N, false, false, false>(tm, f.a, x, k);  // U x = y
    }
    tm.sync();
}

} // namespace detail

/**
 * @brief Solve `A X = B` from the factor. `b` and `x` are `N x k` row-major, `x` may be `b`.
 *
 * Collective; zeros when the factorisation failed.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void luSolve(const Team& tm, const TeamLu<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::luSolveImpl<false>(tm, f, b, x, k);
}

/** @brief Solve `A^T X = B` from the factor; same shapes and rules as `luSolve`. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void luSolveT(const Team& tm, const TeamLu<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::luSolveImpl<true>(tm, f, b, x, k);
}

/**
 * @brief The determinant, `0` when the factorisation failed. Not collective: every thread
 *        may call it after `lu` and gets the same value (a fixed-order product).
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() T luDet(const TeamLu<T, N>& f)
{
    T d = T(0);
    if (*f.info == 0) {
        d = T(1);
        for (std::size_t k = 0; k < N; ++k) {
            d *= f.a[k * N + k];
            if (static_cast<std::size_t>(f.piv[k]) != k) {
                d = -d;
            }
        }
    }
    return d;
}

/** @brief The inverse into `out` (`N x N`, row-major): `N` column solves against the identity. Collective. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void luInverse(const Team& tm, const TeamLu<T, N>& f, T* out)
{
    for (std::size_t e = static_cast<std::size_t>(tm.rank); e < N * N; e += static_cast<std::size_t>(tm.size)) {
        out[e] = (e / N == e % N) ? T(1) : T(0);
    }
    tm.sync();
    detail::luSolveImpl<false>(tm, f, out, out, N);
}

} // namespace team
} // namespace linalg
} // namespace aether
