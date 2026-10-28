// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Qr.h
 * @brief `aether::linalg::team::qr`: Householder QR of a tall `M x N` matrix by a
 *        team, and the least-squares solve built on it.
 *
 * The rows are cut into fixed tiles of `qrTile` rows. Each tile's partial sum
 * (a column norm, or a reflector applied to a column) is a serial loop over
 * the tile; the partials are then added in tile order by one thread. The
 * tiling does not depend on the team size, so the factor and the solution
 * are bitwise the same for any team size. Reflectors follow LAPACK `geqr2`:
 * `H = I - tau v v^T`, `v = (1, a[j+1.., j])`, `R` on and above the diagonal.
 *
 * `qrSolve` minimises `|A x - b|` for full-rank `A` (`M >= N`) without forming
 * the normal equations, so the conditioning is `cond(A)`, not its square.
 * `qrGramSolve` gives `(A^T A)^-1 B` from the same factor. A column whose `R` diagonal is exactly zero sets `info` to that column
 * (1-based) and the solve writes zeros: rank deficiency is reported only when
 * it is exact.
 *
 * Scratch: `a` is `M*N` elements, `tau` `N`, `work` `qrWorkSize(M, max(N, k))`, `info` 1.
 */

#include "aether/linalg/team/Triangular.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {
namespace team {

/** @brief Views into the scratch of one `M x N` QR factorisation (`M` is a run-time argument). */
template<class T, std::size_t N>
struct TeamQr {
    T* a;                 ///< `M*N`, row-major: `R` on and above the diagonal, the reflectors below.
    T* tau;               ///< `N` reflector scales.
    T* work;              ///< `qrWorkSize(M, max(N, k))` partial sums.
    std::int32_t* info;   ///< 1: 0, or the first column (1-based) with a zero `R` diagonal.
};

namespace detail {

/**
 * @brief Apply reflector `j` (`v` = column `j` of `qr` below the diagonal, 1 on it) to the
 *        `cols` columns of `x` (`M x cols`, row stride `ld`, first column `c0`).
 *
 * The dot products `w_c = v . x[:, c]` are tile partials combined in tile order.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void reflect(const Team& tm, const T* qr, T tau, std::size_t m, std::size_t j, T* x,
                                 std::size_t ld, std::size_t c0, std::size_t cols, T* work)
{
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    const std::size_t nt = (m + qrTile - 1) / qrTile;
    for (std::size_t e = rank; e < cols * nt; e += size) {
        const std::size_t c = c0 + e / nt;
        const std::size_t t = e % nt;
        std::size_t lo = t * qrTile;
        const std::size_t hi = (t + 1) * qrTile < m ? (t + 1) * qrTile : m;
        T s = T(0);
        if (lo < j) {
            lo = j;
        }
        for (std::size_t i = lo; i < hi; ++i) {
            s += (i == j ? T(1) : qr[i * N + j]) * x[i * ld + c];
        }
        work[e] = s;
    }
    tm.sync();
    for (std::size_t q = rank; q < cols; q += size) {
        T w = T(0);
        for (std::size_t t = 0; t < nt; ++t) {
            w += work[q * nt + t];
        }
        work[q * nt] = tau * w;
    }
    tm.sync();
    for (std::size_t e = rank; e < (m - j) * cols; e += size) {
        const std::size_t i = j + e / cols;
        const std::size_t q = e % cols;
        x[i * ld + c0 + q] -= work[q * nt] * (i == j ? T(1) : qr[i * N + j]);
    }
    tm.sync();
}

} // namespace detail

/**
 * @brief Factor `a_in` (`M x N`, row-major, `M >= N`) into `f`. `a_in` may be `f.a`. Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void qr(const Team& tm, const TeamQr<T, N>& f, const T* a_in, std::size_t m)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::team: T must be float or double and 1 <= N <= 4096");
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    const std::size_t nt = (m + qrTile - 1) / qrTile;
    T* a = f.a;
    T* work = f.work;
    detail::copyIn(tm, a, a_in, m * N);
    tm.sync();
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t t = rank; t < nt; t += size) {  // tile partials of |a[j+1:, j]|^2
            std::size_t lo = t * qrTile;
            const std::size_t hi = (t + 1) * qrTile < m ? (t + 1) * qrTile : m;
            if (lo < j + 1) {
                lo = j + 1;
            }
            T s = T(0);
            for (std::size_t i = lo; i < hi; ++i) {
                s += a[i * N + j] * a[i * N + j];
            }
            work[t] = s;
        }
        tm.sync();
        if (rank == 0) {
            T xs = T(0);
            for (std::size_t t = 0; t < nt; ++t) {
                xs += work[t];
            }
            const T alpha = a[j * N + j];
            T tau = T(0);
            T scale = T(1);
            if (xs != T(0)) {
                const T norm = math::sqrt(alpha * alpha + xs);
                const T beta = alpha < T(0) ? norm : -norm;
                tau = (beta - alpha) / beta;
                scale = T(1) / (alpha - beta);
                a[j * N + j] = beta;
            }
            f.tau[j] = tau;
            work[nt] = scale;  // slot past the tile partials
        }
        tm.sync();
        const T tau = f.tau[j];
        if (tau != T(0)) {
            const T scale = work[nt];
            for (std::size_t i = j + 1 + rank; i < m; i += size) {
                a[i * N + j] *= scale;
            }
            tm.sync();
            detail::reflect<N>(tm, a, tau, m, j, a, N, j + 1, N - 1 - j, work);
        }
        tm.sync();
    }
    if (rank == 0) {
        *f.info = 0;
        for (std::size_t j = 0; j < N; ++j) {
            if (a[j * N + j] == T(0)) {
                *f.info = static_cast<std::int32_t>(j + 1);
                break;
            }
        }
    }
    tm.sync();
}

/**
 * @brief Least squares `min |A X - B|` from the factor of `A` (`M x N`).
 *
 * `b` is `M x k` row-major and is OVERWRITTEN with `Q^T B`; `x` is `N x k`. `work` needs
 * `qrWorkSize(M, max(N, k))` elements. Zeros when the factorisation failed. Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void qrSolve(const Team& tm, const TeamQr<T, N>& f, std::size_t m, T* b, T* x, std::size_t k = 1)
{
    if (*f.info == 0) {
        for (std::size_t j = 0; j < N; ++j) {
            const T tau = f.tau[j];
            if (tau != T(0)) {
                detail::reflect<N>(tm, f.a, tau, m, j, b, k, 0, k, f.work);
            }
        }
    }
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    for (std::size_t e = rank; e < N * k; e += size) {
        x[e] = (*f.info == 0) ? b[e] : T(0);
    }
    tm.sync();
    if (*f.info == 0) {
        detail::substitute<N, false, false, false>(tm, f.a, x, k);
    }
}

/**
 * @brief Solve `(A^T A) X = B` from the factor of `A`: `R^T Y = B`, then `R X = Y`, never forming `A^T A`.
 *
 * `b` and `x` are `N x k` row-major, `x` may be `b`. Zeros when the factorisation failed. Collective.
 */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void qrGramSolve(const Team& tm, const TeamQr<T, N>& f, const T* b, T* x, std::size_t k = 1)
{
    detail::copyIn(tm, x, b, N * k);
    tm.sync();
    if (*f.info != 0) {
        detail::zeroOut(tm, x, N * k);
        tm.sync();
        return;
    }
    detail::substitute<N, true, true, false>(tm, f.a, x, k);    // R^T Y = B
    detail::substitute<N, false, false, false>(tm, f.a, x, k);  // R X = Y
}

} // namespace team
} // namespace linalg
} // namespace aether
