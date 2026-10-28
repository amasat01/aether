// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Triangular.h
 * @brief Team substitution against a packed triangle: the one routine under
 *        `luSolve`, `luSolveT`, `choSolve` and `qrSolve`.
 *
 * Column-oriented: step `i` divides row `i` of the right-hand sides by the
 * diagonal (unless the triangle has a unit diagonal), then one axpy per
 * remaining row. Every entry of the solution is updated by one thread in a
 * fixed order, with no reduction across threads, so the bits do not depend on
 * the team size.
 */

#include "aether/linalg/team/Team.h"

namespace aether {
namespace linalg {
namespace team {
namespace detail {

/**
 * @brief Solve `op(M) X = X` in place for the `K` columns of `x` (`N x K`, row-major).
 *
 * `M` is `N x N` row-major (row stride `N`). `Forward` runs rows `0..N-1`
 * (lower triangle), otherwise `N-1..0` (upper). `Trans` reads the coefficient
 * of row `r` at step `i` from `m[i*N + r]` instead of `m[r*N + i]`, i.e. solves
 * against the transpose of what is stored. `Unit` skips the division.
 */
template<std::size_t N, bool Forward, bool Trans, bool Unit, class T>
AETHER_DEVICEHOST() void substitute(const Team& tm, const T* m, T* x, std::size_t k)
{
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    for (std::size_t s = 0; s < N; ++s) {
        const std::size_t i = Forward ? s : N - 1 - s;
        if constexpr (!Unit) {
            const T d = m[i * N + i];
            for (std::size_t c = rank; c < k; c += size) {
                x[i * k + c] /= d;
            }
            tm.sync();
        }
        const std::size_t r0 = Forward ? i + 1 : 0;
        const std::size_t rows = Forward ? N - 1 - i : i;
        for (std::size_t f = rank; f < rows * k; f += size) {
            const std::size_t r = r0 + f / k;
            const std::size_t c = f % k;
            const T coef = Trans ? m[i * N + r] : m[r * N + i];
            x[r * k + c] -= coef * x[i * k + c];
        }
        tm.sync();
    }
}

/** @brief `x[0 .. n) = (b == x) ? x : b`, strided over the team; the caller syncs. */
template<class T>
AETHER_DEVICEHOST() void copyIn(const Team& tm, T* x, const T* b, std::size_t n)
{
    if (b != x) {
        for (std::size_t e = static_cast<std::size_t>(tm.rank); e < n; e += static_cast<std::size_t>(tm.size)) {
            x[e] = b[e];
        }
    }
}

/** @brief `x[0 .. n) = 0`, strided over the team; the caller syncs. */
template<class T>
AETHER_DEVICEHOST() void zeroOut(const Team& tm, T* x, std::size_t n)
{
    for (std::size_t e = static_cast<std::size_t>(tm.rank); e < n; e += static_cast<std::size_t>(tm.size)) {
        x[e] = T(0);
    }
}

} // namespace detail
} // namespace team
} // namespace linalg
} // namespace aether
