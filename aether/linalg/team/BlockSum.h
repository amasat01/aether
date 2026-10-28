// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BlockSum.h
 * @brief `aether::linalg::team::blockSum`: the sum over rows of per-row outer
 *        products, `G = sum_r u_r v_r^T`, ordered.
 *
 * The Gram matrix of a tall Jacobian (`J^T J`: `u = v = J`) and the gradient
 * `J^T r` (`v` one column wide). One thread owns each entry `G(i, j)` and adds the rows
 * in index order, serially: no atomics, no cross-thread floating reduction, so
 * the result equals the serial loop to the compiler's contraction and does not depend on the
 * team size.
 */

#include "aether/linalg/team/Team.h"

namespace aether {
namespace linalg {
namespace team {

/**
 * @brief `g[i*Q + j] = sum_{r < count} u[row(r)*strideU + i] * v[row(r)*strideV + j]`.
 *
 * `row(r)` is `first + r`, or `index[first + r]` when `index` is not null (an active-set
 * map). With `rowsPer = M > 1` each sample owns `M` consecutive rows (`M x P` per sample, as in
 * `transpose(A) A` of a per-lane `M x P` block): `count` counts samples, sample `r` is
 * `first + r` or `index[first + r]`, and its rows `sample*M .. sample*M + M - 1` are added in
 * order, so the sum equals the expanded row list's. `rowsPer = 1` is the plain form, bit for bit.
 * Rows are visited in ascending `r`. Entry `(i, j)` is written to
 * `g[i * strideGI + j * strideGJ]`; the defaults (0) mean a contiguous `P x Q` row-major block
 * (`strideGI = Q`, `strideGJ = 1`). A payload-first plane over `K` keys, entry-major with the key
 * fastest, is `g + key`, `strideGI = Q * K`, `strideGJ = K`, so the sum lands in the user plane
 * with no copy. The values do not depend on the layout. Collective; `g` is complete on return.
 *
 * @tparam P  Width of `u` (rows of `g`).
 * @tparam Q  Width of `v` (columns of `g`); `Q = 1` gives `J^T r`.
 */
template<std::size_t P, std::size_t Q = P, class T>
AETHER_DEVICEHOST() void blockSum(const Team& tm, T* g, const T* u, const T* v, std::size_t first, std::size_t count,
                                  std::size_t strideU, std::size_t strideV, const std::int32_t* index = nullptr,
                                  std::size_t strideGI = 0, std::size_t strideGJ = 0, std::size_t rowsPer = 1)
{
    static_assert(detail::supported_v<T, P> && detail::supported_v<T, Q>,
                  "aether::linalg::team: T must be float or double and 1 <= P, Q <= 4096");
    for (std::size_t e = static_cast<std::size_t>(tm.rank); e < P * Q; e += static_cast<std::size_t>(tm.size)) {
        const std::size_t i = e / Q;
        const std::size_t j = e % Q;
        T s = T(0);
        for (std::size_t r = 0; r < count; ++r) {
            const std::size_t sample = index != nullptr ? static_cast<std::size_t>(index[first + r]) : first + r;
            for (std::size_t q = 0; q < rowsPer; ++q) {
                const std::size_t row = sample * rowsPer + q;
                s += u[row * strideU + i] * v[row * strideV + j];
            }
        }
        g[i * (strideGI != 0 ? strideGI : Q) + j * (strideGJ != 0 ? strideGJ : 1)] = s;
    }
    tm.sync();
}

} // namespace team
} // namespace linalg
} // namespace aether
