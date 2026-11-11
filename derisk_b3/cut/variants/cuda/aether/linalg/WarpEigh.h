// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file WarpEigh.h
 * @brief `aether::linalg::warp::eighValues`: the eigenvalues of ONE 32 x 32 symmetric matrix worked on by ONE warp.
 *
 * The one-thread `eigh` (`Eigh.h`) keeps the 32 x 32 matrix in local memory, so a lane fills a
 * fraction of a wave and waits on memory. This is its warp form: the matrix sits in a shared tile
 * and lane `k` owns the `k`-th row and column. It is the SAME algorithm, op for op: the cyclic
 * `(p, q)` order, the skip test, the rotation scalars, the `k` updates, the final sum of squares
 * and the info bits are those of `Eigh.h`, so eigenvalues and `info` are BIT-IDENTICAL to the
 * one-thread form (compiled with the same flags). Rotations stay sequential; what runs lane
 * parallel is the `k` loop of each rotation. Eigenvectors are not formed.
 *
 * Collective: all 32 lanes of a warp call it with the same matrix and a warp-private scratch tile.
 * Device only; the header holds nothing in a host-compiler translation unit.
 *
 * Cost note: every lane computes the rotation scalars redundantly, so on hardware with a slow
 * FP64 pipe the double form is slower than the one-thread form; it is meant for `float`.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/linalg/Eigh.h"


namespace aether {
namespace linalg {
namespace warp {

/** @brief Shared-memory scratch of one warp's `eighValues`: a padded 32 x 32 tile and the sorted values. */
template<class T>
struct EighValuesScratch {
    T s[32 * 33];  ///< The symmetric matrix, row `i` at `i * 33` (the pad keeps rows bank-conflict free).
    T w[32];       ///< The eigenvalues in sorted order.
};

/** @brief What `eighValues` returns to a lane. */
template<class T>
struct EighValues {
    T w;                 ///< This lane's eigenvalue: lane `l` holds the `l`-th smallest.
    std::int32_t info;   ///< As `Eigh::info` (bit 1 not converged, bit 2 close eigenvalues); the same on every lane.
};

/**
 * @brief Eigenvalues of the symmetrised `32 x 32` operand `a`, ascending, one lane each.
 *
 * @param sc Warp-private shared scratch; contents on entry are ignored and on return are scratch.
 * @param a  The matrix, read as `Eigh.h` reads it (`detail::symAt`).
 */
template<std::size_t N, class T, class A>
    requires(N == 32 && detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICE() EighValues<T> eighValues(EighValuesScratch<T>& sc, const A& a)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::warp: T must be float or double");
    constexpr std::size_t ld = 33;
    constexpr T eps = detail::epsilon_v<T>;
    const std::size_t lane = static_cast<std::size_t>(threadIdx.x) & 31u;
    T* const s = sc.s;
    for (std::size_t j = 0; j < N; ++j) {
        const std::size_t lo = lane < j ? lane : j, hi = lane < j ? j : lane;
        s[lane * ld + j] = detail::symAt<N>(a, lo, hi);
    }
    __syncwarp();
    // The sums of squares run in `Eigh.h`'s row-major order on one lane and are broadcast.
    T fro2 = T(0);
    if (lane == 0) {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                fro2 += s[i * ld + j] * s[i * ld + j];
    }
    fro2 = __shfl_sync(0xffffffffu, fro2, 0);
    constexpr std::size_t sweeps = eighSweeps(N);
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < N; ++p) {
            for (std::size_t q = p + 1; q < N; ++q) {
                const T app = s[p * ld + p];
                const T aqq = s[q * ld + q];
                const T apq = s[p * ld + q];
                if (!(math::abs(apq) <= eps * math::sqrt(math::abs(app * aqq)))) {
                    rotated = true;
                    const T theta = (aqq - app) / (T(2) * apq);
                    const T sg = theta < T(0) ? T(-1) : T(1);
                    const T t = sg / (math::abs(theta) + math::sqrt(theta * theta + T(1)));
                    const T c = T(1) / math::sqrt(t * t + T(1));
                    const T sn = t * c;
                    if (lane != p && lane != q) {
                        const T akp = s[lane * ld + p];
                        const T akq = s[lane * ld + q];
                        const T np = c * akp - sn * akq;
                        const T nq = sn * akp + c * akq;
                        s[lane * ld + p] = np;
                        s[p * ld + lane] = np;
                        s[lane * ld + q] = nq;
                        s[q * ld + lane] = nq;
                    }
                    if (lane == 0) {
                        s[p * ld + p] = app - t * apq;
                        s[q * ld + q] = aqq + t * apq;
                        s[p * ld + q] = T(0);
                        s[q * ld + p] = T(0);
                    }
                    __syncwarp();
                }
            }
        }
        if (!rotated) {
            break;  // a sweep that rotated nothing is followed by sweeps that do the same
        }
    }
    T off2 = T(0);
    if (lane == 0) {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                if (j != i)
                    off2 += s[i * ld + j] * s[i * ld + j];
    }
    off2 = __shfl_sync(0xffffffffu, off2, 0);
    const T mine = s[lane * ld + lane];
    sc.w[lane] = mine;
    __syncwarp();
    std::size_t rank = 0;
    for (std::size_t j = 0; j < N; ++j) {
        const T o = sc.w[j];
        rank += (o < mine) || (o == mine && j < lane);
    }
    __syncwarp();
    sc.w[rank] = mine;
    __syncwarp();
    const T tol = static_cast<T>(N) * eps;
    std::int32_t info = !(off2 <= tol * tol * fro2) ? 1 : 0;
    const T gap = tol * math::sqrt(fro2);
    if (lane + 1 < N && !(sc.w[lane + 1] - sc.w[lane] > gap)) {
        info |= 2;
    }
    for (int o = 16; o > 0; o >>= 1) {
        info |= __shfl_xor_sync(0xffffffffu, info, o);
    }
    __syncwarp();
    return {sc.w[lane], info};
}

} // namespace warp
} // namespace linalg
} // namespace aether
