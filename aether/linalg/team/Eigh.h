// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Eigh.h
 * @brief `aether::linalg::team::eigh`: eigenvalues and eigenvectors of the
 *        symmetrised input by cyclic Jacobi, run by a team.
 *
 * The rotations keep the one-thread order (`p < q`, ascending, `eighSweeps(N)`
 * sweeps, the same skip test) and one thread computes each rotation; the
 * team splits the rotation's `k` loop, whose entries are independent. Every
 * entry therefore sees the same arithmetic as in `linalg::eigh`: the result
 * agrees with the one-thread result to a few ulp (a compiler may contract the
 * two loops differently) and is bitwise independent of the team size. The rotations are sequential, so this is slow at large `N`.
 * `info` is the same bit mask.
 *
 * Scratch: `w` `N`, `v` `N*N`, `work` `eighWorkSize(N)` elements, `info` 1.
 */

#include "aether/linalg/Eigh.h"
#include "aether/linalg/team/Team.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {
namespace team {

/** @brief Views into the scratch of one `N x N` eigendecomposition. */
template<class T, std::size_t N>
struct TeamEigh {
    T* w;                 ///< `N`: eigenvalues, ascending.
    T* v;                 ///< `N*N`, row-major; column `j` is the eigenvector of `w[j]`.
    T* work;              ///< `eighWorkSize(N)`: the working symmetric matrix and the rotation.
    std::int32_t* info;   ///< 1: bit 1 not converged, bit 2 neighbouring eigenvalues too close.
};

/** @brief Decompose the symmetric part of `a_in` (`N x N`, row-major) into `r`. Collective. */
template<std::size_t N, class T>
AETHER_DEVICEHOST() void eigh(const Team& tm, const TeamEigh<T, N>& r, const T* a_in)
{
    static_assert(detail::supported_v<T, N>, "aether::linalg::team: T must be float or double and 1 <= N <= 4096");
    const std::size_t rank = static_cast<std::size_t>(tm.rank);
    const std::size_t size = static_cast<std::size_t>(tm.size);
    T* v = r.v;
    T* s = r.work;
    T* rot = r.work + N * N;  // c, sn
    constexpr T eps = linalg::detail::epsilon_v<T>;
    for (std::size_t e = rank; e < N * N; e += size) {
        const std::size_t i = e / N;
        const std::size_t j = e % N;
        s[e] = T(0.5) * a_in[i * N + j] + T(0.5) * a_in[j * N + i];
        v[e] = (i == j) ? T(1) : T(0);
    }
    tm.sync();
    if constexpr (N == 1) {
        if (rank == 0) {
            r.w[0] = a_in[0];
            *r.info = 0;
        }
        tm.sync();
        return;
    } else {
        T fro2 = T(0);
        for (std::size_t e = 0; e < N * N; ++e) {  // every thread, same order: no broadcast needed
            fro2 += s[e] * s[e];
        }
        tm.sync();  // every thread has read s before the first rotation writes it
        constexpr std::size_t sweeps = linalg::eighSweeps(N);
        for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
            for (std::size_t p = 0; p + 1 < N; ++p) {
                for (std::size_t q = p + 1; q < N; ++q) {
                    if (rank == 0) {
                        const T app = s[p * N + p];
                        const T aqq = s[q * N + q];
                        const T apq = s[p * N + q];
                        if (!(math::abs(apq) <= eps * math::sqrt(math::abs(app * aqq)))) {
                            const T theta = (aqq - app) / (T(2) * apq);
                            const T sg = theta < T(0) ? T(-1) : T(1);
                            const T t = sg / (math::abs(theta) + math::sqrt(theta * theta + T(1)));
                            const T c = T(1) / math::sqrt(t * t + T(1));
                            rot[0] = c;
                            rot[1] = t * c;
                            s[p * N + p] = app - t * apq;
                            s[q * N + q] = aqq + t * apq;
                            s[p * N + q] = T(0);
                            s[q * N + p] = T(0);
                        } else {
                            rot[0] = T(0);  // c is positive for a real rotation: 0 marks a skip
                        }
                    }
                    tm.sync();
                    const T c = rot[0];
                    if (c != T(0)) {
                        const T sn = rot[1];
                        for (std::size_t k = rank; k < N; k += size) {
                            if (k != p && k != q) {
                                const T akp = s[k * N + p];
                                const T akq = s[k * N + q];
                                const T np = c * akp - sn * akq;
                                const T nq = sn * akp + c * akq;
                                s[k * N + p] = np;
                                s[p * N + k] = np;
                                s[k * N + q] = nq;
                                s[q * N + k] = nq;
                            }
                            const T vp = v[k * N + p];
                            const T vq = v[k * N + q];
                            v[k * N + p] = c * vp - sn * vq;
                            v[k * N + q] = sn * vp + c * vq;
                        }
                    }
                    tm.sync();
                }
            }
        }
        if (rank == 0) {
            T off2 = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                r.w[i] = s[i * N + i];
                for (std::size_t j = 0; j < N; ++j) {
                    if (j != i) {
                        off2 += s[i * N + j] * s[i * N + j];
                    }
                }
            }
            std::int32_t info = 0;
            const T tol = static_cast<T>(N) * eps;
            if (!(off2 <= tol * tol * fro2)) {
                info |= 1;
            }
            for (std::size_t i = 0; i + 1 < N; ++i) {
                std::size_t m = i;
                for (std::size_t j = i + 1; j < N; ++j) {
                    if (r.w[j] < r.w[m]) {
                        m = j;
                    }
                }
                if (m != i) {
                    const T tw = r.w[i];
                    r.w[i] = r.w[m];
                    r.w[m] = tw;
                    for (std::size_t k = 0; k < N; ++k) {
                        const T tv = v[k * N + i];
                        v[k * N + i] = v[k * N + m];
                        v[k * N + m] = tv;
                    }
                }
            }
            const T gap = tol * math::sqrt(fro2);
            for (std::size_t i = 0; i + 1 < N; ++i) {
                if (!(r.w[i + 1] - r.w[i] > gap)) {
                    info |= 2;
                }
            }
            *r.info = info;
        }
        tm.sync();
        for (std::size_t j = rank; j < N; j += size) {
            std::size_t big = 0;
            for (std::size_t k = 1; k < N; ++k) {
                if (math::abs(v[k * N + j]) > math::abs(v[big * N + j])) {
                    big = k;
                }
            }
            if (v[big * N + j] < T(0)) {
                for (std::size_t k = 0; k < N; ++k) {
                    v[k * N + j] = -v[k * N + j];
                }
            }
        }
        tm.sync();
    }
}

} // namespace team
} // namespace linalg
} // namespace aether
