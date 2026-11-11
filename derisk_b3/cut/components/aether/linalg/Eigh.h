// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Eigh.h
 * @brief `aether::linalg::Eigh`: eigenvalues and eigenvectors of the
 *        symmetrised input by cyclic Jacobi.
 *
 * The input is `S = (A + A^T) / 2`. Rotations run in a fixed order
 * (`p < q`, ascending) for a fixed number of sweeps (`eighSweeps`), each
 * rotation skipped when `|s_pq| <= eps * sqrt(|s_pp s_qq|)`; there is no
 * early exit, so the control flow does not depend on the data (`eighValues` alone stops after a
 * sweep that rotated nothing, which changes no bit). The
 * eigenvalues come back ascending and every eigenvector (a column of `v`) is
 * signed so its largest-magnitude component, the first of equals, is positive.
 *
 * `info` is a bit mask: bit 1 the off-diagonal part is still above
 * `N * eps * ||S||_F` after the sweeps (or a NaN got in), bit 2 two
 * neighbouring eigenvalues are within `N * eps * ||S||_F` of each other, so
 * their eigenvectors are not determined. The values are returned either way.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/linalg/detail/Common.h"
#include "aether/math/math.h"

namespace aether {
namespace linalg {

/** @brief Jacobi sweeps run for dimension `n`: 2, 6, 8, 12, 15 up to 2, 4, 8, 16, 32. */
AETHER_DEVICEHOST() constexpr std::size_t eighSweeps(std::size_t n)
{
    return n <= 2 ? 2 : n <= 4 ? 6 : n <= 8 ? 8 : n <= 16 ? 12 : 15;
}

/** @brief Result of `eigh`: ascending eigenvalues, eigenvectors as columns, status bits. */
template<class T, std::size_t N>
struct Eigh {
    T w[N];             ///< Eigenvalues, ascending.
    T v[N * N];         ///< Row-major; column `j` is the eigenvector of `w[j]`.
    std::int32_t info;  ///< Bit 1: not converged; bit 2: neighbouring eigenvalues too close.
};

/** @brief Result of `eighValues`: the eigenvalues and status bits of `Eigh`, without the eigenvectors. */
template<class T, std::size_t N>
struct EighValues {
    T w[N];             ///< Eigenvalues, ascending.
    std::int32_t info;  ///< As `Eigh::info`.
};

namespace detail {

/** @brief The Jacobi run behind `eigh` (`WithV`) and `eighValues` (eigenvalues and `info`, `v` unused). */
template<bool WithV, std::size_t N, class T, class A>
AETHER_DEVICEHOST() void eighRun(T* w, [[maybe_unused]] T* v, std::int32_t& info, const A& a)
{
    static_assert(supported_v<T, N>, "aether::linalg: T must be float or double and 1 <= N <= 32");
    info = 0;
    zeroFill(w, N);
    if constexpr (WithV) {
        zeroFill(v, N * N);
    }
    if constexpr (N == 1) {
        w[0] = elemAt<1>(a, 0, 0);
        if constexpr (WithV) {
            v[0] = T(1);
        }
    } else {
        constexpr T eps = detail::epsilon_v<T>;
        T s[N * N];
        T fro2 = T(0);
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            AETHER_NO_UNROLL()
            for (std::size_t j = 0; j < N; ++j) {
                s[i * N + j] = (j >= i) ? detail::symAt<N>(a, i, j) : s[j * N + i];
                fro2 += s[i * N + j] * s[i * N + j];
            }
            if constexpr (WithV) {
                v[i * N + i] = T(1);
            }
        }
        constexpr std::size_t sweeps = eighSweeps(N);
        AETHER_NO_UNROLL()
        for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
            bool rotated = false;
            AETHER_NO_UNROLL()
            for (std::size_t p = 0; p + 1 < N; ++p) {
                AETHER_NO_UNROLL()
                for (std::size_t q = p + 1; q < N; ++q) {
                    const T app = s[p * N + p];
                    const T aqq = s[q * N + q];
                    const T apq = s[p * N + q];
                    if (!(math::abs(apq) <= eps * math::sqrt(math::abs(app * aqq)))) {
                        const T theta = (aqq - app) / (T(2) * apq);
                        const T sg = theta < T(0) ? T(-1) : T(1);
                        const T t = sg / (math::abs(theta) + math::sqrt(theta * theta + T(1)));
                        const T c = T(1) / math::sqrt(t * t + T(1));
                        const T sn = t * c;
                        rotated = true;
                        s[p * N + p] = app - t * apq;
                        s[q * N + q] = aqq + t * apq;
                        s[p * N + q] = T(0);
                        s[q * N + p] = T(0);
                        AETHER_NO_UNROLL()
                        for (std::size_t k = 0; k < N; ++k) {
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
                            if constexpr (WithV) {
                                const T vp = v[k * N + p];
                                const T vq = v[k * N + q];
                                v[k * N + p] = c * vp - sn * vq;
                                v[k * N + q] = sn * vp + c * vq;
                            }
                        }
                    }
                }
            }
            if constexpr (!WithV) {
                if (!rotated) {
                    break;  // a sweep that rotated nothing leaves `s` as it is: so would every later one
                }
            }
        }
        T off2 = T(0);
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i < N; ++i) {
            w[i] = s[i * N + i];
            AETHER_NO_UNROLL()
            for (std::size_t j = 0; j < N; ++j) {
                if (j != i) {
                    off2 += s[i * N + j] * s[i * N + j];
                }
            }
        }
        const T tol = static_cast<T>(N) * eps;
        if (!(off2 <= tol * tol * fro2)) {
            info |= 1;
        }
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i + 1 < N; ++i) {
            std::size_t m = i;
            AETHER_NO_UNROLL()
            for (std::size_t j = i + 1; j < N; ++j) {
                if (w[j] < w[m]) {
                    m = j;
                }
            }
            if (m != i) {
                const T tw = w[i];
                w[i] = w[m];
                w[m] = tw;
                if constexpr (WithV) {
                    AETHER_NO_UNROLL()
                    for (std::size_t k = 0; k < N; ++k) {
                        const T tv = v[k * N + i];
                        v[k * N + i] = v[k * N + m];
                        v[k * N + m] = tv;
                    }
                }
            }
        }
        if constexpr (WithV) {
            AETHER_NO_UNROLL()
            for (std::size_t j = 0; j < N; ++j) {
                std::size_t big = 0;
                AETHER_NO_UNROLL()
                for (std::size_t k = 1; k < N; ++k) {
                    if (math::abs(v[k * N + j]) > math::abs(v[big * N + j])) {
                        big = k;
                    }
                }
                if (v[big * N + j] < T(0)) {
                    AETHER_NO_UNROLL()
                    for (std::size_t k = 0; k < N; ++k) {
                        v[k * N + j] = -v[k * N + j];
                    }
                }
            }
        }
        const T gap = tol * math::sqrt(fro2);
        AETHER_NO_UNROLL()
        for (std::size_t i = 0; i + 1 < N; ++i) {
            if (!(w[i + 1] - w[i] > gap)) {
                info |= 2;
            }
        }
    }
}

} // namespace detail

/**
 * @brief Eigendecomposition `S = V diag(w) V^T` of the symmetric part of `a` into
 *        `r`, in place: the result is built where it lives, never returned or
 *        copied by value.
 *
 * @tparam N  Dimension.
 * @param r   The decomposition, overwritten.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class T, class A>
    requires(detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICEHOST() void eigh(Eigh<T, N>& r, const A& a)
{
    detail::eighRun<true, N>(r.w, r.v, r.info, a);
}

/**
 * @brief The eigenvalues and `info` of `eigh`, bit for bit, without the eigenvectors: `V` is never
 *        accumulated, and the sweeps stop after one in which no rotation passed the skip test.
 *
 * @tparam N  Dimension.
 * @param r   Eigenvalues and status, overwritten.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class T, class A>
    requires(detail::operand<A> && std::is_same_v<T, detail::value_t<A>>)
AETHER_DEVICEHOST() void eighValues(EighValues<T, N>& r, const A& a)
{
    detail::eighRun<false, N>(r.w, static_cast<T*>(nullptr), r.info, a);
}

/**
 * @brief Eigendecomposition `S = V diag(w) V^T` of the symmetric part of `a`, returned by value.
 *
 * @tparam N  Dimension.
 * @param a   `N x N`, row-major.
 */
template<std::size_t N, class A>
    requires detail::operand<A>
AETHER_DEVICEHOST() Eigh<detail::value_t<A>, N> eigh(const A& a)
{
    Eigh<detail::value_t<A>, N> r;
    eigh<N>(r, a);
    return r;
}

} // namespace linalg
} // namespace aether
