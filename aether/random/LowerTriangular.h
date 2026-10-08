// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file LowerTriangular.h
 * @brief `aether::random::LowerTriangular<Real,VD>`: packed lower-triangular
 *        matrix (POD) plus a host `cholesky` factorisation, used by
 *        `Generator::multivariateNormal` to realise a correlated draw
 *        `x = mean + L z`.
 *
 * `L` (with `L L^T = Sigma`) is computed once on the host via
 * `aether::random::cholesky(covariance)` and captured by value into the draw
 * node (`detail::MultivariateNormalLeaf`), so no per-thread / per-sample
 * Cholesky ever runs on the device.
 */

#include <cmath>
#include <cstddef>

#include "aether/macros.h"

namespace aether {
namespace random {

/**
 * @brief Row-major packed lower-triangular matrix (component `(r,c)`, `c <=
 *        r`, stored at `r*(r+1)/2 + c`). Trivially copyable, so it passes
 *        into kernels / draw nodes by value.
 *
 * @tparam Real  Scalar type.
 * @tparam VD    Matrix dimension.
 */
template<class Real, std::size_t VD>
struct LowerTriangular {
    /** @brief Matrix dimension (rows == cols). */
    static constexpr std::size_t rows = VD;
    /** @brief Number of stored (lower-triangular) entries. */
    static constexpr std::size_t packedSize = VD * (VD + 1) / 2;

    Real data[packedSize];

    /** @brief Mutable access to entry `(r, c)` (requires `c <= r`). */
    AETHER_DEVICEHOST() constexpr Real& at(std::size_t r, std::size_t c) { return data[r * (r + 1) / 2 + c]; }
    /** @brief Read access to entry `(r, c)` (requires `c <= r`). */
    AETHER_DEVICEHOST() constexpr Real at(std::size_t r, std::size_t c) const { return data[r * (r + 1) / 2 + c]; }
};

/**
 * @brief Cholesky factorisation `Sigma = L L^T` (HOST ONLY — `std::sqrt`).
 *
 * @tparam Real  Scalar type.
 * @tparam VD    Dimension.
 * @param cov    Symmetric positive-definite covariance (row-major `VD x VD`).
 * @return Lower-triangular factor `L`.
 */
template<class Real, std::size_t VD>
inline LowerTriangular<Real, VD> cholesky(const Real (&cov)[VD][VD])
{
    LowerTriangular<Real, VD> L{};
    for (std::size_t r = 0; r < VD; ++r) {
        for (std::size_t c = 0; c <= r; ++c) {
            Real sum = cov[r][c];
            for (std::size_t k = 0; k < c; ++k)
                sum -= L.at(r, k) * L.at(c, k);
            if (r == c)
                L.at(r, c) = std::sqrt(sum);
            else
                L.at(r, c) = sum / L.at(c, c);
        }
    }
    return L;
}

} // namespace random
} // namespace aether
