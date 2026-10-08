// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file free.h
 * @brief Ambient / free-function spelling for `aether::random`.
 *
 * A zero-handle convenience over the `Generator` handle. The free
 * distribution functions use a fixed compile-time `DEFAULT_SEED`
 * (device-safe, header-only, no mutable global state), and `seed(s)` is a
 * named `Generator` factory for controlled seeding:
 * @code
 * out[i] = aether::random::normal<double, 3>(0.0, 1.0);            // default seed
 * out[i] = aether::random::seed(42).normal<double, 3>(0.0, 1.0);   // chosen seed
 * @endcode
 *
 * There is intentionally no settable ambient seed: a host-set device global
 * would hit the multi-DSO "invalid device symbol" pitfall in shared-library
 * / nanobind builds. Use `seed(s)` or a `Generator` directly when the seed
 * matters.
 *
 * Scalar parameters only: matching `Generator`'s own narrowing, no
 * `Item<Real,VD>`-valued free-function overload is provided — call
 * `Generator(...)`/`seed(...)` directly for a per-component parameter.
 */

#include <cstddef>
#include <cstdint>

#include "aether/macros.h"
#include "aether/random/Generator.h"
#include "aether/random/LowerTriangular.h"
#include "aether/view/Item.h"

namespace aether {
namespace random {

/** @brief Seed used by the free distribution functions (fixed, reproducible). */
inline constexpr std::uint64_t DEFAULT_SEED = 0xFE7A5EED0C0FFEE5ULL;

/** @brief Named `Generator` factory — `aether::random::seed(s)` == `Generator(s)`. */
AETHER_DEVICEHOST() inline Generator seed(std::uint64_t s) { return Generator(s); }

// ----- uniform ---------------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto uniform(Real lo, Real hi)
{
    return Generator(DEFAULT_SEED).uniform<Real, VD>(lo, hi);
}

// ----- normal ------------------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto normal(Real mean, Real sd)
{
    return Generator(DEFAULT_SEED).normal<Real, VD>(mean, sd);
}

// ----- lognormal -----------------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto lognormal(Real mu, Real sigma)
{
    return Generator(DEFAULT_SEED).lognormal<Real, VD>(mu, sigma);
}

// ----- exponential ---------------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto exponential(Real lambda)
{
    return Generator(DEFAULT_SEED).exponential<Real, VD>(lambda);
}

// ----- bernoulli -----------------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto bernoulli(Real p)
{
    return Generator(DEFAULT_SEED).bernoulli<Real, VD>(p);
}

// ----- uniformInt ------------------------------------------------------------
template<class Int, std::size_t VD>
AETHER_DEVICEHOST() auto uniformInt(Int lo, Int hi)
{
    return Generator(DEFAULT_SEED).uniformInt<Int, VD>(lo, hi);
}

// ----- multivariateNormal ----------------------------------------------------
template<class Real, std::size_t VD>
AETHER_DEVICEHOST() auto multivariateNormal(const Item<Real, VD>& mean, const LowerTriangular<Real, VD>& L)
{
    return Generator(DEFAULT_SEED).multivariateNormal<Real, VD>(mean, L);
}

} // namespace random
} // namespace aether
