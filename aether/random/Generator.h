// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Generator.h
 * @brief `aether::random::Generator` — the seed-carrying handle and
 *        distribution factory for the aether random module.
 *
 * A `Generator` is a tiny POD carrying one mixed 64-bit key. Build it on the
 * host, pass it into kernels or host loops BY VALUE (exactly like a `View`);
 * there is no global mutable state anywhere in this module, so results are
 * reproducible and independent of thread / kernel-launch order.
 *
 * @code
 * aether::random::Generator rng(0xC0FFEEu);
 * out[i] = rng.normal<double, 3>(0.0, 1.0);      // per-component, composes
 * aether::Vec3d local = rng.substream(7).uniform<double, 3>(-1.0, 1.0);
 * @endcode
 *
 * Consumer spelling: `Generator`, `substream(k)`, `.uniform<Real,D>(lo,hi)`,
 * `.normal<Real,D>(mean,sd)`. Two deliberate narrowings:
 *
 *  - **Parameters are a scalar or an `Item<Real,VD>`**. A fully general
 *    expression-valued overload set is not provided: every measured call
 *    site passes a plain scalar or a small vector. See
 *    `detail/DistributionLeaf.h`'s own note.
 *  - **`uniformInt`/`multivariateNormal` narrow further**: `uniformInt` has
 *    only the scalar `lo`/`hi` overload; `multivariateNormal` has only the
 *    `(Item<Real,VD> mean, LowerTriangular<Real,VD> L)` overload — a
 *    correlated draw's mean is inherently per-component, so there is no
 *    scalar form to offer.
 *
 * One seed names one stream on both arms — see `detail/Backend.h`.
 */

#include <cstdint>
#include <cstddef>

#include "aether/macros.h"
#include "aether/random/LowerTriangular.h"
#include "aether/random/detail/Backend.h"
#include "aether/random/detail/DistributionLeaf.h"
#include "aether/random/detail/MultivariateNormalLeaf.h"
#include "aether/view/Item.h"

namespace aether {
namespace random {

/**
 * @brief Seed-carrying RNG handle and distribution factory.
 *
 * Trivially copyable POD (one `std::uint64_t`); pass by value into kernels.
 */
class Generator {
public:
    /**
     * @brief Construct from a user seed and an optional stream id.
     *
     * @param seed      64-bit seed.
     * @param streamId  Optional independent-stream selector (default 0).
     */
    AETHER_DEVICEHOST() constexpr explicit Generator(std::uint64_t seed, std::uint64_t streamId = 0)
        : key_{ detail::splitmix64(seed ^ detail::splitmix64(streamId)) }
    {
    }

    /**
     * @brief An independent sub-stream generator — a PURE function of `k`.
     *
     * Distinct `k` give decorrelated streams, which is the natural way to get
     * many distinct local draws (e.g. what an SDE integrator's Wiener-
     * increment/Poisson-jump chain relies on): `substream(k)` never depends
     * on how many times it has been called, only on `k`. Keying is fixed,
     * verbatim.
     */
    AETHER_DEVICEHOST() constexpr Generator substream(std::uint64_t k) const
    {
        return Generator::fromKey_(detail::splitmix64(key_ ^ detail::splitmix64(k ^ salt_)));
    }

    // ----- uniform: U(lo, hi] --------------------------------------------

    /** @brief Per-component uniform draws with a shared scalar range. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto uniform(Real lo, Real hi) const
    {
        return detail::UniformLeaf<Real, VD, Real, Real>(key_, lo, hi);
    }
    /** @brief Per-component uniform draws with per-component ranges. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto uniform(const Item<Real, VD>& lo, const Item<Real, VD>& hi) const
    {
        return detail::UniformLeaf<Real, VD, Item<Real, VD>, Item<Real, VD>>(key_, lo, hi);
    }

    // ----- normal: N(mean, sd^2) -----------------------------------------

    /** @brief Per-component normal draws with shared scalar parameters. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto normal(Real mean, Real sd) const
    {
        return detail::NormalLeaf<Real, VD, Real, Real>(key_, mean, sd);
    }
    /** @brief Per-component normal draws with per-component parameters. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto normal(const Item<Real, VD>& mean, const Item<Real, VD>& sd) const
    {
        return detail::NormalLeaf<Real, VD, Item<Real, VD>, Item<Real, VD>>(key_, mean, sd);
    }

    // ----- lognormal: exp(N(mu, sigma^2)) ---------------------------------

    /** @brief Per-component lognormal draws with shared scalar parameters. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto lognormal(Real mu, Real sigma) const
    {
        return detail::LognormalLeaf<Real, VD, Real, Real>(key_, mu, sigma);
    }
    /** @brief Per-component lognormal draws with per-component parameters. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto lognormal(const Item<Real, VD>& mu, const Item<Real, VD>& sigma) const
    {
        return detail::LognormalLeaf<Real, VD, Item<Real, VD>, Item<Real, VD>>(key_, mu, sigma);
    }

    // ----- exponential: -log(u) / lambda -----------------------------------

    /** @brief Per-component exponential draws with a shared scalar rate. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto exponential(Real lambda) const
    {
        return detail::ExponentialLeaf<Real, VD, Real>(key_, lambda);
    }
    /** @brief Per-component exponential draws with per-component rates. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto exponential(const Item<Real, VD>& lambda) const
    {
        return detail::ExponentialLeaf<Real, VD, Item<Real, VD>>(key_, lambda);
    }

    // ----- bernoulli: P(1) = p ---------------------------------------------

    /** @brief Per-component bernoulli draws with a shared scalar rate. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto bernoulli(Real p) const
    {
        return detail::BernoulliLeaf<Real, VD, Real>(key_, p);
    }
    /** @brief Per-component bernoulli draws with per-component rates. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto bernoulli(const Item<Real, VD>& p) const
    {
        return detail::BernoulliLeaf<Real, VD, Item<Real, VD>>(key_, p);
    }

    // ----- uniformInt: integer in [lo, hi] (scalar only) -------------------

    /** @brief Per-component integer draws, uniform over `[lo, hi]`. */
    template<class Int, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto uniformInt(Int lo, Int hi) const
    {
        return detail::UniformIntLeaf<Int, VD>(key_, lo, hi);
    }

    // ----- multivariateNormal: x = mean + L z (correlated) -----------------

    /** @brief Correlated draw `x = mean + L z`; assign standalone or materialise into an `Item`. */
    template<class Real, std::size_t VD>
    AETHER_DEVICEHOST() constexpr auto multivariateNormal(
        const Item<Real, VD>& mean, const LowerTriangular<Real, VD>& L) const
    {
        return detail::MultivariateNormalLeaf<Real, VD>(key_, mean, L);
    }

    /** @brief The generator's mixed key — the value every draw is positioned by. */
    AETHER_DEVICEHOST() constexpr std::uint64_t key() const { return key_; }

private:
    AETHER_DEVICEHOST() constexpr Generator()
        : key_{ 0 }
    {
    }
    AETHER_DEVICEHOST() static constexpr Generator fromKey_(std::uint64_t key)
    {
        Generator g;
        g.key_ = key;
        return g;
    }

    /** @brief `substream`'s decorrelation salt (fixed). */
    static constexpr std::uint64_t salt_ = 0x2545F4914F6CDD1DULL;
    std::uint64_t key_;
};

} // namespace random
} // namespace aether
