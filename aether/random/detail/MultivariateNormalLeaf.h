// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file MultivariateNormalLeaf.h
 * @brief `aether::random::detail::MultivariateNormalLeaf` — correlated draw
 *        `x = mean + L z`.
 *
 * A correlated multivariate-normal draw cannot be evaluated one component
 * at a time in general: every component shares the same underlying
 * standard-normal vector `z`. This leaf takes the simple road: it is a
 * plain `PerComponentLeaf` — exactly the same shape as `UniformLeaf`/
 * `NormalLeaf`/... — whose `eval<I>` regenerates `z_0 .. z_I` from the
 * shared counters every time component `I` is read:
 *
 *     eval<I>(i) = mean[I] + sum_{c=0}^{I} L(I,c) * standardNormal(seed^salt, i.global(), c)
 *
 * Reading all `VD` components therefore redraws `z_0` `VD` times, `z_1`
 * `VD-1` times, ..., `O(VD^2)` draws per sample total — correct (every
 * component sees the same `z` sequence, since `standardNormal` is a pure
 * function of `(seed, global, counter)`) but not the cheapest possible
 * evaluation. This is the documented revival lever: a multi-load
 * assignment specialization that draws `z` once per sample and writes
 * every component together would cut the redundant draws to `O(VD)` —
 * worth building if/when a profile shows this leaf on a hot path.
 *
 * Because this leaf is an ordinary `PerComponentLeaf`, it composes exactly
 * like every other distribution leaf, and is picked up by the same packet
 * path (`detail/PacketSupport.h`) via the same `isRandomPerComponentLeaf`
 * marker.
 */

#include <cstddef>
#include <cstdint>

#include "aether/index/Offset.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/random/LowerTriangular.h"
#include "aether/random/detail/Backend.h"
#include "aether/random/detail/DistributionLeaf.h"
#include "aether/view/Item.h"

/// @cond INTERNAL

namespace aether {
namespace random {
namespace detail {

/**
 * @brief Correlated multivariate-normal draw node (`x = mean + L z`).
 *
 * @tparam Real  Scalar type.
 * @tparam VD    Dimension (not enforced here — larger `VD` simply costs more
 *               redundant `standardNormal` draws per read, see the file
 *               docstring).
 */
template<class Real, std::size_t VD>
class MultivariateNormalLeaf : public PerComponentLeaf<MultivariateNormalLeaf<Real, VD>, Real, VD> {
    using Base = PerComponentLeaf<MultivariateNormalLeaf<Real, VD>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt (fixed). */
    static constexpr std::uint64_t salt = 0xC0AC29B7C97C50DDULL;

    AETHER_DEVICEHOST() constexpr MultivariateNormalLeaf(
        std::uint64_t seed, const Item<Real, VD>& mean, const LowerTriangular<Real, VD>& L)
        : Base(seed)
        , L_{ L }
    {
        for (std::size_t k = 0; k < VD; ++k)
            mean_[k] = mean.data()[k];
    }

    /**
     * @brief L1 element protocol (canonical): component `I` of sample `i`.
     *        Regenerates `z_0..z_I` (see file docstring) rather than sharing
     *        a single draw across components — correct, not cheapest.
     */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "MultivariateNormalLeaf: rank-1 expression — exactly one component index");
        return row_<Is...>(this->seed_ ^ salt, i.global());
    }

    /** @brief Alias for `eval<I>(i)`. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Real get(const SampleIndex& i) const
    {
        return eval<I>(i);
    }
    /** @overload Positions by a flat sample index. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Real get(offset_t i) const
    {
        return eval<I>(SampleIndex::make(static_cast<std::size_t>(i)));
    }
    /** @brief INDEX-FREE draw: a fixed generator-local slot, in its own domain. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Real get() const
    {
        return row_<I>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u);
    }

private:
    /** @brief `mean[I] + sum_{c<=I} L(I,c) * standardNormal(seed, global, c)`. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Real row_(std::uint64_t seed, offset_t global) const
    {
        static_assert(I < VD, "MultivariateNormalLeaf: invalid component index");
        Real acc = mean_[I];
        for (std::size_t c = 0; c <= I; ++c)
            acc += L_.at(I, c) * standardNormal<Real>(seed, global, static_cast<std::uint64_t>(c));
        return acc;
    }

    Real mean_[VD];
    LowerTriangular<Real, VD> L_;
};

} // namespace detail
} // namespace random
} // namespace aether

/// @endcond
