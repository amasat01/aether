// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DistributionLeaf.h
 * @brief `aether::random::detail::UniformLeaf`/`NormalLeaf`/`LognormalLeaf`/
 *        `ExponentialLeaf`/`BernoulliLeaf`/`UniformIntLeaf` — per-component
 *        distribution leaves: aether expressions whose every component is an
 *        independent draw.
 *
 * Each leaf follows the rank-generic expression protocol
 * (`aether/expr/Expression.h`): `eval<Is...>(SampleIndex)` is the canonical
 * accessor, with `get<I>(...)` spellings built on top of it. `isLeaf = false`
 * makes every composite capture the leaf by value, so a temporary returned by
 * a `Generator` factory cannot dangle inside a larger expression.
 *
 * A draw is a pure function of `(seed ^ salt, i.global(), dim)`, independent
 * of thread id, launch shape or evaluation order, so a parallel fill and a
 * serial fill are bit-identical. The index-free `get<I>()` overload draws
 * from a separate domain (seed XORed with `INDEX_FREE_DOMAIN`), so it never
 * collides with an indexed array fill at sample 0.
 *
 * Distribution parameters are each either a broadcast scalar or an
 * `Item<Real,VD>` evaluated per component.
 */

#include <cstdint>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/Offset.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/macros.h"
#include "aether/random/detail/Backend.h"

/// @cond INTERNAL

namespace aether {
namespace random {
namespace detail {

/**
 * @brief Domain constant XORed into the seed for index-free (local) draws so
 *        they never collide with an indexed array fill at the same position.
 */
inline constexpr std::uint64_t INDEX_FREE_DOMAIN = 0xD1B54A32D192ED03ULL;

/**
 * @brief Read component `Is...` of a distribution parameter.
 *
 * A scalar parameter broadcasts (every component sees the same value); an
 * expression parameter (in practice an `Item<Real,VD>` by value) is evaluated
 * component-wise. One helper rather than two leaf specializations keeps the
 * leaves readable and keeps the broadcast decision at ONE place.
 */
template<std::size_t... Is, class P>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr auto paramValue(const P& p, const SampleIndex& i)
{
    if constexpr (aether_expression<P>) {
        return p.template eval<Is...>(i);
    } else {
        (void)i;
        return p;
    }
}

/**
 * @brief Common shape of every per-component distribution leaf: element type,
 *        rank-1 `element_extents`, the capture-by-value `isLeaf = false`
 *        convention, and the seed slot.
 *
 * CRTP through `Expression` (not deducing-this: nvcc's device compilation
 * targets C++20), so `Derived` must be the concrete leaf.
 */
template<class Derived, class Real, std::size_t VD>
class PerComponentLeaf : public Expression<Derived, Real> {
public:
    /** @brief Scalar element type of the draws. */
    using element_type = Real;
    /** @brief Rank-1 element shape: `VD` independent components. */
    using element_extents = extents<VD>;
    /** @brief Capture by value in composites. */
    static constexpr bool isLeaf = false;
    /**
     * @brief Marks every per-component random leaf so the packet path's
     *        ADL-found `packetGet` overload can find it generically. Every
     *        leaf in this module, including `MultivariateNormalLeaf` (which
     *        also derives from this base), inherits it from here.
     */
    static constexpr bool isRandomPerComponentLeaf = true;

protected:
    AETHER_DEVICEHOST() constexpr explicit PerComponentLeaf(std::uint64_t seed)
        : seed_{ seed }
    {
    }

    std::uint64_t seed_;
};

/**
 * @brief `lo + (hi - lo) * U(0,1]` per component.
 *
 * `LoE`/`HiE` are each either `Real` (broadcast scalar) or `Item<Real,VD>`.
 * The salt decorrelates this distribution's stream from every other one
 * drawn from the same generator.
 */
template<class Real, std::size_t VD, class LoE, class HiE>
class UniformLeaf : public PerComponentLeaf<UniformLeaf<Real, VD, LoE, HiE>, Real, VD> {
    using Base = PerComponentLeaf<UniformLeaf<Real, VD, LoE, HiE>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0x243F6A8885A308D3ULL;

    AETHER_DEVICEHOST() constexpr UniformLeaf(std::uint64_t seed, const LoE& lo, const HiE& hi)
        : Base(seed)
        , lo_{ lo }
        , hi_{ hi }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "UniformLeaf: rank-1 expression — exactly one component index");
        const Real lo = static_cast<Real>(paramValue<Is...>(lo_, i));
        const Real hi = static_cast<Real>(paramValue<Is...>(hi_, i));
        return lo + (hi - lo) * uniform01<Real>(this->seed_ ^ salt, i.global(), counter_<Is...>());
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
        const SampleIndex i0 = SampleIndex::make(std::size_t{ 0 });
        const Real lo        = static_cast<Real>(paramValue<I>(lo_, i0));
        const Real hi        = static_cast<Real>(paramValue<I>(hi_, i0));
        return lo + (hi - lo) * uniform01<Real>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u, counter_<I>());
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() static constexpr std::uint64_t counter_()
    {
        static_assert(I < VD, "UniformLeaf: invalid component index");
        return static_cast<std::uint64_t>(I);
    }

    // Captured by value: a parameter is either a scalar or a small
    // `Item<Real,VD>`, and a `Generator` factory's return value routinely
    // outlives the call expression that built its arguments.
    const LoE lo_;
    const HiE hi_;
};

/**
 * @brief `mean + sd * N(0,1)` per component.
 *
 * `MeanE`/`SdE` are each either `Real` (broadcast scalar) or `Item<Real,VD>`.
 */
template<class Real, std::size_t VD, class MeanE, class SdE>
class NormalLeaf : public PerComponentLeaf<NormalLeaf<Real, VD, MeanE, SdE>, Real, VD> {
    using Base = PerComponentLeaf<NormalLeaf<Real, VD, MeanE, SdE>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0x13198A2E03707344ULL;

    AETHER_DEVICEHOST() constexpr NormalLeaf(std::uint64_t seed, const MeanE& mean, const SdE& sd)
        : Base(seed)
        , mean_{ mean }
        , sd_{ sd }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "NormalLeaf: rank-1 expression — exactly one component index");
        const Real mean = static_cast<Real>(paramValue<Is...>(mean_, i));
        const Real sd   = static_cast<Real>(paramValue<Is...>(sd_, i));
        return mean + sd * standardNormal<Real>(this->seed_ ^ salt, i.global(), counter_<Is...>());
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
        const SampleIndex i0 = SampleIndex::make(std::size_t{ 0 });
        const Real mean      = static_cast<Real>(paramValue<I>(mean_, i0));
        const Real sd        = static_cast<Real>(paramValue<I>(sd_, i0));
        return mean + sd * standardNormal<Real>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u, counter_<I>());
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() static constexpr std::uint64_t counter_()
    {
        static_assert(I < VD, "NormalLeaf: invalid component index");
        return static_cast<std::uint64_t>(I);
    }

    // Captured by value — see UniformLeaf's note.
    const MeanE mean_;
    const SdE sd_;
};

/**
 * @brief `exp(mu + sigma * N(0,1))` per component.
 *
 * `MuE`/`SigmaE` are each either `Real` (broadcast scalar) or `Item<Real,VD>`.
 */
template<class Real, std::size_t VD, class MuE, class SigmaE>
class LognormalLeaf : public PerComponentLeaf<LognormalLeaf<Real, VD, MuE, SigmaE>, Real, VD> {
    using Base = PerComponentLeaf<LognormalLeaf<Real, VD, MuE, SigmaE>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0xA4093822299F31D0ULL;

    AETHER_DEVICEHOST() constexpr LognormalLeaf(std::uint64_t seed, const MuE& mu, const SigmaE& sigma)
        : Base(seed)
        , mu_{ mu }
        , sigma_{ sigma }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "LognormalLeaf: rank-1 expression — exactly one component index");
        const Real mu    = static_cast<Real>(paramValue<Is...>(mu_, i));
        const Real sigma = static_cast<Real>(paramValue<Is...>(sigma_, i));
        const Real z     = standardNormal<Real>(this->seed_ ^ salt, i.global(), counter_<Is...>());
        return math::exp(mu + sigma * z);
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
        const SampleIndex i0 = SampleIndex::make(std::size_t{ 0 });
        const Real mu         = static_cast<Real>(paramValue<I>(mu_, i0));
        const Real sigma      = static_cast<Real>(paramValue<I>(sigma_, i0));
        const Real z          = standardNormal<Real>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u, counter_<I>());
        return math::exp(mu + sigma * z);
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() static constexpr std::uint64_t counter_()
    {
        static_assert(I < VD, "LognormalLeaf: invalid component index");
        return static_cast<std::uint64_t>(I);
    }

    // Captured by value — see UniformLeaf's note.
    const MuE mu_;
    const SigmaE sigma_;
};

/**
 * @brief `-log(u) / lambda` per component.
 *
 * No lower-endpoint guard is needed: `uniform01` is `(0,1]` on both arms, so
 * `u` is always strictly positive and `log(u)` is always safe.
 */
template<class Real, std::size_t VD, class LambdaE>
class ExponentialLeaf : public PerComponentLeaf<ExponentialLeaf<Real, VD, LambdaE>, Real, VD> {
    using Base = PerComponentLeaf<ExponentialLeaf<Real, VD, LambdaE>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0x082EFA98EC4E6C89ULL;

    AETHER_DEVICEHOST() constexpr ExponentialLeaf(std::uint64_t seed, const LambdaE& lambda)
        : Base(seed)
        , lambda_{ lambda }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "ExponentialLeaf: rank-1 expression — exactly one component index");
        const Real u      = uniform01<Real>(this->seed_ ^ salt, i.global(), counter_<Is...>());
        const Real lambda = static_cast<Real>(paramValue<Is...>(lambda_, i));
        return -math::log(u) / lambda;
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
        const SampleIndex i0 = SampleIndex::make(std::size_t{ 0 });
        const Real u          = uniform01<Real>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u, counter_<I>());
        const Real lambda     = static_cast<Real>(paramValue<I>(lambda_, i0));
        return -math::log(u) / lambda;
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() static constexpr std::uint64_t counter_()
    {
        static_assert(I < VD, "ExponentialLeaf: invalid component index");
        return static_cast<std::uint64_t>(I);
    }

    // Captured by value — see UniformLeaf's note.
    const LambdaE lambda_;
};

/**
 * @brief `(u < p) ? 1 : 0` per component, returned as `Real`.
 */
template<class Real, std::size_t VD, class PE>
class BernoulliLeaf : public PerComponentLeaf<BernoulliLeaf<Real, VD, PE>, Real, VD> {
    using Base = PerComponentLeaf<BernoulliLeaf<Real, VD, PE>, Real, VD>;

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0x452821E638D01377ULL;

    AETHER_DEVICEHOST() constexpr BernoulliLeaf(std::uint64_t seed, const PE& p)
        : Base(seed)
        , p_{ p }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Real eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "BernoulliLeaf: rank-1 expression — exactly one component index");
        const Real u = uniform01<Real>(this->seed_ ^ salt, i.global(), counter_<Is...>());
        const Real p = static_cast<Real>(paramValue<Is...>(p_, i));
        return u < p ? Real(1) : Real(0);
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
        const SampleIndex i0 = SampleIndex::make(std::size_t{ 0 });
        const Real u          = uniform01<Real>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u, counter_<I>());
        const Real p          = static_cast<Real>(paramValue<I>(p_, i0));
        return u < p ? Real(1) : Real(0);
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() static constexpr std::uint64_t counter_()
    {
        static_assert(I < VD, "BernoulliLeaf: invalid component index");
        return static_cast<std::uint64_t>(I);
    }

    // Captured by value — see UniformLeaf's note.
    const PE p_;
};

/**
 * @brief `lo + (bits % (hi - lo + 1))` per component, integer-only: draws
 *        via `randomBits64`, never `uniform01`, so this leaf never touches
 *        floating point. The modulo mapping's bias is `< range / 2^64`,
 *        immaterial for any representable `[lo, hi]`.
 *
 * Only the scalar `lo`/`hi` form is provided; `Generator::uniformInt` has no
 * expression-valued overload.
 */
template<class Int, std::size_t VD>
class UniformIntLeaf : public PerComponentLeaf<UniformIntLeaf<Int, VD>, Int, VD> {
    using Base = PerComponentLeaf<UniformIntLeaf<Int, VD>, Int, VD>;
    static_assert(std::is_integral_v<Int>, "uniformInt requires an integer type");

public:
    /** @brief Per-distribution decorrelation salt. */
    static constexpr std::uint64_t salt = 0xBE5466CF34E90C6CULL;

    AETHER_DEVICEHOST() constexpr UniformIntLeaf(std::uint64_t seed, Int lo, Int hi)
        : Base(seed)
        , lo_{ lo }
        , hi_{ hi }
    {
    }

    /** @brief Canonical accessor: component `I` of sample `i`. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr Int eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == 1, "UniformIntLeaf: rank-1 expression — exactly one component index");
        return sample_<Is...>(this->seed_ ^ salt, i.global());
    }

    /** @brief Alias for `eval<I>(i)`. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Int get(const SampleIndex& i) const
    {
        return eval<I>(i);
    }
    /** @overload Positions by a flat sample index. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Int get(offset_t i) const
    {
        return eval<I>(SampleIndex::make(static_cast<std::size_t>(i)));
    }
    /** @brief INDEX-FREE draw: a fixed generator-local slot, in its own domain. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Int get() const
    {
        return sample_<I>(this->seed_ ^ salt ^ INDEX_FREE_DOMAIN, 0u);
    }

private:
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr Int sample_(std::uint64_t seed, offset_t pos) const
    {
        static_assert(I < VD, "UniformIntLeaf: invalid component index");
        const std::uint64_t bits  = randomBits64(seed, pos, static_cast<std::uint64_t>(I));
        const std::uint64_t range = static_cast<std::uint64_t>(static_cast<long>(hi_) - static_cast<long>(lo_)) + 1u;
        return static_cast<Int>(static_cast<long>(lo_) + static_cast<long>(bits % range));
    }

    // Captured by value — see UniformLeaf's note.
    const Int lo_;
    const Int hi_;
};

} // namespace detail
} // namespace random
} // namespace aether

/// @endcond
