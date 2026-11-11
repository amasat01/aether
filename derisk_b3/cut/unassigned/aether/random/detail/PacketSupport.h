// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketSupport.h
 * @brief CPU-SIMD ("packet") support for the random leaves — host-only.
 *
 * One `packetGet` overload, keyed on the `isRandomPerComponentLeaf` marker
 * every leaf in this module inherits (`DistributionLeaf.h`'s
 * `PerComponentLeaf` base), covers Uniform, Normal, Lognormal, Exponential,
 * Bernoulli, UniformInt and MultivariateNormal alike.
 *
 * Mechanism. `aether::detail::RecursivePacketAssign::eval`
 * (`backend/cpu/packet/Assign.h`) calls unqualified `packetGet<Is...>(from,
 * pi)`; when `from` is one of our leaves (in `aether::random::detail`),
 * ordinary lookup plus argument-dependent lookup finds this overload with
 * no other wiring. `Leaf::isRandomPerComponentLeaf` is `requires`-gated so
 * this overload set never competes with `View`/`Item`/`PacketItem`/`Sum`/
 * `CWiseScale`'s own overloads (`backend/cpu/packet/LoadStore.h`) for an
 * unrelated leaf type.
 *
 * Bit-identity by construction. Every lane `k` of a packet calls the leaf's
 * own scalar `eval<Is...>(SampleIndex::make(pi.base_ + k))` — the same
 * function `View::operator[] = expr` calls for a plain scalar fill — so the
 * packed result is bit-identical to the scalar path by construction, not by
 * a numerically-equal-but-differently-computed formula. The tail
 * (`!pi.full()`) uses a masked load over a zero-initialised buffer so no
 * uninitialised lane is ever read. No vectorized Philox: `n` independent
 * scalar Philox streams are drawn and then packed.
 */

#include <cstddef>

#include "aether/backend/cpu/simd/simd.h"
#include "aether/index/SampleIndex.h"
#include "aether/random/detail/DistributionLeaf.h"

/// @cond INTERNAL

namespace aether {
namespace random {
namespace detail {

/**
 * @brief Packet load for ANY per-component random leaf: `W` per-lane scalar
 *        draws packed via store-array-then-load (`simd::Packet` has no lane
 *        setter, so there is no other way to fill one).
 */
template<std::size_t... Is, class Leaf, std::size_t W>
    requires(Leaf::isRandomPerComponentLeaf)
simd::Packet<typename Leaf::element_type, W> packetGet(const Leaf& leaf, const PacketIndex<W>& pi)
{
    using DataT = typename Leaf::element_type;
    DataT tmp[W] = {};
    const std::size_t n = pi.active_;
    for (std::size_t k = 0; k < n; ++k)
        tmp[k] = leaf.template eval<Is...>(SampleIndex::make(pi.base_ + k));
    if (pi.full())
        return simd::Packet<DataT, W>::load(tmp);
    return simd::Packet<DataT, W>::maskLoad(tmp, simd::PacketMask<DataT, W>::firstN(n));
}

} // namespace detail
} // namespace random
} // namespace aether

/// @endcond
