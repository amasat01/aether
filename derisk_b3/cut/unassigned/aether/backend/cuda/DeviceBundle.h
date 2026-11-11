// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DeviceBundle.h
 * @brief `aether::DeviceBundle`: the device analog of `backend/cpu/
 *        PacketItem.h`'s `PacketItem` — `W` samples per thread,
 *        register-resident (GPUs have no SIMD registers, so
 *        "vectorization" here means `W` separate scalars per element that
 *        the compiler keeps live in registers; the actual 128-bit
 *        vectorization lives at the memory boundary — `backend/cuda/bundle/
 *        LoadStore.h`'s `double2`/`float4` reinterpret loads/stores).
 *
 * Compiled in both modes: "device" names the concept (one
 * thread/loop-iteration processes `W` consecutive samples), not a CUDA-only
 * capability — in `AETHER_CPP_MODE` builds every bundle operation is a
 * plain `W`-iteration host loop over the same per-lane scalar formulas, so
 * `AETHER_HAS_CUDA` is only tested internally (`backend/cuda/bundle/
 * LoadStore.h`) to decide whether a leaf `View` access also gets a
 * vectorized `double2`/`float4` memory instruction; the `DeviceBundle` type
 * itself, defined here, has no CUDA dependency at all.
 *
 * `eval<Is...>(SampleIndex)` reuses `SampleIndex::work()` as the lane
 * selector (0..W-1) rather than being a hard compile-error guard the way
 * `PacketItem::eval()` is. This is the mechanism that lets every existing
 * expression node — not just `Sum`/`CWiseScale`, but the geometric/
 * quaternion/structural/product nodes too (`Cross`/`QuatMul`/`QuatConj`/
 * `Segment`/`Transpose`/...) — compose over `DeviceBundle` operands for
 * free, with no per-node bundle-aware overload needed:
 * `backend/cuda/bundle/LoadStore.h`'s generic `bundleGet` fallback drives
 * this same reuse from the other direction (it loops `bi.lane(k)` through
 * the ordinary scalar `eval<Is...>(SampleIndex)` protocol for any node it
 * does not specifically recurse into) — see that header's docstring.
 * `PacketItem` cannot take this shortcut because CPU packet lanes are not
 * addressed by anything resembling a `SampleIndex::work()` slot; aether's
 * `SampleIndex` already carries one, originally meant for block-local/
 * shared-memory indexing, and a `DeviceBundle` lane is exactly that: a
 * thread-local sub-index. Every node's arithmetic formula only ever
 * combines across the static `Is...` element index, never across
 * samples/lanes, so lifting the scalar formula unchanged over `i.work()`
 * is sound by construction — not a coincidence of this particular node
 * set.
 *
 * Storage is element-major, lane-minor: `Carray<T, Size*W>` with element
 * `Is...`'s `W` lanes stored contiguously (`data_[offset(Is...)*W + lane]`).
 * This matches the shape a vectorized `double2`/`float4` load naturally
 * deposits into (W consecutive samples of one element, read in one 128-bit
 * transaction) — `backend/cuda/bundle/LoadStore.h`'s `bundleStore<Is...>`
 * overload for `DeviceBundle` writes exactly that contiguous slice.
 */

#include <cstddef>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Register-resident bundle of `W` samples' worth of `extents<Es...>`
 *        (L3): one `Carray<T,W>`-shaped lane-group per element of
 *        `element_extents`, `Size` elements total. Rebases onto the
 *        `Expression` CRTP exactly like `Item`/`PacketItem` (`isLeaf =
 *        true`).
 */
template<class T, std::size_t W, std::size_t... Es>
class DeviceBundle : public Expression<DeviceBundle<T, W, Es...>, T> {
public:
    using element_type = T;
    /** @brief L3 element protocol: a `DeviceBundle`'s own shape IS its element shape. */
    using element_extents = extents<Es...>;
    /** @brief L3 element protocol: `DeviceBundle` is always a leaf. */
    static constexpr bool isLeaf = true;

    /** @brief Number of modes. */
    static constexpr std::size_t Rank = sizeof...(Es);
    /** @brief Total element count — the product of `Es...` (excludes the `W` lane factor). */
    static constexpr std::size_t Size = (Es * ... * std::size_t{ 1 });
    /** @brief Samples bundled per thread/iteration. */
    static constexpr std::size_t width = W;

    // No AETHER_DEVICEHOST() here — see the note on aether::View's/Item's
    // default ctor (macros.h #20012-D: nvcc auto-annotates `= default`).
    constexpr DeviceBundle() = default;

    /** @brief Lane `k`'s scalar for component `Is...` (compile-time multi-index) — read/write. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr T& at(std::size_t lane)
    {
        return data_[offset_(Is...) * W + lane];
    }
    /** @overload */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr const T& at(std::size_t lane) const
    {
        return data_[offset_(Is...) * W + lane];
    }

    /**
     * @brief Scalar element-protocol hook (see file docstring): `i.work()`
     *        selects the LANE, `Is...` selects the element — every existing
     *        expression node composes over `DeviceBundle` operands through
     *        this, UNCHANGED.
     */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr T& eval(const SampleIndex& i)
    {
        return at<Is...>(static_cast<std::size_t>(i.work()));
    }
    /** @overload */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr const T& eval(const SampleIndex& i) const
    {
        return at<Is...>(static_cast<std::size_t>(i.work()));
    }

    /** @brief Raw pointer to the underlying storage (element-major, lane-minor — see file docstring). */
    AETHER_DEVICEHOST() constexpr T* data() { return data_.data(); }
    AETHER_DEVICEHOST() constexpr const T* data() const { return data_.data(); }
    /** @brief Total scalar count, `Size * W`. */
    AETHER_DEVICEHOST() static constexpr std::size_t size() { return Size * W; }

private:
    template<class... Idxs>
    AETHER_DEVICEHOST() static constexpr std::size_t offset_(Idxs... idxs)
    {
        const detail::Carray<std::size_t, Rank> idxArr{ static_cast<std::size_t>(idxs)... };
        const detail::Carray<std::size_t, Rank> extArr{ Es... };
        std::size_t offset = 0;
        for (std::size_t i = 0; i < Rank; ++i)
            offset = offset * extArr[i] + idxArr[i];
        return offset;
    }

    // Size*W can never be 0 (matches Item<T,Es...>::Size's own convention —
    // no aether shape names a 0-sized mode, and W is always >= 1).
    detail::Carray<T, Size * W> data_{};
};

namespace detail {

/** @brief `DeviceBundle<T,W,Extents's static modes...>` — mirrors
 *         `ItemFromExtents` (`expr/Assign.h`) / `PacketItemFromExtents`
 *         (`backend/cpu/Tiled.h`) for the bundle layer. */
template<class T, std::size_t W, class Extents, class Seq>
struct DeviceBundleFromExtentsImpl;
template<class T, std::size_t W, class Extents, std::size_t... Is>
struct DeviceBundleFromExtentsImpl<T, W, Extents, std::index_sequence<Is...>> {
    using type = DeviceBundle<T, W, Extents::static_extent(Is)...>;
};
template<class T, std::size_t W, class Extents>
using DeviceBundleFromExtents = typename DeviceBundleFromExtentsImpl<T, W, Extents, std::make_index_sequence<Extents::Rank>>::type;

} // namespace detail

} // namespace aether
