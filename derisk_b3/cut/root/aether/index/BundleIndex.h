// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BundleIndex.h
 * @brief `aether::BundleIndex<W>`: the device analog of `backend/cpu/simd/
 *        PacketMask.h`'s tail-handling idiom for `DeviceBundle<T,W,Es...>`
 *        (`backend/cuda/DeviceBundle.h`) — a contiguous group of `W`
 *        samples processed by one thread.
 *
 * Mirrors `SampleIndex`'s own two factories (`index/SampleIndex.h`):
 * `make(threadIdx.x, blockIdx.x, blockDim.x)` for device launch grids
 * (`base sample = W * (blockIdx*blockDim + threadIdx)`, i.e. thread `t`
 * covers samples `[W*global, W*global+W)`) and `make(index)` for a flat
 * host loop. `base_` is `offset_t` (32-bit) — the same narrow address
 * chain every other index type in this library uses; see `index/Offset.h`.
 *
 * `lane(k)` returns the scalar `SampleIndex` for lane `k` within this
 * bundle: `global() == base()+k`, `work() == k`. This is the mechanism
 * `DeviceBundle::eval<Is...>(SampleIndex)` reuses to let every existing
 * expression node (`Sum`/`CWiseScale`, and — via `backend/cuda/
 * bundle/LoadStore.h`'s generic `bundleGet` fallback — the geometric/
 * quaternion/structural nodes too) compose over `DeviceBundle` operands
 * with no bundle-specific overload of their own; see `DeviceBundle.h`'s
 * file docstring for the full argument.
 *
 * `mask(n)` — PacketMask-style `firstN` (`backend/cpu/simd/PacketMask.h`)
 * for the tail bundle straddling the end of an `n`-sample array: lane `k`
 * is active iff `base()+k < n`. Deliberately not embedded as extra state
 * inside `BundleIndex` itself (unlike `PacketIndex`'s `active_` member) —
 * a `BundleIndex` always denotes a logically-full `W`-wide span; masking is
 * a separate, on-demand query against the actual array length, matching
 * `backend/cuda/bundle/Assign.h`'s `bundleAssign(dest, expr, bi[, mask])`
 * shape.
 */

#include <cstddef>

#include "aether/index/Offset.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief A contiguous group of `W` samples, one thread's worth of bundled
 *        work. `W` is expected to be 2 or 4 (the 128-bit-LDG/STG-eligible
 *        widths `backend/cuda/bundle/LoadStore.h` targets), though nothing
 *        here hard-codes that — other widths simply never hit the
 *        vectorized memory path and always take the scalar per-lane
 *        fallback (still correct).
 */
template<std::size_t W>
struct BundleIndex {
    /** @brief Number of samples this bundle covers. */
    static constexpr std::size_t width = W;

    /** @brief The global sample index of lane 0. */
    offset_t base_;

    /**
     * @brief Construct from GPU built-in thread/block indices — mirrors
     *        `SampleIndex::make`'s three-argument device factory exactly,
     *        scaled by `W`: thread `t` in block `b` (block size `d`) owns
     *        samples `[W*(b*d+t), W*(b*d+t)+W)`.
     */
    AETHER_DEVICEHOST() static constexpr BundleIndex make(offset_t threadIdxx, offset_t blockIdxx, offset_t blockDimx)
    {
        return BundleIndex{ static_cast<offset_t>(W) * (blockIdxx * blockDimx + threadIdxx) };
    }

    /**
     * @brief Construct from a flat host bundle index — bundle `i` covers
     *        samples `[W*i, W*i+W)`. Keeps a `std::size_t` parameter so the
     *        ubiquitous host spelling needs no cast at the call site (same
     *        convention as `SampleIndex::make(std::size_t)`).
     */
    AETHER_DEVICEHOST() static constexpr BundleIndex make(std::size_t i)
    {
        return BundleIndex{ static_cast<offset_t>(W * i) };
    }

    /** @brief The global sample index of lane 0. */
    AETHER_DEVICEHOST() constexpr offset_t base() const { return base_; }

    /** @brief The scalar `SampleIndex` for lane `k` (0 <= k < W): `global()
     *         == base()+k`, `work() == k` (the lane selector — see file
     *         docstring). */
    AETHER_DEVICEHOST() constexpr SampleIndex lane(std::size_t k) const
    {
        return SampleIndex{ base_ + static_cast<offset_t>(k), static_cast<offset_t>(k) };
    }

    /** @brief Per-lane active mask for a bundle straddling the end of an
     *         `n`-sample array (PacketMask-style `firstN`). */
    struct Mask {
        offset_t active_;

        /** @brief `true` when lane `k` is within bounds. */
        AETHER_DEVICEHOST() constexpr bool lane(std::size_t k) const { return static_cast<offset_t>(k) < active_; }
        /** @brief `true` when every lane is active (a full, non-tail bundle). */
        AETHER_DEVICEHOST() constexpr bool full() const { return active_ == static_cast<offset_t>(W); }
    };

    /** @brief The lane-active mask for this bundle against an `n`-sample array. */
    AETHER_DEVICEHOST() constexpr Mask mask(offset_t n) const
    {
        const offset_t remaining = (base_ < n) ? (n - base_) : offset_t{ 0 };
        return Mask{ remaining < static_cast<offset_t>(W) ? remaining : static_cast<offset_t>(W) };
    }
};

} // namespace aether
