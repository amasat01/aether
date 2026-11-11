// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file SampleIndex.h
 * @brief `aether::SampleIndex` / `aether::PacketIndex`.
 *
 * `SampleIndex` carries global vs. work (block-local / shared-memory)
 * indexing.
 *
 * `PacketIndex` is a minimal shape (a `base_` + lane width `Width`) —
 * enough to compile; the CPU SIMD packet layer (`backend/cpu/`) gives it
 * life.
 */

#include <cstddef>

#include "aether/index/Offset.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Sample index: both the global (grid-wide) index and the work
 *        (block-local / shared-memory) index for one sample.
 *
 * Both members are `offset_t` (32-bit). This is the HEAD of the address
 * chain — a `std::size_t` here forces `blockIdx.x * blockDim.x + threadIdx.x`
 * into a 64-bit multiply-add and then a truncation at the first `mapping`
 * call, which is precisely the seam `index/Offset.h` documents. The GPU
 * built-ins (`threadIdx.x` & co.) are already `unsigned`, so the three-
 * argument factory is an EXACT match with no conversion at all.
 */
struct SampleIndex {
    /**
     * @brief Construct from GPU built-in thread/block indices.
     *
     * @param threadIdxx `threadIdx.x`.
     * @param blockIdxx  `blockIdx.x`.
     * @param blockDimx  `blockDim.x`.
     * @return `SampleIndex` with `global() == blockIdxx*blockDimx + threadIdxx`
     *         and `work() == threadIdxx`.
     */
    AETHER_DEVICEHOST() static constexpr SampleIndex make(
        offset_t threadIdxx, offset_t blockIdxx, offset_t blockDimx)
    {
        return SampleIndex{ blockIdxx * blockDimx + threadIdxx, threadIdxx };
    }

    /**
     * @brief Construct from a flat host index — `global() == work() == index`.
     *        Keeps a `std::size_t` parameter so the ubiquitous host spelling
     *        `for (std::size_t idx = ...) SampleIndex::make(idx)` needs no
     *        cast at the call site; the narrowing happens once, here.
     */
    AETHER_DEVICEHOST() static constexpr SampleIndex make(std::size_t index)
    {
        return SampleIndex{ static_cast<offset_t>(index), static_cast<offset_t>(index) };
    }

    /** @brief The global (grid-wide) sample index. */
    AETHER_DEVICEHOST() constexpr offset_t global() const { return globalIdx_; }
    /** @brief The block-local index used for work (shared) memory arrays. */
    AETHER_DEVICEHOST() constexpr offset_t work() const { return workIdx_; }

    /** @brief Public data members (POD layout). */
    offset_t globalIdx_;
    offset_t workIdx_;
};

/**
 * @brief SIMD packet index — represents a contiguous group of `Width`
 *        samples starting at `base_`. A minimal shape; `backend/cpu/`
 *        gives it life.
 */
template<std::size_t Width>
struct PacketIndex {
    /** @brief Number of SIMD lanes. */
    static constexpr std::size_t width = Width;

    std::size_t base_;
    std::size_t active_;

    /** @brief A full packet (all lanes active). */
    AETHER_DEVICEHOST() static constexpr PacketIndex make(std::size_t base) { return PacketIndex{ base, Width }; }
    /** @brief A tail packet with `remaining` active lanes. */
    AETHER_DEVICEHOST() static constexpr PacketIndex makeTail(std::size_t base, std::size_t remaining)
    {
        return PacketIndex{ base, remaining };
    }

    /** @brief `true` when every lane is active. */
    AETHER_DEVICEHOST() constexpr bool full() const { return active_ == Width; }

    /** @brief The scalar `SampleIndex` for lane `k` within this packet. */
    AETHER_DEVICEHOST() constexpr SampleIndex scalar(std::size_t k) const { return SampleIndex::make(base_ + k); }
};

} // namespace aether
