// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file L2Cache.h
 * @brief Runtime detection of the L2 cache size for tile-size sizing.
 *
 * Detected once per process via `std::call_once` and cached. Detection
 * uses POSIX `sysconf(_SC_LEVEL2_CACHE_SIZE)` on Linux, falling back to a
 * 1 MiB default when `sysconf` returns 0 (some containers, virtualised
 * environments, or non-Linux platforms). This narrows to `std::size_t`
 * (aether has no separate index typedef for a BYTE COUNT — `offset_t`/
 * `idx_t` are for element indices only, `aether/index/Offset.h`'s own
 * docstring — a cache size in bytes is a capacity, not an offset).
 *
 * Consumed by `aether::optimalTileSize`/`aether::packetBatchedFor`
 * (`aether/backend/cpu/Tiled.h`).
 */

#include <cstddef>
#include <mutex>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace aether {
namespace detail {

/** @brief Conservative fallback when `sysconf` returns 0 (1 MiB). Modern
 *  x86-64 cores typically have 256 KiB - 2 MiB private L2; 1 MiB is a
 *  safe middle ground that doesn't over-tile on small L2 cores nor
 *  under-tile on larger ones. */
inline constexpr std::size_t L2_CACHE_SIZE_FALLBACK = std::size_t{ 1 } << 20;

/**
 * @brief Return the per-core L2 cache size in bytes.
 *
 * Detected once per process and cached. On non-Linux, or when `sysconf`
 * fails, returns `L2_CACHE_SIZE_FALLBACK`.
 *
 * Note: `_SC_LEVEL2_CACHE_SIZE` reports the per-core L2 on most modern
 * x86-64 systems where L2 is core-private; on older shared-L2 designs it
 * would over-report. The downstream tile sizer (`optimalTileSize`) derates
 * by a factor (default 0.5) to leave headroom for state + per-step
 * scratch, so a small over-report is harmless.
 */
inline std::size_t l2CacheSize()
{
    static std::size_t cached = 0;
    static std::once_flag flag;
    std::call_once(flag, []() {
#if defined(_SC_LEVEL2_CACHE_SIZE)
        const long sz = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
        cached = (sz > 0) ? static_cast<std::size_t>(sz) : L2_CACHE_SIZE_FALLBACK;
#else
        cached = L2_CACHE_SIZE_FALLBACK;
#endif
    });
    return cached;
}

} // namespace detail
} // namespace aether
