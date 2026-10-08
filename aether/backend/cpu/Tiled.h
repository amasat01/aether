// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Tiled.h
 * @brief SIMD-packet loop drivers: `packetFor`/`packetFlatFor` (context-aware
 *        — `omp_in_parallel()` -> `omp for nowait`; explicit `parallel=true`
 *        -> a new `omp parallel for` region; else serial),
 *        `packetCapture<T>(expr, pi) -> PacketItem`, `packetEval`/
 *        `packetEvalParallel`, and an L2-cache-sized tiled form
 *        (`optimalTileSize`/`packetBatchedFor`).
 *
 * `packetFor`/`packetFlatFor` iterate `[begin,end)` flat, no tile
 * subdivision. `packetBatchedFor` instead tiles the range so each tile's
 * working set fits in L2 cache (`aether/backend/cpu/L2Cache.h`), sized
 * entirely from `detail::l2CacheSize()` — no GPU device properties involved.
 *
 * Tails are handled the same way as the rest of the packet layer:
 * `PacketMask::firstN` on the last, partial packet (via `packetGet`/
 * `packetStore`'s own `pi.full()` branch, `backend/cpu/packet/
 * LoadStore.h`).
 */

#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <utility>

#include "aether/backend/cpu/L2Cache.h"
#include "aether/backend/cpu/PacketItem.h"
#include "aether/backend/cpu/packet/Assign.h"
#include "aether/backend/cpu/packet/LoadStore.h"
#include "aether/backend/cpu/simd/simd.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"

namespace aether {

namespace detail {

/**
 * @brief `PacketItem<T, W, Extents's static modes...>` — the type
 *        `packetCapture` materializes, mirroring `detail::ItemFromExtents`
 *        (`expr/Assign.h`) for the packet layer.
 */
template<class T, std::size_t W, class Extents, class Seq>
struct PacketItemFromExtentsImpl;
template<class T, std::size_t W, class Extents, std::size_t... Is>
struct PacketItemFromExtentsImpl<T, W, Extents, std::index_sequence<Is...>> {
    using type = PacketItem<T, W, Extents::static_extent(Is)...>;
};
template<class T, std::size_t W, class Extents>
using PacketItemFromExtents =
    typename PacketItemFromExtentsImpl<T, W, Extents, std::make_index_sequence<Extents::Rank>>::type;

/**
 * @brief Compile-time-unrolled capture: `out.packet<Is...>() =
 *        packetGet<Is...>(from, pi)` for every static multi-index of
 *        `Extents` — same recursion shape as `RecursivePacketAssign`
 *        (`packet/Assign.h`), writing into a register-resident
 *        `PacketItem` instead of a `View`.
 */
template<class Extents, std::size_t Mode, std::size_t W, std::size_t... Is>
struct RecursivePacketCapture {
    template<class Expr, class Out>
    static void eval(const PacketIndex<W>& pi, Out& out, const Expr& from)
    {
        if constexpr (Mode == Extents::Rank) {
            out.template packet<Is...>() = packetGet<Is...>(from, pi);
        } else {
            loop<0>(pi, out, from);
        }
    }

private:
    template<std::size_t K, class Expr, class Out>
    static void loop(const PacketIndex<W>& pi, Out& out, const Expr& from)
    {
        if constexpr (K < Extents::static_extent(Mode)) {
            RecursivePacketCapture<Extents, Mode + 1, W, Is..., K>::eval(pi, out, from);
            loop<K + 1>(pi, out, from);
        }
    }
};

} // namespace detail

/**
 * @brief Materialize a lazy expression into a register-resident
 *        `PacketItem` — the SIMD analogue of `Item(expr)`'s scalar
 *        materialization (`expr/Assign.h`).
 */
template<class T, class Expr, std::size_t W>
auto packetCapture(const Expr& expr, const PacketIndex<W>& pi)
{
    using Extents = typename Expr::element_extents;
    detail::PacketItemFromExtents<T, W, Extents> item;
    detail::RecursivePacketCapture<Extents, 0, W>::eval(pi, item, expr);
    return item;
}

/**
 * @brief Flat SIMD packet loop — no tiling, no OMP dispatch. Calls
 *        `body(PacketIndex<W>)` for every full packet in `[begin,end)`,
 *        then one masked tail packet if `end - begin` is not a multiple of
 *        `W`.
 */
template<class T, class Func>
void packetFlatFor(std::size_t begin, std::size_t end, Func&& body)
{
    constexpr std::size_t W = simd::PreferredWidth<T>;
    std::size_t i = begin;
    for (; i + W <= end; i += W)
        body(PacketIndex<W>::make(i));
    if (i < end)
        body(PacketIndex<W>::makeTail(i, end - i));
}

/**
 * @brief Context-aware SIMD packet loop, flat (no tile subdivision — see
 *        file docstring):
 *          - already inside an `omp parallel` region: work-share the full
 *            packets via `#pragma omp for schedule(static) nowait`, tail
 *            on one thread via `#pragma omp single nowait`;
 *          - outside and `parallel == true`: open a new `omp parallel for`
 *            region;
 *          - outside and `parallel == false` (default): serial — delegates
 *            to `packetFlatFor`.
 */
template<class T, class Func>
void packetFor(std::size_t begin, std::size_t end, Func&& body, bool parallel = false)
{
    constexpr std::size_t W = simd::PreferredWidth<T>;
    const std::size_t nFullPackets = (end - begin) / W;
    const std::size_t tailStart = begin + nFullPackets * W;

    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (std::size_t p = 0; p < nFullPackets; ++p)
            body(PacketIndex<W>::make(begin + p * W));
#pragma omp single nowait
        if (tailStart < end)
            body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
    } else if (parallel) {
#pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < nFullPackets; ++p)
            body(PacketIndex<W>::make(begin + p * W));
        if (tailStart < end)
            body(PacketIndex<W>::makeTail(tailStart, end - tailStart));
    } else {
        packetFlatFor<T>(begin, end, std::forward<Func>(body));
    }
}

/**
 * @brief Evaluate `expr` into `dest` over `[0, dest.samples())` using SIMD
 *        packets — serial (flat) dispatch.
 */
template<class ExprL, class ExprR>
void packetEval(ExprL& dest, const ExprR& expr)
{
    using T = typename ExprL::element_type;
    constexpr std::size_t W = simd::PreferredWidth<T>;
    packetFlatFor<T>(0, dest.samples(), [&](const PacketIndex<W>& pi) { packetAssign(dest, expr, pi); });
}

/**
 * @brief OpenMP-parallel `packetEval`: opens a new parallel region and
 *        work-shares packets across threads.
 */
template<class ExprL, class ExprR>
void packetEvalParallel(ExprL& dest, const ExprR& expr)
{
    using T = typename ExprL::element_type;
    constexpr std::size_t W = simd::PreferredWidth<T>;
    packetFor<T>(
        0, dest.samples(), [&](const PacketIndex<W>& pi) { packetAssign(dest, expr, pi); }, true);
}

// =============================================================================
//  L2-tiled packet loop
// =============================================================================

/** @brief Default tile size (samples) when the caller passes no per-sample
 *  working-set estimate to `optimalTileSize`. */
inline constexpr std::size_t DEFAULT_TILE_SIZE = 256;

/** @brief Cache-line size in bytes (modern x86-64 / Apple Silicon). Used by
 *  `optimalTileSize` to round tile boundaries to a multiple of one cache
 *  line x packet width — prevents false sharing on the edges between
 *  adjacent threads' tiles. */
inline constexpr std::size_t CACHE_LINE_BYTES = 64;

/** @brief Fraction of L2 reserved for the working set inside one tile.
 *  Leaves ~half of L2 for state + per-step scratch that the body lambda
 *  accesses besides the per-sample working set declared via
 *  `bytesPerSample`. */
inline constexpr double L2_WORKING_SET_FRACTION = 0.5;

/** @brief Lower bound on the tile size in samples — keeps OMP overhead
 *  amortised even when working-set estimates are pathological. */
inline constexpr std::size_t MIN_TILE_SAMPLES = 64;

/** @brief Upper bound on the tile size in samples — caps the tile when
 *  working-set estimates are very small, preventing one thread from
 *  monopolising work. */
inline constexpr std::size_t MAX_TILE_SAMPLES = 4096;

/**
 * @brief Compute an L2-resident tile size for the caller's working set —
 *        `cudaDeviceProp`-free.
 *
 * Returns a tile size `K` (in samples) such that:
 *  - `K * bytesPerSample <= L2_WORKING_SET_FRACTION * L2_per_core`
 *  - `K` is a multiple of `CACHE_LINE_BYTES / sizeof(T)` rounded up to a
 *    multiple of the SIMD packet width `W = simd::PreferredWidth<T>`, so
 *    adjacent tiles never share a cache line (false-sharing free).
 *  - `K` is clamped to `[MIN_TILE_SAMPLES, MAX_TILE_SAMPLES]`.
 *
 * When `bytesPerSample == 0` returns `DEFAULT_TILE_SIZE`.
 *
 * @tparam T             Scalar type — controls the SIMD packet width.
 * @param bytesPerSample Estimated per-sample working set the body touches
 *                       inside the tile.
 * @param l2Override     Optional L2 size in bytes (0 = auto-detect via
 *                       `detail::l2CacheSize()`).
 */
template<class T = double>
std::size_t optimalTileSize(std::size_t bytesPerSample, std::size_t l2Override = 0)
{
    if (bytesPerSample == 0)
        return DEFAULT_TILE_SIZE;

    constexpr std::size_t W = simd::PreferredWidth<T>;
    constexpr std::size_t cacheLineSamples = (CACHE_LINE_BYTES + sizeof(T) - 1) / sizeof(T);
    constexpr std::size_t alignment = (cacheLineSamples > W) ? cacheLineSamples : W;

    const std::size_t l2 = (l2Override > 0) ? l2Override : detail::l2CacheSize();
    const std::size_t budget = static_cast<std::size_t>(L2_WORKING_SET_FRACTION * static_cast<double>(l2));
    std::size_t K = budget / bytesPerSample;

    /* Round DOWN to a multiple of alignment so the tile end falls on a
     * cache-line boundary; this prevents threads from sharing the last
     * cache line of one tile with the first of the next. */
    K = (K / alignment) * alignment;

    /* Clamp to sane bounds; ensure the alignment is preserved at the lower
     * end (alignment is always <= MIN_TILE_SAMPLES in practice). */
    if (K < MIN_TILE_SAMPLES)
        K = ((MIN_TILE_SAMPLES + alignment - 1) / alignment) * alignment;
    if (K > MAX_TILE_SAMPLES)
        K = (MAX_TILE_SAMPLES / alignment) * alignment;

    return K;
}

namespace detail {

/** @brief Process a single tile `[tileStart, tileEnd)` with SIMD packets —
 *  the per-tile body `packetBatchedFor` calls for every tile. */
template<class T, class Func>
void processTile(std::size_t tileStart, std::size_t tileEnd, Func&& body)
{
    constexpr std::size_t W = simd::PreferredWidth<T>;
    std::size_t i = tileStart;
    for (; i + W <= tileEnd; i += W)
        body(PacketIndex<W>::make(i));
    if (i < tileEnd)
        body(PacketIndex<W>::makeTail(i, tileEnd - i));
}

} // namespace detail

/**
 * @brief Context-aware SIMD packet loop with L2-derived tile sizing:
 *          - already inside an `omp parallel` region: work-share the tiles
 *            via `#pragma omp for schedule(static) nowait`;
 *          - outside and `parallel == true`: open a new `omp parallel for`
 *            region over the tiles;
 *          - outside and `parallel == false` (default): serial tile loop.
 *
 * Thread-count clamp: when the L2-derived tile is large enough that the
 * workload would produce fewer tiles than `nthreads * 2`, the tile is
 * capped (rounded down to the packet width) so every thread gets at least
 * 2 tiles under `schedule(static)` — without this clamp a small workload
 * (e.g. an acceptance test with a few hundred samples) leaves threads idle.
 *
 * @tparam T             Scalar type — controls the SIMD packet width.
 * @tparam Func          Callable with signature `void(PacketIndex<W>)`.
 * @param begin          First sample index (inclusive).
 * @param end            Past-the-end sample index.
 * @param body           Lambda to invoke per packet.
 * @param bytesPerSample Estimated per-sample working set inside the body —
 *                       drives the L2-derived tile size
 *                       (`optimalTileSize<T>`).
 * @param parallel       When not already in a parallel region, open one if
 *                       true.
 * @param tileOverride   Optional explicit tile size in samples (0 = use
 *                       the L2-derived size). Useful for tests.
 */
template<class T, class Func>
void packetBatchedFor(std::size_t begin, std::size_t end, Func&& body, std::size_t bytesPerSample,
    bool parallel = false, std::size_t tileOverride = 0)
{
    std::size_t tile = (tileOverride > 0) ? tileOverride : optimalTileSize<T>(bytesPerSample);

    if (omp_in_parallel() && (end > begin)) {
        const std::size_t nthreads = static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t targetTiles = nthreads * std::size_t{ 2 };
        const std::size_t maxTile = (end - begin + targetTiles - 1) / targetTiles;
        if (maxTile > 0 && tile > maxTile) {
            constexpr std::size_t W = simd::PreferredWidth<T>;
            const std::size_t aligned = (maxTile / W) * W;
            tile = (aligned > 0) ? aligned : W;
        }
    }

    const std::size_t nTilesTotal = (end > begin) ? (end - begin + tile - 1) / tile : 0;

    if (omp_in_parallel()) {
#pragma omp for schedule(static) nowait
        for (std::size_t t = 0; t < nTilesTotal; ++t) {
            const std::size_t tileStart = begin + t * tile;
            const std::size_t tileEnd = std::min(tileStart + tile, end);
            detail::processTile<T>(tileStart, tileEnd, body);
        }
    } else if (parallel) {
#pragma omp parallel for schedule(static)
        for (std::size_t t = 0; t < nTilesTotal; ++t) {
            const std::size_t tileStart = begin + t * tile;
            const std::size_t tileEnd = std::min(tileStart + tile, end);
            detail::processTile<T>(tileStart, tileEnd, body);
        }
    } else {
        for (std::size_t t = 0; t < nTilesTotal; ++t) {
            const std::size_t tileStart = begin + t * tile;
            const std::size_t tileEnd = std::min(tileStart + tile, end);
            detail::processTile<T>(tileStart, tileEnd, body);
        }
    }
}

} // namespace aether
