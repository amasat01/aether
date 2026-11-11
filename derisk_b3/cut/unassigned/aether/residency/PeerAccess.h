// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PeerAccess.h
 * @brief `aether::Route` + `aether::peerRoute`: the explicit, per-ordered-
 *        pair CUDA peer-access decision that `aether/chunk/Copy.h`'s
 *        CUDA<->CUDA path never makes on its own — that path is a bare
 *        `cudaMemcpy(..., cudaMemcpyDeviceToDevice)`, with no
 *        `cudaDeviceCanAccessPeer`/`cudaDeviceEnablePeerAccess` call
 *        anywhere, leaving P2P-vs-staging entirely to the driver's own
 *        transparent UVA behavior — plus `aether::BounceBuffer`, the
 *        pinned host staging chunk a transport owns for a `Route::STAGED`
 *        move.
 *
 * `peerRoute(a, b, forceStaging)` never allocates and never launches a
 * kernel: `cudaDeviceCanAccessPeer` is a pure query, cached per ordered
 * pair (`a`->`b` and `b`->`a` cached separately, since the CUDA contract
 * does not guarantee symmetry even though every topology measured so far
 * is symmetric in practice); `cudaDeviceEnablePeerAccess` is invoked
 * lazily, once per ordered pair, on the first `Route::P2P` verdict for
 * that pair — repeat pairs are cache-suppressed rather than re-issuing the
 * call and relying on catching `cudaErrorPeerAccessAlreadyEnabled` every
 * time (that status is still tolerated, defensively, the one time it can
 * fire).
 *
 * `forceStaging` is a blunt, test-only override: when set, every
 * CUDA<->CUDA pair — same-device included — routes `STAGED`, so the
 * staging path (device -> pinned host -> device) is exercisable and
 * bit-comparable against a direct copy on a single GPU, without a second
 * physical device.
 *
 * Host-only header: uses `std::map`/`std::string`; never include this from
 * a device-compiled translation unit (mirrors `aether/err/Error.h`).
 */

#include <cstddef>
#include <map>
#include <string>
#include <utility>

#include "aether/chunk/Chunk.h"
#include "aether/device/Device.h"
#include "aether/err/Error.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

/**
 * @brief How a transport moves bytes between two `Device`s.
 *
 * - `SAME`   — both endpoints are the identical CUDA device: an
 *              intra-device copy, no peer-access question at all.
 * - `P2P`    — distinct CUDA devices, `cudaDeviceCanAccessPeer` true in
 *              both directions, peer access enabled: a direct
 *              `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` crosses the
 *              PCIe/NVLink fabric without staging.
 * - `STAGED` — distinct CUDA devices without usable peer access in both
 *              directions, or `forceStaging` set on any CUDA pair (same
 *              device included): the move goes through the transport's
 *              own pinned `BounceBuffer` — never the driver's implicit
 *              UVA fallback.
 * - `HOST`   — at least one endpoint is not `kDLCUDA` (CPU or CUDAHost),
 *              or this build has no CUDA backend at all
 *              (`AETHER_CPP_MODE`): the existing `Copy.h` host-addressable
 *              paths apply verbatim; peer access is not this enum's
 *              concern for either case.
 */
enum class Route { SAME, P2P, STAGED, HOST };

#ifdef AETHER_HAS_CUDA
namespace detail {

/** @brief `cudaDeviceCanAccessPeer(devA, devB)`, cached per ordered pair. */
inline bool canAccessPeerCached(int devA, int devB)
{
    static std::map<std::pair<int, int>, bool> cache;
    const std::pair<int, int> key{ devA, devB };
    const auto it = cache.find(key);
    if (it != cache.end())
        return it->second;
    int can = 0;
    err::checkCuda(cudaDeviceCanAccessPeer(&can, devA, devB), "cudaDeviceCanAccessPeer",
        "CUDA[" + std::to_string(devA) + "] -> CUDA[" + std::to_string(devB) + "]", 0);
    return cache.emplace(key, can != 0).first->second;
}

/**
 * @brief Lazily `cudaDeviceEnablePeerAccess(devB)` from `devA`'s context,
 *        once per ordered pair — restores the caller's prior current
 *        device afterward, mirroring `Chunk::allocate`'s own `kDLCUDA`
 *        scoping (`aether/chunk/Chunk.h`).
 */
inline void enablePeerAccessCached(int devA, int devB)
{
    static std::map<std::pair<int, int>, bool> enabled;
    const std::pair<int, int> key{ devA, devB };
    if (enabled.count(key) != 0)
        return;

    int prior = 0;
    err::checkCuda(cudaGetDevice(&prior), "cudaDeviceEnablePeerAccess", "PeerAccess", 0);
    err::checkCuda(cudaSetDevice(devA), "cudaDeviceEnablePeerAccess", "PeerAccess", 0);
    const cudaError_t status = cudaDeviceEnablePeerAccess(devB, 0);
    err::checkCuda(cudaSetDevice(prior), "cudaDeviceEnablePeerAccess", "PeerAccess", 0); // restore first, even on failure below
    if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
        err::fail("cudaDeviceEnablePeerAccess",
            "CUDA[" + std::to_string(devA) + "] -> CUDA[" + std::to_string(devB) + "]", 0,
            cudaGetErrorString(status));
    }
    enabled.emplace(key, true);
}

} // namespace detail
#endif

/**
 * @brief Decide the `Route` for a move from `a` to `b`. Lazily enables
 *        peer access (both directions) on the first `P2P` verdict for an
 *        ordered pair.
 *
 * In an `AETHER_CPP_MODE` build (no CUDA backend) this always returns
 * `Route::HOST`, unconditionally — there is no CUDA device kind to reason
 * about peer access over.
 */
inline Route peerRoute(const Device& a, const Device& b, bool forceStaging = false)
{
#ifndef AETHER_HAS_CUDA
    (void)a;
    (void)b;
    (void)forceStaging;
    return Route::HOST;
#else
    if (!a.is_cuda() || !b.is_cuda())
        return Route::HOST;
    if (forceStaging)
        return Route::STAGED;
    if (a == b)
        return Route::SAME;
    if (detail::canAccessPeerCached(a.id(), b.id()) && detail::canAccessPeerCached(b.id(), a.id())) {
        detail::enablePeerAccessCached(a.id(), b.id());
        detail::enablePeerAccessCached(b.id(), a.id());
        return Route::P2P;
    }
    return Route::STAGED;
#endif
}

#ifdef AETHER_HAS_CUDA
/**
 * @brief Pinned host bounce buffer: the transport's own staging memory for
 *        a `Route::STAGED` move — sized to the largest chunk moved through
 *        it so far, grown, never shrunk. CUDA-backend-only: staging
 *        through pinned host memory is meaningless without a CUDA device
 *        on either end.
 *
 * Move-only (mirrors `Chunk`): a `BounceBuffer` owns one pinned
 * allocation, freed on destruction via its `Chunk` member.
 */
class BounceBuffer {
public:
    BounceBuffer() = default;
    BounceBuffer(const BounceBuffer&) = delete;
    BounceBuffer& operator=(const BounceBuffer&) = delete;
    BounceBuffer(BounceBuffer&&) = default;
    BounceBuffer& operator=(BounceBuffer&&) = default;

    /** @brief Grow the buffer to >= `bytes` if it is not already that
     *         large; a no-op otherwise (grown, never shrunk). */
    void ensure(std::size_t bytes)
    {
        if (bytes <= capacity_)
            return;
        chunk_    = Chunk::allocate(Device(kDLCUDAHost), bytes);
        capacity_ = bytes;
    }

    /** @brief The backing pinned chunk — `.size() == capacity()` after `ensure()`. */
    Chunk& chunk() { return chunk_; }
    /** @brief Bytes currently allocated (the high-water mark, not any one `ensure()` request). */
    std::size_t capacity() const { return capacity_; }

private:
    Chunk chunk_;
    std::size_t capacity_ = 0;
};
#endif

} // namespace aether
