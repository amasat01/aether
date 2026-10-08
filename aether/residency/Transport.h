// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Transport.h
 * @brief `aether::transport<T>`: the C++20 concept every single-node or
 *        future multi-node data-mover for residency (`ReplicaSet::
 *        broadcast`, a partitioned array's block-to-block movement) must
 *        satisfy, plus `StreamTransport`, the peer-access-aware
 *        implementation.
 *
 * Multi-node transports (NCCL, MPI, an RDMA/peer-to-peer fabric) are
 * consumers of this same concept; none is provided here.
 *
 * `StreamTransport` widens beyond a bare delegate to `aether::copy`/
 * `copyAsync`: `route(a, b)` exposes its `aether::Route` decision for a
 * device pair (`aether/residency/PeerAccess.h`), a `forceStaging` switch
 * forces every CUDA<->CUDA pair through the explicit STAGED path (its own
 * pinned `BounceBuffer`) for testing, and — CUDA builds only — every async
 * move records a `cudaEvent` afterward, observable via `lastEvent()`/
 * `wait(event)`/`fence()`.
 */

#include <concepts>
#include <vector>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/residency/PeerAccess.h"

#ifdef AETHER_HAS_CUDA
#include "aether/err/Error.h"
#endif

namespace aether {

#ifdef AETHER_HAS_CUDA
/** @brief A type satisfies `transport` when it offers a blocking `copy`
 *         and an asynchronous `copyAsync` between two `Chunk`s. */
template<class T>
concept transport = requires(T t, Chunk& dst, const Chunk& src, Stream stream) {
    { t.copy(dst, src) } -> std::same_as<void>;
    { t.copyAsync(dst, src, stream) } -> std::same_as<void>;
};
#else
/** @brief A type satisfies `transport` when it offers a blocking `copy`
 *         between two `Chunk`s. No `copyAsync` requirement in this
 *         build: `aether::copyAsync` itself does not exist outside a CUDA
 *         backend (`aether/chunk/Copy.h`). */
template<class T>
concept transport = requires(T t, Chunk& dst, const Chunk& src) {
    { t.copy(dst, src) } -> std::same_as<void>;
};
#endif

/**
 * @brief The peer-access-aware `transport` implementation. `Route::STAGED`
 *        moves go through this transport's own pinned `BounceBuffer`
 *        (`aether/residency/PeerAccess.h`) explicitly; every other route
 *        delegates verbatim to `aether::copy`/`aether::copyAsync`
 *        (`aether/chunk/Copy.h`) — no behavior of its own beyond that, so
 *        it still satisfies the concept trivially.
 *
 * A `BounceBuffer` and any recorded `cudaEvent`s are per-instance (not
 * process-global, unlike `PeerAccess.h`'s peer-access cache) — reusing one
 * `StreamTransport` for two moves whose STAGED legs may still be in
 * flight on different streams races on the shared bounce chunk; issuing
 * every `Route::STAGED` move for one transport instance on a single
 * stream (or fully synchronizing between them) is the safe usage this
 * type assumes.
 */
class StreamTransport {
public:
    StreamTransport() = default;
    /** @brief `forceStaging=true` routes every CUDA<->CUDA pair (same
     *         device included) through `Route::STAGED` — a test-only
     *         override. */
    explicit StreamTransport(bool forceStaging)
        : forceStaging_{ forceStaging }
    {
    }

#ifdef AETHER_HAS_CUDA
    // Move-only: the BounceBuffer + recorded cudaEvent handles below must
    // not be duplicated by a copy (mirrors Chunk's own move-only contract).
    StreamTransport(const StreamTransport&)            = delete;
    StreamTransport& operator=(const StreamTransport&) = delete;
    StreamTransport(StreamTransport&&)                 = default;
    StreamTransport& operator=(StreamTransport&&)      = default;

    /** @brief Destroys every `cudaEvent` this transport has recorded. */
    ~StreamTransport()
    {
        for (cudaEvent_t ev : events_)
            cudaEventDestroy(ev);
    }
#endif

    /** @brief This transport's `Route` decision for a move from `a` to
     *         `b` (`aether::peerRoute`, honoring `forceStaging()`). */
    Route route(const Device& a, const Device& b) const { return peerRoute(a, b, forceStaging_); }

    /** @brief `true` when this transport forces every CUDA<->CUDA pair through `Route::STAGED`. */
    bool forceStaging() const { return forceStaging_; }

    /**
     * @brief Blocking copy: `Route::STAGED` goes device -> pinned bounce
     *        -> device explicitly through this transport's own
     *        `BounceBuffer`; every other route (`SAME`, `P2P`, `HOST`)
     *        delegates verbatim to `aether::copy`.
     */
    void copy(Chunk& dst, const Chunk& src)
    {
#ifdef AETHER_HAS_CUDA
        if (route(src.device(), dst.device()) == Route::STAGED) {
            stagedCopy_(dst, src);
            return;
        }
#endif
        aether::copy(dst, src);
    }

#ifdef AETHER_HAS_CUDA
    /**
     * @brief Async copy on `stream` (CUDA builds only): same STAGED-vs-
     *        delegate split as `copy`, then records a `cudaEvent` on
     *        `stream` after the move is enqueued — `lastEvent()`/
     *        `wait()`/`fence()` observe it.
     */
    void copyAsync(Chunk& dst, const Chunk& src, Stream stream)
    {
        if (route(src.device(), dst.device()) == Route::STAGED)
            stagedCopyAsync_(dst, src, stream);
        else
            aether::copyAsync(dst, src, stream);
        recordEvent_(stream);
    }

    /** @brief The event recorded after the most recent async move issued
     *         through this transport (`nullptr` before any). */
    cudaEvent_t lastEvent() const { return events_.empty() ? nullptr : events_.back(); }

    /**
     * @brief Block the calling host thread until `event` completes.
     *        `event` need not be one this transport recorded — any
     *        `cudaEvent_t` is legal, e.g. an event a test records itself
     *        around a producer kernel it wants ordered against a
     *        subsequent move. A `nullptr` event is a no-op (mirrors
     *        `lastEvent()`'s "before any move" sentinel).
     */
    void wait(cudaEvent_t event) const
    {
        if (event == nullptr)
            return;
        err::checkCuda(cudaEventSynchronize(event), "cudaEventSynchronize", "StreamTransport::wait", 0);
    }

    /** @brief Block until every event this transport has itself recorded
     *         (every async move issued through it so far) completes. */
    void fence() const
    {
        for (cudaEvent_t ev : events_)
            wait(ev);
    }
#endif

private:
#ifdef AETHER_HAS_CUDA
    /** @brief `Route::STAGED`, blocking: `src` -> `bounce_` -> `dst`, two
     *         ordinary `aether::copy` legs over a `Chunk::borrow()`-wrapped
     *         view of the bounce chunk (no new copy primitive). */
    void stagedCopy_(Chunk& dst, const Chunk& src)
    {
        bounce_.ensure(src.size());
        Chunk stage = Chunk::borrow(bounce_.chunk().data(), src.size(), bounce_.chunk().device());
        aether::copy(stage, src);
        aether::copy(dst, stage);
    }

    /** @brief `Route::STAGED`, async: both legs on the same `stream`, so
     *         the driver's own same-stream FIFO ordering (not an event)
     *         guarantees the second leg only starts once the first has
     *         landed in `bounce_`. */
    void stagedCopyAsync_(Chunk& dst, const Chunk& src, Stream stream)
    {
        bounce_.ensure(src.size());
        Chunk stage = Chunk::borrow(bounce_.chunk().data(), src.size(), bounce_.chunk().device());
        aether::copyAsync(stage, src, stream);
        aether::copyAsync(dst, stage, stream);
    }

    void recordEvent_(Stream stream)
    {
        cudaEvent_t ev;
        err::checkCuda(cudaEventCreate(&ev), "cudaEventCreate", "StreamTransport", 0);
        err::checkCuda(cudaEventRecord(ev, stream), "cudaEventRecord", "StreamTransport", 0);
        events_.push_back(ev);
    }

    BounceBuffer bounce_;
    std::vector<cudaEvent_t> events_;
#endif
    bool forceStaging_ = false;
};

static_assert(transport<StreamTransport>, "aether::StreamTransport must satisfy aether::transport");

} // namespace aether
