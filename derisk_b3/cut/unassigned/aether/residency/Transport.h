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
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
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

    /**
     * @brief ONE ordered copy for every pair, on `stream`: `dst` is written after the work already queued
     *        on `stream`, and every device-side leg is enqueued on it.
     *
     * CUDA<->CUDA and CUDA<->CUDAHost go direct (`aether::copyAsync`, no event recorded). A PAGEABLE host
     * chunk on one side is STAGED through this transport's pinned bounce buffer: host->device copies the
     * bytes into the bounce chunk now and enqueues the upload (the call returns before it ran; the next
     * staged call waits for it before reusing the bounce chunk), device->host downloads into the bounce
     * chunk and BLOCKS on `stream` before copying into `dst`. The staged routes allocate and synchronize:
     * not capture-legal; the direct ones are. A pair with no CUDA device chunk is host-only: blocks on
     * `stream`, then `aether::copy`. `AETHER_CPP_MODE`: every pair is host-only, `stream` ignored.
     *
     * @throws aether::Error on a size mismatch or a failing CUDA call.
     */
    void copyOrdered(Chunk& dst, const Chunk& src, Stream stream)
    {
#ifdef AETHER_HAS_CUDA
        if (dst.size() != src.size()) {
            err::fail("copyOrdered", detail::pathLabel(src, dst), src.size(),
                "size mismatch: dst.size()=" + std::to_string(dst.size()));
        }
        const bool srcCuda = src.device().type() == kDLCUDA, dstCuda = dst.device().type() == kDLCUDA;
        if (!srcCuda && !dstCuda) {
            err::checkCuda(cudaStreamSynchronize(stream), "copyOrdered", detail::pathLabel(src, dst), src.size());
            aether::copy(dst, src);
            return;
        }
        const bool srcPageable = src.device().type() == kDLCPU, dstPageable = dst.device().type() == kDLCPU;
        if (!srcPageable && !dstPageable) {
            aether::copyAsync(dst, src, stream);
            return;
        }
        const std::size_t bytes = src.size();
        if (bytes == 0)
            return;
        const std::string where = detail::pathLabel(src, dst);
        if (bounceBusy_) {
            err::checkCuda(cudaEventSynchronize(bounceDone_.get()), "copyOrdered", where, bytes);
            bounceBusy_ = false;
        }
        {
            detail::RelaxedCapture relaxed;
            bounce_.ensure(bytes);
        }
        Chunk stage = Chunk::borrow(bounce_.chunk().data(), bytes, bounce_.chunk().device());
        if (srcPageable) {
            std::memcpy(stage.data(), src.data(), bytes);
            aether::copyAsync(dst, stage, stream);
            // An event, not the stream: the caller may destroy `stream` before the next staged copy waits.
            if (!bounceDone_) {
                cudaEvent_t ev;
                err::checkCuda(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate", where, bytes);
                bounceDone_.reset(ev);
            }
            err::checkCuda(cudaEventRecord(bounceDone_.get(), stream), "cudaEventRecord", where, bytes);
            bounceBusy_ = true;
        } else {
            aether::copyAsync(stage, src, stream);
            err::checkCuda(cudaStreamSynchronize(stream), "copyOrdered", where, bytes);
            std::memcpy(dst.data(), stage.data(), bytes);
        }
#else
        (void)stream;
        aether::copy(dst, src);
#endif
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
    struct EventDeleter {
        void operator()(cudaEvent_t ev) const noexcept { cudaEventDestroy(ev); }
    };
    /** @brief `true` while the last staged upload may still be reading `bounce_`; `bounceDone_` completes with it. */
    bool bounceBusy_ = false;
    std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, EventDeleter> bounceDone_;
#endif
    bool forceStaging_ = false;
};

static_assert(transport<StreamTransport>, "aether::StreamTransport must satisfy aether::transport");

} // namespace aether
