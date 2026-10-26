// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file StreamOrdered.h
 * @brief The stream-ordered side of `Chunk`: the `Stream` alias, the capture
 *        query, and the process-wide queue that defers a free requested
 *        while a capture makes it illegal.
 *
 * A `Chunk` allocated with a stream (`Chunk::allocate(device, bytes, stream)`)
 * frees with `cudaFreeAsync` on that stream. A free of that kind is not
 * capturable work: issued while the owning stream (or the legacy stream) is
 * capturing it would invalidate the capture, so it is QUEUED instead and
 * released by the next stream-ordered allocation or release made outside a
 * capture, by `drainDeferredFrees()`, and at process exit. The queue is
 * guarded by a mutex: any number of threads may allocate and free at once.
 *
 * `AETHER_CPP_MODE`: `Stream` is a plain `void*`, nothing is ever capturing,
 * the queue stays empty — the same API, the synchronous host behaviour.
 */

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <vector>

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

#ifdef AETHER_HAS_CUDA
/** @brief CUDA stream alias. */
using Stream = cudaStream_t;
#else
/** @brief Placeholder stream handle: this build has no CUDA backend, nothing is ever enqueued. */
using Stream = void*;
#endif

namespace detail {

#ifdef AETHER_HAS_CUDA
/**
 * @brief Per-thread relaxed capture mode while alive (when `active`): the calling thread's allocations,
 *        frees and queries then no longer invalidate another stream's global-mode capture.
 */
class RelaxedCapture {
public:
    explicit RelaxedCapture(bool active = true)
        : active_{ active }
    {
        if (active_)
            cudaThreadExchangeStreamCaptureMode(&mode_);
    }
    RelaxedCapture(const RelaxedCapture&)            = delete;
    RelaxedCapture& operator=(const RelaxedCapture&) = delete;
    ~RelaxedCapture()
    {
        if (active_)
            cudaThreadExchangeStreamCaptureMode(&mode_);
    }

private:
    bool active_;
    cudaStreamCaptureMode mode_ = cudaStreamCaptureModeRelaxed;
};

/** @brief Makes `device` the calling thread's current device while alive. */
class DeviceGuard {
public:
    explicit DeviceGuard(int device)
    {
        cudaGetDevice(&prior_);
        switched_ = prior_ != device;
        if (switched_)
            cudaSetDevice(device);
    }
    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    ~DeviceGuard()
    {
        if (switched_)
            cudaSetDevice(prior_);
    }

private:
    int prior_     = 0;
    bool switched_ = false;
};
#endif

} // namespace detail

/**
 * @brief Whether `stream` is being captured, or a capture elsewhere makes work on it illegal.
 *
 * Error 906 (`cudaErrorStreamCaptureImplicit`, a blocking capture covers the legacy stream) and any other
 * failure of the query both answer `true`: the safe answer is "do not issue the call now". `nullptr` names
 * the legacy stream. Always `false` in `AETHER_CPP_MODE`.
 */
inline bool streamCapturing(Stream stream) noexcept
{
#ifdef AETHER_HAS_CUDA
    detail::RelaxedCapture relaxed;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        cudaGetLastError();
        return true;
    }
    return status != cudaStreamCaptureStatusNone;
#else
    (void)stream;
    return false;
#endif
}

namespace detail {

/** @brief One stream-ordered release: a device pointer to free on `stream`, or a pinned host pointer. */
struct PendingFree {
    void* ptr;
    int device;
    Stream stream;
    bool pinned;
};

/** @brief The process-wide queue of releases deferred by a capture. Never destroyed (exit order). */
class DeferredFrees {
public:
    static DeferredFrees& instance()
    {
        static DeferredFrees* queue = [] {
            auto* q = new DeferredFrees;
            std::atexit([] { instance().drain(true, true); });
            return q;
        }();
        return *queue;
    }

    void push(const PendingFree& f)
    {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.push_back(f);
    }

    std::size_t size()
    {
        std::lock_guard<std::mutex> lock(mu_);
        return pending_.size();
    }

    bool exiting()
    {
        std::lock_guard<std::mutex> lock(mu_);
        return exiting_;
    }

    /** @brief Test hook: leave the exiting state a `drain(.., true)` set. */
    void resumeAfterExit()
    {
        std::lock_guard<std::mutex> lock(mu_);
        exiting_ = false;
    }

    /**
     * @brief Release every queued free whose stream is no longer blocked (all of them when `all`, in which
     *        case the queue goes into the exiting state: later releases are dropped to the driver).
     */
    void drain(bool all, bool markExiting) noexcept
    {
        std::vector<PendingFree> todo;
        {
            std::lock_guard<std::mutex> lock(mu_);
            todo.swap(pending_);
            if (markExiting)
                exiting_ = true;
        }
        std::vector<PendingFree> keep;
        for (const PendingFree& f : todo) {
            if (!all && blocked(f))
                keep.push_back(f);
            else
                release(f, all);
        }
        if (!keep.empty()) {
            std::lock_guard<std::mutex> lock(mu_);
            pending_.insert(pending_.end(), keep.begin(), keep.end());
        }
    }

    /** @brief A release that a capture makes illegal now. Pinned frees look at the legacy stream only. */
    static bool blocked(const PendingFree& f) noexcept
    {
        return streamCapturing(nullptr) || (!f.pinned && streamCapturing(f.stream));
    }

    /** @brief Issue the release (`synchronous`: the blocking free, for exit). Never throws. */
    static void release([[maybe_unused]] const PendingFree& f, [[maybe_unused]] bool synchronous) noexcept
    {
#ifdef AETHER_HAS_CUDA
        RelaxedCapture relaxed;
        if (f.pinned) {
            cudaFreeHost(f.ptr);
        } else {
            DeviceGuard device(f.device);
            if (synchronous)
                cudaFree(f.ptr);
            else
                cudaFreeAsync(f.ptr, f.stream);
        }
        cudaGetLastError();
#endif
    }

private:
    DeferredFrees() = default;

    std::mutex mu_;
    std::vector<PendingFree> pending_;
    bool exiting_ = false;
};

/**
 * @brief Release a stream-ordered allocation: freed on its stream now, or queued while a capture blocks it.
 *        After the exit drain the memory is left to the driver. Never throws.
 */
inline void releaseStreamOrdered(void* ptr, int device, Stream stream, bool pinned) noexcept
{
    DeferredFrees& queue = DeferredFrees::instance();
    if (queue.exiting())
        return;
    queue.drain(false, false);
    const PendingFree f{ ptr, device, stream, pinned };
    if (DeferredFrees::blocked(f))
        queue.push(f);
    else
        DeferredFrees::release(f, false);
}

} // namespace detail

/** @brief Release every deferred free whose capture has ended; those still blocked stay queued. */
inline void drainDeferredFrees() noexcept { detail::DeferredFrees::instance().drain(false, false); }

/** @brief Number of frees currently deferred by a capture. */
inline std::size_t deferredFreeCount() { return detail::DeferredFrees::instance().size(); }

} // namespace aether
