// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Launch.h
 * @brief CUDA launch helpers: `launchConfig(n, block)` -> a `{blocks,
 *        threads}` 1D grid shape; `AETHER_CHECK_CUDA(call)` (throws
 *        `aether::Error` with the CUDA error string on failure);
 *        `checkLastLaunch(name)` (`cudaGetLastError` +
 *        `cudaDeviceSynchronize`, throwing on either).
 *
 * `checkLastLaunch` exists because a failed launch otherwise leaves the
 * destination buffer holding whatever it held before, and a later
 * comparison blames a "value divergence" whose real cause is that nothing
 * ran.
 *
 * Both helpers reuse `err::checkCuda` (`aether/err/Error.h`) rather than
 * duplicating the check-and-throw shape — same message format
 * (`"<operation> failed on <where>, size=<bytes> bytes (<detail>)"`) as
 * every other host-side failure in this library.
 *
 * Host-only, CUDA-backend-only: the entire file is a no-op (empty) when
 * `AETHER_HAS_CUDA` is not defined — mirrors `aether/chunk/Copy.h`'s
 * `Stream`/`copyAsync` gating.
 */

#include "aether/macros.h"

#ifdef AETHER_HAS_CUDA

#include <cstddef>
#include <string>

#include "aether/err/Error.h"

#include <cuda_runtime.h>

namespace aether {
namespace cuda {

/** @brief A 1D CUDA launch grid shape: `blocks` blocks of `threads` threads each. */
struct LaunchConfig {
    int blocks;
    int threads;
};

/**
 * @brief 1D launch config covering `n` work-items with `block`-sized
 *        blocks (default 256) — `blocks = ceil(n / block)`.
 */
inline LaunchConfig launchConfig(std::size_t n, int block = 256)
{
    const int threads = block;
    const int blocks = static_cast<int>(
        (n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));
    return LaunchConfig{ blocks, threads };
}

/**
 * @brief 1D launch config covering `n` samples, `W` per thread —
 *        `blocks = ceil(ceil(n/W) / block)`: first fold `n` down to the
 *        number of threads needed (one per `W`-sample bundle), then apply
 *        the ordinary per-block ceil-div. Every thread's `BundleIndex<W>::
 *        make(threadIdx.x, blockIdx.x, blockDim.x)` may therefore cover a
 *        bundle straddling `n` (the tail) — callers mask via `BundleIndex<
 *        W>::mask(n)` (see `index/BundleIndex.h`).
 */
inline LaunchConfig bundleLaunchConfig(std::size_t n, std::size_t W, int block = 256)
{
    const std::size_t threadsNeeded = (n + W - 1) / W;
    return launchConfig(threadsNeeded, block);
}

/**
 * @brief Assert that the kernel launched immediately before this call
 *        actually started and completed.
 *
 * @param name  the kernel's name, for the thrown message.
 * @throws aether::Error  if the launch failed or the kernel's execution
 *         failed (`cudaGetLastError` then `cudaDeviceSynchronize`).
 */
inline void checkLastLaunch(const char* name)
{
    err::checkCuda(cudaGetLastError(), std::string("checkLastLaunch(") + name + ")", "CUDA[launch]", 0);
    err::checkCuda(cudaDeviceSynchronize(), std::string("checkLastLaunch(") + name + ")", "CUDA[execution]", 0);
}

} // namespace cuda
} // namespace aether

/**
 * @brief Check a CUDA runtime API call's status; throw `aether::Error`
 *        naming the failing call and the CUDA error string on failure
 *        (host-side). Reuses `aether::err::checkCuda`.
 */
#define AETHER_CHECK_CUDA(call) ::aether::err::checkCuda((call), #call, "CUDA", 0)

#endif // AETHER_HAS_CUDA
