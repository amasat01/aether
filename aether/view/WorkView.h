// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file WorkView.h
 * @brief `aether::make_work_view`: a `View` factory over a shared-memory
 *        span, device-legal (`AETHER_DEVICEHOST()` — unlike `view/View.h`'s
 *        `make_view`, which is host-only: it throws an `aether::Error` on
 *        a bad span, and device code cannot throw). A work view's sample
 *        mode is `block_samples` (the block-local count, not the
 *        grid-global sample count) — indexed via `SampleIndex::work()`
 *        (the same field `DeviceBundle::eval()` reuses as a lane selector
 *        — see `backend/cuda/DeviceBundle.h`) rather than
 *        `SampleIndex::global()`.
 *
 * Volatile-composable via the result's own `as_volatile()` (`view/View.h`)
 * — no separate mechanism needed here: `make_work_view` just returns an
 * ordinary `View`, and every `View` operation (including `.as_volatile()`,
 * `.as_readonly()`, expression-leaf use, `operator[]`) already applies to
 * it unchanged.
 *
 * The AETHER_SHARED() + "carve" idiom
 * CUDA allows exactly one dynamically-sized `extern __shared__` array per
 * kernel; multiple logical work views share that one allocation, each
 * `make_work_view` call "carving" a different byte offset out of it. Two
 * shapes:
 *
 *   (a) Statically sized (no carving needed) — declare a fixed-size
 *       `AETHER_SHARED()` array per logical buffer (each becomes its own
 *       `__shared__` symbol; the compiler lays them out, no manual offset
 *       arithmetic):
 *   @code
 *       AETHER_KERNEL() void kernel(Vec3dView global, offset_t blockSamples)
 *       {
 *           AETHER_SHARED() double buf[3 * 128]; // 128 = max block size
 *           auto work = aether::make_work_view<double, 3>(buf, blockSamples);
 *           const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
 *           if (i.global() >= global.samples()) return;
 *           work(0, i.work()) = global(0, i.global()); // global -> shared
 *           __syncthreads();                            // ... compute ...
 *       }
 *   @endcode
 *
 *   (b) One `extern __shared__` array, multiple logical buffers carved out
 *       by byte offset (needed when the kernel wants more than one
 *       dynamically-sized shared buffer — CUDA only grants one `extern
 *       __shared__` symbol per kernel):
 *   @code
 *       AETHER_KERNEL() void kernel(Vec3dView a, Vec3dView b, offset_t blockSamples)
 *       {
 *           extern __shared__ unsigned char raw[];
 *           auto workA = aether::make_work_view<double, 3>(
 *               reinterpret_cast<double*>(raw), blockSamples);
 *           auto workB = aether::make_work_view<double, 3>(
 *               reinterpret_cast<double*>(raw + 3 * blockSamples * sizeof(double)),
 *               blockSamples);
 *           // launch with a dynamic shared-mem size covering both carves.
 *       }
 *   @endcode
 *
 * Bundle-aware staging composes directly: a `BundleIndex<W>` built from the
 * same `threadIdx`/`blockIdx`/`blockDim` addresses `W` consecutive
 * work-local slots of a work view exactly the way it addresses `W`
 * consecutive global samples of an ordinary view (`backend/cuda/bundle/
 * {LoadStore,Assign}.h` do not care which `View` they are handed) — a
 * global-to-shared staging kernel can therefore load via a bundle from the
 * global view and store via a bundle into the work view, or vice versa, in
 * the same vectorized idiom `out[bi] = a + s*b` already uses.
 */

#include <cstddef>

#include "aether/device/Device.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/macros.h"
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Construct a `View<T, extents<Es..., dyn>>` over a shared-memory
 *        span: `block_samples` becomes the (block-local) trailing sample
 *        mode, indexed via `SampleIndex::work()`/bundle lanes rather than
 *        `.global()`. No addressability guard (unlike `make_view`):
 *        device code cannot throw, and shared-memory spans are already
 *        bounded by the launch's own `<<<..., sharedBytes>>>` argument.
 */
template<class T, std::size_t... Es>
AETHER_DEVICEHOST() constexpr View<T, extents<Es..., dyn>> make_work_view(T* shared_ptr, offset_t block_samples)
{
    using Ext = extents<Es..., dyn>;
    const Ext ext(static_cast<std::size_t>(block_samples));
    const typename layout_right::mapping<Ext> map(ext);
    return View<T, Ext>(shared_ptr, map, Device(kDLCUDA));
}

} // namespace aether
