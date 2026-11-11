// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Fill.h
 * @brief `fill`: set every element of a `Chunk` to one repeated value, on a stream.
 *
 * No kernels: a device chunk is filled by `cudaMemsetAsync` (every byte of the value equal) or, for any
 * other value, by one `cudaMemset2DAsync` per byte of the element writing that byte plane with a strided
 * 1-byte memset. Both are capture-legal, so `fill` may sit inside a stream capture and replays with the
 * graph. Host kinds (`kDLCPU`, `kDLCUDAHost`) are filled by the calling thread, `stream` ignored: ordering a
 * pinned fill against work already queued on a stream is the caller's business.
 */

#include <cstddef>
#include <cstring>
#include <span>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/StreamOrdered.h"
#include "aether/err/Error.h"

namespace aether {

/**
 * @brief Fill `dst` with the element `value` (its raw bytes, any width >= 1) repeated across the chunk.
 *
 * @throws aether::Error  if `value` is empty, `dst.size()` is not a multiple of `value.size()`, `dst` is a
 *         `kDLCUDA` chunk in a build with no CUDA backend, or a CUDA call fails.
 */
inline void fill(Chunk& dst, std::span<const std::byte> value, Stream stream)
{
    const std::size_t width = value.size();
    const std::size_t bytes = dst.size();
    if (width == 0 || bytes % width != 0) {
        err::fail("fill", err::device_label(dst.device()), bytes,
            "value width " + std::to_string(width) + " does not divide the chunk");
    }
    if (bytes == 0)
        return;

    bool uniform = true;
    for (std::size_t j = 1; j < width; ++j)
        uniform = uniform && value[j] == value[0];

    if (dst.device().type() != kDLCUDA) {
        if (uniform) {
            std::memset(dst.data(), static_cast<int>(value[0]), bytes);
        } else {
            for (std::size_t i = 0; i < bytes; i += width)
                std::memcpy(dst.data() + i, value.data(), width);
        }
        return;
    }
#ifdef AETHER_HAS_CUDA
    const std::string where = err::device_label(dst.device());
    detail::DeviceGuard device(dst.device().id());
    if (uniform) {
        err::checkCuda(cudaMemsetAsync(dst.data(), static_cast<int>(value[0]), bytes, stream), "fill", where, bytes);
        return;
    }
    for (std::size_t j = 0; j < width; ++j) {
        err::checkCuda(cudaMemset2DAsync(dst.data() + j, width, static_cast<int>(value[j]), 1, bytes / width, stream),
            "fill", where, bytes);
    }
#else
    (void)stream;
    err::fail("fill", err::device_label(dst.device()), bytes,
        "this build has no CUDA backend (AETHER_CPP_MODE / no CUDA toolchain)");
#endif
}

} // namespace aether
