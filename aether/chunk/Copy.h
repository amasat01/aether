// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Copy.h
 * @brief `copy`/`copyAsync`: byte-range transfers between `Chunk`s.
 *
 * Legal device-pair matrix for BLOCKING `copy` (see the note above
 * `copy()`): CPU<->CPU, CPU<->CUDAHost, CUDAHost<->CUDA, CUDA<->CUDA.
 * CPU<->CUDA direct is deliberately ILLEGAL — staging through CUDAHost is
 * the caller's business.
 *
 * `copyAsync` is narrower still: no `async` specialisation exists for any
 * pair touching plain (pageable) CPU memory — only CUDAHost<->CUDAHost/
 * CUDA and CUDA<->CUDA are async-capable — so `copyAsync` throws on every
 * CPU-involving pair, not just CPU<->CUDA.
 */

#include <cstddef>
#include <cstring>
#include <string>

#include "aether/chunk/Chunk.h"
#include "aether/err/Error.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

#ifdef AETHER_HAS_CUDA
/** @brief CUDA stream alias — declared only when this build has a CUDA backend. */
using Stream = cudaStream_t;
#endif

namespace detail {

enum class CopyPath {
    CpuCpu,
    CpuCudaHost,
    CudaHostCpu,
    CudaHostCudaHost,
    CudaHostCuda,
    CudaCudaHost,
    CudaCuda,
    Illegal,
};

inline CopyPath classify(DLDeviceType from, DLDeviceType to)
{
    if (from == kDLCPU && to == kDLCPU)
        return CopyPath::CpuCpu;
    if (from == kDLCPU && to == kDLCUDAHost)
        return CopyPath::CpuCudaHost;
    if (from == kDLCUDAHost && to == kDLCPU)
        return CopyPath::CudaHostCpu;
    if (from == kDLCUDAHost && to == kDLCUDAHost)
        return CopyPath::CudaHostCudaHost;
    if (from == kDLCUDAHost && to == kDLCUDA)
        return CopyPath::CudaHostCuda;
    if (from == kDLCUDA && to == kDLCUDAHost)
        return CopyPath::CudaCudaHost;
    if (from == kDLCUDA && to == kDLCUDA)
        return CopyPath::CudaCuda;
    return CopyPath::Illegal;
}

inline std::string pathLabel(const Chunk& src, const Chunk& dst)
{
    return err::device_label(src.device()) + " -> " + err::device_label(dst.device());
}

} // namespace detail

/**
 * @brief Blocking byte-range copy from `src` to `dst`.
 *
 * `dst` and `src` must have equal `.size()`; a mismatch throws. Both the
 * size check and the legal-path check run BEFORE the zero-byte short
 * circuit, so they still fire for a 0-byte pair (this is what lets a
 * `AETHER_CPP_MODE` build exercise the illegal-path check on a `kDLCUDA`
 * `Chunk` obtained via `Chunk::allocate(Device(kDLCUDA), 0)`, without ever
 * needing a real CUDA allocation — see `tests/test_Copy.cpp`).
 *
 * @throws aether::Error on a size mismatch, an illegal device pair
 *         (including direct CPU<->CUDA), or an underlying `cudaMemcpy`
 *         failure.
 */
inline void copy(Chunk& dst, const Chunk& src)
{
    if (dst.size() != src.size()) {
        err::fail("copy", detail::pathLabel(src, dst), src.size(),
            "size mismatch: dst.size()=" + std::to_string(dst.size()));
    }

    const auto path = detail::classify(src.device().type(), dst.device().type());
    if (path == detail::CopyPath::Illegal) {
        err::fail("copy", detail::pathLabel(src, dst), src.size(),
            "illegal device pair (CPU<->CUDA direct is not supported; stage through CUDAHost)");
    }

    const std::size_t bytes = src.size();
    if (bytes == 0)
        return;

    switch (path) {
    case detail::CopyPath::CpuCpu:
    case detail::CopyPath::CpuCudaHost:
    case detail::CopyPath::CudaHostCpu:
    case detail::CopyPath::CudaHostCudaHost:
        // Both endpoints are host-addressable — avoid the CUDA runtime
        // entirely.
        std::memcpy(dst.data(), src.data(), bytes);
        return;
#ifdef AETHER_HAS_CUDA
    case detail::CopyPath::CudaHostCuda:
        err::checkCuda(cudaMemcpy(dst.data(), src.data(), bytes, cudaMemcpyHostToDevice), "copy",
            detail::pathLabel(src, dst), bytes);
        return;
    case detail::CopyPath::CudaCudaHost:
        err::checkCuda(cudaMemcpy(dst.data(), src.data(), bytes, cudaMemcpyDeviceToHost), "copy",
            detail::pathLabel(src, dst), bytes);
        return;
    case detail::CopyPath::CudaCuda:
        err::checkCuda(cudaMemcpy(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice), "copy",
            detail::pathLabel(src, dst), bytes);
        return;
#endif
    case detail::CopyPath::Illegal:
    default:
        // Unreachable: the illegal-path check above already threw.
        return;
    }
}

#ifdef AETHER_HAS_CUDA
/**
 * @brief Asynchronous byte-range copy from `src` to `dst` on `stream`.
 *
 * Only declared when this build has a CUDA backend (absent in
 * `AETHER_CPP_MODE`). Same size-mismatch check as `copy`, but a NARROWER
 * legal-path set: only CUDAHost<->CUDAHost, CUDAHost<->CUDA, and
 * CUDA<->CUDA are async-capable — every CPU-involving pair throws here
 * even though `copy()` allows CPU<->CUDAHost.
 *
 * @throws aether::Error on a size mismatch, a non-async-capable device
 *         pair, or an underlying `cudaMemcpyAsync` failure.
 */
inline void copyAsync(Chunk& dst, const Chunk& src, Stream stream)
{
    if (dst.size() != src.size()) {
        err::fail("copyAsync", detail::pathLabel(src, dst), src.size(),
            "size mismatch: dst.size()=" + std::to_string(dst.size()));
    }

    const auto path = detail::classify(src.device().type(), dst.device().type());

    // Initialised to a harmless sentinel purely so -Wmaybe-uninitialized
    // cannot flag it: every switch arm below either sets `kind` or calls
    // the [[noreturn]] err::fail(), so this value is never actually read.
    cudaMemcpyKind kind = cudaMemcpyDefault;
    switch (path) {
    case detail::CopyPath::CudaHostCudaHost:
        kind = cudaMemcpyHostToHost;
        break;
    case detail::CopyPath::CudaHostCuda:
        kind = cudaMemcpyHostToDevice;
        break;
    case detail::CopyPath::CudaCudaHost:
        kind = cudaMemcpyDeviceToHost;
        break;
    case detail::CopyPath::CudaCuda:
        kind = cudaMemcpyDeviceToDevice;
        break;
    case detail::CopyPath::CpuCpu:
    case detail::CopyPath::CpuCudaHost:
    case detail::CopyPath::CudaHostCpu:
    case detail::CopyPath::Illegal:
    default:
        err::fail("copyAsync", detail::pathLabel(src, dst), src.size(),
            "no async path for plain (pageable) CPU memory — use copy(), staged through "
            "CUDAHost");
        return; // unreachable — err::fail is [[noreturn]]
    }

    const std::size_t bytes = src.size();
    if (bytes == 0)
        return;
    err::checkCuda(cudaMemcpyAsync(dst.data(), src.data(), bytes, kind, stream), "copyAsync",
        detail::pathLabel(src, dst), bytes);
}
#endif

} // namespace aether
