// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Copy.h
 * @brief `aether::copyAsync(View, View, Stream)`: view-level transfer
 *        over the existing `Chunk`-pair machinery (`aether/chunk/Copy.h`).
 *
 * `aether::View` has no `upload`/`download`/`fetch` of its own — every
 * existing transfer is either `Array`-level (`Array::upload()`/
 * `download()`, blocking or `Stream`'d) or `Chunk`-level (`aether::copy`/
 * `copyAsync`, `chunk/Copy.h`). A caller holding two bare `View`s (e.g.
 * one sliced out of a scratch allocation nobody wrapped in an `Array`)
 * had no async transfer to reach for at all.
 *
 * This wraps each view's own backing span in a non-owning `Chunk`
 * (`Chunk::borrow` — the same technique `Array::copyRowsPitch_` already
 * uses for its own pitched realloc copies) and defers straight to
 * `aether::copyAsync(Chunk&, const Chunk&, Stream)` for the transfer
 * itself, so the legal device-pair matrix, the CPU-pair refusal, and the
 * underlying `cudaMemcpyAsync` call are the same code `chunk/Copy.h`
 * already audits — this file adds no new transport rule, only a
 * view-shaped entry point onto the existing one.
 *
 * CUDA-only, like every other `Stream`-taking call in this library
 * (`aether::Stream` is itself declared only under `AETHER_HAS_CUDA`,
 * `chunk/Copy.h`) — there is no `AETHER_CPP_MODE` counterpart to widen
 * into; a pure-C++ build has no stream/async concept at all, so this
 * header contributes nothing there (see `tests/test_ViewCopyAsync.cu`'s
 * own note on how the CPU-pair refusal is still exercised without one).
 */

#ifdef AETHER_HAS_CUDA

#include <cstddef>
#include <string>
#include <type_traits>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/err/Error.h"
#include "aether/residency/Partition.h" // detail::partitionStridesOf -- the per-mode stride reader Array.h itself already reuses from here
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Asynchronous `src` -> `dst` transfer on `stream`, at VIEW
 *        granularity.
 *
 * `dst` and `src` must share the same `Extents`/`Layout` (a single pair of
 * template parameters both views bind to — a mismatched RANK or Layout is
 * a compile error, never reaches this body) and their element types must
 * agree up to `const` (`src` read-only is fine; `dst` must not be). Beyond
 * that static shape, the two views' runtime extents and per-mode strides
 * must also be identical — two `Array<double,3>`s with different
 * `capacity()` share `Extents`/`Layout` but pitch their components
 * differently, exactly the case this catches. A mismatch on either throws
 * `aether::Error` rather than silently transferring the wrong bytes.
 *
 * Both views' own backing span (`data()` through `mapping().
 * required_span_size()` elements) is wrapped as a non-owning `Chunk`
 * (`Chunk::borrow`) and handed to `aether::copyAsync(Chunk&, const Chunk&,
 * Stream)` (`chunk/Copy.h`) for the actual transfer — so it throws on
 * exactly the pairs THAT function refuses (every CPU-involving pair; only
 * CUDAHost<->CUDAHost/CUDA and CUDA<->CUDA are async-capable) and accepts
 * exactly the pairs it accepts: this file adds no new transport rule.
 *
 * @throws aether::Error  on an extents/strides mismatch, or anything
 *         `aether::copyAsync(Chunk&, const Chunk&, Stream)` itself throws
 *         (a non-async-capable device pair, or an underlying
 *         `cudaMemcpyAsync` failure).
 */
template<class TDst, class TSrc, class Extents, class Layout>
void copyAsync(View<TDst, Extents, Layout> dst, const View<TSrc, Extents, Layout> src, Stream stream)
{
    static_assert(std::is_same_v<std::remove_const_t<TSrc>, TDst>,
        "aether::copyAsync(View, View, Stream): dst/src element types must agree up to const "
        "(src read-only is fine; dst must not be const)");

    const std::string where = err::device_label(src.device()) + " -> " + err::device_label(dst.device());
    constexpr std::size_t Rank = Extents::Rank;

    for (std::size_t i = 0; i < Rank; ++i) {
        if (dst.extent(i) != src.extent(i)) {
            err::fail("copyAsync(View)", where, static_cast<std::size_t>(dst.size()) * sizeof(TDst),
                "extents mismatch at mode " + std::to_string(i) + ": dst.extent=" + std::to_string(dst.extent(i))
                    + " src.extent=" + std::to_string(src.extent(i)));
        }
    }

    const auto dstStrides = detail::partitionStridesOf(dst);
    const auto srcStrides = detail::partitionStridesOf(src);
    for (std::size_t i = 0; i < Rank; ++i) {
        if (dstStrides[i] != srcStrides[i]) {
            err::fail("copyAsync(View)", where, static_cast<std::size_t>(dst.size()) * sizeof(TDst),
                "strides mismatch at mode " + std::to_string(i) + ": dst.stride=" + std::to_string(dstStrides[i])
                    + " src.stride=" + std::to_string(srcStrides[i]));
        }
    }

    const std::size_t bytes = dst.mapping().required_span_size() * sizeof(TDst);
    Chunk dstChunk = Chunk::borrow(dst.data(), bytes, dst.device());
    // `src.data()` is `const TSrc*` (or `TDst*` if the caller passed a
    // mutable view as `src`, e.g. Array::deviceView() before .as_const());
    // Chunk::borrow's `void*` parameter has no const-preserving overload,
    // and this Chunk is only ever read from below (the source side of a
    // one-way copy) -- same idiom as aether/view/MakeRuntimeView.h's own
    // `const_cast<void*>(static_cast<const void*>(v.data()))`.
    Chunk srcChunk = Chunk::borrow(
        const_cast<void*>(static_cast<const void*>(src.data())), bytes, src.device());

    copyAsync(dstChunk, srcChunk, stream);
}

} // namespace aether

#endif // AETHER_HAS_CUDA
