// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Partition.h
 * @brief `aether::PartitionSpec` / `aether::padded_block_samples` /
 *        `aether::partition_view`: slicing a batched `View`'s
 *        trailing SAMPLE mode into `parts` even, PADDED blocks.
 *
 * This module's SCOPE is precise: distribution itself (moving a partition's
 * block to its own device/node) is a separate concern
 * (`aether/residency/Transport.h`'s concept seam); what this file provides
 * is the LAYOUT ALGEBRA a partitioned array needs regardless of where its
 * blocks eventually live — the offset/stride arithmetic for "block r of
 * `parts`, each block padded up to a `pad_to` granularity" (`Chunk`'s
 * 256-byte allocation alignment, `aether/chunk/Chunk.h`, is the analogous
 * granularity at the byte level; `pad_to` here is in SAMPLES, the caller's
 * choice).
 *
 * `partition_view` slices the ORIGINAL view's own memory — it allocates
 * nothing and copies nothing. `padded_block_samples(n, parts, pad_to)` is
 * the block PITCH: how far apart (in samples) two consecutive blocks start.
 * Every block but possibly the last has exactly that many REAL samples;
 * the trailing block's real extent is whatever remains of `n` — SHORT when
 * `n` is not an exact multiple of the pitch — while block boundaries
 * themselves still fall at regular, PADDED intervals regardless of any one
 * block's real occupancy ("real extent, padded stride" — see
 * `partition_view`'s own docstring for the closed-form derivation).
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/layout/detail/Carray.h"
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Partition parameters: split the SAMPLE mode into `parts`
 *        blocks, each block pitch rounded up to a `pad_to`-sample
 *        granularity (default 32).
 */
struct PartitionSpec {
    std::size_t parts;
    std::size_t pad_to = 32;
};

/**
 * @brief The per-block PITCH (in samples): `ceil(n / parts)`,
 *        rounded up to the next multiple of `pad_to`. This is the stride
 *        between consecutive blocks' start offsets, NOT necessarily any
 *        one block's real occupancy — see `partition_view`.
 *
 * @throws aether::Error  if `parts == 0` or `pad_to == 0`.
 */
inline std::size_t padded_block_samples(std::size_t n, std::size_t parts, std::size_t pad_to = 32)
{
    if (parts == 0)
        err::fail("padded_block_samples", "n/a", n, "parts must be > 0");
    if (pad_to == 0)
        err::fail("padded_block_samples", "n/a", n, "pad_to must be > 0");

    const std::size_t perPart = (n + parts - 1) / parts; // ceil-div
    return ((perPart + pad_to - 1) / pad_to) * pad_to; // round up to pad_to
}

namespace detail {

/**
 * @brief Per-mode strides (element units), read from an existing `View`'s
 *        own mapping — `layout_right`'s implicit row-major formula, or
 *        `layout_stride`'s explicit per-mode strides. Mirrors
 *        `aether::fromView`'s identical derivation (`aether/view/
 *        RuntimeView.h`) — kept local rather than shared because that
 *        header's helper is a private implementation detail of a different
 *        (runtime-descriptor) conversion, not a public API.
 */
template<class ViewT>
Carray<std::size_t, ViewT::extents_type::Rank> partitionStridesOf(const ViewT& v)
{
    using Layout  = typename ViewT::layout_type;
    using Extents = typename ViewT::extents_type;
    Carray<std::size_t, Extents::Rank> strides{};
    if constexpr (std::is_same_v<Layout, layout_right>) {
        std::size_t acc = 1;
        for (std::size_t i = Extents::Rank; i-- > 0;) {
            strides[i] = acc;
            acc *= static_cast<std::size_t>(v.extent(i));
        }
    } else if constexpr (std::is_same_v<Layout, layout_stride>) {
        for (std::size_t i = 0; i < Extents::Rank; ++i)
            strides[i] = v.mapping().stride(i);
    } else {
        static_assert(!sizeof(Layout*), "partition_view: unsupported Layout policy (only layout_right/layout_stride)");
    }
    return strides;
}

} // namespace detail

/**
 * @brief A `View` over block `r` of `spec.parts`  — slices `view`'s
 *        own memory (no allocation, no copy).
 *
 * Requires exactly one dynamic mode (`Extents::RankDynamic == 1`), the
 * TRAILING one — the SAMPLE mode every batched `*View` alias appends
 * (partitioning is scoped to this mode only).
 *
 * CLOSED FORM: let `N = view.samples()`, `padded =
 * padded_block_samples(N, spec.parts, spec.pad_to)`.
 *   - `offset = r * padded` (elements) — added to `view.data()` (the
 *     sample mode's own stride is always 1 in this library's convention,
 *     so an element offset needs no further scaling: "offset =
 *     r*padded*elem-stride").
 *   - `real = min(padded, N - offset)` when `offset < N`, else `0` — the
 *     block's own runtime SAMPLE extent (SHORT only for a trailing block
 *     that does not fill its full pitch).
 *   - every OTHER mode's extent and EVERY mode's stride are copied
 *     VERBATIM from `view`'s own mapping — block boundaries fall at
 *     regular `padded`-sample intervals, but a block's internal layout
 *     (how far apart its own components sit in memory) is exactly
 *     `view`'s own, unchanged (the "padded stride": a uniform
 *     block PITCH, not a re-derived per-block component stride).
 *
 * The result is a `layout_stride` view (still DLPack-expressible) —
 * even when `view` itself is `layout_right`, a non-full, non-first block
 * is not compact under the trailing mode's own leading-mode products, so
 * the strided mapping is the only one that can represent it in general.
 *
 * @throws aether::Error  if `r >= spec.parts`.
 */
template<class T, class Extents, class Layout>
View<T, Extents, layout_stride> partition_view(
    const View<T, Extents, Layout>& view, const PartitionSpec& spec, std::size_t r)
{
    static_assert(Extents::Rank >= 1, "partition_view: Extents must have rank >= 1");
    static_assert(Extents::RankDynamic == 1, "partition_view: exactly one dynamic (SAMPLE) mode is supported");
    static_assert(Extents::static_extent(Extents::Rank - 1) == dyn,
        "partition_view: the trailing mode must be the dynamic SAMPLE mode");

    if (r >= spec.parts) {
        err::fail("partition_view", err::device_label(view.device()), 0,
            "block index " + std::to_string(r) + " >= spec.parts=" + std::to_string(spec.parts));
    }

    const std::size_t n      = static_cast<std::size_t>(view.samples());
    const std::size_t padded = padded_block_samples(n, spec.parts, spec.pad_to);
    const std::size_t offset = r * padded;
    const std::size_t real   = (offset < n) ? std::min(padded, n - offset) : std::size_t{ 0 };

    const auto strides = detail::partitionStridesOf(view);

    Extents newExt(real);
    typename layout_stride::mapping<Extents> map(newExt, strides);
    T* newData = view.data() + offset;
    return View<T, Extents, layout_stride>(newData, map, view.device());
}

} // namespace aether
