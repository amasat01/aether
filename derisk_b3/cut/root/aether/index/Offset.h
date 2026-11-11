// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Offset.h
 * @brief `aether::offset_t`: the index/offset width for every address
 *        computation in aether — a 32-bit unsigned integer.
 *
 * Why 32 bits: address arithmetic width is a register-pressure decision on
 * the GPU, not a capacity one. A 64-bit running index or byte address
 * occupies a register pair and every step of the chain costs an
 * `IADD.CC`/`IADD.X` couple; the 32-bit form keeps one register per
 * pending index and lets ptxas fold the widening into the load itself
 * (`ISCADD Rd.CC, Ridx, base, 0x3` + `SHR.U32`/`IADD.X`). Measured on a
 * `matmat33` kernel at sm_61 with a single-variable probe (identical SoA
 * 3x3 product, only the index type differing): 40 registers at 32 bits vs
 * 48 at 64. Paired with the destination staging in `expr/nodes/Product.h`'s
 * `AssignDispatch<MatMat>` this reaches 40 registers overall.
 *
 * The seam rule: a half-narrowed chain is worse than a wide one. Narrowing
 * only part of the chain (e.g. `layout_right::mapping::operator()` alone,
 * leaving `SampleIndex`, `extent()` and `View`'s accessors at
 * `std::size_t`) costs address instructions at every 32<->64 hop and buys
 * no registers. So the width is uniform across the whole hot chain —
 * `SampleIndex::global()`/`work()`, `extents::extent()`, both
 * `mapping::operator()`s, and `View::operator()`/`eval()`/`extent()`/
 * `samples()`/`size()`. The only legal widening is the final subscript
 * (`data_[offset]`), where the compiler folds it into the addressing mode.
 *
 * The host subscript is the one exception, and only for the two built-in
 * layouts: `View`'s element access computes the same fold at `std::size_t`
 * width there (`detail::elementOffset`, `aether/layout/Layout.h`). The
 * register argument is a device one; on the host a 32-bit fold widened at
 * the subscript is a wrapping expression the compiler cannot treat as
 * affine in the sample index, which keeps a host sample loop from
 * vectorising. Every view's span fits `offset_t`, so both widths name the
 * same element.
 *
 * What stays 64-bit:
 *   - Compile-time shape: `extents::static_extent()`, `Rank`, `aether::dyn`
 *     (`SIZE_MAX`) — pure template arithmetic, no runtime cost.
 *   - The API boundary: `extents`'s constructor, `make_view`'s dynamic-value
 *     arguments, `Array`'s sample count and byte sizes, and every
 *     `Chunk`/allocation size — bytes are not offsets.
 *   - `required_span_size()` and `extents::extentWide()`, deliberately: the
 *     guard below has to be able to see a span that does not fit, which it
 *     could not do if it were computed in the narrow type it is checking.
 *
 * The guard: `offset_max` is the largest addressable element offset.
 * `make_view` (`aether/view/View.h`) throws `aether::Error` when a view's
 * `required_span_size()` exceeds it, so the narrowing can never silently
 * wrap: the failure is a host-side throw at construction, before any
 * device code can compute a truncated address. The check lives in
 * `make_view` and not in `View`'s constructor because that constructor is
 * `AETHER_DEVICEHOST()` and device code cannot throw.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/macros.h"
#include "aether/typedefs.h"

namespace aether {

/** @brief The index/offset type — every runtime address computation's
 *  width. Derived from `aether::index_base_t` (`aether/typedefs.h`), the
 *  single point of change for the whole index family — never re-declared
 *  independently here. */
using offset_t = index_base_t;

/** @brief Invariant: `offset_t` and `idx_t` are the same type — one
 *  switch (`AETHER_INDEX_T`, `aether/typedefs.h`) controls both. */
static_assert(std::is_same_v<offset_t, idx_t>, "offset_t must track idx_t/index_base_t (aether/typedefs.h)");

/** @brief Largest element offset an `offset_t` can address; the `make_view` guard's bound. */
inline constexpr std::size_t offset_max = static_cast<std::size_t>(UINT32_MAX);

/**
 * @brief `true` when `span` (an element count, computed wide) is addressable
 *        by `offset_t`. The predicate behind the `make_view` guard —
 *        exposed so any later view factory gates on the same condition
 *        rather than restating it.
 */
AETHER_DEVICEHOST() constexpr bool offset_fits(std::size_t span)
{
    return span <= offset_max;
}

} // namespace aether
