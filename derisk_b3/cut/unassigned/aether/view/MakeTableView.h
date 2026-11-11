// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file MakeTableView.h
 * @brief `aether::makeTableView()` — split out of `aether/view/
 *        TableView.h` (device-safety split).
 *
 * Deliberately unannotated (no `AETHER_DEVICEHOST()`), matching
 * `make_view`'s own convention — it delegates to `make_view`, which throws
 * `aether::Error` on an out-of-range span, a host-only operation.
 * NVRTC's JIT mode rejects an entirely unannotated function outright, even
 * bodiless/uncalled, unlike nvcc's normal offline device pass — see
 * `TableView.h`'s own note. Include this header explicitly (or
 * `aether/aether.h`, which does) wherever `makeTableView()` is called —
 * every existing call site already reaches it through `aether.h`
 * unchanged.
 */

#include "aether/view/MakeView.h"
#include "aether/view/TableView.h"
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Build a `TableView<T, extents<Es...>>` over `ptr + offset`.
 *
 * @tparam Es       One value per dimension: a compile-time size or
 *                   `aether::dyn`. Given explicitly, as in `make_view`.
 * @param  ptr      Base pointer the table's first element (before `offset`
 *                   is applied) would sit at were `offset == 0`; `T` may
 *                   already be `const`.
 * @param  dev      The device `ptr` is resident on.
 * @param  offset   Element offset of the table's first element from `ptr`
 *                   (`offset_t`) — folded into the pointer here, not
 *                   stored (see `TableView.h`'s file docstring's "Offset"
 *                   note).
 * @param  dynVals  Runtime extents, in dimension order, one per
 *                   `aether::dyn` slot in `Es...` — forwarded verbatim to
 *                   `make_view`, which validates the count and the span
 *                   guard.
 * @throws aether::Error  under the same conditions `make_view`'s
 *         foreign-pointer overload does (span guard; null pointer with
 *         non-zero span).
 */
template<std::size_t... Es, class T, class... DynVals>
TableView<T, extents<Es...>> makeTableView(T* ptr, Device dev, offset_t offset, DynVals... dynVals)
{
    return make_view<const T, Es...>(ptr + offset, dev, dynVals...);
}

} // namespace aether
