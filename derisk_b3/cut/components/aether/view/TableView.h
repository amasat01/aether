// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file TableView.h
 * @brief `aether::TableView`: a thin facade naming `aether::View`'s
 *        existing row-major, non-owning, read-only multi-dimensional
 *        access pattern.
 *
 * `TableView<Carrier, Extents...>` is, op-for-op,
 * `aether::View<const T, extents<Es...>, layout_right>`. No new storage or
 * behaviour is added here — this header adds no members and no runtime
 * cost over `View` itself: `TableView` is a plain alias, `makeTableView`
 * a thin pointer-arithmetic wrapper over the existing `make_view`
 * foreign-pointer overload.
 *
 * `makeTableView` is deliberately unannotated (no `AETHER_DEVICEHOST()`),
 * matching `make_view`'s own convention: it delegates to `make_view`,
 * which throws `aether::Error` on an out-of-range span — a host-only
 * operation. Once built, the resulting `TableView` (being nothing but a
 * `View`) is fully `AETHER_DEVICEHOST()` on every accessor and crosses a
 * kernel boundary by value like any other `View`: build host-side, read
 * device-side.
 *
 * `makeTableView()` itself lives in `aether/view/MakeTableView.h` — an
 * unannotated free function is rejected outright by NVRTC's JIT mode even
 * bodiless/uncalled (a stricter rule than nvcc's normal offline device
 * pass, which tolerates it fine since every call site is host-only).
 * `TableView<T,Extents>` itself is a plain alias over `View` (nothing to
 * guard) and stays here — this header is standalone under NVRTC. Include
 * `aether/view/MakeTableView.h` (or `aether/aether.h`, which does)
 * wherever `makeTableView()` is called.
 */

#include "aether/view/View.h"

namespace aether {

/**
 * @brief Dimensioned, read-only, non-owning view over `T`, over
 *        `aether::View`.
 *
 * @tparam T        Element type (the view is always over `const T` —
 *                   `TableView` is read-only by construction).
 * @tparam Extents  An `aether::extents<Es...>` — one static size or
 *                   `aether::dyn` per dimension.
 */
template<class T, class Extents>
using TableView = View<const T, Extents, layout_right>;

} // namespace aether
