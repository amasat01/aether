// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file MakeView.h
 * @brief `aether::make_view()` — the host-only `View` factories split out
 *        of `aether/view/View.h` (device-safety split).
 *
 * `View` itself never called `err::` or needed `Chunk` — only these two
 * factory overloads did (the `Chunk` one directly; the foreign-pointer one
 * only for its `err::fail` diagnostics). Splitting them out here, rather
 * than guarding them in place ("split, don't guard, where a split is
 * possible"): `View.h` stays a plain class header with no
 * `<string>`/`aether/err/Error.h`/`aether/chunk/Chunk.h` dependency at
 * all, so it (and `aether/device.h`, which includes it but not this
 * header) parse under NVRTC. Include this header explicitly (or
 * `aether/aether.h`, which does) wherever `make_view()` is called —
 * every existing call site already reaches it through `aether.h`
 * unchanged.
 */

#include <cstddef>
#include <string>

#include "aether/chunk/Chunk.h"
#include "aether/device/Device.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Construct a `View<T, extents<Es...>>` over `chunk`.
 *
 * `dynVals...` are `chunk`'s dynamic-mode values, in order (matching
 * `extents<Es...>`'s own constructor). Throws `aether::Error` when
 * `chunk.size() < mapping.required_span_size() * sizeof(T)`, and when that
 * span exceeds `offset_max`, so aether's 32-bit `offset_t` addressing can
 * never silently wrap: the failure is a host throw at construction, before
 * any device code computes a truncated address.
 */
template<class T, std::size_t... Es, class... DynVals>
View<T, extents<Es...>> make_view(const Chunk& chunk, DynVals... dynVals)
{
    using Ext = extents<Es...>;
    Ext ext(dynVals...);
    typename layout_right::mapping<Ext> map(ext);

    const std::size_t span = map.required_span_size();
    if (!offset_fits(span)) {
        err::fail("make_view", err::device_label(chunk.device()), chunk.size(),
            "required span " + std::to_string(span) + " elements exceeds the addressable maximum "
                + std::to_string(offset_max) + " (aether addresses elements with a 32-bit offset_t)");
    }

    const std::size_t needBytes = span * sizeof(T);
    if (chunk.size() < needBytes) {
        err::fail("make_view", err::device_label(chunk.device()), chunk.size(),
            "chunk too small for the requested extents: need " + std::to_string(needBytes) + " bytes");
    }

    return View<T, Ext>(reinterpret_cast<T*>(chunk.data()), map, chunk.device());
}

/**
 * @brief Construct a `View<T, extents<Es...>>` over foreign, caller-owned
 *        memory — no size check beyond null-with-size: a null `ptr`
 *        is only rejected when the requested extents span a non-zero
 *        number of elements. The addressability guard applies here too
 *        (see the chunk overload above); since it needs no allocation,
 *        this overload is also how that guard is tested.
 */
template<class T, std::size_t... Es, class... DynVals>
View<T, extents<Es...>> make_view(T* ptr, Device dev, DynVals... dynVals)
{
    using Ext = extents<Es...>;
    Ext ext(dynVals...);
    typename layout_right::mapping<Ext> map(ext);

    const std::size_t span = map.required_span_size();
    if (!offset_fits(span)) {
        err::fail("make_view", err::device_label(dev), 0,
            "required span " + std::to_string(span) + " elements exceeds the addressable maximum "
                + std::to_string(offset_max) + " (aether addresses elements with a 32-bit offset_t)");
    }

    if (ptr == nullptr && span > 0) {
        err::fail("make_view", err::device_label(dev), 0,
            "null pointer with a non-zero required span size");
    }

    return View<T, Ext>(ptr, map, dev);
}

} // namespace aether
