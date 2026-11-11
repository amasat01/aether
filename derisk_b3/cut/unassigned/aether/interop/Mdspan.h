// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Mdspan.h
 * @brief `aether::interop::toMdspan`/`fromMdspan`: host-only `View` <->
 *        `std::mdspan` adapter for `layout_right` cases.
 *
 * Host-only, C++23-guarded: never included by the umbrella
 * (`aether/aether.h`) — mirrors `aether/dtype/Format.h`'s "explicit
 * include only" convention (a device-visible TU, or a host TU built at
 * the C++20 floor, must never be forced to see this). Include it directly
 * when you need the adapter.
 *
 * Feature-gated the way every other optional backend in this library
 * already is (`AETHER_HAS_CUDA`): `__has_include(<mdspan>)`, ANDed with
 * the existing `AETHER_HAS_CXX23` gate (`aether/macros.h`). A toolchain
 * can accept the C++23 language while its libstdc++ does not yet ship the
 * `<mdspan>` library header at all (`std::mdspan`/P0009 lands in a later
 * libstdc++ than the language feature does — a toolchain fact, not a code
 * defect), so the body below compiles to an inert (empty) translation
 * unit on such a toolchain and activates automatically once its
 * libstdc++/libc++ carries `<mdspan>`. Untested on a toolchain missing
 * it as a result; no substitute type (e.g. `cuda::std::mdspan`)
 * or hand-rolled alternative is used in its place, since either would
 * change what "View <-> std::mdspan" means.
 */

#include "aether/macros.h"

#if defined(AETHER_HAS_CXX23) && defined(__has_include)
#if __has_include(<mdspan>)
#define AETHER_HAS_STD_MDSPAN 1
#endif
#endif

#ifdef AETHER_HAS_STD_MDSPAN

#include <array>
#include <cstddef>
#include <mdspan>
#include <utility>

#include "aether/device/Device.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/view/View.h"

namespace aether {
namespace interop {

namespace detail {

/** @brief `std::extents<std::size_t, ...>` matching `Ext` (an
 *         `aether::extents<Es...>`) mode-for-mode. `aether::dyn` and
 *         `std::dynamic_extent` are both `std::size_t(-1)` by their
 *         respective standards, so the VALUES already agree — this alias
 *         just spells the pack out for `std::extents`'s template head. */
template<class Ext, std::size_t... Is>
constexpr auto stdExtentsTypeOf(std::index_sequence<Is...>)
    -> std::extents<std::size_t, Ext::static_extent(Is)...>;

template<class Ext>
using StdExtents = decltype(stdExtentsTypeOf<Ext>(std::make_index_sequence<Ext::Rank>{}));

/** @brief The reverse mapping: `aether::extents<Es...>` matching an
 *         `std::mdspan` extents type `Ext` mode-for-mode. */
template<class Ext, std::size_t... Is>
constexpr auto aetherExtentsTypeOf(std::index_sequence<Is...>)
    -> aether::extents<(Ext::static_extent(Is) == std::dynamic_extent ? aether::dyn : Ext::static_extent(Is))...>;

template<class Ext>
using AetherExtents = decltype(aetherExtentsTypeOf<Ext>(std::make_index_sequence<Ext::rank()>{}));

} // namespace detail

/**
 * @brief `View<T, Extents, layout_right>` -> `std::mdspan<T,
 *        detail::StdExtents<Extents>, std::layout_right>` — zero-copy,
 *        host-only.
 */
template<class T, class Extents>
auto toMdspan(const View<T, Extents, layout_right>& v)
{
    using StdExt = detail::StdExtents<Extents>;
    std::array<std::size_t, Extents::RankDynamic> dynVals{};
    std::size_t k = 0;
    for (std::size_t i = 0; i < Extents::Rank; ++i) {
        if (Extents::static_extent(i) == aether::dyn)
            dynVals[k++] = static_cast<std::size_t>(v.extent(i));
    }
    return std::mdspan<T, StdExt, std::layout_right>(v.data(), StdExt(dynVals));
}

/**
 * @brief `std::mdspan<T, Ext, std::layout_right>` -> `aether::View<T,
 *        detail::AetherExtents<Ext>, layout_right>` — zero-copy,
 *        host-only. `device` is supplied by the caller (an `std::mdspan`
 *        carries no device tag of its own).
 */
template<class T, class Ext>
View<T, detail::AetherExtents<Ext>, layout_right> fromMdspan(const std::mdspan<T, Ext, std::layout_right>& md, Device device)
{
    using AExt = detail::AetherExtents<Ext>;
    std::array<std::size_t, AExt::RankDynamic> dynVals{};
    std::size_t k = 0;
    for (std::size_t i = 0; i < Ext::rank(); ++i) {
        if (Ext::static_extent(i) == std::dynamic_extent)
            dynVals[k++] = md.extent(i);
    }
    const AExt ext = [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
        if constexpr (AExt::RankDynamic == 0) {
            return AExt{};
        } else {
            return AExt(dynVals[Ks]...);
        }
    }(std::make_index_sequence<AExt::RankDynamic>{});

    typename layout_right::mapping<AExt> map(ext);
    return View<T, AExt, layout_right>(md.data_handle(), map, device);
}

} // namespace interop
} // namespace aether

#endif // AETHER_HAS_STD_MDSPAN
