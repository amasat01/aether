// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Layout.h
 * @brief `aether::layout_right` / `aether::layout_stride`: the mapping
 *        policies from a multi-index to a flat offset.
 *
 * `layout_right` is THE SoA anchor: for `extents<C, dyn>` (a `VecView<T,C>`)
 * `operator()(c, i)` folds to `c*N + i` and for `extents<R, C, dyn>`
 * (a `MatView<T,R,C>`) to `(r*C + c)*N + n`. Both mappings are
 * mdspan-mirrored in name only; there is no `std::` dependence.
 *
 * Both `operator()`s fold in `aether::offset_t` (32-bit) end to end and
 * RETURN it — see `aether/index/Offset.h` for why the width is uniform and
 * why a half-narrowed chain is worse than a wide one. `View`'s HOST element
 * access instead uses `detail::elementOffset` (bottom of this file), the
 * same fold at full width. `required_span_size()`
 * deliberately stays `std::size_t`: it is the bounds guard's input, not a
 * hot address computation, and must be able to represent the very spans
 * the guard rejects.
 */

#include <cstddef>
#include <utility>

#include "aether/index/Offset.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Row-major layout policy: the rightmost index varies fastest.
 *        Default layout for `View` and `Array`.
 */
struct layout_right {
    template<class Extents>
    class mapping {
    public:
        using extents_type = Extents;

        // No AETHER_DEVICEHOST() here: a `= default` special member on its
        // first declaration is automatically host+device eligible under
        // nvcc, which warns (#20012-D) that an explicit annotation is
        // redundant (see aether/device/Device.h's own note).
        constexpr mapping() = default;
        AETHER_DEVICEHOST() constexpr explicit mapping(const Extents& ext)
            : extents_(ext)
        {
        }

        AETHER_DEVICEHOST() constexpr const Extents& extents() const { return extents_; }

        /** @brief Row-major fold: `((idx0)*extent(1) + idx1)*extent(2) + idx2 ...`,
         *         entirely in `offset_t`. */
        template<class... Idxs>
        AETHER_DEVICEHOST() constexpr offset_t operator()(Idxs... idxs) const
        {
            static_assert(sizeof...(Idxs) == Extents::Rank,
                "layout_right::mapping::operator(): wrong number of indices");
            return fold_(std::make_index_sequence<Extents::Rank>{}, static_cast<offset_t>(idxs)...);
        }

        /** @brief Product of every extent — the number of elements this mapping
         *         spans, at FULL width (the bounds guard's input; see the file note). */
        AETHER_DEVICEHOST() constexpr std::size_t required_span_size() const
        {
            std::size_t total = 1;
            for (std::size_t i = 0; i < Extents::Rank; ++i)
                total *= extents_.extentWide(i);
            return total;
        }

    private:
        // The fold as a pack expansion over the modes, not a loop: every
        // `extent(Ks)` is then a constant mode index (a static extent folds
        // to a literal), so the whole offset is straight-line code before
        // any loop pass runs and a host sample loop around it is innermost.
        template<std::size_t... Ks, class... I>
        AETHER_DEVICEHOST() constexpr offset_t fold_(std::index_sequence<Ks...>, I... idx) const
        {
            offset_t offset = 0;
            ((offset = offset * extents_.extent(Ks) + idx), ...);
            return offset;
        }

        Extents extents_{};
    };
};

/**
 * @brief Strided layout policy: an explicit per-mode stride (in elements)
 *        replaces the row-major formula. This is the shape foreign
 *        (e.g. DLPack) strides land in — construct via `mapping(extents,
 *        strides)`.
 */
struct layout_stride {
    template<class Extents>
    class mapping {
    public:
        using extents_type = Extents;
        static constexpr std::size_t Rank = Extents::Rank;

        // No AETHER_DEVICEHOST() here — see the note on layout_right::mapping()'s default ctor above.
        constexpr mapping() = default;
        AETHER_DEVICEHOST() constexpr mapping(
            const Extents& ext, const detail::Carray<std::size_t, Rank>& strides)
            : extents_(ext)
            , strides_(strides)
        {
        }

        AETHER_DEVICEHOST() constexpr const Extents& extents() const { return extents_; }
        AETHER_DEVICEHOST() constexpr std::size_t stride(std::size_t i) const { return strides_[i]; }

        /** @brief `dot(idxs, strides)`, in `offset_t` (same narrow chain
         *         as `layout_right`; the strides themselves stay stored at full
         *         width so `required_span_size()` below can still see an
         *         out-of-range span and the bounds guard can reject it). */
        template<class... Idxs>
        AETHER_DEVICEHOST() constexpr offset_t operator()(Idxs... idxs) const
        {
            static_assert(sizeof...(Idxs) == Rank,
                "layout_stride::mapping::operator(): wrong number of indices");
            const detail::Carray<offset_t, Rank> idxArr{ static_cast<offset_t>(idxs)... };
            offset_t offset = 0;
            for (std::size_t i = 0; i < Rank; ++i)
                offset += idxArr[i] * static_cast<offset_t>(strides_[i]);
            return offset;
        }

        /**
         * @brief `0` if any extent is `0`; otherwise `1 + sum((extent(i)-1) *
         *        stride(i))` — the one-past-the-last offset a compliant
         *        strided layout can address (mdspan's own formula).
         */
        AETHER_DEVICEHOST() constexpr std::size_t required_span_size() const
        {
            std::size_t span = 1;
            for (std::size_t i = 0; i < Rank; ++i) {
                const std::size_t e = extents_.extentWide(i);
                if (e == 0)
                    return 0;
                span += (e - 1) * strides_[i];
            }
            return span;
        }

    private:
        Extents extents_{};
        detail::Carray<std::size_t, Rank> strides_{};
    };
};

namespace detail {

template<class Map>
inline constexpr bool isLayoutRightMapping = false;
template<class E>
inline constexpr bool isLayoutRightMapping<layout_right::mapping<E>> = true;
template<class Map>
inline constexpr bool isLayoutStrideMapping = false;
template<class E>
inline constexpr bool isLayoutStrideMapping<layout_stride::mapping<E>> = true;

template<class Map, std::size_t... Ks, class... I>
AETHER_DEVICEHOST() constexpr std::size_t wideRightOffset(const Map& map, std::index_sequence<Ks...>, I... idx)
{
    std::size_t offset = 0;
    ((offset = offset * map.extents().extentWide(Ks) + idx), ...);
    return offset;
}

template<class Map, std::size_t... Ks, class... I>
AETHER_DEVICEHOST() constexpr std::size_t wideStrideOffset(const Map& map, std::index_sequence<Ks...>, I... idx)
{
    return (std::size_t{ 0 } + ... + (idx * map.stride(Ks)));
}

/**
 * @brief The element offset a `View` subscripts its pointer with:
 *        `map(idxs...)` on the device, the same fold at `std::size_t` width
 *        on the host.
 *
 * The device keeps the narrow `offset_t` chain (`aether/index/Offset.h`:
 * a register-pressure decision). On the host a 32-bit fold zero-extended
 * into the address is a WRAPPING expression to the compiler, so a host
 * sample loop over `c*N + i` is not an affine access ("complicated access
 * pattern") and the loop does not vectorise; the same fold at full width is
 * affine in `i`. The indices are taken through `offset_t` first and every
 * view's span fits `offset_t` (the `make_view` guard), so the wide fold
 * names the very element the narrow one does — only its spelling changes.
 * A user-supplied layout keeps its own `operator()`.
 */
template<class Map, class... Idxs>
AETHER_DEVICEHOST() constexpr auto elementOffset(const Map& map, Idxs... idxs)
{
#if defined(__CUDA_ARCH__)
    return map(idxs...);
#else
    if constexpr (isLayoutRightMapping<Map>) {
        static_assert(sizeof...(Idxs) == Map::extents_type::Rank,
            "layout_right::mapping::operator(): wrong number of indices");
        return wideRightOffset(map, std::make_index_sequence<sizeof...(Idxs)>{},
            static_cast<std::size_t>(static_cast<offset_t>(idxs))...);
    } else if constexpr (isLayoutStrideMapping<Map>) {
        static_assert(sizeof...(Idxs) == Map::Rank,
            "layout_stride::mapping::operator(): wrong number of indices");
        return wideStrideOffset(map, std::make_index_sequence<sizeof...(Idxs)>{},
            static_cast<std::size_t>(static_cast<offset_t>(idxs))...);
    } else {
        return map(idxs...);
    }
#endif
}

} // namespace detail

} // namespace aether
