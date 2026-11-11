// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Extents.h
 * @brief `aether::extents<Es...>`: static|dynamic shape, mdspan-mirrored
 *        naming but with no `std::` dependence.
 *
 * `aether::dyn` is the dynamic-extent sentinel (mirrors `std::dynamic_extent`).
 * An `extents<Es...>` mode is either a compile-time constant (any value other
 * than `dyn`) or dynamic (`dyn`, resolved at construction time). All-static
 * extents (`rank_dynamic() == 0`) are default-constructible and carry no
 * per-instance state beyond the (empty) dynamic-value storage — see
 * `aether::detail::Carray<T,0>`.
 */

#include <cstddef>
#include <cstdint>
#include <utility>

#include "aether/index/Offset.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"

namespace aether {

/** @brief The dynamic-extent sentinel. */
inline constexpr std::size_t dyn = SIZE_MAX;

namespace detail {

template<std::size_t... Es>
inline constexpr std::size_t count_dynamic = ((Es == aether::dyn ? std::size_t{ 1 } : std::size_t{ 0 }) + ... + std::size_t{ 0 });

} // namespace detail

/**
 * @brief Compile-time-and-runtime shape: `Es...` are the modes, each
 *        either a static extent or `aether::dyn`.
 *
 * The constructor takes exactly `rank_dynamic()` `std::size_t` values, in
 * order, one per `dyn` slot.
 */
template<std::size_t... Es>
class extents {
public:
    /** @brief Number of modes. */
    static constexpr std::size_t Rank = sizeof...(Es);
    /** @brief Number of `dyn` modes. */
    static constexpr std::size_t RankDynamic = detail::count_dynamic<Es...>;

    /** @brief Number of modes (mdspan-mirrored name). */
    AETHER_DEVICEHOST() static constexpr std::size_t rank() { return Rank; }
    /** @brief Number of `dyn` modes (mdspan-mirrored name). */
    AETHER_DEVICEHOST() static constexpr std::size_t rank_dynamic() { return RankDynamic; }

    /** @brief The compile-time extent of mode `i` (`dyn` if that mode is dynamic). */
    AETHER_DEVICEHOST() static constexpr std::size_t static_extent(std::size_t i)
    {
        return staticExtents_()[i];
    }

    /**
     * @brief The runtime extent of mode `i` — resolves `dyn` modes from
     *        stored values. Returns `offset_t`: this is the value the
     *        row-major fold multiplies by, so it is part of the narrow hot
     *        chain. Storage stays `std::size_t` (the constructor's API is
     *        unchanged) — truncating a stored 64-bit value at the read is
     *        free on both targets (it is the low half of the register pair),
     *        whereas doing the arithmetic wide is not.
     *
     * Use `extentWide()` where the untruncated value is the point, i.e. the
     * span guard.
     */
    AETHER_DEVICEHOST() constexpr offset_t extent(std::size_t i) const { return static_cast<offset_t>(extentWide(i)); }

    /**
     * @brief The runtime extent of mode `i` at full `std::size_t` width —
     *        the accessor `required_span_size()` (and therefore the span
     *        guard) reads, so an over-large shape is still visible as
     *        over-large rather than silently wrapped by the very narrowing
     *        the guard exists to police.
     */
    AETHER_DEVICEHOST() constexpr std::size_t extentWide(std::size_t i) const
    {
        const std::size_t s = staticExtents_()[i];
        // `if constexpr` (not a runtime branch): when RankDynamic == 0,
        // `dynamic_` is the zero-size Carray specialization with no
        // operator[] at all — the body below must not even be compiled
        // for that instantiation, not merely unreached at runtime.
        if constexpr (RankDynamic > 0) {
            if (s != dyn)
                return s;
            return dynamic_[dynamic_index_(i)];
        } else {
            return s;
        }
    }

    /** @brief Default-constructible when `rank_dynamic() == 0` (all-static extents). */
    constexpr extents() = default;

    /** @brief Construct from exactly `rank_dynamic()` dynamic-mode values, in order. */
    template<class... DynVals>
        requires(sizeof...(DynVals) == RankDynamic && RankDynamic > 0)
    AETHER_DEVICEHOST() constexpr explicit extents(DynVals... vals)
        : dynamic_{ static_cast<std::size_t>(vals)... }
    {
    }

private:
    // A local (not `static`) constexpr Carray, rebuilt on every call: nvcc
    // rejects an ODR-used `static constexpr` class-template member of
    // aggregate type referenced from device code ("identifier ... is
    // undefined in device code") unless it is separately `__device__`-
    // annotated, which the macro surface has no vocabulary for. Es... are
    // compile-time constants, so this is a pure register-resident local —
    // no actual storage duplication cost.
    AETHER_DEVICEHOST() static constexpr detail::Carray<std::size_t, Rank> staticExtents_() { return { Es... }; }

    /** @brief Index into `dynamic_` for static mode index `i` — the count of
     *         `dyn` slots strictly before `i` among `Es...`.
     *
     * A fold over the modes, not a loop: for a constant `i` (every
     * `layout_right` fold, every `eval<Is...>`) it is a compile-time
     * constant before any loop pass runs, so a host sample loop over a
     * `View` read never carries an inner loop GCC's vectoriser has to
     * unroll first (it declines a nest with "two or more consecutive inner
     * loops"). For a runtime `i` it is the same count, branch-free. */
    AETHER_DEVICEHOST() static constexpr std::size_t dynamic_index_(std::size_t i)
    {
        return dynamicIndexFold_(i, std::make_index_sequence<Rank>{});
    }

    template<std::size_t... Ks>
    AETHER_DEVICEHOST() static constexpr std::size_t dynamicIndexFold_(std::size_t i, std::index_sequence<Ks...>)
    {
        return (std::size_t{ 0 } + ... + static_cast<std::size_t>(Ks < i && Es == dyn));
    }

    detail::Carray<std::size_t, RankDynamic> dynamic_{};
};

} // namespace aether
