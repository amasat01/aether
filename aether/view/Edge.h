// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Edge.h
 * @brief The edge rules of an indexed read: what an index outside `[0, extent)` means.
 *
 * `clamp` holds the index at the nearest valid one, `wrap` takes it modulo the extent, `inside` tells whether the index is valid
 * (a `zero` read yields nothing outside) and `unchecked` passes it through on the caller's promise that it is valid. Each returns
 * the index as `std::size_t`, ready for a view's call form.
 */

#include <cstddef>
#include <cstdint>

#include "aether/macros.h"

namespace aether::edge {

/** @brief The index held within `[0, extent)`; `extent` must be at least 1. */
template<class Index>
AETHER_DEVICEHOST() constexpr std::size_t clamp(Index index, std::size_t extent)
{
    const auto last = static_cast<std::int64_t>(extent) - 1;
    const auto value = static_cast<std::int64_t>(index);
    return static_cast<std::size_t>(value < 0 ? 0 : (value > last ? last : value));
}

/** @brief The index taken modulo the extent, a negative index counting from the end; `extent` must be at least 1. */
template<class Index>
AETHER_DEVICEHOST() constexpr std::size_t wrap(Index index, std::size_t extent)
{
    const auto n = static_cast<std::int64_t>(extent);
    // Two remainders and no branch: a device compiler was measured to fold the form "remainder, then add the extent
    // when negative" wrongly for a constant extent of 3, giving an index of -1.
    return static_cast<std::size_t>(((static_cast<std::int64_t>(index) % n) + n) % n);
}

/** @brief Whether the index lies within `[0, extent)`. */
template<class Index>
AETHER_DEVICEHOST() constexpr bool inside(Index index, std::size_t extent)
{
    const auto value = static_cast<std::int64_t>(index);
    return value >= 0 && static_cast<std::uint64_t>(value) < extent;
}

/** @brief The index as it is: the caller states it is valid, and the extent is not read. */
template<class Index>
AETHER_DEVICEHOST() constexpr std::size_t unchecked(Index index, std::size_t)
{
    return static_cast<std::size_t>(index);
}

} // namespace aether::edge
