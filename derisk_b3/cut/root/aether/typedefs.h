// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file typedefs.h
 * @brief `aether::idx_t`/`aether::dims_t` — the canonical index/dimension
 *        alias family, one switch wide.
 *
 * This header is aether's single point of change for the index/dimension
 * width. Consumers use `idx_t`/`dims_t`, never a raw `std::uint32_t`/`int`.
 * A width switch here changes `View`/`Chunk` layout — rebuild the world,
 * never mixable across binaries in one process (DLPack strides stay
 * `int64` regardless, so export is unaffected). Host-side sizes (byte
 * counts, capacities) stay `std::size_t` — this is the element/dimension
 * index family, `aether::offset_t`'s own (`aether/index/Offset.h`).
 * Included by `aether/aether.h`, and by `device.h`.
 */

#ifndef AETHER_INDEX_T
#define AETHER_INDEX_T std::uint32_t
#endif

#include <cstdint>

namespace aether {

/** @brief The index/dimension base type — the ONE switch. */
using index_base_t = AETHER_INDEX_T;
/** @brief Array-index/size alias, aether's own width. */
using idx_t = index_base_t;
/** @brief Dimension-count alias, aether's own width. */
using dims_t = index_base_t;

} // namespace aether
