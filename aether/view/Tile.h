// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Tile.h
 * @brief `aether::Tile`: a thin facade naming `aether::Item` + the
 *        existing `Expression::segment<Off,Len>()` slice mechanism.
 *
 * `Tile<DataT, SlotDim, NumSlots>` is just `Item<T, D*K>` with a thin slot
 * accessor that delegates to the existing `segment<offset, len>`
 * machinery: `Item<T,Es...>` (`aether/view/Item.h`) is the register-
 * resident all-static storage, and `Expression::segment<Off,Len>()`
 * (`aether/expr/Expression.h`, inherited by every expression leaf) is the
 * slot mechanism. `Tile` is therefore a plain alias, and `slot<s>()` a
 * free function forwarding to `segment<s*D, D>()` — no new members, no
 * new runtime.
 *
 * `PacketTile<DataT, SlotDim, NumSlots, W>` is the same shape one layer
 * down, over `aether::backend::cpu::PacketItem<T, W, Es...>`
 * (`aether/backend/cpu/PacketItem.h`) in place of `Item` — host-only, like
 * `PacketItem` itself: no `AETHER_DEVICEHOST()` anywhere on this facade's
 * `PacketTile`/packet-`slot()` surface, matching `PacketItem`'s own
 * un-annotated convention (SIMD intrinsics never run on device).
 */

#include "aether/view/Item.h"

namespace aether {

/**
 * @brief Compile-time-fixed `K`-slot stack tile of `D`-dim vectors, over
 *        `aether::Item`.
 *
 * @tparam T  Scalar component type.
 * @tparam D  Components per slot ("SlotDim").
 * @tparam K  Number of slots ("NumSlots").
 */
template<class T, std::size_t D, std::size_t K>
using Tile = Item<T, D * K>;

/**
 * @brief Slot `s`'s `D`-component sub-view of `tile`, over
 *        `Expression::segment<s*D, D>()`.
 *
 * `s` and `D` are both explicit template arguments (`slot<s, D>(tile)`),
 * not `s` alone: `Tile` is a plain alias (`Item<T, D*K>`, above) — the
 * parameter type the compiler actually sees is `Item<T, N>` with
 * `N = D*K`, and template argument deduction cannot invert a product to
 * recover `D` and `K` separately from a single deduced `N` (a
 * non-deduced context; a hard C++ limit, not a design choice). `K` is not
 * a separate parameter here at all: the range check below only ever
 * needs `(s+1)*D <= N`, which `K` never added information beyond.
 *
 * A single overload (`const Item<T,N>&`) covers both a mutable and a
 * `const` `tile` argument: `aether::Expression::segment<Off,Len>()` is
 * itself a `const`-qualified member (`aether/expr/Expression.h`) — there
 * is no separate mutable overload to forward a mutable `tile` to.
 *
 * Read-only: `aether::Expression::segment<Off,Len>()` returns a lazy
 * `detail::Segment` read node (`aether/expr/nodes/Structural.h`'s own
 * docstring — deliberate scope, "no write-back path"), not an assignable
 * component view. A slot therefore participates as an expression operand
 * (`slot<0,3>(tile) + ...`), but `slot<s,D>(tile) = expr` has no
 * equivalent here — see `tests/test_Tile.{cpp,cu}`'s header comment for
 * the disposition this leaves assignment-shaped rows in.
 *
 * @tparam s  Compile-time slot index.
 * @tparam D  Per-slot component count ("SlotDim") — must be given
 *            explicitly; see the note above.
 */
template<std::size_t s, std::size_t D, class T, std::size_t N>
AETHER_DEVICEHOST() constexpr auto slot(const Item<T, N>& tile)
{
    static_assert((s + 1) * D <= N, "aether::slot<s,D>(): slot index out of range");
    return tile.template segment<s * D, D>();
}

} // namespace aether

#include "aether/backend/cpu/PacketItem.h"

namespace aether {

/**
 * @brief Compile-time-fixed `K`-slot stack tile of `W`-lane `D`-dim SIMD
 *        packets, over `aether::backend::cpu::PacketItem`. Host-only (see
 *        the file docstring).
 *
 * @tparam T  Scalar component type.
 * @tparam D  Components per slot ("SlotDim").
 * @tparam K  Number of slots ("NumSlots").
 * @tparam W  SIMD lane count.
 */
template<class T, std::size_t D, std::size_t K, std::size_t W>
using PacketTile = PacketItem<T, W, D * K>;

/**
 * @brief Slot `s`'s `D`-component sub-view of `tile` — the `PacketTile`
 *        counterpart of `slot<s,D>(const Item<T,N>&)` above (same note:
 *        `s` and `D` are both explicit, `K` derived structurally).
 *        Host-only, no `AETHER_DEVICEHOST()` (matches `PacketItem`'s own
 *        convention).
 *
 * @tparam s  Compile-time slot index.
 * @tparam D  Per-slot component count ("SlotDim") — must be given explicitly.
 */
template<std::size_t s, std::size_t D, class T, std::size_t W, std::size_t N>
constexpr auto slot(const PacketItem<T, W, N>& tile)
{
    static_assert((s + 1) * D <= N, "aether::slot<s,D>(): slot index out of range");
    return tile.template segment<s * D, D>();
}

} // namespace aether
