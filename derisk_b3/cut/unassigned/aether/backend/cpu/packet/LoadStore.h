// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file LoadStore.h
 * @brief Packet element access: `packetGet<Is...>(leaf_or_node,
 *        PacketIndex)` for `View` (contiguous SIMD load / `maskLoad` on
 *        tails), `Item` (broadcast), `PacketItem` (passthrough), and the
 *        `Sum`/`CWiseScale` expression nodes (recursion — kept in this ONE
 *        header); `packetStore<Is...>(View&, PacketIndex, val)` for
 *        the write side (full-packet `store` / tail-packet `maskStore`,
 *        picked by `pi.full()` — the masked tail form is this SAME
 *        dispatch, not a separate entry point).
 *
 * `Is...` is aether's rank-generic static multi-index, generalized to
 * match `element_extents`'s rank, exactly as `expr/Assign.h`'s
 * `RecursiveAssign` generalizes a flat index the same way.
 *
 * All `packetGet`/`packetStore` overloads live in `aether::detail`
 * (matching `Sum`/`CWiseScale`'s own home, `expr/nodes/Arithmetic.h`) so
 * ordinary unqualified lookup finds every overload from every other
 * overload's body without ADL games. Forward declarations precede the
 * definitions, needed for nvcc's EDG frontend, which resolves dependent
 * template calls at DEFINITION time rather than instantiation time.
 *
 * Host-only header — no `AETHER_DEVICE` code — but must COMPILE in
 * CUDA-mode builds (host side).
 */

#include <cstddef>
#include <type_traits>

#include "aether/backend/cpu/PacketItem.h"
#include "aether/backend/cpu/simd/simd.h"
#include "aether/expr/nodes/Arithmetic.h"
#include "aether/index/SampleIndex.h"
#include "aether/view/Item.h"
#include "aether/view/View.h"

namespace aether {
namespace detail {

// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations (nvcc EDG two-phase lookup).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, class Extents, class Layout, std::size_t W>
simd::Packet<std::remove_const_t<T>, W> packetGet(const View<T, Extents, Layout>& v, const PacketIndex<W>& pi);

template<std::size_t... Is, class T, std::size_t... Es, std::size_t W>
simd::Packet<T, W> packetGet(const Item<T, Es...>& item, const PacketIndex<W>& pi);

template<std::size_t... Is, class T, std::size_t W, std::size_t... Es>
const simd::Packet<T, W>& packetGet(const PacketItem<T, W, Es...>& item, const PacketIndex<W>& pi);

template<std::size_t... Is, class L, class R, bool sub, std::size_t W>
auto packetGet(const Sum<L, R, sub>& expr, const PacketIndex<W>& pi);

template<std::size_t... Is, class E, std::size_t W>
auto packetGet(const CWiseScale<E>& expr, const PacketIndex<W>& pi);

// ═══════════════════════════════════════════════════════════════════════
//  packetGet — one overload per leaf/node type.
// ═══════════════════════════════════════════════════════════════════════

/**
 * @brief `View` leaf: contiguous SIMD load along the trailing SAMPLE mode.
 *        Under `layout_right` the sample mode is the LAST (fastest-
 *        varying) index, so `&v(Is..., pi.base_)` addresses `W` CONSECUTIVE
 *        elements as `pi.base_` increments by one — the SoA contiguity
 *        `View::eval`'s scalar path already relies on. A tail packet
 *        (`!pi.full()`) loads via `maskLoad`, matching `pi.active_` lanes.
 */
template<std::size_t... Is, class T, class Extents, class Layout, std::size_t W>
simd::Packet<std::remove_const_t<T>, W> packetGet(const View<T, Extents, Layout>& v, const PacketIndex<W>& pi)
{
    using DataT = std::remove_const_t<T>;
    const DataT* ptr = &v(Is..., pi.base_);
    if (pi.full())
        return simd::Packet<DataT, W>::load(ptr);
    return simd::Packet<DataT, W>::maskLoad(ptr, simd::PacketMask<DataT, W>::firstN(pi.active_));
}

/**
 * @brief `Item` leaf: broadcast the (sample-free) scalar value into every
 *        lane — `Item` carries no sample dimension, so every packet lane
 *        sees the SAME value regardless of `pi`.
 */
template<std::size_t... Is, class T, std::size_t... Es, std::size_t W>
simd::Packet<T, W> packetGet(const Item<T, Es...>& item, const PacketIndex<W>&)
{
    return simd::Packet<T, W>::broadcast(item.template get<Is...>());
}

/**
 * @brief `PacketItem` leaf: already a register — passthrough (no load, no
 *        broadcast, `pi` is unused).
 */
template<std::size_t... Is, class T, std::size_t W, std::size_t... Es>
const simd::Packet<T, W>& packetGet(const PacketItem<T, W, Es...>& item, const PacketIndex<W>&)
{
    return item.template packet<Is...>();
}

/** @brief `Sum`/`Diff` node: recurse into both operands, combine. */
template<std::size_t... Is, class L, class R, bool sub, std::size_t W>
auto packetGet(const Sum<L, R, sub>& expr, const PacketIndex<W>& pi)
{
    auto lp = packetGet<Is...>(expr.l_, pi);
    auto rp = packetGet<Is...>(expr.r_, pi);
    if constexpr (sub)
        return lp - rp;
    else
        return lp + rp;
}

/** @brief `CWiseScale` node: recurse, broadcast the scalar factor, multiply. */
template<std::size_t... Is, class E, std::size_t W>
auto packetGet(const CWiseScale<E>& expr, const PacketIndex<W>& pi)
{
    using DataT = typename E::element_type;
    auto p = packetGet<Is...>(expr.expr_, pi);
    return simd::Packet<DataT, W>::broadcast(expr.factor_) * p;
}

// ═══════════════════════════════════════════════════════════════════════
//  packetStore / packetMaskedStore — `View` is the only writable leaf.
// ═══════════════════════════════════════════════════════════════════════

/**
 * @brief Store `val` into `dest` at component `Is...`, sample range `pi`.
 *        Full packets store via SIMD `store`; a tail packet (`!pi.full()`)
 *        stores via `maskStore` — the tail-mask handling L3 calls out.
 */
template<std::size_t... Is, class T, class Extents, class Layout, std::size_t W>
void packetStore(View<T, Extents, Layout>& dest, const PacketIndex<W>& pi, const simd::Packet<T, W>& val)
{
    T* ptr = &dest(Is..., pi.base_);
    if (pi.full())
        simd::Packet<T, W>::store(ptr, val);
    else
        simd::Packet<T, W>::maskStore(ptr, simd::PacketMask<T, W>::firstN(pi.active_), val);
}

} // namespace detail
} // namespace aether
