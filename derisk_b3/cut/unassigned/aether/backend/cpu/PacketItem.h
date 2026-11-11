// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketItem.h
 * @brief `aether::PacketItem`: a register-resident SIMD vector/
 *        matrix — one `Packet<T,W>` per ELEMENT of `element_extents`.
 *
 * `PacketItem<T,W,Es...>` mirrors `Item<T,Es...>`'s own shape
 * parameterization exactly (rank-generic `element_extents` — one operator
 * set, no vector/matrix family split), storing one `Packet<T,W>` per
 * element instead of one `T`.
 *
 * The key use case: single-pass fused evaluation — `packetCapture`
 * (backend/cpu/Tiled.h) materializes an intermediate expression into a
 * `PacketItem` that stays in SIMD registers, avoiding the memory
 * round-trip of a two-pass capture-then-store.
 *
 * Host-only header — no `AETHER_DEVICE` code (SIMD intrinsics never run on
 * device) — but must COMPILE in CUDA-mode builds (host side).
 *
 * Storage is a PLAIN C array (`PacketT data_[Size]`), not
 * `aether::detail::Carray` — deliberately. `aether::detail::Carray`'s
 * whole reason to exist is that `std::array` is unsafe in DEVICE-VISIBLE
 * headers, so its accessors carry `AETHER_DEVICEHOST()` (`__device__`
 * during nvcc's device compilation pass) UNCONDITIONALLY, for every `T`.
 * `Packet<T,W>` holds a raw SIMD register (`__m256d` etc.) — a type nvcc
 * explicitly refuses to touch in device code — and nvcc instantiates a
 * class template's dual-qualified members for every type it is used with
 * ACROSS THE WHOLE TRANSLATION UNIT, not just from code actually reachable
 * from a `__global__` function; `Carray<Packet<double,4>, 3>::%operator[]`/
 * `data()` then fail to compile as device code even though nothing on the
 * device ever calls them (confirmed against this exact build). `PacketItem`
 * is HOST-ONLY BY DESIGN (this file's own docstring above), so the
 * device-safety `Carray` exists for does not even apply here; a plain
 * array has no member functions to accidentally instantiate for the
 * device pass.
 */

#include <cstddef>

#include "aether/backend/cpu/simd/simd.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/layout/detail/Carray.h"

namespace aether {

/**
 * @brief Register-resident SIMD vector/matrix: one `Packet<T,W>` per
 *        element of `extents<Es...>` (see the file docstring for why
 *        storage is a plain array, not `aether::detail::Carray`).
 *
 * Rebases onto the `Expression` CRTP exactly like `Item` (`isLeaf = true`,
 * `element_extents = extents<Es...>`) so it slots into the SAME `Sum`/
 * `CWiseScale` node machinery `expr/nodes/Arithmetic.h` already provides —
 * `backend/cpu/packet/LoadStore.h`'s `packetGet` overload for `PacketItem`
 * is a plain passthrough (the packet is already a register, nothing to
 * load).
 */
template<class T, std::size_t W, std::size_t... Es>
class PacketItem : public Expression<PacketItem<T, W, Es...>, T> {
public:
    using element_type = T;
    /** @brief L2 element protocol: a `PacketItem`'s own shape IS its element shape. */
    using element_extents = extents<Es...>;
    /** @brief L2 element protocol: `PacketItem` is always a leaf. */
    static constexpr bool isLeaf = true;

    /** @brief Number of modes. */
    static constexpr std::size_t Rank = sizeof...(Es);
    /** @brief Total element count — the product of `Es...`. */
    static constexpr std::size_t Size = (Es * ... * std::size_t{ 1 });
    /** @brief SIMD lane count of every packet this `PacketItem` stores. */
    static constexpr std::size_t width = W;

    using PacketT = simd::Packet<T, W>;

    PacketItem() = default;

    /** @brief Read component `Is...` (compile-time multi-index). */
    template<std::size_t... Is>
    PacketT& packet()
    {
        static_assert(sizeof...(Is) == Rank, "PacketItem::packet<Is...>(): wrong number of indices");
        return data_[offset_(Is...)];
    }
    /** @overload packet() const */
    template<std::size_t... Is>
    const PacketT& packet() const
    {
        static_assert(sizeof...(Is) == Rank, "PacketItem::packet<Is...>(): wrong number of indices");
        return data_[offset_(Is...)];
    }

    /**
     * @brief Scalar element-protocol hook — DELIBERATELY a hard compile
     *        error when instantiated: a `PacketItem` is a
     *        register of SIMD packets, not scalars, so the scalar
     *        `RecursiveAssign` path (`expr/Assign.h`) must never reach it.
     *        Use `packetGet<Is...>(item, pi)` (`backend/cpu/packet/
     *        LoadStore.h`) or `item.packet<Is...>()` instead.
     */
    template<std::size_t... Is>
    T& eval(const SampleIndex&)
    {
        static_assert(dependentFalse_<Is...>,
            "PacketItem::eval<Is...>() reached from a scalar expression context;"
            " use packetGet<Is...>(item, pi) or item.packet<Is...>() instead.");
        static T unreachable_{}; // the static_assert above always fires first
        return unreachable_;
    }

    /** @brief Raw pointer to the underlying packet storage (one `Packet<T,W>` per element). */
    PacketT* data() { return data_; }
    const PacketT* data() const { return data_; }
    static constexpr std::size_t size() { return Size; }

private:
    template<std::size_t...>
    static constexpr bool dependentFalse_ = false;

    // Carray<std::size_t,Rank> here is fine for device compilation (no
    // vector/intrinsic type involved) — only `data_` below (storing
    // Packet<T,W>, which DOES hold a raw SIMD register) needs the plain
    // C array; see the file docstring.
    template<class... Idxs>
    static constexpr std::size_t offset_(Idxs... idxs)
    {
        const detail::Carray<std::size_t, Rank> idxArr{ static_cast<std::size_t>(idxs)... };
        const detail::Carray<std::size_t, Rank> extArr{ Es... };
        std::size_t offset = 0;
        for (std::size_t i = 0; i < Rank; ++i)
            offset = offset * extArr[i] + idxArr[i];
        return offset;
    }

    // Size == (Es * ... * 1) can never be 0 for the identity-fold formula
    // above (matches `Item<T,Es...>::Size`'s same convention) unless a
    // caller explicitly names a 0-sized mode, which no aether shape does.
    PacketT data_[Size]{};
};

} // namespace aether
