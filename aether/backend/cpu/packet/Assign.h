// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Assign.h
 * @brief Packet assignment: the rank-generic, compile-time-unrolled
 *        `RecursivePacketAssign` (mirrors `expr/Assign.h`'s scalar
 *        `RecursiveAssign`, generalized to `Packet<T,W>` lanes instead of
 *        `T` scalars) and the `aether::packetAssign(view, expr, pi)`
 *        consumer entry point.
 *
 * Full and tail (masked) packets are both handled by `packetGet`/
 * `packetStore`'s own `pi.full()` branch (`LoadStore.h`) — the masked tail
 * form is that same dispatch, not a separate function; see `LoadStore.h`'s
 * file docstring.
 */

#include <cstddef>
#include <type_traits>

#include "aether/backend/cpu/packet/LoadStore.h"
#include "aether/index/SampleIndex.h"

namespace aether {
namespace detail {

/**
 * @brief Compile-time recursive unroll over `Extents` (L3): enumerates
 *        EVERY static multi-index `Is...` of `Extents` and, at each leaf
 *        multi-index, does `packetStore<Is...>(to, pi, packetGet<Is...>(from, pi))`.
 *        `Extents::Rank == 0` degenerates to one call with an empty `Is...`.
 *
 * Same shape as `expr/Assign.h`'s `RecursiveAssign` (rank-generic, one
 * mode at a time via `Mode`), with `W` threaded through as an extra
 * template parameter (every `PacketIndex<W>`/`Packet<T,W>` in one recursion
 * shares the SAME width, deduced once by `packetAssign` below).
 */
template<class Extents, std::size_t Mode, std::size_t W, std::size_t... Is>
struct RecursivePacketAssign {
    template<class ExprL, class ExprR>
    static void eval(const PacketIndex<W>& pi, ExprL& to, const ExprR& from)
    {
        if constexpr (Mode == Extents::Rank) {
            auto pkt = packetGet<Is...>(from, pi);
            packetStore<Is...>(to, pi, pkt);
        } else {
            loop<0>(pi, to, from);
        }
    }

private:
    template<std::size_t K, class ExprL, class ExprR>
    static void loop(const PacketIndex<W>& pi, ExprL& to, const ExprR& from)
    {
        if constexpr (K < Extents::static_extent(Mode)) {
            RecursivePacketAssign<Extents, Mode + 1, W, Is..., K>::eval(pi, to, from);
            loop<K + 1>(pi, to, from);
        }
    }
};

} // namespace detail

/**
 * @brief `dest[pi] = expr[pi]`, per-lane, over `ExprL::element_extents`
 *        (L3) — the SIMD analogue of `detail::assign` (`expr/Assign.h`).
 */
template<class ExprL, class ExprR, std::size_t W>
void packetAssign(ExprL& dest, const ExprR& expr, const PacketIndex<W>& pi)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "packetAssign: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "packetAssign: element_extents mismatch");
    detail::RecursivePacketAssign<typename ExprL::element_extents, 0, W>::eval(pi, dest, expr);
}

} // namespace aether
