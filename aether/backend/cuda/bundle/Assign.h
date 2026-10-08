// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Assign.h
 * @brief Bundle assignment: `RecursiveBundleAssign` (mirrors
 *        `backend/cpu/packet/Assign.h`'s `RecursivePacketAssign` — rank-
 *        generic compile-time unroll over `element_extents`, `bundleGet` +
 *        `bundleStore` at each leaf multi-index), the public
 *        `bundleAssign`/`bundleAssignAdd`/`bundleAssignSub` entry points
 *        (mirrors `expr/Assign.h`'s `assign`/`assignAdd`/`assignSub`
 *        naming exactly), the masked-tail overload of `bundleAssign`, and
 *        the `BundleRef` assignment-target proxy + `View::operator[]
 *        (BundleIndex<W>)` out-of-line definition (same two-way-need
 *        split as `expr/Assign.h`'s `SampleRef`; see that file's docstring
 *        for why), and `BundleRef`'s read-only sibling
 *        `ConstBundleRef` + `View::operator[](const BundleIndex<W>&) const`
 *        — placed here rather than `expr/Assign.h` (see `ConstBundleRef`'s
 *        own docstring for the DEVIATION rationale: it needs
 *        `DeviceBundleFromExtents`, which only this file already pulls in).
 *
 * The masked-tail overload NEVER attempts a vectorized memory instruction
 * for a partial bundle (L3: "masked tail per-lane") — it reuses the
 * proven scalar `detail::assign` per ACTIVE lane, so bit-identity with
 * the scalar reference holds BY CONSTRUCTION for the tail: this literally
 * IS the scalar path for those lanes, not a re-derivation of it.
 */

#include <cstddef>
#include <type_traits>

#include "aether/backend/cuda/DeviceBundle.h"
#include "aether/backend/cuda/bundle/LoadStore.h"
#include "aether/expr/Assign.h"
#include "aether/expr/Expression.h"
#include "aether/index/BundleIndex.h"
#include "aether/macros.h"
#include "aether/view/View.h"

namespace aether {
namespace detail {

/**
 * @brief Compile-time recursive unroll over `Extents`: enumerates
 *        every static multi-index `Is...` and, at each leaf multi-index,
 *        `bundleGet`s the RHS and `bundleStore`s (Set) or read-modifies-
 *        writes (Add/Sub, via a fresh `bundleGet` of the CURRENT `to`
 *        value) the LHS. Same shape as `RecursivePacketAssign`
 *        (`backend/cpu/packet/Assign.h`), with `AssignOp` threaded through
 *        exactly like `expr/Assign.h`'s scalar `RecursiveAssign`.
 */
template<class Extents, AssignOp Op, std::size_t Mode, std::size_t W, std::size_t... Is>
struct RecursiveBundleAssign {
    template<class ExprL, class ExprR>
    AETHER_DEVICEHOST() static void eval(const BundleIndex<W>& bi, ExprL& to, const ExprR& from)
    {
        if constexpr (Mode == Extents::Rank) {
            const auto rhs = bundleGet<Is...>(from, bi);
            if constexpr (Op == AssignOp::Set) {
                bundleStore<Is...>(to, bi, rhs);
            } else {
                const auto cur = bundleGet<Is...>(to, bi);
                Carray<typename ExprL::element_type, W> combined{};
                for (std::size_t k = 0; k < W; ++k)
                    combined[k] = (Op == AssignOp::Add) ? (cur[k] + rhs[k]) : (cur[k] - rhs[k]);
                bundleStore<Is...>(to, bi, combined);
            }
        } else {
            loop<0>(bi, to, from);
        }
    }

private:
    template<std::size_t K, class ExprL, class ExprR>
    AETHER_DEVICEHOST() static void loop(const BundleIndex<W>& bi, ExprL& to, const ExprR& from)
    {
        if constexpr (K < Extents::static_extent(Mode)) {
            RecursiveBundleAssign<Extents, Op, Mode + 1, W, Is..., K>::eval(bi, to, from);
            loop<K + 1>(bi, to, from);
        }
    }
};

} // namespace detail

/** @brief `dest[bi] = expr[bi]` over a FULL bundle (all `W` lanes valid) —
 *         the SIMD-at-the-memory-boundary analogue of `detail::assign`. */
template<class ExprL, class ExprR, std::size_t W>
AETHER_DEVICEHOST() void bundleAssign(ExprL& dest, const ExprR& expr, const BundleIndex<W>& bi)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "bundleAssign: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "bundleAssign: element_extents mismatch");
    detail::RecursiveBundleAssign<typename ExprL::element_extents, detail::AssignOp::Set, 0, W>::eval(bi, dest, expr);
}

/** @brief `dest[bi] += expr[bi]` over a full bundle. */
template<class ExprL, class ExprR, std::size_t W>
AETHER_DEVICEHOST() void bundleAssignAdd(ExprL& dest, const ExprR& expr, const BundleIndex<W>& bi)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "bundleAssignAdd: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "bundleAssignAdd: element_extents mismatch");
    detail::RecursiveBundleAssign<typename ExprL::element_extents, detail::AssignOp::Add, 0, W>::eval(bi, dest, expr);
}

/** @brief `dest[bi] -= expr[bi]` over a full bundle. */
template<class ExprL, class ExprR, std::size_t W>
AETHER_DEVICEHOST() void bundleAssignSub(ExprL& dest, const ExprR& expr, const BundleIndex<W>& bi)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "bundleAssignSub: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "bundleAssignSub: element_extents mismatch");
    detail::RecursiveBundleAssign<typename ExprL::element_extents, detail::AssignOp::Sub, 0, W>::eval(bi, dest, expr);
}

/**
 * @brief Masked-tail overload (L3): a bundle straddling the end of an
 *        `n`-sample array. NEVER attempts a vectorized memory op for the
 *        partial bundle (see file docstring) — reuses the scalar
 *        `detail::assign` per ACTIVE lane only.
 */
template<class ExprL, class ExprR, std::size_t W>
AETHER_DEVICEHOST() void bundleAssign(
    ExprL& dest, const ExprR& expr, const BundleIndex<W>& bi, const typename BundleIndex<W>::Mask& mask)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "bundleAssign: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "bundleAssign: element_extents mismatch");
    for (std::size_t k = 0; k < W; ++k) {
        if (mask.lane(k))
            detail::assign(bi.lane(k), dest, expr);
    }
}

/**
 * @brief Assignment-target proxy: `view[bi] = expr`, `view[bi] +=
 *        expr`, `view[bi] -= expr`, `view[bi].get()` (register
 *        materialization into a `DeviceBundle`). Returned by `View::
 *        operator[](BundleIndex<W>)` — mirrors `SampleRef` exactly:
 *        "bundle code reads EXACTLY like Item code".
 */
template<class V, std::size_t W>
class BundleRef {
public:
    using element_type = typename V::element_type;
    using element_extents = typename V::element_extents;

    AETHER_DEVICEHOST() constexpr BundleRef(V& view, const BundleIndex<W>& bi)
        : view_(view)
        , bi_(bi)
    {
    }

    /** @brief `view[bi] = expr` — overwrite. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DEVICEHOST() BundleRef& operator=(const E& expr)
    {
        bundleAssign(view_, expr, bi_);
        return *this;
    }

    /** @brief `view[bi] += expr`. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DEVICEHOST() BundleRef& operator+=(const E& expr)
    {
        bundleAssignAdd(view_, expr, bi_);
        return *this;
    }

    /** @brief `view[bi] -= expr`. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DEVICEHOST() BundleRef& operator-=(const E& expr)
    {
        bundleAssignSub(view_, expr, bi_);
        return *this;
    }

    /** @brief Register materialization: a `DeviceBundle` snapshot of this bundle. */
    AETHER_DEVICEHOST() detail::DeviceBundleFromExtents<element_type, W, element_extents> get() const
    {
        detail::DeviceBundleFromExtents<element_type, W, element_extents> result;
        bundleAssign(result, view_, bi_);
        return result;
    }

private:
    V& view_;
    BundleIndex<W> bi_;
};

/**
 * @brief READ-ONLY counterpart of `BundleRef`: `cv[bi].get()` only — no
 *        `= += -=`. Returned by `View::operator[](const BundleIndex<W>&)
 *        const`, always instantiated with a CONST-qualified `V`. DEFINED
 *        HERE rather than in `aether/expr/Assign.h` — see `view/View.h`'s
 *        `ConstBundleRef` forward declaration for the full rationale:
 *        `.get()`'s return type, `DeviceBundleFromExtents<...>`, needs
 *        `backend/cuda/DeviceBundle.h` complete, a dependency that would
 *        otherwise have to enter `expr/Assign.h` — and `aether/device.h`
 *        already includes `expr/Assign.h`, so that would silently widen
 *        `device.h`'s NVRTC-audited transitive closure past what has
 *        already been verified there. This file already carries both
 *        `expr/Assign.h` and `DeviceBundle.h`, so placing `ConstBundleRef`
 *        beside `BundleRef` (its mutable sibling, which needs the
 *        identical two dependencies) adds nothing new to any umbrella's
 *        reachable set.
 */
template<class V, std::size_t W>
class ConstBundleRef {
public:
    using element_type = typename V::element_type;
    using element_extents = typename V::element_extents;

    AETHER_DEVICEHOST() constexpr ConstBundleRef(V& view, const BundleIndex<W>& bi)
        : view_(view)
        , bi_(bi)
    {
    }

    /** @brief Register materialization: a `DeviceBundle` snapshot of this
     *         bundle (mirrors `BundleRef::get()` exactly — `bundleAssign`'s
     *         source operand is read-only regardless of `view_`'s own
     *         const-qualification). */
    AETHER_DEVICEHOST() detail::DeviceBundleFromExtents<element_type, W, element_extents> get() const
    {
        detail::DeviceBundleFromExtents<element_type, W, element_extents> result;
        bundleAssign(result, view_, bi_);
        return result;
    }

private:
    V& view_;
    BundleIndex<W> bi_;
};

// ---------------------------------------------------------------------------
// Out-of-line definitions declared in view/View.h (see that file's forward
// declarations + this file's docstring for why).
// ---------------------------------------------------------------------------

template<class T, class Extents, class Layout, bool Volatile, bool ReadOnly>
template<std::size_t W>
AETHER_DEVICEHOST() constexpr BundleRef<View<T, Extents, Layout, Volatile, ReadOnly>, W> View<T, Extents, Layout,
    Volatile, ReadOnly>::operator[](const BundleIndex<W>& bi)
{
    return BundleRef<View<T, Extents, Layout, Volatile, ReadOnly>, W>(*this, bi);
}

/** @brief The CONST bundle overload's out-of-line definition. */
template<class T, class Extents, class Layout, bool Volatile, bool ReadOnly>
template<std::size_t W>
AETHER_DEVICEHOST() constexpr ConstBundleRef<const View<T, Extents, Layout, Volatile, ReadOnly>, W> View<T, Extents,
    Layout, Volatile, ReadOnly>::operator[](const BundleIndex<W>& bi) const
{
    return ConstBundleRef<const View<T, Extents, Layout, Volatile, ReadOnly>, W>(*this, bi);
}

} // namespace aether
