// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file LoadStore.h
 * @brief Bundle element access: `bundleGet<Is...>(leaf_or_node,
 *        BundleIndex<W>) -> Carray<T,W>` (one scalar per lane) for:
 *          - `View` leaf — vectorized `double2`/`float4` reinterpret load
 *            when the sample mode is contiguous (`layout_right`, always
 *            stride-1 on its trailing mode — see `backend/cpu/packet/
 *            LoadStore.h`'s identical observation) and the base address is
 *            16-byte aligned and `(T,W)` is a 128-bit-eligible combo
 *            (double x2 or float x4); else a scalar per-lane fallback —
 *            same values either way, only the SASS differs.
 *          - `Item` leaf — broadcast (mirrors `backend/cpu/packet/
 *            LoadStore.h`'s `Item` overload exactly).
 *          - `DeviceBundle` leaf — passthrough (already register-resident).
 *          - `Sum`/`CWiseScale` nodes — recurse (mirrors the CPU packet
 *            layer's exact node coverage — everything else is reached
 *            through the generic fallback below).
 *          - every other `aether_expression` (the geometric/quaternion/
 *            structural/product nodes: `Cross`/`QuatMul`/`QuatConj`/
 *            `Segment`/...) — a generic fallback that loops `bi.lane(k)`
 *            through the ordinary scalar `eval<Is...>(SampleIndex)`
 *            protocol. This is sound and cheap specifically because those
 *            nodes' leaves, in every bundle-composed expression, are
 *            already-materialized `DeviceBundle`s (register reads via
 *            `DeviceBundle::eval()`'s `work()`-as-lane trick — see
 *            `DeviceBundle.h`'s docstring) rather than `View`s straight off
 *            memory; the generic loop therefore costs nothing beyond what
 *            the node algebra already does; it is never the path that
 *            touches device memory (that is always the View overload
 *            above), so it needs no vectorization of its own.
 *
 *        `bundleStore<Is...>(dest, BundleIndex<W>, Carray<T,W>)` for the
 *        write side: `View` (vectorized STG under the same conditions as
 *        the load, scalar fallback otherwise) and `DeviceBundle` (used when
 *        materializing a `BundleRef::get()` result, `backend/cuda/bundle/
 *        Assign.h`).
 *
 * Compiled in both modes: only the vectorized `View` fast path is
 * `AETHER_HAS_CUDA`-gated; everything else — broadcast, passthrough,
 * Sum/CWiseScale recursion, the generic fallback, and the View scalar
 * fallback — is plain, portable C++ and is `AETHER_CPP_MODE`'s "lowers to a
 * scalar loop" behaviour.
 *
 * Forward declarations precede the definitions (nvcc EDG two-phase lookup —
 * the same convention `backend/cpu/packet/LoadStore.h` documents and uses).
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/backend/cuda/DeviceBundle.h"
#include "aether/expr/nodes/Arithmetic.h"
#include "aether/index/BundleIndex.h"
#include "aether/layout/Layout.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"
#include "aether/view/Item.h"
#include "aether/view/View.h"

#ifdef AETHER_HAS_CUDA
#include <vector_functions.h>
#include <vector_types.h>
#endif

namespace aether {
namespace detail {

// ═══════════════════════════════════════════════════════════════════════
//  Overload-resolution guard: which types get a SPECIFIC bundleGet
//  overload below (everything else falls to the generic template at the
//  bottom of this file). Ordinary C++ partial ordering between function
//  templates already prefers a more-constrained overload over a bare
//  `template<class E>` one, but nvcc's EDG frontend has documented quirks
//  elsewhere in this codebase — this trait makes the exclusion EXPLICIT
//  (a `requires` clause on the generic fallback) rather than relying on
//  that ordering alone.
// ═══════════════════════════════════════════════════════════════════════

template<class E>
struct IsBundleSpecialLeaf_ : std::false_type { };
template<class T, class Extents, class Layout, bool Volatile, bool ReadOnly>
struct IsBundleSpecialLeaf_<View<T, Extents, Layout, Volatile, ReadOnly>> : std::true_type { };
template<class T, std::size_t... Es>
struct IsBundleSpecialLeaf_<Item<T, Es...>> : std::true_type { };
template<class T, std::size_t W, std::size_t... Es>
struct IsBundleSpecialLeaf_<DeviceBundle<T, W, Es...>> : std::true_type { };
template<class L, class R, bool sub>
struct IsBundleSpecialLeaf_<Sum<L, R, sub>> : std::true_type { };
template<class E>
struct IsBundleSpecialLeaf_<CWiseScale<E>> : std::true_type { };

// ═══════════════════════════════════════════════════════════════════════
//  Forward declarations.
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, class Extents, class Layout, bool Volatile, bool ReadOnly, std::size_t W>
AETHER_DEVICEHOST() Carray<std::remove_const_t<T>, W> bundleGet(
    const View<T, Extents, Layout, Volatile, ReadOnly>& v, const BundleIndex<W>& bi);

template<std::size_t... Is, class T, std::size_t... Es, std::size_t W>
AETHER_DEVICEHOST() Carray<T, W> bundleGet(const Item<T, Es...>& item, const BundleIndex<W>& bi);

template<std::size_t... Is, class T, std::size_t W, std::size_t... Es>
AETHER_DEVICEHOST() Carray<T, W> bundleGet(const DeviceBundle<T, W, Es...>& b, const BundleIndex<W>& bi);

template<std::size_t... Is, class L, class R, bool sub, std::size_t W>
AETHER_DEVICEHOST() Carray<typename L::element_type, W> bundleGet(const Sum<L, R, sub>& expr, const BundleIndex<W>& bi);

template<std::size_t... Is, class E, std::size_t W>
AETHER_DEVICEHOST() Carray<typename E::element_type, W> bundleGet(const CWiseScale<E>& expr, const BundleIndex<W>& bi);

/** @brief Generic fallback — every OTHER `aether_expression` (see file docstring). */
/// \cond AETHER_DOXYGEN_REQUIRES_OWN_LINE_WORKAROUND
template<std::size_t... Is, class E, std::size_t W>
    requires(!IsBundleSpecialLeaf_<E>::value)
AETHER_DEVICEHOST() Carray<typename E::element_type, W> bundleGet(const E& expr, const BundleIndex<W>& bi);
/// \endcond

// ═══════════════════════════════════════════════════════════════════════
//  Alignment/eligibility helpers.
// ═══════════════════════════════════════════════════════════════════════

/** @brief `true` when `p` is 16-byte aligned — the runtime half of the
 *         vectorization guard (the compile-time half is `Layout ==
 *         layout_right` — see the `bundleGet`/`bundleStore` View overloads). */
AETHER_DEVICEHOST() inline bool bundleAligned16_(const void* p)
{
    return (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0u;
}

/** @brief `true` exactly for the two 128-bit-LDG/STG-eligible `(T,W)`
 *         combos this vectorization targets ("double2/float4 reinterpret").
 *         Every other combination (incl. `W` not in `{2,4}`) always takes the
 *         scalar per-lane path — still correct, just not vectorized. */
template<class T, std::size_t W>
inline constexpr bool bundleVectorEligible_
    = (std::is_same_v<T, double> && W == 2) || (std::is_same_v<T, float> && W == 4);

// ═══════════════════════════════════════════════════════════════════════
//  bundleGet — View leaf (the vectorized payoff).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, class Extents, class Layout, bool Volatile, bool ReadOnly, std::size_t W>
AETHER_DEVICEHOST() Carray<std::remove_const_t<T>, W> bundleGet(
    const View<T, Extents, Layout, Volatile, ReadOnly>& v, const BundleIndex<W>& bi)
{
    using DataT = std::remove_const_t<T>;
    const std::size_t base = static_cast<std::size_t>(bi.base());
    Carray<DataT, W> out{};

#ifdef AETHER_HAS_CUDA
    // Volatile/ReadOnly are excluded from the vectorized fast path: both
    // change `operator()`'s return type away from a plain `T&` (Volatile ->
    // `volatile T&`, ReadOnly -> `T` by value, see View.h), so `&v(...)`
    // below either yields the wrong pointer qualification or is outright
    // ill-formed (address-of a prvalue) when ReadOnly. Neither combination
    // is a vectorization target here (ReadOnly's `__ldg` is its own, separate,
    // orthogonal mechanism) — both simply fall back to the scalar loop,
    // still correct.
    if constexpr (std::is_same_v<Layout, layout_right> && !Volatile && !ReadOnly && bundleVectorEligible_<DataT, W>) {
        const DataT* ptr = &v(Is..., base);
        if (bundleAligned16_(ptr)) {
            // Device: the typed 128-bit load (LDG.128). Host: the same 16
            // bytes through AETHER_BITCOPY, which carries no 16-byte
            // alignment claim — GCC may merge this load with the scalar
            // fallback's two loads below and hoist it above the
            // `bundleAligned16_` test, and a typed `double2` access would
            // then hoist as an ALIGNED move that faults on the misaligned
            // samples the test exists to route around.
            if constexpr (std::is_same_v<DataT, double>) {
#if defined(__CUDA_ARCH__)
                const double2 packed = *reinterpret_cast<const double2*>(ptr);
#else
                double2 packed;
                AETHER_BITCOPY(&packed, ptr, sizeof(packed));
#endif
                out[0] = packed.x;
                out[1] = packed.y;
            } else {
#if defined(__CUDA_ARCH__)
                const float4 packed = *reinterpret_cast<const float4*>(ptr);
#else
                float4 packed;
                AETHER_BITCOPY(&packed, ptr, sizeof(packed));
#endif
                out[0] = packed.x;
                out[1] = packed.y;
                out[2] = packed.z;
                out[3] = packed.w;
            }
            return out;
        }
    }
#endif
    for (std::size_t k = 0; k < W; ++k)
        out[k] = v(Is..., base + k);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  bundleGet — Item leaf (broadcast).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, std::size_t... Es, std::size_t W>
AETHER_DEVICEHOST() Carray<T, W> bundleGet(const Item<T, Es...>& item, const BundleIndex<W>&)
{
    Carray<T, W> out{};
    const T v = item.template get<Is...>();
    for (std::size_t k = 0; k < W; ++k)
        out[k] = v;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  bundleGet — DeviceBundle leaf (passthrough).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, std::size_t W, std::size_t... Es>
AETHER_DEVICEHOST() Carray<T, W> bundleGet(const DeviceBundle<T, W, Es...>& b, const BundleIndex<W>&)
{
    Carray<T, W> out{};
    for (std::size_t k = 0; k < W; ++k)
        out[k] = b.template at<Is...>(k);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  bundleGet — Sum/CWiseScale nodes (recurse).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class L, class R, bool sub, std::size_t W>
AETHER_DEVICEHOST() Carray<typename L::element_type, W> bundleGet(const Sum<L, R, sub>& expr, const BundleIndex<W>& bi)
{
    const auto lp = bundleGet<Is...>(expr.l_, bi);
    const auto rp = bundleGet<Is...>(expr.r_, bi);
    Carray<typename L::element_type, W> out{};
    for (std::size_t k = 0; k < W; ++k)
        out[k] = sub ? (lp[k] - rp[k]) : (lp[k] + rp[k]);
    return out;
}

template<std::size_t... Is, class E, std::size_t W>
AETHER_DEVICEHOST() Carray<typename E::element_type, W> bundleGet(const CWiseScale<E>& expr, const BundleIndex<W>& bi)
{
    const auto p = bundleGet<Is...>(expr.expr_, bi);
    Carray<typename E::element_type, W> out{};
    for (std::size_t k = 0; k < W; ++k)
        out[k] = expr.factor_ * p[k];
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  bundleGet — generic fallback (every other aether_expression node; see
//  file docstring).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class E, std::size_t W>
    requires(!IsBundleSpecialLeaf_<E>::value)
AETHER_DEVICEHOST() Carray<typename E::element_type, W> bundleGet(const E& expr, const BundleIndex<W>& bi)
{
    Carray<typename E::element_type, W> out{};
    for (std::size_t k = 0; k < W; ++k)
        out[k] = expr.template eval<Is...>(bi.lane(k));
    return out;
}

// ═══════════════════════════════════════════════════════════════════════
//  bundleStore — View (the vectorized STG payoff) and DeviceBundle (materialization
//  destination, `BundleRef::get()`).
// ═══════════════════════════════════════════════════════════════════════

template<std::size_t... Is, class T, class Extents, class Layout, bool Volatile, bool ReadOnly, std::size_t W>
AETHER_DEVICEHOST() void bundleStore(
    View<T, Extents, Layout, Volatile, ReadOnly>& dest, const BundleIndex<W>& bi, const Carray<T, W>& lanes)
{
    const std::size_t base = static_cast<std::size_t>(bi.base());

#ifdef AETHER_HAS_CUDA
    if constexpr (std::is_same_v<Layout, layout_right> && bundleVectorEligible_<T, W>) {
        T* ptr = &dest(Is..., base);
        if (bundleAligned16_(ptr)) {
            // Host stores go through AETHER_BITCOPY for the reason given in
            // bundleGet above; the device keeps the typed 128-bit store.
            if constexpr (std::is_same_v<T, double>) {
#if defined(__CUDA_ARCH__)
                *reinterpret_cast<double2*>(ptr) = make_double2(lanes[0], lanes[1]);
#else
                const double2 packed = make_double2(lanes[0], lanes[1]);
                AETHER_BITCOPY(ptr, &packed, sizeof(packed));
#endif
            } else {
#if defined(__CUDA_ARCH__)
                *reinterpret_cast<float4*>(ptr) = make_float4(lanes[0], lanes[1], lanes[2], lanes[3]);
#else
                const float4 packed = make_float4(lanes[0], lanes[1], lanes[2], lanes[3]);
                AETHER_BITCOPY(ptr, &packed, sizeof(packed));
#endif
            }
            return;
        }
    }
#endif
    for (std::size_t k = 0; k < W; ++k)
        dest(Is..., base + k) = lanes[k];
}

template<std::size_t... Is, class T, std::size_t W, std::size_t... Es>
AETHER_DEVICEHOST() void bundleStore(DeviceBundle<T, W, Es...>& dest, const BundleIndex<W>&, const Carray<T, W>& lanes)
{
    for (std::size_t k = 0; k < W; ++k)
        dest.template at<Is...>(k) = lanes[k];
}

} // namespace detail
} // namespace aether
