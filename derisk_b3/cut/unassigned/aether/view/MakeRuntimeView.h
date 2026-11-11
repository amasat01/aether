// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file MakeRuntimeView.h
 * @brief `aether::RuntimeView::as<>()` and `aether::fromView()` — both
 *        split out of `aether/view/RuntimeView.h`, plus the
 *        `<string>`/`err::`-using helpers only `as<>()` calls.
 *
 * `RuntimeView::as<>()` is host-only by design (it throws) — this is
 * where its definition lives now; `RuntimeView.h` keeps only the
 * declaration, guarded (`#if !defined(__CUDACC_RTC__)` — NVRTC-only,
 * never true under nvcc's own host or device pass; see that header's note
 * on why a guard rather than a split, unlike everything else here, and
 * why `__CUDACC_RTC__` rather than `__CUDA_ARCH__`).
 * `fromView()` is a plain free function (no such constraint) and moved
 * out wholesale — its own docstring already said "host-only". Include
 * this header explicitly (or `aether/aether.h`, which does) wherever
 * `.as<>()` or `fromView()` is actually called.
 */

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"
#include "aether/view/RuntimeView.h"
#include "aether/view/View.h"

namespace aether {
namespace detail {

/**
 * @brief Compile-time list (as a `Carray<std::size_t, Ext::RankDynamic>`) of
 *        the mode indices that are `dyn` in `Ext`, in ascending order —
 *        the inverse of `extents::dynamic_index_`'s per-mode lookup: this
 *        builds the whole list once so a runtime array of dynamic-mode
 *        values can be zipped back into an `Ext`'s constructor argument
 *        pack (see `buildExtentsFromRuntime` below).
 */
template<class Ext>
AETHER_DEVICEHOST() constexpr Carray<std::size_t, Ext::RankDynamic> dynModeIndices()
{
    Carray<std::size_t, Ext::RankDynamic> out{};
    std::size_t k = 0;
    for (std::size_t i = 0; i < Ext::Rank; ++i) {
        if (Ext::static_extent(i) == dyn) {
            if constexpr (Ext::RankDynamic > 0) {
                out[k] = i;
            }
            ++k;
        }
    }
    return out;
}

template<class Ext, std::size_t... Ks>
Ext buildExtentsFromRuntimeImpl(const Carray<offset_t, RuntimeRankCap>& rt, std::index_sequence<Ks...>)
{
    if constexpr (sizeof...(Ks) == 0) {
        return Ext{};
    } else {
        constexpr auto idx = dynModeIndices<Ext>();
        return Ext(static_cast<std::size_t>(rt[idx[Ks]])...);
    }
}

/**
 * @brief Build an `Ext` (`extents<Es...>`) object from a runtime array
 *        holding one value per mode (both static and dynamic slots) — only
 *        the `dyn` slots are actually read, in mode order, matching `Ext`'s
 *        own constructor's expected argument order.
 */
template<class Ext>
Ext buildExtentsFromRuntime(const Carray<offset_t, RuntimeRankCap>& rt)
{
    return buildExtentsFromRuntimeImpl<Ext>(rt, std::make_index_sequence<Ext::RankDynamic>{});
}

/** @brief Human-readable-enough (no `std::string` dependence beyond what
 *         `aether::err::fail` already needs) label for a `DType` in a
 *         `RuntimeView::as<>()` rejection message — deliberately not
 *         `aether::to_string(DType)` (`aether/dtype/Format.h`): that header
 *         is excluded from the umbrella by design (device-safety), and this
 *         header is included from the umbrella, so pulling it in here
 *         would silently re-widen what every umbrella consumer compiles. */
inline std::string dtypeLabel(const DType& dt)
{
    return "DType(code=" + std::to_string(static_cast<unsigned>(dt.code()))
        + ", bits=" + std::to_string(static_cast<unsigned>(dt.bits()))
        + ", lanes=" + std::to_string(static_cast<unsigned>(dt.lanes())) + ")";
}

} // namespace detail

// Guarded to match the declaration's guard in aether/view/RuntimeView.h
// (see that header's note on `as<>()` — a class member can't be split the
// way a free function like `make_view`/`reshaped` can without breaking every
// existing `rt.as<T>()` call site).
#if !defined(__CUDACC_RTC__)
template<class StaticViewT>
StaticViewT RuntimeView::as() const
{
    using Ext = typename StaticViewT::extents_type;
    using Layout = typename StaticViewT::layout_type;
    using T = typename StaticViewT::element_type;
    static_assert(std::is_same_v<Layout, layout_right>,
        "RuntimeView::as<>(): checked promotion only supports layout_right (row-major) static views");
    static_assert(Ext::Rank <= RuntimeRankCap, "RuntimeView::as<>(): StaticViewT's rank exceeds RuntimeRankCap");

    const DType wantDtype = dtype_of<std::remove_const_t<T>>();
    if (!(dtype == wantDtype)) {
        err::fail("RuntimeView::as", err::device_label(device), 0,
            "dtype mismatch: runtime=" + detail::dtypeLabel(dtype) + " requested=" + detail::dtypeLabel(wantDtype));
    }

    if (rank != Ext::Rank) {
        err::fail("RuntimeView::as", err::device_label(device), 0,
            "rank mismatch: runtime rank=" + std::to_string(rank) + " requested Rank=" + std::to_string(Ext::Rank));
    }

    for (std::size_t i = 0; i < Ext::Rank; ++i) {
        const std::size_t s = Ext::static_extent(i);
        if (s != dyn && static_cast<offset_t>(s) != extents[i]) {
            err::fail("RuntimeView::as", err::device_label(device), 0,
                "extent mismatch at mode " + std::to_string(i) + ": runtime=" + std::to_string(extents[i])
                    + " requested (static)=" + std::to_string(s));
        }
    }

    switch (device.type()) {
    case kDLCPU:
        break;
#ifdef AETHER_HAS_CUDA
    case kDLCUDA:
    case kDLCUDAHost:
        break;
#endif
    default:
        err::fail("RuntimeView::as", err::device_label(device), 0,
            "unsupported/unknown device kind for this build");
    }

    // Row-major (layout_right) stride pattern implied by `extents` alone,
    // computed without going through `Ext` (works even when a mode this
    // library treats as static is, at this call, being checked against a
    // runtime value that already matched above) — the expected stride at
    // mode k is the product of every extent to its right.
    detail::Carray<offset_t, RuntimeRankCap> expectedStrides{};
    offset_t acc = 1;
    for (std::size_t i = Ext::Rank; i-- > 0;) {
        expectedStrides[i] = acc;
        acc *= extents[i];
    }
    for (std::size_t i = 0; i < Ext::Rank; ++i) {
        if (strides[i] != expectedStrides[i]) {
            err::fail("RuntimeView::as", err::device_label(device), 0,
                "stride pattern mismatch at mode " + std::to_string(i) + ": runtime stride="
                    + std::to_string(strides[i]) + " expected (layout_right)=" + std::to_string(expectedStrides[i]));
        }
    }

    const Ext extObj = detail::buildExtentsFromRuntime<Ext>(extents);
    const typename Layout::template mapping<Ext> map(extObj);
    return StaticViewT(reinterpret_cast<T*>(data), map, device);
}
#endif // !defined(__CUDACC_RTC__)

/**
 * @brief Build a `RuntimeView` from a static `View` — always succeeds
 *        (host-only; no throw): every `View` this library can construct
 *        has already passed the span guard at `make_view`, so there is
 *        nothing left to reject here.
 *
 * `Volatile` views are rejected at compile time (`static_assert`): a
 * `RuntimeView`'s access through `aether/eval/Runtime.h` is never
 * volatile-qualified, so silently dropping that qualifier would be a
 * semantic footgun (volatile is a device-work-view CSE-avoidance
 * property) rather than a harmless erasure.
 */
template<class T, class Extents, class Layout, bool Volatile>
RuntimeView fromView(const View<T, Extents, Layout, Volatile>& v)
{
    static_assert(!Volatile,
        "fromView: a volatile-qualified View cannot be converted to a RuntimeView (volatile is a "
        "device-work-view property; RuntimeView access through eval::runtimeEval is never volatile-qualified)");
    static_assert(Extents::Rank <= RuntimeRankCap, "fromView: Extents::Rank exceeds RuntimeView's RuntimeRankCap");

    RuntimeView rt;
    rt.data = const_cast<void*>(static_cast<const volatile void*>(v.data()));
    rt.dtype = dtype_of<std::remove_cv_t<T>>();
    rt.device = v.device();
    rt.rank = Extents::Rank;
    for (std::size_t i = 0; i < Extents::Rank; ++i)
        rt.extents[i] = v.extent(i);

    if constexpr (std::is_same_v<Layout, layout_right>) {
        offset_t acc = 1;
        for (std::size_t i = Extents::Rank; i-- > 0;) {
            rt.strides[i] = acc;
            acc *= rt.extents[i];
        }
    } else if constexpr (std::is_same_v<Layout, layout_stride>) {
        for (std::size_t i = 0; i < Extents::Rank; ++i)
            rt.strides[i] = static_cast<offset_t>(v.mapping().stride(i));
    } else {
        static_assert(!sizeof(Layout*), "fromView: unsupported Layout policy (only layout_right/layout_stride)");
    }

    return rt;
}

} // namespace aether
