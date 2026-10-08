// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file AccumPlane.h
 * @brief `aether::accum::Plane`/`PlaneView`: a per-sample accumulation
 *        plane that lets many independent producers accumulate into one
 *        per-sample vector quantity through a single `+=` syntax, backed
 *        by either an atomic-safe or a serialized `double` policy.
 */

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

#include "aether/accum/atomic.h"
#include "aether/array/Array.h"
#include "aether/err/Error.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"
#include "aether/view/Item.h"
#include "aether/view/View.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {
namespace accum {

// =====================================================================
//  Policy tags
// =====================================================================

/// @brief `double` plane: hardware `atomicAdd` + exact 2Sum residual in a
/// companion lane, folded at delivery. Concurrency-safe, order-independent.
struct CompensatedAtomic { };

/// @brief `double` plane: direct `+=`, no atomics, no companion lane. The
/// caller owes serialization of the producers writing one sample.
struct Serialized { };

// =====================================================================
//  Policy traits
// =====================================================================

/** @brief Compile-time description of a policy. Primary template
 *         deliberately left UNDEFINED — an unknown policy tag is a hard
 *         compile error at the plane, never a silent default. */
template<class Policy>
struct PolicyTraits;

template<>
struct PolicyTraits<CompensatedAtomic> {
    static constexpr bool concurrentSafe   = true;
    static constexpr bool hasCompanionLane = true;
};

template<>
struct PolicyTraits<Serialized> {
    static constexpr bool concurrentSafe   = false;
    static constexpr bool hasCompanionLane = false;
};

// =====================================================================
//  Scalar -> default policy
// =====================================================================

/** @brief The policy a plane uses when the caller does not name one.
 *         Primary template deliberately UNDEFINED. */
template<class Real>
struct DefaultPolicy;

template<>
struct DefaultPolicy<double> {
    using T = CompensatedAtomic;
};

template<class Real>
using DefaultPolicyT = typename DefaultPolicy<Real>::T;

// =====================================================================
//  Per-(scalar, policy) storage and operations
// =====================================================================

namespace detail {

/**
 * @brief The storage element and the three operations (`zero`,
 *        `accumulate`, `deliver`) of one (scalar, policy) pair. Primary
 *        template deliberately UNDEFINED.
 */
template<class Real, class Policy>
struct PolicyOps;

/// @brief `double` + compensated atomic. Two lanes; delivery folds them.
template<>
struct PolicyOps<double, CompensatedAtomic> {
    using StoreT = double;

    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() static StoreT zeroStore() { return 0.0; }

    template<class ViewT>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() static void accumulate(
        ViewT& value, ViewT& comp, const SampleIndex& i, double term)
    {
        atomic::compensatedSum(value, comp, i, term);
    }

    template<class ViewT>
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() static double deliver(
        const ViewT& value, const ViewT& comp, const SampleIndex& i)
    {
        return value(i.global()) + comp(i.global());
    }
};

/// @brief `double` + direct serialized `+=`. One lane; delivery is a read.
template<>
struct PolicyOps<double, Serialized> {
    using StoreT = double;

    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() static StoreT zeroStore() { return 0.0; }

    template<class ViewT>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() static void accumulate(
        ViewT& value, ViewT& /*comp*/, const SampleIndex& i, double term)
    {
        value(i.global()) = value(i.global()) + term;
    }

    template<class ViewT>
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() static double deliver(
        const ViewT& value, const ViewT& /*comp*/, const SampleIndex& i)
    {
        return value(i.global());
    }
};

/**
 * @brief Slice lane `d` (`d < VD`) out of the owning `VD`-lane view `full`
 *        (`View<T, extents<VD,dyn>, layout_stride>`, `Array`'s own view
 *        shape) as a scalar `View<T, extents<dyn>, layout_stride>`: one
 *        pointer offset by lane `d`'s own stride, with the sample-mode
 *        stride carried straight through.
 */
template<class T, std::size_t VD>
[[nodiscard]] AETHER_DEVICEHOST() inline View<T, extents<dyn>, layout_stride> laneView_(
    const View<T, extents<VD, dyn>, layout_stride>& full, std::size_t d)
{
    const std::size_t cStride = full.mapping().stride(0);
    const std::size_t sStride = full.mapping().stride(1);
    extents<dyn> ext(static_cast<std::size_t>(full.samples()));
    aether::detail::Carray<std::size_t, 1> strides{ { sStride } };
    typename layout_stride::mapping<extents<dyn>> map(ext, strides);
    return View<T, extents<dyn>, layout_stride>(full.data() + d * cStride, map, full.device());
}

} // namespace detail

// =====================================================================
//  LaneView -- what `plane[i]` is
// =====================================================================

template<class Real, std::size_t VD, class Policy>
class PlaneView;

/**
 * @brief The per-sample accumulator handle returned by
 *        `PlaneView::operator[]`. Holds a reference to the plane view and
 *        the sample index only — no buffered state, so it may be created
 *        and discarded freely (`acc[i] += workVec3` reads the way the
 *        kernels already read).
 */
template<class Real, std::size_t VD, class Policy>
class LaneView {
public:
    using PlaneViewT = PlaneView<Real, VD, Policy>;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() LaneView(PlaneViewT& plane, const SampleIndex& i)
        : plane_{ plane }
        , i_{ i }
    {
    }

    /// @brief Accumulate a `VD`-component term into this sample's lanes.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() LaneView& operator+=(const Expression<E, typename E::element_type>& term)
    {
        plane_.accumulate(i_, term);
        return *this;
    }

    /// @brief Named spelling of `operator+=`.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() LaneView& accumulate(const Expression<E, typename E::element_type>& term)
    {
        plane_.accumulate(i_, term);
        return *this;
    }

    /// @brief This sample's delivered value (see `PlaneView::delivered`).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Item<Real, VD> delivered() const
    {
        return plane_.delivered(i_);
    }

    /// @brief Zero this sample's lanes, unconditionally.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void zero() { plane_.zero(i_); }

private:
    PlaneViewT& plane_;
    const SampleIndex i_;
};

// =====================================================================
//  PlaneView -- the non-owning view handed to a kernel
// =====================================================================

/**
 * @brief Non-owning device-bindable view of an accumulation plane (public
 *        alias `aether::AccumPlaneView`), passed by value — the only way a
 *        kernel touches the plane.
 *
 * @tparam Real    the scalar the plane delivers (only `double` is
 *                 supported).
 * @tparam VD      lanes per sample (3 for a vec3 acceleration).
 * @tparam Policy  `CompensatedAtomic` or `Serialized`.
 */
template<class Real, std::size_t VD, class Policy = DefaultPolicyT<Real>>
class PlaneView {
public:
    using OpsT      = detail::PolicyOps<Real, Policy>;
    using StoreT     = typename OpsT::StoreT;
    using TraitsT    = PolicyTraits<Policy>;
    using ArrayT     = Array<StoreT, VD>;
    using StoreViewT = typename ArrayT::ViewT;
    using LaneT      = LaneView<Real, VD, Policy>;
    using ItemT      = Item<Real, VD>;
    using PolicyT    = Policy;

    /** @brief Lanes per sample. */
    static constexpr std::size_t VecDims = VD;
    /** @brief @see PolicyTraits::concurrentSafe. */
    static constexpr bool concurrentSafe = TraitsT::concurrentSafe;
    /** @brief Whether a companion residual lane exists. */
    static constexpr bool hasCompanionLane = TraitsT::hasCompanionLane;

    PlaneView() = default;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() PlaneView(const StoreViewT& value, const StoreViewT& comp)
        : value_{ value }
        , comp_{ comp }
    {
    }

    /** @brief Number of samples the plane covers. */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() offset_t size() const { return value_.samples(); }

    /** @brief `plane[i] += term` -- the accumulation surface. */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() LaneT operator[](const SampleIndex& i) { return LaneT{ *this, i }; }

    /**
     * @brief Accumulate a `VD`-component term into sample `i`'s lanes. The
     *        lanes are visited in ascending component order, so a host run
     *        and a device run accumulate in the same order and produce the
     *        same bits.
     */
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void accumulate(const SampleIndex& i, const Expression<E, typename E::element_type>& term)
    {
        static_assert(E::element_extents::Rank == 1, "AccumPlane: accumulation term must be rank-1 (a vector)");
        static_assert(E::element_extents::static_extent(0) == VD,
            "AccumPlane: accumulation term must have exactly as many components as the plane has lanes");
        const E& e = static_cast<const E&>(term);
        accumulateAll_(i, e, std::make_index_sequence<VD>{});
    }

    /**
     * @brief Zero sample `i`'s lanes -- UNCONDITIONALLY, including the
     *        companion lane. No guard, no owner, no "first producer" flag.
     */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void zero(const SampleIndex& i)
    {
        zeroAll_(value_, i, std::make_index_sequence<VD>{});
        if constexpr (hasCompanionLane)
            zeroAll_(comp_, i, std::make_index_sequence<VD>{});
    }

    /**
     * @brief Sample `i`'s accumulated value, in `Real`. One fold for the
     *        whole step; returns a register-resident `Item`.
     */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() ItemT delivered(const SampleIndex& i) const
    {
        return deliveredAll_(i, std::make_index_sequence<VD>{});
    }

    /// @brief Terminal, written straight into a destination expression: `plane.deliver(out, i)` is `out[i] = plane.delivered(i)`.
    template<class OutE>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void deliver(OutE& out, const SampleIndex& i) const
    {
        out[i] = delivered(i);
    }

    /** @brief The value-lane VIEW — diagnostics/tests, not part of the accumulation surface. */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() const StoreViewT& valueView() const { return value_; }
    /** @brief The companion-lane view; a 0-sample view unless `hasCompanionLane`. */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() const StoreViewT& compView() const { return comp_; }

private:
    template<class E, std::size_t... d>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void accumulateAll_(const SampleIndex& i, const E& e, std::index_sequence<d...>)
    {
        (accumulateComponent_<d>(i, e), ...);
    }
    template<std::size_t d, class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void accumulateComponent_(const SampleIndex& i, const E& e)
    {
        auto vh = detail::laneView_<StoreT, VD>(value_, d);
        auto ch = detail::laneView_<StoreT, VD>(comp_, d);
        const double term = static_cast<double>(e.template eval<d>(i));
        OpsT::accumulate(vh, ch, i, term);
    }

    template<std::size_t... d>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void zeroAll_(const StoreViewT& lane, const SampleIndex& i, std::index_sequence<d...>)
    {
        (zeroComponent_<d>(lane, i), ...);
    }
    template<std::size_t d>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void zeroComponent_(const StoreViewT& lane, const SampleIndex& i)
    {
        auto h            = detail::laneView_<StoreT, VD>(lane, d);
        h(i.global())      = OpsT::zeroStore();
    }

    template<std::size_t... d>
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() ItemT deliveredAll_(const SampleIndex& i, std::index_sequence<d...>) const
    {
        ItemT out;
        ((out.template get<d>() = deliveredComponent_<d>(i)), ...);
        return out;
    }
    template<std::size_t d>
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real deliveredComponent_(const SampleIndex& i) const
    {
        const auto vh = detail::laneView_<StoreT, VD>(value_, d);
        const auto ch = detail::laneView_<StoreT, VD>(comp_, d);
        return static_cast<Real>(OpsT::deliver(vh, ch, i));
    }

    StoreViewT value_{};
    StoreViewT comp_{};
};

// =====================================================================
//  Plane -- the owning allocation
// =====================================================================

/**
 * @brief Owning per-sample accumulation plane: `VD` lanes of `double`, plus
 *        the companion lane when the policy has one. Host-side type; hand
 *        `deviceView()` (CUDA) or `hostView()` to the code that
 *        accumulates.
 */
template<class Real, std::size_t VD = 3, class Policy = DefaultPolicyT<Real>>
class Plane {
public:
    using OpsT    = detail::PolicyOps<Real, Policy>;
    using StoreT  = typename OpsT::StoreT;
    using TraitsT = PolicyTraits<Policy>;
    using ArrayT  = Array<StoreT, VD>;
    using ViewT   = PlaneView<Real, VD, Policy>;
    using PolicyT = Policy;

    static constexpr std::size_t VecDims   = VD;
    static constexpr bool concurrentSafe   = TraitsT::concurrentSafe;
    static constexpr bool hasCompanionLane = TraitsT::hasCompanionLane;

    /// @brief Allocate a plane for `n` samples. Storage is UNINITIALIZED
    /// (matches `Array`'s own "raw allocate" convention) — call
    /// `zeroHost()`/`zeroDeviceAsync()` before first use.
    explicit Plane(std::size_t n)
        : value_(n)
        , comp_(hasCompanionLane ? n : std::size_t{ 0 })
    {
    }

    /// @brief Number of samples.
    [[nodiscard]] std::size_t samples() const { return value_.samples(); }

    /// @brief Host-side view.
    [[nodiscard]] ViewT hostView()
    {
        if constexpr (hasCompanionLane)
            return ViewT{ value_.hostView(), comp_.hostView() };
        else
            return ViewT{ value_.hostView(), typename ArrayT::ViewT{} };
    }

    /// @brief Device-side view -- what a kernel takes, by value. `AETHER_CPP_MODE`:
    /// the same single chunk as `hostView()` (mirrors `Array::deviceView()`).
    [[nodiscard]] ViewT deviceView()
    {
        if constexpr (hasCompanionLane)
            return ViewT{ value_.deviceView(), comp_.deviceView() };
        else
            return ViewT{ value_.deviceView(), typename ArrayT::ViewT{} };
    }

    /// @brief Upload the host lanes to the device lanes. No-op in `AETHER_CPP_MODE`.
    void upload()
    {
        value_.upload();
        if constexpr (hasCompanionLane)
            comp_.upload();
    }
    /// @brief Download the device lanes into the host lanes. No-op in `AETHER_CPP_MODE`.
    void download()
    {
        value_.download();
        if constexpr (hasCompanionLane)
            comp_.download();
    }

#ifdef AETHER_HAS_CUDA
    /**
     * @brief Zero the plane on the device: one `cudaMemsetAsync` over the
     *        whole allocation per lane (both `samples()` and any reserved
     *        padding). A byte-wise zero is the policy's zero element for
     *        `double` (see `zeroIsAllZeroBytes()`).
     */
    void zeroDeviceAsync(cudaStream_t stream = 0)
    {
        err::checkCuda(cudaMemsetAsync(value_.deviceView().data(), 0, laneBytes_(value_), stream), "Plane::zeroDeviceAsync",
            err::device_label(value_.deviceView().device()), laneBytes_(value_));
        if constexpr (hasCompanionLane) {
            err::checkCuda(cudaMemsetAsync(comp_.deviceView().data(), 0, laneBytes_(comp_), stream), "Plane::zeroDeviceAsync",
                err::device_label(comp_.deviceView().device()), laneBytes_(comp_));
        }
    }
#endif

    /// @brief The host mirror of `zeroDeviceAsync()`: one unconditional pass over the whole host plane.
    void zeroHost()
    {
        StoreT* base = value_.hostView().data();
        std::fill_n(base, laneCount_(value_), OpsT::zeroStore());
        if constexpr (hasCompanionLane) {
            StoreT* cbase = comp_.hostView().data();
            std::fill_n(cbase, laneCount_(comp_), OpsT::zeroStore());
        }
    }

    /** @brief Is an all-zero byte pattern the policy's zero element? (Always
     *         `true` here — `double`'s `+0.0` is all-zero-bytes — checked at
     *         run time since the answer is about object representation.) */
    [[nodiscard]] static bool zeroIsAllZeroBytes()
    {
        const StoreT z = OpsT::zeroStore();
        unsigned char bytes[sizeof(StoreT)];
        std::memcpy(bytes, &z, sizeof(StoreT));
        for (unsigned char b : bytes) {
            if (b != 0u)
                return false;
        }
        return true;
    }

    /// @brief The underlying value-lane `Array` (allocation, upload/download).
    [[nodiscard]] ArrayT& valueArray() { return value_; }
    [[nodiscard]] const ArrayT& valueArray() const { return value_; }
    /// @brief The underlying companion-lane `Array`; 0 samples unless `hasCompanionLane`.
    [[nodiscard]] ArrayT& compArray() { return comp_; }
    [[nodiscard]] const ArrayT& compArray() const { return comp_; }

private:
    [[nodiscard]] static std::size_t laneCount_(const ArrayT& a) { return a.capacity() * VD; }
    [[nodiscard]] static std::size_t laneBytes_(const ArrayT& a) { return laneCount_(a) * sizeof(StoreT); }

    ArrayT value_;
    ArrayT comp_;
};

} // namespace accum

/// @brief Owning per-sample accumulation plane. @see accum::Plane.
template<class Real, std::size_t VD = 3, class Policy = accum::DefaultPolicyT<Real>>
using AccumPlane = accum::Plane<Real, VD, Policy>;

/// @brief Non-owning device-bindable accumulation-plane view. @see accum::PlaneView.
template<class Real, std::size_t VD = 3, class Policy = accum::DefaultPolicyT<Real>>
using AccumPlaneView = accum::PlaneView<Real, VD, Policy>;

} // namespace aether
