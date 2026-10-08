// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file View.h
 * @brief `aether::View`: a typed, non-owning, zero-copy tensor over a
 *        `Chunk` (or foreign memory) — the aether tensor type.
 *
 * The `View` is already the lightweight descriptor — no separate "handle"
 * type, no work-vs-global-reference split (that distinction is just which
 * `Chunk`/pointer the `View` was built over).
 *
 * The host-only `make_view()` factories (over a `Chunk` or foreign memory
 * — both throw `aether::Error` on a bad span) live in `aether/view/
 * MakeView.h`, taking `<string>`/`aether/err/Error.h`/`aether/chunk/
 * Chunk.h` with them. `View` itself never calls `err::` or touches
 * `Chunk` (only `make_view`'s `Chunk` overload did) — this header is
 * standalone under NVRTC; include `aether/view/MakeView.h` explicitly
 * when you need `make_view`.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include "aether/device/Device.h"
#include "aether/expr/Expression.h"
#include "aether/index/BundleIndex.h"
#include "aether/index/Offset.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Assignment-target proxy returned by `View::operator[]`.
 *        Forward-declared here, defined in `aether/expr/Assign.h` (which
 *        needs `View`'s full definition to wrap it — a genuine two-way
 *        need; see `Item.h`'s equivalent note for the ctor/assign split).
 */
template<class V>
class SampleRef;

/**
 * @brief Read-only assignment-target-shaped proxy returned by `View::
 *        operator[](const SampleIndex&) const` (the const read path).
 *        Forward-declared here, defined in `aether/expr/Assign.h` beside
 *        `SampleRef` — same two-way need (`.get()` materializes an `Item`,
 *        which needs `view/Item.h` complete). Carries no assignment
 *        operators (that is the whole point: a `const View` is read-only) —
 *        see `aether/expr/Assign.h`'s own docstring for the full contract.
 */
template<class V>
class ConstSampleRef;

/**
 * @brief Bundle assignment-target proxy returned by `View::operator[]
 *        (BundleIndex<W>)`. Forward-declared here, defined in
 *        `aether/backend/cuda/bundle/Assign.h` — same two-way need as
 *        `SampleRef` above (that header needs `View`'s full definition to
 *        wrap it; `View` only needs the name to declare the member).
 */
template<class V, std::size_t W>
class BundleRef;

/**
 * @brief Read-only counterpart of `BundleRef`, returned by `View::
 *        operator[](const BundleIndex<W>&) const`. Forward-declared here,
 *        defined in `aether/backend/cuda/bundle/Assign.h` beside
 *        `BundleRef`, rather than in `expr/Assign.h`: `.get()` materializes
 *        a `DeviceBundleFromExtents<...>` (`aether/backend/cuda/
 *        DeviceBundle.h`), a dependency `expr/Assign.h` does not otherwise
 *        carry and which `aether/device.h` also does not currently reach
 *        through `expr/Assign.h` — pulling it in there would silently
 *        widen `device.h`'s NVRTC-audited transitive closure. Defining it
 *        beside `BundleRef` (which already sees both `expr/Assign.h` and
 *        `DeviceBundle.h`) avoids that without touching `device.h` itself.
 *        Carries no assignment operators.
 */
template<class V, std::size_t W>
class ConstBundleRef;

namespace detail {

/**
 * @brief Leaf-protocol helper: `extents<>` over the first `Rank - 1`
 *        modes of `Ext` (the static prefix, dropping the trailing mode
 *        whatever it is). Purely mechanical — asserts nothing about the
 *        dropped mode — so it stays well-formed for every `Ext` this
 *        library instantiates a `View` over, including non-batched shapes
 *        and `reshaped<...>` results whose dynamic mode is not last
 *        (`tests/test_Reshape.*`'s `reshaped<dyn, Minor>` case). The "only
 *        the trailing mode may be dyn for expression use" invariant is
 *        instead enforced where it can be without eagerly instantiating
 *        for every `View`: inside `View::eval<Is...>` below, a member
 *        function template only instantiated when actually called by
 *        expression machinery (unlike a nested type alias, which the
 *        language instantiates unconditionally with the enclosing class).
 */
template<class Ext, class Seq>
struct ViewLeafExtentsImpl;
template<class Ext, std::size_t... Is>
struct ViewLeafExtentsImpl<Ext, std::index_sequence<Is...>> {
    using type = extents<Ext::static_extent(Is)...>;
};
template<class Ext>
using ViewLeafExtents
    = typename ViewLeafExtentsImpl<Ext, std::make_index_sequence<(Ext::Rank >= 1) ? Ext::Rank - 1 : 0>>::type;

} // namespace detail

/**
 * @brief Typed, non-owning view of `T` over `Extents`, addressed through
 *        `Layout` (default `layout_right`, the SoA anchor).
 *
 * Gains the expression leaf protocol unconditionally: `isLeaf`,
 * `element_extents` (the static prefix — see `detail::ViewLeafExtents`
 * above) and `eval<Is...>(SampleIndex)` (`= (*this)(Is..., i.global())`),
 * plus `operator[](SampleIndex)` returning a `SampleRef` assignment
 * proxy. `View` rebases onto `Expression<View<...>, T>` (CRTP).
 *
 * `Volatile` is a 4th, defaulted (`false`) NTTP — every existing
 * `View<T,Extents,Layout>` spelling is unaffected. When `true`,
 * `pointer`/`reference` become `volatile T*`/`volatile T&` so every load and
 * store through this view's `operator()`/`eval()` is a genuine memory access
 * that the compiler may not CSE, hoist, or route through a stack-backed
 * mirror ("volatile" is a view property, not a separate handle tier).
 * Reached only via `as_volatile()` below; `make_view()`/`Array` never
 * produce one directly.
 *
 * `ReadOnly` is a 5th, defaulted (`false`) NTTP. When `true`, `reference`
 * becomes `T` (by value, not `T&` — see below) and every device-compiled
 * `operator()`/`eval()` load routes through `__ldg`
 * (`#if defined(__CUDA_ARCH__)`; the host path is unchanged, `__ldg` has no
 * host meaning). Returning by value rather than by reference is deliberate,
 * not incidental: `__ldg` reads through the read-only data cache into a
 * register — there is no addressable lvalue to hand back — and it also
 * means a `ReadOnly` view is naturally non-assignable at the element level
 * (`SampleRef::operator=`'s `to.eval<Is...>(i) = ...` fails to compile, an
 * lvalue-required error, since `eval()` no longer returns a reference): the
 * type system enforces "read-only" for free, with no extra guard needed.
 * `ReadOnly` and `Volatile` are contradictory (a genuine memory access that
 * also claims cache-only read-through makes no sense) and combining them is
 * a `static_assert` error. Reached only via `as_readonly()` below.
 */
template<class T, class Extents, class Layout = layout_right, bool Volatile = false, bool ReadOnly = false>
class View : public Expression<View<T, Extents, Layout, Volatile, ReadOnly>, T> {
    static_assert(!(Volatile && ReadOnly), "View: Volatile and ReadOnly are contradictory");

public:
    using element_type = T;
    using extents_type = Extents;
    using layout_type = Layout;
    using mapping_type = typename Layout::template mapping<Extents>;
    /** @brief The (possibly volatile-qualified) underlying pointer type. */
    using pointer = std::conditional_t<Volatile, volatile T*, T*>;
    /** @brief The element access type: `T` by value when `ReadOnly` (see
     *         class docstring), else the (possibly volatile-qualified)
     *         element reference type. */
    using reference = std::conditional_t<ReadOnly, T, std::conditional_t<Volatile, volatile T&, T&>>;

    /** @brief The static prefix of `Extents` (drops the trailing — batch/
     *         sample — mode). See `detail::ViewLeafExtents`. */
    using element_extents = detail::ViewLeafExtents<Extents>;
    /** @brief `View` is always a leaf. */
    static constexpr bool isLeaf = true;
    /** @brief Whether this view's element access is volatile-qualified. */
    static constexpr bool isVolatile = Volatile;
    /** @brief Whether this view's device loads route through `__ldg`. */
    static constexpr bool isReadOnly = ReadOnly;

    // No AETHER_DEVICEHOST() here: a `= default` special member on its first
    // declaration is automatically host+device eligible under nvcc, which
    // warns (#20012-D) that an explicit annotation is redundant (see
    // aether/device/Device.h's own note).
    constexpr View() = default;

    AETHER_DEVICEHOST() constexpr View(T* data, const mapping_type& map, Device dev)
        : data_(data)
        , map_(map)
        , dev_(dev)
    {
    }

    /** @brief The underlying (typed, possibly volatile-qualified) pointer. */
    AETHER_DEVICEHOST() constexpr pointer data() const { return data_; }
    /** @brief The device this view's memory lives on. */
    AETHER_DEVICEHOST() constexpr Device device() const { return dev_; }
    /** @brief The runtime extent of mode `i` (`offset_t` — part of the
     *         narrow address chain; see `aether/index/Offset.h`). */
    AETHER_DEVICEHOST() constexpr offset_t extent(std::size_t i) const { return map_.extents().extent(i); }
    /** @brief Number of modes. A compile-time mode count, not an offset — stays `std::size_t`. */
    AETHER_DEVICEHOST() static constexpr std::size_t rank() { return Extents::Rank; }
    /**
     * @brief The batch-mode extent, `extent(rank() - 1)`. Provided
     *        unconditionally (any rank >= 1); meaningful for the batched
     *        views this library actually builds — every `*View` alias
     *        appends a trailing dynamic sample mode. Calling it on a
     *        rank-0 view is a caller error (`rank() - 1` underflows
     *        `std::size_t`).
     */
    AETHER_DEVICEHOST() constexpr offset_t samples() const { return extent(rank() - 1); }
    /**
     * @brief Product of every extent — the logical element count.
     *
     * Returns `offset_t`, so the ubiquitous guard `if (i.global() >=
     * v.samples()) return;` is a single 32-bit `ISETP` rather than a
     * two-instruction 64-bit compare holding an extra register pair live.
     * Never wraps: no view can exist whose span exceeds `offset_max` (the
     * `make_view` guard below), and `size() <= required_span_size()` for
     * both layouts.
     */
    AETHER_DEVICEHOST() constexpr offset_t size() const
    {
        offset_t total = 1;
        for (std::size_t i = 0; i < Extents::Rank; ++i)
            total *= extent(i);
        return total;
    }
    /** @brief The underlying layout mapping. */
    AETHER_DEVICEHOST() constexpr const mapping_type& mapping() const { return map_; }

    /**
     * @brief Element access — `data()[mapping()(idxs...)]` (on the host the
     *        same offset at full width, `detail::elementOffset`). When
     *        `ReadOnly`, the device-compiled load routes through `__ldg`
     *        (the read-only data-cache hint, SASS `LDG.E.CI`); the host
     *        path is unchanged (`__ldg` has no host meaning).
     */
    template<class... Idxs>
    AETHER_DEVICEHOST() constexpr reference operator()(Idxs... idxs) const
    {
        if constexpr (ReadOnly) {
#if defined(__CUDA_ARCH__)
            return __ldg(&data_[map_(idxs...)]);
#else
            return data_[detail::elementOffset(map_, idxs...)];
#endif
        } else {
            return data_[detail::elementOffset(map_, idxs...)];
        }
    }

    /** @brief A read-only view of the same data (Volatile- and ReadOnly-preserving). */
    AETHER_DEVICEHOST() constexpr View<const T, Extents, Layout, Volatile, ReadOnly> as_const() const
    {
        return View<const T, Extents, Layout, Volatile, ReadOnly>(data_, map_, dev_);
    }

    /**
     * @brief A volatile-qualified view of the same data — every
     *        `operator()`/`eval()` access through the result becomes a real
     *        load/store the compiler may not CSE, hoist, or stack-mirror.
     *        Intended for shared-memory scratch that multiple statements
     *        deliberately re-read. Calling this on an already-volatile
     *        view is a static-assert error — `as_volatile()` is not
     *        idempotent-by-design, it is the single entry point from the
     *        (default) non-volatile side.
     */
    AETHER_DEVICEHOST() constexpr View<T, Extents, Layout, true, ReadOnly> as_volatile() const
    {
        static_assert(!Volatile, "View::as_volatile(): this view is already volatile-qualified");
        static_assert(!ReadOnly, "View::as_volatile(): ReadOnly and Volatile are contradictory");
        return View<T, Extents, Layout, true, ReadOnly>(data_, map_, dev_);
    }

    /**
     * @brief A read-only-qualified view of the same data — every
     *        device-compiled `operator()`/`eval()` load through the
     *        result routes through `__ldg` (see the class docstring above
     *        for why `reference` becomes `T` by value here, and why that
     *        alone makes the result non-assignable). Calling this on an
     *        already-read-only or already-volatile view is a
     *        `static_assert` error — mirrors `as_volatile()`'s own
     *        single-entry-point convention.
     */
    AETHER_DEVICEHOST() constexpr View<T, Extents, Layout, Volatile, true> as_readonly() const
    {
        static_assert(!ReadOnly, "View::as_readonly(): this view is already read-only-qualified");
        static_assert(!Volatile, "View::as_readonly(): ReadOnly and Volatile are contradictory");
        return View<T, Extents, Layout, Volatile, true>(data_, map_, dev_);
    }

    /**
     * @brief A writable rank-1 view of static component `I` of a rank-2
     *        (component, sample) view — e.g. component 0 ("x") of an
     *        `aether::Array<double,3>`'s `hostView()`.
     *
     *        A `Vec3dArr` (`Array<double,3>`) stores its x/y/z components as
     *        three separate sample-length runs (the SoA layout — see this
     *        class's own docstring), so "the x component" is itself an
     *        ordinary rank-1 array, once you know where it starts and how
     *        far apart consecutive samples are. This reads both numbers off
     *        this view's own mapping (start = `I * mapping().stride(0)`,
     *        spacing = `mapping().stride(1)`) rather than assuming
     *        `samples()`: an owning `Array` pitches components at its
     *        (possibly larger, capacity-rounded) `capacity()`, not
     *        `samples()`, while a view built by hand over a scratch slot
     *        pitches however its builder chose — so the two can never
     *        silently disagree. Writes through the result land directly in
     *        this view's own backing memory, no copy involved: write via
     *        `arr.hostView().component<0>`, `arr.upload()`, and the new
     *        values are visible through `arr.deviceView()`.
     *
     * @tparam I  Which component (0-based). Checked against this view's
     *            static component extent at compile time (a `static_assert`,
     *            not a runtime check) — the component count is always known
     *            at compile time for a `View`, so an out-of-range `I` is
     *            always a programming error, never a data-dependent one.
     */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr View<T, extents<dyn>, layout_stride> component()
    {
        static_assert(Extents::Rank == 2,
            "View::component<I>(): source must be a rank-2 (component, sample) view");
        static_assert(std::is_same_v<Layout, layout_stride>,
            "View::component<I>(): source must be a layout_stride view (the shape "
            "Array::hostView()/deviceView() return) -- layout_right has no per-mode "
            "stride to read the component pitch off");
        static_assert(I < Extents::static_extent(0),
            "View::component<I>(): I is out of range for this view's component extent");
        using ExtT = extents<dyn>;
        using MapT = typename layout_stride::template mapping<ExtT>;
        const std::size_t componentPitch = map_.stride(0);
        const std::size_t samplePitch    = map_.stride(1);
        return View<T, ExtT, layout_stride>(
            data_ + I * componentPitch, MapT(ExtT(samples()), { samplePitch }), dev_);
    }

    /** @brief Read-only counterpart of `component()` (see above): identical
     *         pointer arithmetic, returns a `View<const T, ...>` so the
     *         result cannot be written through. */
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr View<const T, extents<dyn>, layout_stride> component() const
    {
        static_assert(Extents::Rank == 2,
            "View::component<I>(): source must be a rank-2 (component, sample) view");
        static_assert(std::is_same_v<Layout, layout_stride>,
            "View::component<I>(): source must be a layout_stride view (the shape "
            "Array::hostView()/deviceView() return) -- layout_right has no per-mode "
            "stride to read the component pitch off");
        static_assert(I < Extents::static_extent(0),
            "View::component<I>(): I is out of range for this view's component extent");
        using ExtT = extents<dyn>;
        using MapT = typename layout_stride::template mapping<ExtT>;
        const std::size_t componentPitch = map_.stride(0);
        const std::size_t samplePitch    = map_.stride(1);
        return View<const T, ExtT, layout_stride>(
            data_ + I * componentPitch, MapT(ExtT(samples()), { samplePitch }), dev_);
    }

    /**
     * @brief `Is...` matches `element_extents` (the static prefix);
     *        `i.global()` supplies the trailing (batch/sample) index —
     *        `eval<Is...>(i) = (*this)(Is..., i.global())`.
     *
     * A member function template, so — unlike `element_extents` above —
     * this is only instantiated when actually called (by expression
     * machinery: `Sum`/`CWiseScale`/`detail::RecursiveAssign`), which is
     * exactly where this invariant belongs: a `View` whose Extents does
     * not carry `dyn` as the trailing mode compiles fine (e.g.
     * `reshaped<dyn, Minor>(...)`, `tests/test_Reshape.*`) right up until
     * someone actually tries to use it as an expression leaf.
     */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr reference eval(const SampleIndex& i) const
    {
        static_assert(sizeof...(Is) == element_extents::Rank,
            "View::eval<Is...>(): Is... must match element_extents's rank");
        static_assert(Extents::Rank >= 1 && Extents::static_extent(Extents::Rank - 1) == dyn,
            "View::eval(): only the TRAILING mode may be dyn for expression use (L2)");
        return (*this)(Is..., i.global());
    }

    /**
     * @brief Assignment-target entry point: `v[i] = expr` / `v[i] += expr`
     *        / `v[i] -= expr` / `v[i].get()` (register materialization).
     *        Declared here, defined out-of-line in `aether/expr/Assign.h`
     *        (see `SampleRef`'s forward declaration above for why).
     */
    AETHER_DEVICEHOST() constexpr SampleRef<View> operator[](const SampleIndex& i);

    /**
     * @brief The const read path: `cv[i].get()` / `cv[i].eval<Is...>` —
     *        no `= += -=` (this overload returns `ConstSampleRef<const
     *        View>`, which declares none); no implicit conversion is
     *        added. `const View` is itself a conforming expression leaf —
     *        see the `static_assert` below and `aether::aether_expression`'s
     *        own `remove_cv_t` note (`aether/expr/Expression.h`). Declared
     *        here, defined out-of-line in `aether/expr/Assign.h` (see
     *        `ConstSampleRef`'s forward declaration above for why).
     */
    AETHER_DEVICEHOST() constexpr ConstSampleRef<const View> operator[](const SampleIndex& i) const;

    /**
     * @brief Assignment-target entry point: `v[bi] = expr` / `v[bi]
     *        += expr` / `v[bi] -= expr` / `v[bi].get()` (register
     *        materialization into a `DeviceBundle<T,W,...>`) for a
     *        `BundleIndex<W>` (W consecutive samples, one thread's worth).
     *        Declared here, defined out-of-line in
     *        `aether/backend/cuda/bundle/Assign.h` (see `BundleRef`'s
     *        forward declaration above for why).
     */
    template<std::size_t W>
    AETHER_DEVICEHOST() constexpr BundleRef<View, W> operator[](const BundleIndex<W>& bi);

    /**
     * @brief Const counterpart of the bundle entry point above:
     *        `cv[bi].get()` only, no assignment operators. Declared here,
     *        defined out-of-line in `aether/backend/cuda/bundle/Assign.h`
     *        (see `ConstBundleRef`'s forward declaration above for why).
     */
    template<std::size_t W>
    AETHER_DEVICEHOST() constexpr ConstBundleRef<const View, W> operator[](const BundleIndex<W>& bi) const;

private:
    pointer data_ = nullptr;
    mapping_type map_{};
    Device dev_{};
};

/**
 * @brief A `const View` must also conform to `aether_expression` (the
 *        const read path's whole point — `const View` is a leaf exactly
 *        like the mutable one). Not checkable inside the class body
 *        above: `derived_from`'s `is_base_of_v` needs a complete type on
 *        both sides, and `View<...>` is definitionally incomplete
 *        anywhere within its own definition (a self-referential check —
 *        `E::element_extents` reads as invalid from inside the
 *        still-being-defined class). Pinned instead as a namespace-scope
 *        check on one representative instantiation (the batched-vector
 *        shape every `*View` alias in `aether/aether.h` is built from)
 *        right after the template — this is what catches a future
 *        regression in either `View`'s CRTP base or the concept's
 *        `remove_cv_t` handling (`aether/expr/Expression.h`), rather than
 *        only where a `ConstSampleRef` first happens to be used.
 */
static_assert(aether_expression<View<double, extents<3, dyn>>>,
    "View: the mutable View must satisfy aether_expression");
static_assert(aether_expression<const View<double, extents<3, dyn>>>,
    "View: a const-qualified View must satisfy aether_expression -- "
    "see aether::aether_expression's remove_cv_t note");

} // namespace aether
