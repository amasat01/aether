// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Assign.h
 * @brief Expression assignment: the compile-time recursive-unroll evaluator
 *        (`detail::RecursiveAssign`, over `element_extents` — rank-generic,
 *        since aether has no separate vector/matrix family), the
 *        `SampleRef` assignment-target proxy returned by `View::
 *        %operator[]`, its read-only sibling `ConstSampleRef` (returned by
 *        the const overload, `= += -=`-free by construction), `SampleRef`'s
 *        three scalar-RHS overloads (defined in `aether/expr/nodes/
 *        Constant.h` — see their in-class declarations below for why), and
 *        the out-of-line definitions this header exists to break a cycle for:
 *
 *   - `View<T,Extents,Layout>::%operator[](SampleIndex)`, both the mutable and
 *     the CONST overload (DECLARED in `view/View.h`, which only
 *     forward-declares `SampleRef`/`ConstSampleRef`).
 *   - `Item<T,Es...>`'s converting constructor + `operator=` FROM an
 *     expression (DECLARED in `view/Item.h`).
 *
 * WHY the split (not a straight `#include`): `SampleRef::get()` materializes
 * an `Item` (needs `view/Item.h` complete) and `View::%operator[]` wraps a
 * `View` (needs `view/View.h` complete) — but `Item`'s OWN expression ctor
 * needs this header's `RecursiveAssign`, and `View`'s `%operator[]` needs
 * this header's `SampleRef`. `Item.h`/`View.h` neither one can `#include`
 * this header (this header `#include`s them both) without a physical cycle;
 * declaring the two out-of-line members in their home headers and defining
 * them HERE is the standard C++ way to express a genuine two-way need.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include "aether/dtype/WorkingType.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/view/Item.h"
#include "aether/view/View.h"

namespace aether {
namespace detail {

/** @brief Which per-element operation `RecursiveAssign` performs. */
enum class AssignOp : unsigned char { Set, Add, Sub };

/**
 * @brief ★ THE STORE TERMINAL - the ONE place an expression's working
 *        carrier is encoded back into what the destination actually holds.
 *
 * Keyed on the DESTINATION SLOT's own type, never on the tree's `element_type`.
 * The difference matters in both directions:
 *  - a memory-backed slot (`View<BandedReal,...>`, `Item<BandedReal,N>`) is the
 *    storage scalar, so this is the single encode a whole chain pays;
 *  - a register-resident temporary that ALREADY holds the carrier (the fused
 *    caches in `aether/expr/nodes/Product.h`) is its own working type, so
 *    `WorkingType` is the identity primary there and NOTHING is emitted - keying
 *    on `element_type` instead would pack into every staging buffer and unpack
 *    straight back out.
 *
 * `+=`/`-=` read the stored value back through `toWorking`, combine in the
 * carrier and store ONCE. The `Slot == working_type_t<Slot>` arm is the
 * previous expression character for character, which is what keeps `double`/
 * `float` codegen identical AND keeps a `volatile`-qualified `View` slot (whose
 * reference does not bind to `const T&`) on exactly the path it was on before.
 */
template<AssignOp Op, class Ref, class W>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr void storeWorking(Ref&& dst, const W& w)
{
    using Slot = std::remove_cvref_t<Ref>;
    if constexpr (Op == AssignOp::Set) {
        dst = fromWorkingValue<Slot>(w);
    } else if constexpr (std::is_same_v<Slot, working_type_t<Slot>>) {
        if constexpr (Op == AssignOp::Add) {
            dst += w;
        } else {
            dst -= w;
        }
    } else {
        const working_type_t<Slot> cur = WorkingType<Slot>::toWorking(dst);
        if constexpr (Op == AssignOp::Add) {
            dst = fromWorkingValue<Slot>(cur + w);
        } else {
            dst = fromWorkingValue<Slot>(cur - w);
        }
    }
}

/**
 * @brief Host-only `always_inline` for the assign path below
 *        (`RecursiveAssign`, `AssignDispatch`, `assign*`, `SampleRef`).
 *
 * `RecursiveAssign` is pure template recursion that must vanish into its
 * caller. Left to GCC's early-inlining budget, a multi-component
 * `view[i] = expr` keeps it out of line until after the early scalar
 * replacement of aggregates, so the expression temporary it reads stays an
 * addressable stack object whose end-of-life clobber (a volatile statement)
 * then blocks if-conversion of a host sample loop. Forced inline, the
 * temporary is scalarised early and the loop vectorises. Device codegen is
 * untouched (the macro is empty under `__CUDA_ARCH__`).
 */
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
#define AETHER_DETAIL_UNROLL_INLINE __attribute__((always_inline))
#else
#define AETHER_DETAIL_UNROLL_INLINE
#endif

/**
 * @brief Compile-time recursive unroll over `Extents`: enumerates
 *        EVERY static multi-index `Is...` of `Extents` (rank-generic — one
 *        mode at a time, `Mode` counting up from 0 to `Extents::Rank`) and,
 *        at each leaf multi-index, performs `to.eval<Is...>(i) <op>=
 *        from.eval<Is...>(i)`. `Extents::Rank == 0` (a scalar shape)
 *        degenerates to exactly one call with an EMPTY `Is...` pack.
 *
 * Pure recursive template structs (no lambdas — nvcc device code stays on
 * a classic template-recursion idiom, generalized to a full multi-index
 * since aether's `element_extents` may be rank > 1).
 */
template<class Extents, AssignOp Op, std::size_t Mode, std::size_t... Is>
struct RecursiveAssign {
    template<class ExprL, class ExprR>
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() static constexpr void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        if constexpr (Mode == Extents::Rank) {
            storeWorking<Op>(to.template eval<Is...>(i), evalW<Is...>(from, i));
        } else {
            loop<0>(i, to, from);
        }
    }

private:
    template<std::size_t K, class ExprL, class ExprR>
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() static constexpr void loop(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        if constexpr (K < Extents::static_extent(Mode)) {
            RecursiveAssign<Extents, Op, Mode + 1, Is..., K>::eval(i, to, from);
            loop<K + 1>(i, to, from);
        }
    }
};

/**
 * @brief An indirection: the top-level entry point every
 *        `assign`/`assignAdd`/`assignSub` call routes through, keyed on
 *        `ExprR`'s TYPE (a class template, unlike the free functions below —
 *        see the note on why this exists).
 *
 * The primary template just forwards to the plain per-element
 * `RecursiveAssign` loop — behaviourally IDENTICAL to what `assign()` did
 * before this indirection existed. `aether/expr/nodes/
 * Product.h` adds PARTIAL SPECIALIZATIONS of this template for
 * `MatVec<M,V>`/`MatMat<L,R>` that cache the reused operand in a register-
 * resident `Item` once instead of recomputing it per output component
 * (a "fused multi-load" pattern).
 *
 * WHY an indirection through a CLASS template, when `assign()` itself is
 * already a function template: a qualified call (`detail::assign(...)`) to
 * a FUNCTION template only ever considers overloads visible by ordinary
 * lookup at the call's OWN definition point (no ADL for a qualified-id, and
 * qualified lookup is not re-run at instantiation) — so a `Product.h`
 * overload of the free function `assign()`, declared AFTER this header,
 * would never be found by `SampleRef::operator=`'s call below, no matter
 * the `#include` order in the final translation unit. A CLASS template's
 * partial specializations do not have this problem: `AssignDispatch<ExprR,
 * Op>` is a dependent TYPE at THIS header's parse time (its arguments are
 * still template parameters), so which specialization it names is resolved
 * at the point of INSTANTIATION — by then, any specialization declared
 * anywhere earlier in the same translation unit (Product.h included) is
 * visible.
 */
template<class ExprR, AssignOp Op>
struct AssignDispatch {
    template<class ExprL>
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() static constexpr void eval(const SampleIndex& i, ExprL& to, const ExprR& from)
    {
        RecursiveAssign<typename ExprL::element_extents, Op, 0>::eval(i, to, from);
    }
};

/** @brief `to = from`, element-by-element, over `ExprL::element_extents`. */
template<class ExprL, class ExprR>
AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr void assign(const SampleIndex& i, ExprL& to, const ExprR& from)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "assign: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "assign: element_extents mismatch");
    AssignDispatch<ExprR, AssignOp::Set>::eval(i, to, from);
}

/** @brief `to += from`, element-by-element. */
template<class ExprL, class ExprR>
AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr void assignAdd(const SampleIndex& i, ExprL& to, const ExprR& from)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "assignAdd: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "assignAdd: element_extents mismatch");
    AssignDispatch<ExprR, AssignOp::Add>::eval(i, to, from);
}

/** @brief `to -= from`, element-by-element. */
template<class ExprL, class ExprR>
AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr void assignSub(const SampleIndex& i, ExprL& to, const ExprR& from)
{
    static_assert(std::is_same_v<typename ExprL::element_type, typename ExprR::element_type>,
        "assignSub: element_type mismatch");
    static_assert(std::is_same_v<typename ExprL::element_extents, typename ExprR::element_extents>,
        "assignSub: element_extents mismatch");
    AssignDispatch<ExprR, AssignOp::Sub>::eval(i, to, from);
}

/** @brief `Item<T, Extents's static modes...>` — the register-materialized
 *         type `SampleRef::get()` returns for a given (`T`, `element_extents`). */
template<class T, class Extents, class Seq>
struct ItemFromExtentsImpl;
template<class T, class Extents, std::size_t... Is>
struct ItemFromExtentsImpl<T, Extents, std::index_sequence<Is...>> {
    using type = Item<T, Extents::static_extent(Is)...>;
};
template<class T, class Extents>
using ItemFromExtents = typename ItemFromExtentsImpl<T, Extents, std::make_index_sequence<Extents::Rank>>::type;

} // namespace detail

/**
 * @brief Assignment-target proxy: `view[i] = expr`, `view[i] += expr`,
 *        `view[i] -= expr`, `view[i].get()` (register materialization into
 *        an `Item`). Returned by `View::%operator[](SampleIndex)`.
 */
template<class V>
class SampleRef {
public:
    using element_type = typename V::element_type;
    using element_extents = typename V::element_extents;

    AETHER_DEVICEHOST() constexpr SampleRef(V& view, const SampleIndex& i)
        : view_(view)
        , i_(i)
    {
    }

    /** @brief `view[i] = expr` — overwrite. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr SampleRef& operator=(const E& expr)
    {
        detail::assign(i_, view_, expr);
        return *this;
    }

    /** @brief `view[i] += expr`. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr SampleRef& operator+=(const E& expr)
    {
        detail::assignAdd(i_, view_, expr);
        return *this;
    }

    /** @brief `view[i] -= expr`. */
    template<class E>
        requires(aether_expression<E> && std::is_same_v<typename E::element_type, element_type>
            && std::is_same_v<typename E::element_extents, element_extents>)
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr SampleRef& operator-=(const E& expr)
    {
        detail::assignSub(i_, view_, expr);
        return *this;
    }

    /**
     * @brief `view[i] = scalar` — broadcast overwrite. A member function
     *        TEMPLATE, deliberately: `S` is deduced
     *        EXACTLY from the argument (never implicitly converted to a
     *        fixed `element_type` parameter) so the narrowing check in the
     *        out-of-line body is a real gate rather than a silent standard
     *        conversion swallowing e.g. `v[i] = 2` on a `double` view.
     *        `requires(!aether_expression<S>)` keeps this overload out of
     *        the expression-typed `operator=` above's way (an expression
     *        operand would otherwise make BOTH templates viable with the
     *        identical `(const T&)` signature) rather than relying on
     *        partial-ordering tie-breaking. DEFINED out-of-line in
     *        `aether/expr/nodes/Constant.h` (wraps the scalar in a
     *        `Constant<element_type,element_extents>` leaf and forwards to
     *        `detail::assign` — needs that leaf's full definition; no
     *        physical `#include` cycle: `Constant.h` needs nothing from THIS
     *        header beyond `SampleRef`'s own declaration, the same one-way
     *        split `Item`'s expression ctor/assign use above).
     */
    template<class S>
        requires(!aether_expression<S>)
    AETHER_DEVICEHOST() constexpr SampleRef& operator=(const S& scalar);

    /** @brief `view[i] += scalar`. See `operator=(const S&)` above. */
    template<class S>
        requires(!aether_expression<S>)
    AETHER_DEVICEHOST() constexpr SampleRef& operator+=(const S& scalar);

    /** @brief `view[i] -= scalar`. See `operator=(const S&)` above. */
    template<class S>
        requires(!aether_expression<S>)
    AETHER_DEVICEHOST() constexpr SampleRef& operator-=(const S& scalar);

    /** @brief Register materialization: an `Item` snapshot of this sample. */
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr detail::ItemFromExtents<element_type, element_extents> get() const
    {
        detail::ItemFromExtents<element_type, element_extents> result;
        detail::assign(i_, result, view_);
        return result;
    }

private:
    V& view_;
    SampleIndex i_;
};

/**
 * @brief READ-ONLY counterpart of `SampleRef` (the const read path):
 *        `cv[i].get()` (register materialization into an `Item`) and
 *        `cv[i].eval<Is...>` (direct per-component forwarding read, no
 *        `Item` built) — deliberately NO `= += -=`. Returned by `View::
 *        %operator[](const SampleIndex&) const`, always instantiated with a
 *        CONST-qualified `V` (e.g. `ConstSampleRef<const View<...>>`), which
 *        is what makes a `const View` a genuine read-only expression LEAF
 *        rather than merely an accessor missing a write path (`const View`
 *        itself satisfies `aether_expression` — `view/View.h`'s own
 *        `static_assert`, backed by `aether::aether_expression`'s
 *        `remove_cv_t` handling in `aether/expr/Expression.h`). The code
 *        generator's "bool-readable proxy" need
 *        is served by `.get()`; no implicit conversion operator is added —
 *        a proxy with an implicit `operator bool()` is the kind of silent
 *        coercion this type deliberately forbids.
 */
template<class V>
class ConstSampleRef {
public:
    using element_type = typename V::element_type;
    using element_extents = typename V::element_extents;

    AETHER_DEVICEHOST() constexpr ConstSampleRef(V& view, const SampleIndex& i)
        : view_(view)
        , i_(i)
    {
    }

    /** @brief Direct per-component read at this proxy's bound `SampleIndex`,
     *         forwarded straight to the wrapped `View`'s own `eval<Is...>`
     *         — the finer-grained sibling of `.get()` below (no `Item`
     *         materialized). */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr decltype(auto) eval() const
    {
        return view_.template eval<Is...>(i_);
    }

    /** @brief Register materialization: an `Item` snapshot of this sample. */
    AETHER_DETAIL_UNROLL_INLINE AETHER_DEVICEHOST() constexpr detail::ItemFromExtents<element_type, element_extents> get() const
    {
        detail::ItemFromExtents<element_type, element_extents> result;
        detail::assign(i_, result, view_);
        return result;
    }

private:
    V& view_;
    SampleIndex i_;
};

// ---------------------------------------------------------------------------
// Out-of-line definitions declared in view/View.h and view/Item.h (see the
// file docstring above for why they live here).
// ---------------------------------------------------------------------------

template<class T, class Extents, class Layout, bool Volatile, bool ReadOnly>
AETHER_DEVICEHOST() constexpr SampleRef<View<T, Extents, Layout, Volatile, ReadOnly>> View<T, Extents, Layout, Volatile,
    ReadOnly>::operator[](const SampleIndex& i)
{
    return SampleRef<View<T, Extents, Layout, Volatile, ReadOnly>>(*this, i);
}

/** @brief The CONST overload's out-of-line definition (see
 *         `view/View.h`'s forward declaration + this file's docstring). */
template<class T, class Extents, class Layout, bool Volatile, bool ReadOnly>
AETHER_DEVICEHOST() constexpr ConstSampleRef<const View<T, Extents, Layout, Volatile, ReadOnly>> View<T, Extents, Layout,
    Volatile, ReadOnly>::operator[](const SampleIndex& i) const
{
    return ConstSampleRef<const View<T, Extents, Layout, Volatile, ReadOnly>>(*this, i);
}

// Unconstrained at the template head (see Item.h's note on the ctor); validated here instead.
template<class T, std::size_t... Es>
template<class E>
AETHER_DEVICEHOST() constexpr Item<T, Es...>::Item(const E& expr)
{
    static_assert(aether_expression<E>, "Item(expr): E must satisfy aether_expression");
    static_assert(std::is_same_v<typename E::element_type, T>, "Item(expr): element_type mismatch");
    static_assert(std::is_same_v<typename E::element_extents, typename Item<T, Es...>::element_extents>,
        "Item(expr): element_extents mismatch");
    detail::assign(SampleIndex::make(0), *this, expr);
}

template<class T, std::size_t... Es>
template<class E>
AETHER_DEVICEHOST() constexpr Item<T, Es...>& Item<T, Es...>::operator=(const E& expr)
{
    static_assert(aether_expression<E>, "Item::operator=(expr): E must satisfy aether_expression");
    static_assert(std::is_same_v<typename E::element_type, T>, "Item::operator=(expr): element_type mismatch");
    static_assert(std::is_same_v<typename E::element_extents, typename Item<T, Es...>::element_extents>,
        "Item::operator=(expr): element_extents mismatch");
    detail::assign(SampleIndex::make(0), *this, expr);
    return *this;
}

} // namespace aether

// Internal codegen helper, confined to this header: undef so it does not
// leak into every TU that includes aether/aether.h (grep confirms no other
// header or test references it).
#undef AETHER_DETAIL_UNROLL_INLINE
