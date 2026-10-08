// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Item.h
 * @brief `aether::Item`: a register-resident, all-static-extent value
 *        type — `Item<double,3>` is `Vec3d`, `Item<double,3,3>` is `Mat33d`.
 *
 * `Item` rebases onto the `Expression` CRTP (see the class docstring
 * below) — storage/access (construction, element access, the
 * `Zeros()`/`Ones()`/`filled()` factories) is otherwise unremarkable.
 * Arithmetic operators (`+`, `-`, `s*e`, ...) live in `aether/expr/Operations.h`,
 * concept-gated over any `aether_expression`, `Item` included.
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/**
 * @brief Compile-time unroller writing `1` on the diagonal of a square
 *        matrix `Item` — helper for `Item::Identity()`.
 */
template<std::size_t D>
struct IdentityFill {
    template<class MatT>
    AETHER_DEVICEHOST() static constexpr void eval(MatT& m)
    {
        m.template get<D, D>() = typename MatT::element_type{ 1 };
        IdentityFill<D - 1>::eval(m);
    }
};
template<>
struct IdentityFill<0> {
    template<class MatT>
    AETHER_DEVICEHOST() static constexpr void eval(MatT& m)
    {
        m.template get<0, 0>() = typename MatT::element_type{ 1 };
    }
};

} // namespace detail

/**
 * @brief Register-resident value with all-static extents `Es...` (no `dyn`
 *        mode is permitted — `Array`, not `Item`, carries the batch mode).
 *        Row-major element order (matches `layout_right`): for `Item<T,R,C>`,
 *        `(r, c)` maps to offset `r*C + c`.
 *
 * Rebases `Item` onto the `Expression` CRTP: `Item` is always a
 * conforming expression leaf (`isLeaf = true`, `%element_extents =
 * extents<Es...>` — trivially all-static, matching `Item`'s own storage
 * shape) and gains a converting constructor + assignment operator from any
 * matching expression (`aether::aether_expression`), evaluated at
 * `SampleIndex::make(0)` since an `Item` is sample-free. Both are
 * declared here but defined out-of-line in `aether/expr/Assign.h` — `Item`
 * itself must not `#include` that header (it in turn needs `Item`'s full
 * definition to materialize `SampleRef::get()`, a genuine two-way need that
 * a physical `#include` cycle cannot express; the declare/define split
 * breaks the cycle without weakening either side's compile-time checking).
 */
template<class T, std::size_t... Es>
class Item : public Expression<Item<T, Es...>, T> {
    static_assert(((Es != dyn) && ...), "Item: every extent must be static (dyn is not allowed)");

public:
    using element_type = T;
    /** @brief An `Item`'s own shape is its element shape. */
    using element_extents = extents<Es...>;
    /** @brief `Item` is always a leaf. */
    static constexpr bool isLeaf = true;

    /** @brief Number of modes. */
    static constexpr std::size_t Rank = sizeof...(Es);
    /** @brief Total element count — the product of `Es...`. */
    static constexpr std::size_t Size = (Es * ... * std::size_t{ 1 });

    // No AETHER_DEVICEHOST() here — see the note on aether::View's default ctor.
    constexpr Item() = default;

    /**
     * @brief Converting constructor from any matching expression,
     *        evaluated at `SampleIndex::make(0)`. Defined out-of-line in
     *        `aether/expr/Assign.h` (validated there via `static_assert`,
     *        not a `requires`-clause: nvcc 12.9 rejects an out-of-line
     *        member-template definition whose `requires`-clause is not
     *        token-identical to the in-class declaration's, even when the
     *        two spellings are semantically equivalent — e.g. the bare
     *        `%element_extents` this class body can use vs. the
     *        `typename Item<T,Es...>::%element_extents` an out-of-class
     *        definition needs).
     */
    template<class E>
    AETHER_DEVICEHOST() constexpr Item(const E& expr);

    /**
     * @brief Assignment from any matching expression, evaluated at
     *        `SampleIndex::make(0)`. Defined out-of-line in
     *        `aether/expr/Assign.h` — see the ctor's note above.
     */
    template<class E>
    AETHER_DEVICEHOST() constexpr Item& operator=(const E& expr);

    /**
     * @brief N-scalar constructor: `Item<T,Es...>{v0, v1, …}` with exactly
     *        `Size` (the product of `Es...`) values, in the same
     *        row-major order as `operator()`/`get<>`. Compile-time
     *        enforced via a trailing `requires` clause (mirrors
     *        `aether::extents`'s own N-scalar ctor, `aether/layout/
     *        Extents.h`) rather than a body-level `static_assert`: a wrong
     *        count is removed from the overload set entirely, so it can
     *        never become a viable-then-hard-erroring candidate. `Size >=
     *        2` is part of the guard for the same reason it protects the
     *        single-argument case: every alias this library defines
     *        (`Vec3d`/`Vec4d`/`Vec6d`/`Mat33d`/…) has `Size >= 2`, so a
     *        genuine N-scalar call is never a candidate alongside the
     *        converting-expression ctor above (which is deliberately
     *        unconstrained at the signature level — see that ctor's own
     *        note); the never-aliased `Size == 1` case simply falls
     *        through to it, unaffected.
     */
    template<class... Vs>
        requires(sizeof...(Vs) == Size && Size >= 2)
    AETHER_DEVICEHOST() constexpr Item(Vs... vs)
        : data_{ static_cast<T>(vs)... }
    {
    }

    /**
     * @brief 1-component constructor (`Item<T,1>{v}`, also covers the
     *        rank-0 `Item<T>{v}` case, since both have `Size == 1`): the
     *        ctor above's own docstring explains why it is guarded
     *        `Size >= 2` rather than `Size >= 1` — a same-shape
     *        `template<class... Vs> requires(sizeof...(Vs) == 1) Item(Vs...)`
     *        would be a second single-argument candidate alongside the
     *        converting-expression ctor's `template<class E> Item(const E&)`
     *        (deliberately unconstrained at the signature level), and the
     *        two are not orderable by C++20 partial ordering (a trailing
     *        parameter pack vs. a lone parameter do not compare consistently
     *        for a one-argument call) — genuinely ambiguous, not merely
     *        undesirable, so `Size == 1` was left with no N-scalar ctor at
     *        all rather than a broken one.
     *
     *        This overload sidesteps that by not being a template: a plain
     *        (non-template) member function of a class template may still
     *        carry a `requires`-clause naming the class's own parameters
     *        (`Size` here), and per [over.match.best] a non-template
     *        function is preferred over an equally-good template
     *        specialization — so for `Item<T,1>{v}` (`v` of type `T`) this
     *        candidate and the converting ctor's `E=T` instantiation both
     *        produce the identical signature `Item(const T&)`, and overload
     *        resolution picks this one without ever instantiating (let alone
     *        hard-erroring on) the converting ctor's `static_assert
     *        aether_expression<E>` body. `explicit`, unlike the ctor
     *        above: this is the one shape here that could otherwise become
     *        an implicit `T -> Item<T,1>` conversion, which the "no silent
     *        coercion" doctrine (@see `SampleRef::operator=(scalar)`'s own
     *        note, `aether/expr/nodes/Constant.h`) rules out; a brace-init
     *        call shape like `aether::Item<Real,1>{v}` is
     *        direct-initialization, which `explicit` does not affect.
     */
    AETHER_DEVICEHOST() constexpr explicit Item(const T& v)
        requires(Size == 1)
        : data_{ v }
    {
    }

    /**
     * @brief `Is...` matches `%element_extents`; the `SampleIndex` is
     *        ignored (an `Item` is sample-free).
     */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr T& eval(const SampleIndex&)
    {
        return get<Is...>();
    }
    /** @overload */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr const T& eval(const SampleIndex&) const
    {
        return get<Is...>();
    }

    /** @brief An `Item` with every element `0`. */
    AETHER_DEVICEHOST() static constexpr Item Zeros() { return filled(T{ 0 }); }
    /** @brief An `Item` with every element `1`. */
    AETHER_DEVICEHOST() static constexpr Item Ones() { return filled(T{ 1 }); }
    /** @brief An `Item` with every element set to `v`. */
    AETHER_DEVICEHOST() static constexpr Item filled(const T& v)
    {
        Item item;
        for (std::size_t i = 0; i < Size; ++i)
            item.data_[i] = v;
        return item;
    }

    /**
     * @brief The identity matrix — square `Item<T,N,N>` only. Zero-fill
     *        then a compile-time unrolled diagonal write
     *        (`detail::IdentityFill`, declared above this class). The two
     *        conditions are combined into one short-circuited `static_assert`
     *        (rather than two separate ones) so `static_extent(1)` is never
     *        even called — hence never constant-evaluated, hence never an
     *        out-of-bounds `Carray` access — for a non-rank-2 `Item` (`&&`'s
     *        short-circuit applies to constant-expression evaluation the
     *        same way it applies at runtime).
     */
    AETHER_DEVICEHOST() static constexpr Item Identity()
    {
        static_assert(Rank == 2 && element_extents::static_extent(0) == element_extents::static_extent(1),
            "Item::Identity(): only a SQUARE rank-2 (matrix) Item has an identity");
        Item out = Zeros();
        detail::IdentityFill<element_extents::static_extent(0) - 1>::eval(out);
        return out;
    }

    /** @brief Element access by runtime index, row-major over `Es...`. */
    template<class... Idxs>
    AETHER_DEVICEHOST() constexpr T& operator()(Idxs... idxs)
    {
        static_assert(sizeof...(Idxs) == Rank, "Item::operator(): wrong number of indices");
        return data_[offset_(static_cast<std::size_t>(idxs)...)];
    }
    template<class... Idxs>
    AETHER_DEVICEHOST() constexpr const T& operator()(Idxs... idxs) const
    {
        static_assert(sizeof...(Idxs) == Rank, "Item::operator(): wrong number of indices");
        return data_[offset_(static_cast<std::size_t>(idxs)...)];
    }

    /** @brief Element access by compile-time index. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr T& get()
    {
        static_assert(sizeof...(Is) == Rank, "Item::get<Is...>(): wrong number of indices");
        return data_[offset_(Is...)];
    }
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr const T& get() const
    {
        static_assert(sizeof...(Is) == Rank, "Item::get<Is...>(): wrong number of indices");
        return data_[offset_(Is...)];
    }

    AETHER_DEVICEHOST() constexpr T* data() { return data_.data(); }
    AETHER_DEVICEHOST() constexpr const T* data() const { return data_.data(); }
    AETHER_DEVICEHOST() static constexpr std::size_t size() { return Size; }

private:
    template<class... Idxs>
    AETHER_DEVICEHOST() static constexpr std::size_t offset_(Idxs... idxs)
    {
        if constexpr (Rank == 0) {
            return 0;   // a rank-0 Item has one element; Carray<_,0> has no operator[] by design
        } else {
            const detail::Carray<std::size_t, Rank> idxArr{ static_cast<std::size_t>(idxs)... };
            const detail::Carray<std::size_t, Rank> extArr{ Es... };
            std::size_t offset = 0;
            for (std::size_t i = 0; i < Rank; ++i)
                offset = offset * extArr[i] + idxArr[i];
            return offset;
        }
    }

    detail::Carray<T, Size> data_{};
};

namespace detail {
/** @brief An `Item` is a value leaf (see `detail::isValueLeaf`). */
template<class T, std::size_t... Es>
inline constexpr bool isValueLeaf<Item<T, Es...>> = true;
} // namespace detail

} // namespace aether
