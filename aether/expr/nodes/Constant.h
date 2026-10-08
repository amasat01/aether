// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Constant.h
 * @brief Scalar broadcast expression leaf: `Constant<T,Extents>` holds one
 *        `T` by value and answers every `eval<Is...>()` call with it,
 *        regardless of the static multi-index — the expression-tree
 *        counterpart of `SampleRef`'s `= += -=` scalar overloads
 *        (`aether/expr/Assign.h`), which wrap a bare scalar in this leaf
 *        and forward to the ordinary expression assignment path, so a
 *        scalar store pays exactly the same `detail::assign` terminal an
 *        expression store does (one encode at the terminal — no
 *        `Constant`-specific store path).
 *
 * Rank-generic: `Extents` is whatever `element_extents` the destination
 * carries (a vector `extents<3>`, a matrix `extents<3,3>`, a scalar
 * `extents<>`, ...) — `Constant` does not care what shape it is broadcast
 * into, it just hands back the same value for every `Is...`.
 *
 * `working_type` follows `Expression`'s own trait unconditionally — no
 * `Constant`-specific code makes a `BandedReal` constant evaluate in
 * `Band`; `WorkingType<BandedReal>` (declared at the end of
 * `aether/banded/BandedReal.h`, where the carrier type is complete) is what
 * does that, and `Constant` gets it for free by being an ordinary `Expression`
 * CRTP derivative like `Item`/`View`. `eval()` hands back the storage
 * scalar, exactly like any other leaf (`Item`/`View`) — `detail::evalW`,
 * the one place a leaf's read is promoted into the working carrier
 * (`aether/expr/Expression.h`), converts it the same way it converts a
 * `View` read; `Constant` needs no `evalW`/`toWorkingValue` call of its own.
 *
 * This file also carries `SampleRef`'s scalar-overload definitions
 * ----------------------------------------------------------------------
 * `SampleRef::operator= += -=(const S&)` are declared in `aether/expr/
 * Assign.h` (member function templates on `SampleRef`) but defined here —
 * the same declare-here/define-there split `aether/expr/Assign.h`'s own
 * file docstring explains for `View::operator[]`/`Item`'s expression ctor:
 * the bodies construct a `Constant<element_type,element_extents>`, so they
 * need this file's full definition, while `Assign.h` itself does not (a
 * `SampleRef` consumer that never assigns a scalar never needs `Constant.h`
 * at all — no `#include` back into `Assign.h` from here is even required,
 * since `Assign.h` is included below purely for `SampleRef`'s declaration
 * and `detail::assign`/`assignAdd`/`assignSub`).
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Assign.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/**
 * @brief Broadcast leaf: ONE `T`, answered for every static element index
 *        `Is...` of `Extents`. `isLeaf = true` —
 *        `Constant` is a leaf in exactly the sense `Item`/`View` are, it
 *        just happens to hold a value with no addressable storage behind
 *        it (a same-statement temporary, matching `CWiseScale`'s captured
 *        scalar factor in spirit).
 */
template<class T, class Extents>
class Constant : public Expression<Constant<T, Extents>, T> {
public:
    using element_type = T;
    using element_extents = Extents;
    static constexpr bool isLeaf = true;

    AETHER_DEVICEHOST() constexpr explicit Constant(const T& value)
        : value_(value)
    {
    }

    /** @brief Broadcast read: every `Is...` combination answers the SAME
     *         stored value. Returns the STORAGE scalar, matching `Item`/
     *         `View`'s own leaf convention — `detail::evalW` (`aether/expr/
     *         Expression.h`) is what promotes a leaf's read into the working
     *         carrier, and it is idempotent-correct for this exactly like
     *         for any other leaf. */
    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr const T& eval(const SampleIndex&) const
    {
        return value_;
    }

private:
    T value_;
};

/** @brief A `Constant` is a value leaf (see `detail::isValueLeaf`). */
template<class T, class Extents>
inline constexpr bool isValueLeaf<Constant<T, Extents>> = true;

} // namespace detail

/**
 * @brief `aether::constant<Extents>(v)` — a `Constant<T,Extents>` leaf
 *        broadcasting `v` over every component of `Extents` (codegen
 *        ergonomics: an expression tree can name a broadcast scalar
 *        directly, e.g. `aether::constant<extents<3>>(1.0) + other`, the
 *        same shape `SampleRef`'s scalar overloads build internally).
 *        `Extents` is supplied explicitly (nothing in the call deduces it);
 *        `T` deduces from `v`.
 */
template<class Extents, class T>
AETHER_DEVICEHOST() constexpr detail::Constant<T, Extents> constant(const T& v)
{
    return detail::Constant<T, Extents>(v);
}

// ---------------------------------------------------------------------------
// SampleRef's scalar-overload out-of-line definitions (declared in
// aether/expr/Assign.h — see this file's docstring for why they live here).
// ---------------------------------------------------------------------------

template<class V>
template<class S>
    requires(!aether_expression<S>)
AETHER_DEVICEHOST() constexpr SampleRef<V>& SampleRef<V>::operator=(const S& scalar)
{
    static_assert(std::is_same_v<S, element_type>,
        "SampleRef::operator=(scalar): S must be EXACTLY element_type -- no implicit "
        "conversion; spell v[i] = 2.0, not v[i] = 2");
    detail::assign(i_, view_, detail::Constant<element_type, element_extents>(scalar));
    return *this;
}

template<class V>
template<class S>
    requires(!aether_expression<S>)
AETHER_DEVICEHOST() constexpr SampleRef<V>& SampleRef<V>::operator+=(const S& scalar)
{
    static_assert(std::is_same_v<S, element_type>,
        "SampleRef::operator+=(scalar): S must be EXACTLY element_type -- no implicit "
        "conversion; spell v[i] += 2.0, not v[i] += 2");
    detail::assignAdd(i_, view_, detail::Constant<element_type, element_extents>(scalar));
    return *this;
}

template<class V>
template<class S>
    requires(!aether_expression<S>)
AETHER_DEVICEHOST() constexpr SampleRef<V>& SampleRef<V>::operator-=(const S& scalar)
{
    static_assert(std::is_same_v<S, element_type>,
        "SampleRef::operator-=(scalar): S must be EXACTLY element_type -- no implicit "
        "conversion; spell v[i] -= 2.0, not v[i] -= 2");
    detail::assignSub(i_, view_, detail::Constant<element_type, element_extents>(scalar));
    return *this;
}

} // namespace aether
