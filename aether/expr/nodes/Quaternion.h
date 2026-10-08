// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Quaternion.h
 * @brief `aether::detail::QuatMul<L,R>` / `QuatConj<E>` / `QuatRotate<Q,V>` /
 *        `AsPureQuaternion<E>` — back `Expression::quatMul()` /
 *        `quatConj()` / `quatReciprocal()` / `quatRotate()` /
 *        `asPureQuaternion()` / `asBack3DVector()` (declared + defined inline
 *        in `aether/expr/Expression.h`; see that file's docstring for why the
 *        whole member surface lives there). `asBack3DVector()` reuses
 *        `Segment` (`tail<3>`) — no dedicated node needed.
 *        `quatReciprocal()` reuses `quatConj() * rSquaredNorm()`
 *        (`CWiseScale` + `rSquaredNorm()`) — likewise no dedicated node.
 *
 * `QuatMul` is the Hamilton product; `QuatRotate` is the 30-FLOP Rodrigues
 * rotation form (`t = 2*(u x v)`, `result = v + w*t + u x t`, `u` = the
 * quaternion's vector part).
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/macros.h"

namespace aether {
namespace detail {

/** @brief Hamilton product `L * R` of two quaternion expressions (element_extents<4>, w,x,y,z order). */
template<class L, class R>
class QuatMul : public Expression<QuatMul<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_extents, extents<4>>,
        "QuatMul: L must be a quaternion (element_extents<4>)");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "QuatMul: operand element_extents must match");
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>,
        "QuatMul: operand element_type must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr QuatMul(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        const working_type l0 = evalW<0>(l_, i);
        const working_type l1 = evalW<1>(l_, i);
        const working_type l2 = evalW<2>(l_, i);
        const working_type l3 = evalW<3>(l_, i);
        const working_type r0 = evalW<0>(r_, i);
        const working_type r1 = evalW<1>(r_, i);
        const working_type r2 = evalW<2>(r_, i);
        const working_type r3 = evalW<3>(r_, i);
        if constexpr (I == 0) {
            return l0 * r0 - l1 * r1 - l2 * r2 - l3 * r3;
        } else if constexpr (I == 1) {
            return l0 * r1 + l1 * r0 + l2 * r3 - l3 * r2;
        } else if constexpr (I == 2) {
            return l0 * r2 - l1 * r3 + l2 * r0 + l3 * r1;
        } else {
            return l0 * r3 + l1 * r2 - l2 * r1 + l3 * r0;
        }
    }
};

/** @brief Quaternion conjugate (negate the vector part, index 0 = scalar). */
template<class E>
class QuatConj : public Expression<QuatConj<E>, typename E::element_type> {
    static_assert(std::is_same_v<typename E::element_extents, extents<4>>,
        "QuatConj: E must be a quaternion (element_extents<4>)");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename E::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit QuatConj(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        if constexpr (I == 0) {
            return evalW<0>(expr_, i);
        } else {
            return -evalW<I>(expr_, i);
        }
    }
};

/**
 * @brief Rodrigues-formula rotation of 3-vector `V` by unit quaternion `Q`
 *        (element_extents<3> result). 30-FLOP form
 *        (`t = 2*(u x v)`, `result = v + w*t + u x t`, `u` = `Q`'s vector part).
 */
template<class Q, class V>
class QuatRotate : public Expression<QuatRotate<Q, V>, typename Q::element_type> {
    static_assert(std::is_same_v<typename Q::element_extents, extents<4>>,
        "QuatRotate: Q must be a quaternion (element_extents<4>)");
    static_assert(std::is_same_v<typename V::element_extents, extents<3>>,
        "QuatRotate: V must be a 3-vector (element_extents<3>)");
    static_assert(std::is_same_v<typename Q::element_type, typename V::element_type>,
        "QuatRotate: operand element_type must match");

public:
    using element_type = typename Q::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<3>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<Q::isLeaf, const Q&, const Q> q_;
    typename std::conditional_t<V::isLeaf, const V&, const V> v_;

    AETHER_DEVICEHOST() constexpr QuatRotate(const Q& q, const V& v)
        : q_{ q }
        , v_{ v }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        const working_type w = evalW<0>(q_, i);
        const working_type ux = evalW<1>(q_, i);
        const working_type uy = evalW<2>(q_, i);
        const working_type uz = evalW<3>(q_, i);

        const working_type vx = evalW<0>(v_, i);
        const working_type vy = evalW<1>(v_, i);
        const working_type vz = evalW<2>(v_, i);

        /* `element_type(2)`, still: the literal 2 is a STORAGE-side constant
         * (every scalar in this library can be built from an integer literal;
         * a working carrier need not be - `aether::banded::Band` has no such
         * constructor), and mixing it into a carrier-valued product is exactly
         * what the storage/working operator set is for. */
        const working_type tx = element_type(2) * (uy * vz - uz * vy);
        const working_type ty = element_type(2) * (uz * vx - ux * vz);
        const working_type tz = element_type(2) * (ux * vy - uy * vx);

        if constexpr (I == 0) {
            return vx + w * tx + (uy * tz - uz * ty);
        } else if constexpr (I == 1) {
            return vy + w * ty + (uz * tx - ux * tz);
        } else {
            return vz + w * tz + (ux * ty - uy * tx);
        }
    }
};

/** @brief `[x,y,z]` viewed as a pure quaternion `[0,x,y,z]` (element_extents<3> -> <4>). */
template<class E>
class AsPureQuaternion : public Expression<AsPureQuaternion<E>, typename E::element_type> {
    static_assert(std::is_same_v<typename E::element_extents, extents<3>>,
        "AsPureQuaternion: E must be a 3-vector (element_extents<3>)");

public:
    using element_type = typename E::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<4>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<E::isLeaf, const E&, const E> expr_;

    AETHER_DEVICEHOST() constexpr explicit AsPureQuaternion(const E& e)
        : expr_{ e }
    {
    }

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        if constexpr (I == 0) {
            /* `working_type{}` and not `working_type{ 0 }`: value-initialisation
             * is the type's OWN zero, which every carrier can answer; a
             * construction FROM the literal 0 is not (the `AbsOp` note in
             * `aether/expr/Reduce.h` records the same trap). */
            return working_type{};
        } else {
            return evalW<I - 1>(expr_, i);
        }
    }
};

} // namespace detail
} // namespace aether
