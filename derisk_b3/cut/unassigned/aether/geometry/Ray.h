// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Ray.h
 * @brief `aether::geometry::rayTriangle` / `rayAabb` / `raySphere`: ray
 *        intersection on one lane, `float` or `double`.
 *
 * Every primitive returns a `Vec4`: `v[0]` is the hit flag (1 or 0), `v[1..3]`
 * the values below. The values are always the computed ones, also on a miss, so
 * a caller can select between candidates by `hit` and differentiate through
 * the live arm; only a degenerate input (see below) gives zeros.
 *
 *  - `rayTriangle`: `[hit, t, u, v]`, `t` along the ray, `(u, v)` the
 *    barycentric weights of `b` and `c`. Two-sided; hit needs `0 <= t <= tmax`.
 *  - `rayAabb`: `[hit, t_enter, t_exit, 0]`. `t_enter` is negative when the
 *    origin is inside the box. Hit needs the interval to reach `[0, tmax]`.
 *  - `raySphere`: `[hit, t0, t1, 0]`, the two roots in ascending order for a
 *    non-zero direction. Hit needs `t1 >= 0` and `t0 <= tmax`. A tangent ray
 *    is a hit with `t0 == t1`.
 *
 * Safe denominators: every division and square root is guarded BEFORE the
 * select, never after it, so a degenerate input (ray parallel to the triangle,
 * zero-area triangle, zero direction, discriminant below zero, a NaN operand)
 * yields finite values, and a derivative taken from these formulas cannot meet
 * `0 * inf`. A triangle denominator that is not a non-zero number gives
 * `t = u = v = 0`. A slab with a zero direction component uses `+-1e18` in
 * place of the reciprocal (never 0, which would report a ray inside the slab
 * as a miss; the sign follows the zero's sign bit). A sphere with a negative
 * discriminant or a zero direction gives `t0 = t1 = 0`.
 *
 * No silhouette term: `hit` is a comparison and has zero derivative, and the
 * jump of `t` at a triangle edge or a sphere's rim is not modelled.
 *
 * Operands are length-3 pointers or per-sample operands (`view[i]`, `Item`),
 * read element by element. The arithmetic is plain `a * b + c` in the order
 * written, so a build that does not contract products matches the C99 twin of
 * `aether_dsc.c_core` (`ray_tri`, `ray_aabb`, `ray_sphere`) bit for bit.
 */

#include "aether/linalg/detail/Common.h"
#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace geometry {

/** @brief Four values of one lane, returned by value. */
template<class Real>
struct Vec4 {
    Real v[4];
};

namespace detail {

/** @brief Stand-in reciprocal of a zero slab direction component. */
inline constexpr double slab_big = 1e18;

template<class Real, class A>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real at3(const A& a, std::size_t k)
{
    return static_cast<Real>(linalg::detail::elemAt<1>(a, k, 0));
}

template<class Real>
struct V3 {
    Real x, y, z;
};

template<class Real, class A>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() V3<Real> load3(const A& a)
{
    return { at3<Real>(a, 0), at3<Real>(a, 1), at3<Real>(a, 2) };
}

template<class Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() V3<Real> sub(V3<Real> a, V3<Real> b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

template<class Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Real dot(V3<Real> a, V3<Real> b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<class Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() V3<Real> cross(V3<Real> a, V3<Real> b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

/** @brief One slab: the `(lo, hi)` parameter interval of the axis, written to `t0`, `t1` ascending. */
template<class Real>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void slab(Real o, Real d, Real lo, Real hi, Real& t0, Real& t1)
{
    const bool ok  = d > Real(0) || d < Real(0);
    const Real ds  = ok ? d : Real(1);
    const Real inv = ok ? Real(1) / ds : aether::math::copysign(static_cast<Real>(slab_big), d);
    const Real a   = (lo - o) * inv;
    const Real b   = (hi - o) * inv;
    t0 = a < b ? a : b;
    t1 = a < b ? b : a;
}

} // namespace detail

/**
 * @brief Ray `o + t d` against the triangle `(a, b, c)`, Moller-Trumbore.
 * @return `[hit, t, u, v]`.
 */
template<class Real, class O, class D, class A, class B, class C>
AETHER_DEVICEHOST() Vec4<Real> rayTriangle(const O& o, const D& d, const A& a, const B& b, const C& c, Real tmax)
{
    using detail::cross;
    using detail::dot;
    const auto po = detail::load3<Real>(o), pd = detail::load3<Real>(d), pa = detail::load3<Real>(a);
    const auto e1 = detail::sub(detail::load3<Real>(b), pa), e2 = detail::sub(detail::load3<Real>(c), pa);
    const auto p    = cross(pd, e2);
    const Real det  = dot(e1, p);
    const bool ok   = det > Real(0) || det < Real(0);
    const Real dets = ok ? det : Real(1);
    const Real inv  = ok ? Real(1) / dets : Real(0);
    const auto s    = detail::sub(po, pa);
    const Real u    = inv * dot(s, p);
    const auto q    = cross(s, e1);
    const Real v    = inv * dot(pd, q);
    const Real t    = inv * dot(e2, q);
    const bool hit  = ok && u >= Real(0) && v >= Real(0) && u + v <= Real(1) && t >= Real(0) && t <= tmax;
    return { { hit ? Real(1) : Real(0), t, u, v } };
}

/**
 * @brief Ray `o + t d` against the axis-aligned box `[lo, hi]`, slab method.
 * @return `[hit, t_enter, t_exit, 0]`.
 */
template<class Real, class O, class D, class Lo, class Hi>
AETHER_DEVICEHOST() Vec4<Real> rayAabb(const O& o, const D& d, const Lo& lo, const Hi& hi, Real tmax)
{
    const auto po = detail::load3<Real>(o), pd = detail::load3<Real>(d);
    const auto l = detail::load3<Real>(lo), h = detail::load3<Real>(hi);
    Real x0, x1, y0, y1, z0, z1;
    detail::slab(po.x, pd.x, l.x, h.x, x0, x1);
    detail::slab(po.y, pd.y, l.y, h.y, y0, y1);
    detail::slab(po.z, pd.z, l.z, h.z, z0, z1);
    Real tin = x0, tout = x1;
    tin  = y0 > tin ? y0 : tin;
    tout = y1 < tout ? y1 : tout;
    tin  = z0 > tin ? z0 : tin;
    tout = z1 < tout ? z1 : tout;
    const bool hit = tin <= tout && tout >= Real(0) && tin <= tmax;
    return { { hit ? Real(1) : Real(0), tin, tout, Real(0) } };
}

/**
 * @brief Ray `o + t d` against the sphere of centre `c` and radius `r`.
 * @return `[hit, t0, t1, 0]`.
 */
template<class Real, class O, class D, class C>
AETHER_DEVICEHOST() Vec4<Real> raySphere(const O& o, const D& d, const C& c, Real r, Real tmax)
{
    using detail::dot;
    const auto pd = detail::load3<Real>(d);
    const auto oc = detail::sub(detail::load3<Real>(o), detail::load3<Real>(c));
    const Real qa = dot(pd, pd);
    const Real hb = dot(oc, pd);
    const Real qc = dot(oc, oc) - r * r;
    const Real disc = hb * hb - qa * qc;
    const bool ok   = qa > Real(0) && disc >= Real(0);
    const Real qas  = qa > Real(0) ? qa : Real(1);
    const Real ds   = ok ? disc : Real(1);
    const Real sq   = aether::math::sqrt(ds);
    const Real inv  = Real(1) / qas;
    const Real t0   = ok ? (-hb - sq) * inv : Real(0);
    const Real t1   = ok ? (-hb + sq) * inv : Real(0);
    const bool hit  = ok && t1 >= Real(0) && t0 <= tmax;
    return { { hit ? Real(1) : Real(0), t0, t1, Real(0) } };
}

} // namespace geometry
} // namespace aether
