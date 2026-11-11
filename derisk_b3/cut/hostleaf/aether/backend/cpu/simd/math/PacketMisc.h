// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketMisc.h
 * @brief Vectorised `fabs`, `copysign`, `fmax`, `fmin`, `fdim`, `sqrt`,
 *        `rsqrt`, `cbrt`, `hypot` for `aether::simd::Packet<double, W>`.
 *
 * fmax/fmin follow C: a NaN operand is ignored when the other is a number.
 * cbrt: x = m 2^(3q + r), m in [1,2); a quadratic seed for cbrt(m 2^r)
 * (Chebyshev fit on [1,2], 8e-4 relative, times 2^(r/3)), two Halley steps
 * and a final Newton correction with the residual y^3 - t carried in
 * double-double; exact scale by 2^q. hypot: max * sqrt(1 + (min/max)^2),
 * with hypot(+-inf, NaN) = +inf as C requires.
 */

#include "aether/backend/cpu/simd/math/PacketMathCore.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

template<class D>
AETHER_PM_INLINE D vfmax(D a, D b)
{
    D r = select(gt(b, a), b, a);
    r = select(isnan(a), b, r);
    return select(isnan(b), a, r);
}

template<class D>
AETHER_PM_INLINE D vfmin(D a, D b)
{
    D r = select(lt(b, a), b, a);
    r = select(isnan(a), b, r);
    return select(isnan(b), a, r);
}

template<class D>
AETHER_PM_INLINE D vfdim(D a, D b)
{
    const D r = select(gt(a, b), a - b, bc<D>(0.0));
    return select(isnan(a) | isnan(b), a + b, r);
}

/// 1/sqrt(x): the rounded quotient refined by one Newton step on the
/// residual 1 - x y^2 carried exactly (x is pre-scaled by 2^-200 above
/// 2^900 so the exact product stays finite).
template<class D>
AETHER_PM_INLINE D vrsqrt(D x)
{
    const auto big = gt(x, bc<D>(0x1p900));
    const D xs = select(big, x * 0x1p-200, x);
    const D y = 1.0 / sqrt(xs);
    D p, pl;
    twoProd(xs, y, p, pl);
    const D e = residual(p, y, bc<D>(1.0)) - pl * y;
    D r = fma(0.5 * y, e, y);
    r = select(big, r * 0x1p-100, r);
    return select(eq(x, bc<D>(0.0)) | isinf(x) | isnan(x) | lt(x, bc<D>(0.0)), y * select(big, bc<D>(0x1p-100), bc<D>(1.0)), r);
}

inline constexpr double kCbrtSeed[] = { -0.05836172077613443, 0.43356059182365825, 0.625687226564147 };

template<class D>
AETHER_PM_INLINE D vcbrt(D x)
{
    D ax = abs(x);
    const auto sub = lt(ax, bc<D>(0x1p-1022));
    ax = select(sub, ax * 0x1p54, ax);
    const D k = biasedExponent(ax) - select(sub, bc<D>(1077.0), bc<D>(1023.0));
    const D m = fromBits<D>((bits(ax) & 0x000FFFFFFFFFFFFFLL) | bits(bc<D>(1.0)));
    const D q = rintSmall((k - 1.0) * (1.0 / 3.0));
    const D rr = k - 3.0 * q;
    const auto r1 = eq(rr, bc<D>(1.0));
    const auto r2 = eq(rr, bc<D>(2.0));
    const D t = m * select(r1, bc<D>(2.0), select(r2, bc<D>(4.0), bc<D>(1.0)));
    D y = horner(m, kCbrtSeed) * select(r1, bc<D>(1.2599210498948732), select(r2, bc<D>(1.5874010519681996), bc<D>(1.0)));
    for (int i = 0; i < 2; ++i) {
        const D y3 = y * y * y;
        y = y * (y3 + (t + t)) / ((y3 + y3) + t);
    }
    D y2h, y2l, y3h, y3l;
    twoProd(y, y, y2h, y2l);
    twoProd(y2h, y, y3h, y3l);
    const D d = (y3h - t) + (y3l + y2l * y);
    y = y - d / (3.0 * y2h);
    const D r = copysign(y * pow2(q), x);
    return select(eq(ax, bc<D>(0.0)) | isinf(x) | isnan(x), x, r);
}

/// sqrt(x^2 + y^2) with the squares and their sum in double-double and a
/// double-double square root; operands are rescaled by 2^-+600 outside
/// [2^-500, 2^500] so neither the squares nor their error terms over- or
/// underflow.
template<class D>
AETHER_PM_INLINE D vhypot(D x, D y)
{
    const D ax = abs(x);
    const D ay = abs(y);
    const auto yb = gt(ay, ax);
    const D a = select(yb, ay, ax);
    const D b = select(yb, ax, ay);
    const auto big = gt(a, bc<D>(0x1p500));
    const auto tiny = lt(a, bc<D>(0x1p-500));
    const D sc = select(big, bc<D>(0x1p-600), select(tiny, bc<D>(0x1p600), bc<D>(1.0)));
    const D inv = select(big, bc<D>(0x1p600), select(tiny, bc<D>(0x1p-600), bc<D>(1.0)));
    const D as = a * sc, bs = b * sc;
    D A, Ae, B, Be, S, Se;
    twoProd(as, as, A, Ae);
    twoProd(bs, bs, B, Be);
    twoSum(A, B, S, Se);
    D h, hl;
    ddSqrt(S, Se + (Ae + Be), h, hl);
    D r = (h + hl) * inv;
    r = select(eq(a, bc<D>(0.0)), bc<D>(0.0), r);
    r = select(isnan(x) | isnan(y), x + y, r);
    return select(isinf(x) | isinf(y), bc<D>(__builtin_inf()), r);
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
