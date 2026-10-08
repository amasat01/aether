// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketArcTrig.h
 * @brief Vectorised `atan`, `atan2`, `asin`, `acos` for
 *        `aether::simd::Packet<double, W>`, faithfully rounded.
 *
 * atan: reduce t = |x| (a double-double for atan2) to |u| <= tan(pi/8)
 * (u = t, (t-1)/(t+1) with a pi/4 offset, or -1/t with a pi/2 offset; u
 * and the offsets in double-double), then an odd near-minimax polynomial
 * u + u z A(z), degree 23 (Chebyshev-node fit, relative error 2^-58.1 with
 * double coefficients); the offset, u and the polynomial are summed in
 * double-double and rounded once.
 * atan2: t = |y|/|x| as a double-double (exact remainder), atan as above,
 * pi - atan in double-double for x < 0, and the full IEEE special-value
 * table (signed zeros, infinities, NaN).
 * asin/acos: |x| <= 1/2: asin = x + x z R(z) (z = x^2, degree-13 fit of R,
 * relative error 2^-58.6 of asin), acos = pi/2 - asin in double-double;
 * |x| > 1/2: with z = (1 - |x|)/2 (exact) and s = sqrt(z) in double-double,
 * asin = pi/2 - 2 asin(s), acos = 2 asin(s) or pi - 2 asin(s).
 */

#include "aether/backend/cpu/simd/math/PacketMathCore.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

inline constexpr double kPiHi = 3.141592653589793, kPiLo = 1.2246467991473532e-16;
inline constexpr double kPio2Hi = 1.5707963267948966, kPio2Lo = 6.123233995736766e-17;
inline constexpr double kPio4Hi = 0.7853981633974483, kPio4Lo = 3.061616997868383e-17;

/// atan(u) = u + u z A(z), z = u^2 <= tan(pi/8)^2.
inline constexpr double kAtanA[] = {
    0.016284375050043393, -0.03456931506332609, 0.045515447340221615, -0.05230443640686948,
    0.05878927554473807, -0.0666642476020052, 0.07692296368042296, -0.09090908753259112,
    0.11111111105150588, -0.14285714285659779, 0.19999999999999804, -0.3333333333333333,
};

/// asin(x) = x + x z R(z), z = x^2 <= 1/4.
inline constexpr double kAsinR[] = {
    0.02961201126495512, -0.01924167174674304, 0.019554513336123378, 0.0030448799094556773,
    0.009319560794767446, 0.009621842970100282, 0.011566459612121669, 0.01396378001220357,
    0.017352816540325496, 0.02237215744350722, 0.03038194447553234, 0.044642857142551895,
    0.07500000000000118, 0.16666666666666666,
};

/// atan(th + tl) for th >= 0 (th = +inf allowed) as ah + al.
template<class D>
AETHER_PM_INLINE void atanDD(D th, D tl, D& ah, D& al)
{
    const auto big = gt(th, bc<D>(2.414213562373095));
    const auto mid = gt(th, bc<D>(0.41421356237309503)) & ~big;
    // big: u = -1/t
    const D bh = -1.0 / th;
    D bl = (residual(bh, th, bc<D>(-1.0)) - bh * tl) / th;
    bl = select(gt(th, bc<D>(0x1p60)), bc<D>(0.0), bl); // below every rounding of pi/2 + u
    // mid: u = (t - 1)/(t + 1)
    D nh, nl, dh, dl;
    twoSum(th, bc<D>(-1.0), nh, nl);
    twoSum(th, bc<D>(1.0), dh, dl);
    D mh, ml;
    ddDiv(nh, nl + tl, dh, dl + tl, mh, ml);
    const D uh = select(big, bh, select(mid, mh, th));
    const D ul = select(big, bl, select(mid, ml, tl));
    const D baseHi = select(big, bc<D>(kPio2Hi), select(mid, bc<D>(kPio4Hi), bc<D>(0.0)));
    const D baseLo = select(big, bc<D>(kPio2Lo), select(mid, bc<D>(kPio4Lo), bc<D>(0.0)));
    const D z = uh * uh;
    const D p = (uh * z) * horner(z, kAtanA);
    D s, se;
    twoSum(baseHi, uh, s, se);
    fastTwoSum(s, se + ((p + baseLo) + (ul - ul * z)), ah, al);
}

template<class D>
AETHER_PM_INLINE D vatan(D x)
{
    D ah, al;
    atanDD(abs(x), bc<D>(0.0), ah, al);
    return select(isnan(x), x, copysign(ah + al, x));
}

template<class D>
AETHER_PM_INLINE D vatan2(D y, D x)
{
    D ay = abs(y);
    D ax = abs(x);
    // Power-of-two rescale so the remainder below neither overflows nor
    // loses bits to underflow; the quotient is unchanged.
    const auto yb = gt(ay, ax);
    const D mx = select(yb, ay, ax);
    const D mn = select(yb, ax, ay);
    const D sc = select(gt(mx, bc<D>(0x1p900)), bc<D>(0x1p-600),
        select(lt(mn, bc<D>(0x1p-900)) & lt(mx, bc<D>(0x1p400)), bc<D>(0x1p600), bc<D>(1.0)));
    ay = ay * sc;
    ax = ax * sc;
    D t = ay / ax;
    D tl = residual(t, ax, ay) / ax;
    const auto zz = eq(ax, bc<D>(0.0)) & eq(ay, bc<D>(0.0));
    const auto ii = isinf(ax) & isinf(ay);
    t = select(zz, bc<D>(0.0), select(ii, bc<D>(1.0), t));
    tl = select(zz | ii | lt(t, bc<D>(0x1p-900)) | gt(t, bc<D>(0x1p60)), bc<D>(0.0), tl);
    D ah, al;
    atanDD(t, tl, ah, al);
    D s, se;
    twoSum(bc<D>(kPiHi), -ah, s, se);
    const D a = select(signbit(x), s + (se + (kPiLo - al)), ah + al);
    const D r = copysign(a, y);
    return select(isnan(x) | isnan(y), x + y, r);
}

/// asin(sqrt(z)) for z = (1 - |x|)/2 in [0, 1/4], as sh + sl.
template<class D>
AETHER_PM_INLINE void asinOfSqrt(D z, D& ah, D& al)
{
    D sh, sl;
    ddSqrt(z, bc<D>(0.0), sh, sl);
    sl = select(eq(z, bc<D>(0.0)), bc<D>(0.0), sl);
    const D p = (sh * z) * horner(z, kAsinR);
    fastTwoSum(sh, sl + p, ah, al);
}

template<class D>
AETHER_PM_INLINE D vasin(D x)
{
    const D ax = abs(x);
    const D z0 = ax * ax;
    const D small = ax + (ax * z0) * horner(z0, kAsinR);
    const D z = (1.0 - ax) * 0.5;
    D ah, al;
    asinOfSqrt(z, ah, al);
    D h, he;
    twoSum(bc<D>(kPio2Hi), -2.0 * ah, h, he);
    const D large = h + (he + (kPio2Lo - 2.0 * al));
    const D r = select(le(ax, bc<D>(0.5)), small, large);
    return select(isnan(x), x, copysign(r, x));
}

template<class D>
AETHER_PM_INLINE D vacos(D x)
{
    const D ax = abs(x);
    const D z0 = x * x;
    D h, he;
    twoSum(bc<D>(kPio2Hi), -x, h, he);
    const D small = h + (he + (kPio2Lo - (x * z0) * horner(z0, kAsinR)));
    const D z = (1.0 - ax) * 0.5;
    D ah, al;
    asinOfSqrt(z, ah, al);
    const D pos = 2.0 * (ah + al);
    twoSum(bc<D>(kPiHi), -2.0 * ah, h, he);
    const D neg = h + (he + (kPiLo - 2.0 * al));
    const D r = select(le(ax, bc<D>(0.5)), small, select(signbit(x), neg, pos));
    return select(isnan(x), x, r);
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
