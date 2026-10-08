// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketSinCos.h
 * @brief Vectorised `sin`, `cos`, `sincos`, `tan` for
 *        `aether::simd::Packet<double, W>`, faithfully rounded.
 *
 * Reduction x = n*pi/2 + r, |r| <= pi/4, with r carried as a double-double
 * r + rl:
 *   - |x| <= 2^19 (vector path): Cody-Waite with pi/2 split 33 + 53 + 53
 *     bits; n*A is exact, n*B is taken exactly (two-product), so r keeps
 *     about 119 bits of pi/2 — enough for the closest approach of a double
 *     in this range to a multiple of pi/2 (2^-60.5, at x = 45.553...).
 *   - larger finite |x| (per lane, scalar): Payne-Hanek — the bits of 2/pi
 *     that matter for the lane's exponent (a 256-bit window of a 1280-bit
 *     table) times the 53-bit mantissa, in 64-bit integer limbs, giving the
 *     quadrant and 128 fraction bits.
 * Then near-minimax polynomials for sin and cos on |r| <= pi/4 (sin: r -
 * r^3/6 + r^5 S(r^2), degree 15 odd, fit error 2^-64; cos: degree 14 even,
 * 2^-59.5; Chebyshev-node fits with double coefficients) evaluated as a
 * double-double head + tail (the r^3/6 and r^2/2 terms exactly), the rl
 * correction folded into the tail. tan = sin/cos (or -cos/sin)
 * as a double-double quotient. Quadrant selection is branchless.
 * Inf/NaN -> NaN.
 */

#include <cstdint>

#include "aether/backend/cpu/simd/math/PacketMathCore.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

inline constexpr double kTwoOverPi = 0.6366197723675814;
// pi/2 = kPio2A + kPio2B + kPio2C + O(2^-142); kPio2A has 33 significant bits
// so n*kPio2A is exact for |n| < 2^20.
inline constexpr double kPio2A = 0x1.921fb544p+0;
inline constexpr double kPio2B = 0x1.0b4611a626331p-34;
inline constexpr double kPio2C = 0x1.1701b839a252p-88;
inline constexpr double kPio2DdHi = 0x1.921fb54442d18p+0, kPio2DdLo = 0x1.1a62633145c07p-54;
inline constexpr double kTrigVectorBound = 524288.0; // 2^19

/// 2/pi in 64-bit words, most significant first (bit 1 of word 0 is the
/// 2^-1 bit): enough for the Payne-Hanek window of any finite double.
inline constexpr std::uint64_t kTwoOverPiBits[20] = {
    0xA2F9836E4E441529ULL, 0xFC2757D1F534DDC0ULL, 0xDB6295993C439041ULL, 0xFE5163ABDEBBC561ULL,
    0xB7246E3A424DD2E0ULL, 0x06492EEA09D1921CULL, 0xFE1DEB1CB129A73EULL, 0xE88235F52EBB4484ULL,
    0xE99C7026B45F7E41ULL, 0x3991D639835339F4ULL, 0x9C845F8BBDF9283BULL, 0x1FF897FFDE05980FULL,
    0xEF2F118B5A0A6D1FULL, 0x6D367ECF27CB09B7ULL, 0x4F463F669E5FEA2DULL, 0x7527BAC7EBE5F17BULL,
    0x3D0739F78A5292EAULL, 0x6BFB5FB11F8D5D08ULL, 0x56033046FC7B6BABULL, 0xF0CFBC209AF4361DULL,
};

/// sin(r) = r - r^3/6 + r^5 S(z), z = r^2 <= (pi/4)^2. Rel. error of the
/// fit (r^5 S against sin(r)) 2^-64 with double coefficients.
inline constexpr double kSinS[] = {
    -7.595306678125121e-13, 1.6058684329175855e-10, -2.505210723603745e-08, 2.755731922232443e-06,
    -0.00019841269841268963, 0.008333333333333333,
};
inline constexpr double kMinusSixthHi = -0.16666666666666666, kMinusSixthLo = -9.25185853854297e-18;
/// cos(r) = 1 - z/2 + z^2 C(z).
inline constexpr double kCosC[] = {
    -1.1382623647474604e-11, 2.087614614655861e-09, -2.7557317271145144e-07, 2.480158729876456e-05,
    -0.0013888888888887398, 0.041666666666666664,
};

/// Cody-Waite reduction for |x| <= kTrigVectorBound: returns n (integer-
/// valued) with x = n*pi/2 + r + rl.
template<class D>
AETHER_PM_INLINE D trigReduce(D x, D& r, D& rl)
{
    const D n = rintSmall(x * kTwoOverPi);
    const D a = fma(-n, bc<D>(kPio2A), x); // exact (n*kPio2A exact, then Sterbenz)
    D p, pe;
    twoProd(n, bc<D>(kPio2B), p, pe);
    D s, se;
    twoSum(a, -p, s, se);
    const D t = (se - pe) - n * kPio2C;
    fastTwoSum(s, t, r, rl);
    return n;
}

__extension__ typedef unsigned __int128 PmU128;
__extension__ typedef __int128 PmI128;

/// Payne-Hanek reduction of a finite |x| >= 2^19 (scalar): returns the
/// quadrant n mod 4 with x = n*pi/2 + r + rl, |r| <= pi/4. `static`: each
/// TU keeps its own copy, compiled for its own ISA flags (an out-of-line
/// COMDAT shared between TUs built for different targets could run AVX-512
/// code on an SSE2-only path).
static inline int trigReduceLarge(double x, double& r, double& rl)
{
    using U128 = PmU128;
    const std::uint64_t b = __builtin_bit_cast(std::uint64_t, x);
    const bool neg = (b >> 63) != 0;
    const int e = static_cast<int>((b >> 52) & 0x7FF) - 1075; // |x| = m * 2^e
    const std::uint64_t m = (b & 0x000FFFFFFFFFFFFFULL) | 0x0010000000000000ULL;
    // Word j contributes m*T[j]*2^(e - 64(j+1)): a multiple of 4 (drops out
    // of the quadrant) when e - 64(j+1) >= 2, so start at the first word that
    // does not; four words keep 128+ fraction bits.
    const int j0 = e >= 2 ? (e - 2) / 64 : 0;
    const int s = e - 64 * (j0 + 1);
    const U128 p0 = (U128)m * kTwoOverPiBits[j0];
    const U128 p1 = (U128)m * kTwoOverPiBits[j0 + 1];
    const U128 p2 = (U128)m * kTwoOverPiBits[j0 + 2];
    const U128 p3 = (U128)m * kTwoOverPiBits[j0 + 3];
    std::uint64_t L[5]; // L[0] least significant; value = P * 2^(s - 192)
    L[0] = (std::uint64_t)p3;
    U128 t = (U128)(std::uint64_t)p2 + (std::uint64_t)(p3 >> 64);
    L[1] = (std::uint64_t)t;
    t = (U128)(std::uint64_t)p1 + (std::uint64_t)(p2 >> 64) + (std::uint64_t)(t >> 64);
    L[2] = (std::uint64_t)t;
    t = (U128)(std::uint64_t)p0 + (std::uint64_t)(p1 >> 64) + (std::uint64_t)(t >> 64);
    L[3] = (std::uint64_t)t;
    L[4] = (std::uint64_t)(p0 >> 64) + (std::uint64_t)(t >> 64);
    const int k = 192 - s; // bit index of the 2^0 weight
    auto get64 = [&](int pos) -> std::uint64_t {
        const int w = pos >> 6, o = pos & 63;
        const std::uint64_t lo = w < 5 ? L[w] : 0;
        const std::uint64_t hi = w + 1 < 5 ? L[w + 1] : 0;
        return o ? (lo >> o) | (hi << (64 - o)) : lo;
    };
    int q = static_cast<int>(get64(k) & 3);
    const U128 f = ((U128)get64(k - 64) << 64) | get64(k - 128); // fraction * 2^128
    q = (q + static_cast<int>(f >> 127)) & 3;                     // round to nearest quadrant
    const PmI128 fs = (PmI128)f;                              // signed fraction in [-1/2, 1/2)
    const double fh = (double)fs;
    const double fl = (double)(fs - (PmI128)fh);
    double rh, rlo;
    ddMul(fh * 0x1p-128, fl * 0x1p-128, kPio2DdHi, kPio2DdLo, rh, rlo);
    fastTwoSum(rh, rlo, r, rl);
    if (neg) {
        r = -r;
        rl = -rl;
        q = (4 - q) & 3;
    }
    return q;
}

/// sin(r + rl) and cos(r + rl) as double-doubles, |r| <= pi/4.
template<class D>
AETHER_PM_INLINE void sincosKernel(D r, D rl, D& sh, D& sl, D& ch, D& cl)
{
    D z, ze;
    twoProd(r, r, z, ze);
    // sin: r + (-r^3/6 in double-double) + r^5 S(z) + rl cos(r)
    D c3, c3e;
    twoProd(r, z, c3, c3e);
    c3e = c3e + r * ze;
    D t3, t3e;
    twoProd(c3, bc<D>(kMinusSixthHi), t3, t3e);
    t3e = t3e + (c3 * kMinusSixthLo + c3e * kMinusSixthHi);
    D s0, s0e;
    twoSum(r, t3, s0, s0e);
    const D st = (s0e + t3e) + fma(c3 * z, horner(z, kSinS), rl * (1.0 - 0.5 * z));
    fastTwoSum(s0, st, sh, sl);
    const D hz = 0.5 * z;
    const D w = 1.0 - hz;
    const D ct = ((((1.0 - w) - hz) - 0.5 * ze) - r * rl) + (z * z) * horner(z, kCosC);
    fastTwoSum(w, ct, ch, cl);
}

/// sin, cos (rounded) and tan of the reduced argument in quadrant q.
template<class D>
AETHER_PM_INLINE void trigFinish(LaneMask<D> q, D r, D rl, D* sOut, D* cOut, D* tOut)
{
    D sh, sl, ch, cl;
    sincosKernel(r, rl, sh, sl, ch, cl);
    const auto odd = inz(q & 1);
    if (sOut)
        *sOut = negateIf(inz(q & 2), select(odd, ch + cl, sh + sl));
    if (cOut)
        *cOut = negateIf(inz((q + 1) & 2), select(odd, sh + sl, ch + cl));
    if (tOut) {
        D th, tl;
        ddDiv(select(odd, ch, sh), select(odd, cl, sl), select(odd, sh, ch), select(odd, sl, cl), th, tl);
        *tOut = negateIf(odd, th + tl);
    }
}

/// Scalar large-argument path: 0 = sin, 1 = cos, 2 = tan (`static`, as
/// trigReduceLarge).
static inline double trigLarge(double x, int which)
{
    double r, rl, v;
    const std::int64_t q = trigReduceLarge(x, r, rl);
    trigFinish<double>(q, r, rl, which == 0 ? &v : nullptr, which == 1 ? &v : nullptr, which == 2 ? &v : nullptr);
    return v;
}

template<class D>
AETHER_PM_INLINE void trigSmall(D x, D* s, D* c, D* t)
{
    D r, rl;
    const D n = trigReduce(x, r, rl);
    trigFinish(toInt(n), r, rl, s, c, t);
}

template<class D>
AETHER_PM_INLINE LaneMask<D> trigNeedsScalar(D x)
{
    const D ax = abs(x);
    return gt(ax, bc<D>(kTrigVectorBound)) & lt(ax, bc<D>(__builtin_inf()));
}

template<class D>
AETHER_PM_INLINE D vsin(D x)
{
    D s;
    trigSmall<D>(x, &s, nullptr, nullptr);
    s = select(lt(abs(x), bc<D>(0x1p-26)), x, s);
    const auto big = trigNeedsScalar(x);
    if (any<D>(big))
        s = laneFallback<D>(big, x, s, [](double v) { return trigLarge(v, 0); });
    return s;
}

template<class D>
AETHER_PM_INLINE D vcos(D x)
{
    D c;
    trigSmall<D>(x, nullptr, &c, nullptr);
    const auto big = trigNeedsScalar(x);
    if (any<D>(big))
        c = laneFallback<D>(big, x, c, [](double v) { return trigLarge(v, 1); });
    return c;
}

template<class D>
AETHER_PM_INLINE void vsincos(D x, D& s, D& c)
{
    trigSmall<D>(x, &s, &c, nullptr);
    s = select(lt(abs(x), bc<D>(0x1p-26)), x, s);
    const auto big = trigNeedsScalar(x);
    if (any<D>(big)) {
        s = laneFallback<D>(big, x, s, [](double v) { return trigLarge(v, 0); });
        c = laneFallback<D>(big, x, c, [](double v) { return trigLarge(v, 1); });
    }
}

template<class D>
AETHER_PM_INLINE D vtan(D x)
{
    D t;
    trigSmall<D>(x, nullptr, nullptr, &t);
    t = select(lt(abs(x), bc<D>(0x1p-27)), x, t);
    const auto big = trigNeedsScalar(x);
    if (any<D>(big))
        t = laneFallback<D>(big, x, t, [](double v) { return trigLarge(v, 2); });
    return t;
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
