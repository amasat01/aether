// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketExpLog.h
 * @brief Vectorised `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`,
 *        `log10`, `log1p` and `pow` for `aether::simd::Packet<double, W>`
 *        (W = 1, 2, 4, 8).
 *
 * Scheme:
 *   1. Cody-Waite argument reduction — x = n*ln2 + r with |r| <= ln2/2,
 *      ln2 split so that n*ln2_hi is exact;
 *   2. a near-minimax polynomial on the reduced interval (Chebyshev-node
 *      fits in 60-digit arithmetic; the relative error of each fit, with
 *      its coefficients rounded to double, is quoted beside it);
 *   3. reconstruction — scale by 2^n through the exponent field, in two
 *      factors so that subnormal results and overflow come out right.
 *
 * `log` is the atanh form log(m) = 2*atanh((m-1)/(m+1)) carried in
 * double-double, so `log2`/`log10`/`log1p` and `pow` can consume an
 * extended-precision logarithm; `pow` is `exp(y*log|x|)` with the product
 * also in double-double, so the amplification by |y*log x| does not turn
 * into lost ULPs.
 *
 * `exp` and `log` themselves (the two entry points that need no extended
 * precision) take a faster, table-driven route: a 128-entry table of
 * 2^(i/128) (exp) or of (1/c, log c) (log), a degree-5 / degree-7
 * polynomial on the short remainder, no division, the table read by a
 * gather. The scheme is the one of ARM's optimized-routines
 * (https://github.com/ARM-software/optimized-routines, math/exp.c and
 * math/log.c, MIT OR Apache-2.0 WITH LLVM-exception); see NOTICE. Lanes the
 * fast path does not cover (|x| >= 700, NaN for exp; subnormal, <= 0,
 * infinite, NaN and |x - 1| < 1/16 for log) are redone by the double-double
 * scheme above, so every special value keeps its former result. `log` needs
 * a true FMA (exact z/c - 1) and keeps the double-double scheme without one.
 *
 * Accuracy (vs glibc `std::`, every width): see `PacketMath.h`.
 */

#include "aether/backend/cpu/simd/math/PacketExpLogTables.h"
#include "aether/backend/cpu/simd/math/PacketMathCore.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

// ln2 split so n*kLn2Hi is exact for |n| < 2^21 (32 significant bits).
inline constexpr double kLn2Hi = 0.6931471806019545;
inline constexpr double kLn2Lo = -4.2009150726810846e-11;
inline constexpr double kLog2e = 1.4426950408889634;
// Double-double constants (hi, lo).
inline constexpr double kLn2DdHi = 0.6931471805599453, kLn2DdLo = 2.3190468138462996e-17;
inline constexpr double kLn10DdHi = 2.302585092994046, kLn10DdLo = -2.1707562233822494e-16;
inline constexpr double kLog2eDdHi = 1.4426950408889634, kLog2eDdLo = 2.0355273740931033e-17;
inline constexpr double kLog10eDdHi = 0.4342944819032518, kLog10eDdLo = 1.098319650216765e-17;
inline constexpr double kTwoThirdsHi = 0.6666666666666666, kTwoThirdsLo = 3.700743415417188e-17;

/// exp(r) = 1 + r + r^2/2 + r^3 q(r), |r| <= ln2/2. Rel. error of the fit 2^-61.4.
inline constexpr double kExpQ[] = {
    2.0914679376583935e-09, 2.510520637395701e-08, 2.7557273661348637e-07, 2.7557255425746435e-06,
    2.4801587325533363e-05, 0.00019841269874800493, 0.0013888888888883752, 0.008333333333326141,
    0.04166666666666667, 0.1666666666666667,
};

/// expm1(r) = r + r^2/2 + r^3 q(r), |r| <= ln2/2. Rel. error of the fit 2^-58.5.
inline constexpr double kExpm1Q[] = {
    2.091122972975856e-09, 2.5100375832561234e-08, 2.755728298405588e-07, 2.7557268480310024e-06,
    2.4801587317135164e-05, 0.00019841269863040545, 0.0013888888888886554, 0.008333333333330065,
    0.041666666666666664, 0.16666666666666669,
};

/// log(m) = 2s + (2/3)s^3 + (2/5)s^5 + 2s z^3 h(z), s = (m-1)/(m+1),
/// z = s^2 <= 0.0295 (m in [sqrt2/2, sqrt2]). Rel. error of the fit of h
/// 2^-54 (with double coefficients), i.e. 2^-72 of log(m).
inline constexpr double kLogH[] = {
    0.053097569132606826, 0.05236334295140812, 0.05883029381949927, 0.06666657258257504,
    0.0769230776320522, 0.09090909090647106, 0.1111111111111148, 0.14285714285714285,
};
inline constexpr double kTwoFifthsHi = 0.4, kTwoFifthsLo = -2.2204460492503132e-17;

/// Cody-Waite reduction hi + lo = n*ln2 + r + rl (|r| <= ln2/2 + tiny),
/// r + rl exact to about 2^-77 for |hi| < 2^11.
template<class D>
AETHER_PM_INLINE D expReduce(D hi, D lo, D& r, D& rl)
{
    const D n = rintSmall(hi * kLog2e);
    const D a = fma(-n, bc<D>(kLn2Hi), hi); // exact: n*kLn2Hi exact, then Sterbenz
    const D b = fma(-n, bc<D>(kLn2Lo), lo);
    twoSum(a, b, r, rl);
    return n;
}

/// exp(r + rl) for the reduced argument, as ph + pl (in [0.7, 1.42]).
template<class D>
AETHER_PM_INLINE void expKernel(D r, D rl, D& ph, D& pl)
{
    D z, ze;
    twoProd(r, r, z, ze);
    D s, se, t, te;
    twoSum(bc<D>(1.0), r, s, se);
    twoSum(s, 0.5 * z, t, te);
    const D tail = ((se + te) + 0.5 * ze) + fma(z * r, horner(r, kExpQ), rl * (1.0 + r));
    fastTwoSum(t, tail, ph, pl);
}

/// exp(hi + lo), |lo| << ulp(hi). Over/underflow (incl. subnormal results)
/// handled; NaN in -> garbage out (callers select NaN back in).
template<class D>
AETHER_PM_INLINE D expCore(D hi, D lo)
{
    D xc = select(gt(hi, bc<D>(710.0)), bc<D>(710.0), hi);
    xc = select(lt(xc, bc<D>(-746.0)), bc<D>(-746.0), xc);
    lo = select(eq(xc, hi), lo, bc<D>(0.0));
    D r, rl, ph, pl;
    const D n = expReduce(xc, lo, r, rl);
    expKernel(r, rl, ph, pl);
    const D n1 = rintSmall(n * 0.5);
    const D n2 = n - n1;
    return (ph * pow2(n1)) * pow2(n2);
}

/// Double-double `exp` (full range, NaN, over/underflow, subnormal results).
template<class D>
AETHER_PM_INLINE D vexpSlow(D x)
{
    return select(isnan(x), x, expCore(x, bc<D>(0.0)));
}

/// Modular 64-bit lane arithmetic on bit patterns, done in the unsigned lane
/// type so wrap-around is defined (no signed overflow); bit-identical to the
/// two's-complement result.
template<class D>
AETHER_PM_INLINE LaneMask<D> uadd(LaneMask<D> a, LaneMask<D> b)
{
    using U = typename Lanes<D>::U;
    return (LaneMask<D>)((U)a + (U)b);
}
template<class D>
AETHER_PM_INLINE LaneMask<D> usub(LaneMask<D> a, LaneMask<D> b)
{
    using U = typename Lanes<D>::U;
    return (LaneMask<D>)((U)a - (U)b);
}
template<class D>
AETHER_PM_INLINE LaneMask<D> ushl(LaneMask<D> a, int n)
{
    using U = typename Lanes<D>::U;
    return (LaneMask<D>)((U)a << n);
}

/// Table read: lane i of the result is `base[idx[i]]` (a hardware gather
/// where the target has one).
template<class D>
AETHER_PM_INLINE LaneMask<D> gather64(const std::uint64_t* base, LaneMask<D> idx)
{
    if constexpr (kScalar<D>) {
        return static_cast<std::int64_t>(base[idx]);
    }
#if defined(__AVX512F__)
    else if constexpr (std::is_same_v<D, __m512d>) {
        return (LaneMask<D>)_mm512_i64gather_epi64((__m512i)idx, static_cast<const void*>(base), 8);
    }
#endif
#if defined(__AVX2__)
    else if constexpr (std::is_same_v<D, __m256d>) {
        return (LaneMask<D>)_mm256_i64gather_epi64(reinterpret_cast<const long long*>(base), (__m256i)idx, 8);
    }
#endif
    else {
        LaneMask<D> r = {};
        for (std::size_t i = 0; i < Lanes<D>::width; ++i)
            r[i] = static_cast<std::int64_t>(base[idx[i]]);
        return r;
    }
}

/// Table-driven exp for |x| < 700: x = (k/128) ln2 + r, |r| <= ln2/256,
/// exp(x) = 2^(k/128) (1 + tail_i + p(r)) with the 2^(i/128) table.
template<class D>
AETHER_PM_INLINE D vexp(D x)
{
    const D shift = bc<D>(0x1.8p52);
    D kd = x * bc<D>(0x1.71547652b82fep0 * 128.0) + shift;
    const LaneMask<D> ki = bits(kd);
    kd = kd - shift;
    const D r = (x + kd * bc<D>(-0x1.62e42fefa0000p-8)) + kd * bc<D>(-0x1.cf79abc9e3b3ap-47);
    const LaneMask<D> idx = (ki & 127) << 1;
    const D tail = fromBits<D>(gather64<D>(kExpTab, idx));
    const D scale = fromBits<D>(uadd<D>(gather64<D>(kExpTab, idx + 1), ushl<D>(ki, 45)));
    const D r2 = r * r;
    const D tmp = ((tail + r) + r2 * (bc<D>(0x1.ffffffffffdbdp-2) + r * bc<D>(0x1.555555555543cp-3)))
        + (r2 * r2) * (bc<D>(0x1.55555cf172b91p-5) + r * bc<D>(0x1.1111167a4d017p-7));
    const D res = fma(scale, tmp, scale);
    const auto bad = ~lt(abs(x), bc<D>(700.0)); // |x| >= 700, inf, NaN
    if (any<D>(bad))
        return select(bad, vexpSlow(x), res);
    return res;
}

/// exp(x * (cHi + cLo)) with the product carried in double-double.
template<class D>
AETHER_PM_INLINE D expScaled(D x, double bound, double cHi, double cLo)
{
    D xc = select(gt(x, bc<D>(bound)), bc<D>(bound), x);
    xc = select(lt(xc, bc<D>(-bound)), bc<D>(-bound), xc);
    D hi, lo;
    twoProd(xc, bc<D>(cHi), hi, lo);
    lo = fma(xc, bc<D>(cLo), lo);
    return select(isnan(x), x, expCore(hi, lo));
}

template<class D>
AETHER_PM_INLINE D vexp2(D x)
{
    return expScaled(x, 1100.0, kLn2DdHi, kLn2DdLo);
}

template<class D>
AETHER_PM_INLINE D vexp10(D x)
{
    return expScaled(x, 400.0, kLn10DdHi, kLn10DdLo);
}

/// expm1(x) as eh + el for |x| <= 45 (about 2^-100 relative):
/// 2^n (1 + E) - 1 = (2^n - 1) + 2^n E, E = expm1(r + rl) in double-double.
template<class D>
AETHER_PM_INLINE void expm1DD(D x, D& eh, D& el)
{
    D r, rl;
    const D n = expReduce(x, bc<D>(0.0), r, rl);
    D z, ze;
    twoProd(r, r, z, ze);
    D e, ee;
    twoSum(r, 0.5 * z, e, ee);
    const D tail = (ee + 0.5 * ze) + fma(z * r, horner(r, kExpm1Q), rl * (1.0 + r));
    fastTwoSum(e, tail, e, ee);
    const D scale = pow2(n);
    D a, ae, sh, sl;
    twoSum(scale, bc<D>(-1.0), a, ae);
    twoSum(a, scale * e, sh, sl);
    fastTwoSum(sh, (sl + ae) + scale * ee, eh, el);
}

template<class D>
AETHER_PM_INLINE D vexpm1(D x)
{
    D xc = select(gt(x, bc<D>(40.0)), bc<D>(40.0), x);
    xc = select(lt(xc, bc<D>(-40.0)), bc<D>(-40.0), xc);
    D eh, el;
    expm1DD(xc, eh, el);
    D res = eh + el;
    res = select(gt(x, bc<D>(40.0)), expCore(x, bc<D>(0.0)) - 1.0, res);
    res = select(lt(x, bc<D>(-40.0)), bc<D>(-1.0), res);
    res = select(lt(abs(x), bc<D>(0x1p-54)), x, res); // tiny, +-0
    return select(isnan(x), x, res);
}

/// log(x) as an unevaluated double-double hi + lo (about 2^-68 relative),
/// for finite x > 0 (normal or subnormal). Other inputs -> garbage (callers
/// select).
template<class D>
AETHER_PM_INLINE void logDD(D x, D& hi, D& lo)
{
    const auto sub = lt(x, bc<D>(0x1p-1022));
    x = select(sub, x * 0x1p54, x);
    D k = biasedExponent(x) - select(sub, bc<D>(1077.0), bc<D>(1023.0));
    D m = fromBits<D>((bits(x) & 0x000FFFFFFFFFFFFFLL) | bits(bc<D>(1.0)));
    const auto big = gt(m, bc<D>(1.4142135623730951));
    m = select(big, m * 0.5, m);
    k = select(big, k + 1.0, k);

    // s = f / (2 + f) in double-double (f = m - 1 is exact).
    const D f = m - 1.0;
    const D dh = 2.0 + f;
    const D dl = (2.0 - dh) + f;
    const D sh = f / dh;
    D ph, pl;
    twoProd(sh, dh, ph, pl);
    const D sl = (((f - ph) - pl) - sh * dl) / dh;

    // z = s^2, s^3 and s^5 in double-double; (2/3) s^3 and (2/5) s^5.
    D zh, zl;
    twoProd(sh, sh, zh, zl);
    zl = zl + 2.0 * sh * sl;
    D s3h, s3l;
    twoProd(zh, sh, s3h, s3l);
    s3l = s3l + (zl * sh + zh * sl);
    D t3h, t3l;
    twoProd(bc<D>(kTwoThirdsHi), s3h, t3h, t3l);
    t3l = t3l + (kTwoThirdsLo * s3h + kTwoThirdsHi * s3l);
    D s5h, s5l;
    twoProd(s3h, zh, s5h, s5l);
    s5l = s5l + (s3l * zh + s3h * zl);
    D t5h, t5l;
    twoProd(bc<D>(kTwoFifthsHi), s5h, t5h, t5l);
    t5l = t5l + (kTwoFifthsLo * s5h + kTwoFifthsHi * s5l);

    // 2 s z^3 h(z): <= 4e-6 of log(m), plain double is enough.
    const D t7 = (2.0 * sh) * (zh * zh * zh) * horner(zh, kLogH);

    // k ln2 + 2s + t3 + t5 + t7, the large parts summed exactly.
    D H, E1, E2, E3, E4;
    twoSum(k * kLn2Hi, 2.0 * sh, H, E1);
    twoSum(H, t3h, H, E2);
    twoSum(H, t5h, H, E3);
    twoSum(H, t7, H, E4);
    const D rest = ((E1 + E2) + (E3 + E4)) + ((2.0 * sl + t3l) + (t5l + k * kLn2Lo));
    fastTwoSum(H, rest, hi, lo);
}

/// Shared special-value handling of the log family.
template<class D>
AETHER_PM_INLINE D logSpecials(D x, D r)
{
    const D inf = bc<D>(__builtin_inf());
    r = select(lt(x, bc<D>(0.0)), bc<D>(__builtin_nan("")), r);
    r = select(eq(x, bc<D>(0.0)), -inf, r);
    r = select(eq(x, inf), inf, r);
    return select(isnan(x), x, r);
}

/// Double-double `log` (every input).
template<class D>
AETHER_PM_INLINE D vlogSlow(D x)
{
    D h, l;
    logDD(x, h, l);
    return logSpecials(x, h);
}

/// Table-driven log: x = 2^k z, z in [0.6875, 1.375) in cell i with
/// c_i ~ z; log x = k ln2 + log c_i + log1p(z/c_i - 1), z/c_i - 1 exact
/// (FMA), its log1p a degree-7 polynomial, the sum kept as hi + lo.
template<class D>
AETHER_PM_INLINE D vlog(D x)
{
#if defined(__FMA__) || defined(__AVX512F__)
    const LaneMask<D> ix = bits(x);
    const LaneMask<D> tmp = usub<D>(ix, ibc<D>(0x3fe6000000000000LL));
    const LaneMask<D> idx = (srl<D>(tmp, 45) & 127) << 1;
    const LaneMask<D> k = tmp >> 52; // arithmetic: the exponent of x relative to z
    const D kd = fromBits<D>(k + bits(bc<D>(kRoundMagic))) - bc<D>(kRoundMagic);
    const D z = fromBits<D>(usub<D>(ix, tmp & ibc<D>(static_cast<std::int64_t>(0xfff0000000000000ULL))));
    const D invc = fromBits<D>(gather64<D>(kLogTab, idx));
    const D logc = fromBits<D>(gather64<D>(kLogTab, idx + 1));
    const D r = fma(z, invc, bc<D>(-1.0));
    const D w = kd * bc<D>(0x1.62e42fefa3800p-1) + logc; // kd * ln2hi exact
    const D hi = w + r;
    const D lo = ((w - hi) + r) + kd * bc<D>(0x1.ef35793c76730p-45);
    const D r2 = r * r;
    const D p = bc<D>(1.0 / 3.0) + r * bc<D>(-0.25) + r2 * (bc<D>(0.2) + r * bc<D>(-1.0 / 6.0) + r2 * bc<D>(1.0 / 7.0));
    const D res = ((lo + r2 * bc<D>(-0.5)) + (r * r2) * p) + hi;
    const auto bad = ~(ge(x, bc<D>(0x1p-1022)) & le(x, bc<D>(1.7976931348623157e308)))
        | lt(abs(x - bc<D>(1.0)), bc<D>(0x1p-4)); // <= 0, subnormal, inf, NaN, near 1
    if (any<D>(bad))
        return select(bad, vlogSlow(x), res);
    return res;
#else
    return vlogSlow(x);
#endif
}

/// log(x) * (cHi + cLo), product in double-double.
template<class D>
AETHER_PM_INLINE D logScaled(D x, double cHi, double cLo)
{
    D h, l;
    logDD(x, h, l);
    D ph, pe;
    twoProd(h, bc<D>(cHi), ph, pe);
    const D r = ph + ((pe + h * cLo) + l * cHi);
    return logSpecials(x, r);
}

template<class D>
AETHER_PM_INLINE D vlog2(D x)
{
    return logScaled(x, kLog2eDdHi, kLog2eDdLo);
}

template<class D>
AETHER_PM_INLINE D vlog10(D x)
{
    return logScaled(x, kLog10eDdHi, kLog10eDdLo);
}

/// log1p(wh + wl) as h + l (about 2^-68 relative), for wh + wl > -1
/// finite; the rounding of 1 + w is carried exactly and folded back.
template<class D>
AETHER_PM_INLINE void log1pDD(D wh, D wl, D& h, D& l)
{
    D u, ue;
    twoSum(bc<D>(1.0), wh, u, ue);
    const D num = ue + wl;
    const D c = num / u;
    // (beyond 2^512, c is far below the result's last bit and the exact
    // product split of a non-FMA build would overflow)
    const D cl = select(lt(u, bc<D>(0x1p512)), residual(c, u, num) / u, bc<D>(0.0));
    D lh, ll;
    logDD(u, lh, ll);
    D sh, se;
    twoSum(lh, c, sh, se);
    fastTwoSum(sh, (se + ll) + (cl - 0.5 * c * c), h, l);
}

template<class D>
AETHER_PM_INLINE D vlog1p(D x)
{
    D h, l;
    log1pDD(x, bc<D>(0.0), h, l);
    return logSpecials(1.0 + x, select(eq(x, bc<D>(0.0)), x, h + l));
}

/**
 * pow(x, y) = exp(y * log|x|) with C99/IEEE special cases:
 *   pow(x, +-0) = 1 and pow(1, y) = 1 for any x / y (NaN included);
 *   pow(x, 1) = x;
 *   x < 0 finite and y finite non-integer -> NaN;
 *   x < 0 and y an odd integer -> negative result (also -0 / -inf bases);
 *   pow(+-0, y): y < 0 -> +-inf, y > 0 -> +-0 (sign only for odd y);
 *   pow(+-inf, y): y > 0 -> +-inf, y < 0 -> +-0 (sign only for odd y);
 *   pow(x, +-inf): |x| == 1 -> 1, (|x| > 1) == (y > 0) -> +inf, else +0;
 *   any other NaN operand -> NaN; overflow -> +-inf, underflow -> +-0
 *   (subnormal results are produced, not flushed).
 * floating-point exception flags are not raised the way libm raises them.
 */
template<class D>
AETHER_PM_INLINE D vpow(D x, D y)
{
    const D inf = bc<D>(__builtin_inf());
    const D ax = abs(x);
    D lh, ll;
    logDD(ax, lh, ll);
    // |y| > 2^996 only matters through its sign once log|x| != 0 (the
    // result over/underflows either way); clamping keeps the Dekker split
    // of a non-FMA build finite.
    const D ay = abs(y);
    const D yc = select(gt(ay, bc<D>(0x1p996)), copysign(bc<D>(0x1p996), y), y);
    D ph, pe;
    twoProd(yc, lh, ph, pe);
    const D pl = fma(yc, ll, pe);
    D res = expCore(ph, pl);

    const auto isInt = eq(rintNonNeg(ay), ay);
    const D hy = ay * 0.5;
    const auto isOdd = isInt & ne(rintNonNeg(hy), hy) & lt(ay, bc<D>(0x1p53));
    const auto xneg = signbit(x);
    const auto negRes = xneg & isOdd;
    res = negateIf(negRes, res);
    res = select(lt(x, bc<D>(0.0)) & ne(x, -inf) & ~isInt & ~isinf(y), bc<D>(__builtin_nan("")), res);

    const auto yPos = gt(y, bc<D>(0.0));
    const auto axInf = eq(ax, inf);
    const auto axZero = eq(ax, bc<D>(0.0));
    D v = select((yPos & axInf) | (~yPos & axZero), inf, bc<D>(0.0));
    res = select(axInf | axZero, negateIf(negRes, v), res);

    const auto big = gt(ax, bc<D>(1.0));
    v = select(~(big ^ yPos), inf, bc<D>(0.0));
    v = select(eq(ax, bc<D>(1.0)), bc<D>(1.0), v);
    res = select(eq(ay, inf), v, res);

    res = select(isnan(x) | isnan(y), x + y, res);
    res = select(eq(y, bc<D>(1.0)), x, res);
    return select(eq(y, bc<D>(0.0)) | eq(x, bc<D>(1.0)), bc<D>(1.0), res);
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
