// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketHyperbolic.h
 * @brief Vectorised `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
 *        for `aether::simd::Packet<double, W>`, faithfully rounded.
 *
 * Every intermediate that the result is sensitive to is a double-double;
 * only the last addition rounds.
 *   sinh = (E + E/(E+1))/2, tanh = E/(E+2), with E = expm1(|x|) resp.
 *   expm1(2|x|) as a double-double (no cancellation for small |x|);
 *   cosh = (e + 1/e)/2 with e = exp(|x|) as a double-double;
 *   for |x| >= 22, sinh/cosh = exp(|x| - ln2) (the e^-|x| term is below
 *   2^-63 relative), which also keeps the result finite up to the true
 *   overflow threshold; tanh = 1.
 *   asinh = log1p(|x| + x^2/(1 + sqrt(1 + x^2))), acosh = log1p(t +
 *   sqrt(2t + t^2)) with t = x - 1 (exact), atanh = log1p(2|x|/(1-|x|))/2,
 *   each argument formed in double-double and fed to `log1pDD`; beyond
 *   2^28, asinh/acosh = log(|x|) + ln2.
 */

#include "aether/backend/cpu/simd/math/PacketExpLog.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

/// e^ax / 2 = exp(ax - ln2) for ax >= 22 (overflows only where the result does).
template<class D>
AETHER_PM_INLINE D halfExpLarge(D ax)
{
    D h, l;
    twoSum(ax, bc<D>(-kLn2DdHi), h, l);
    return expCore(h, l - kLn2DdLo);
}

template<class D>
AETHER_PM_INLINE D vsinh(D x)
{
    const D ax = abs(x);
    const D axc = select(lt(ax, bc<D>(22.0)), ax, bc<D>(0.0));
    D e, el;
    expm1DD(axc, e, el);
    D dh, dl;
    twoSum(bc<D>(1.0), e, dh, dl);
    D qh, ql;
    ddDiv(e, el, dh, dl + el, qh, ql);
    D sh, sl;
    twoSum(e, qh, sh, sl);
    const D small = 0.5 * (sh + (sl + (el + ql)));
    const D r = select(lt(ax, bc<D>(22.0)), small, halfExpLarge(ax));
    return select(isnan(x), x, copysign(r, x));
}

template<class D>
AETHER_PM_INLINE D vcosh(D x)
{
    const D ax = abs(x);
    const D axc = select(lt(ax, bc<D>(22.0)), ax, bc<D>(0.0));
    D r, rl, ph, pl;
    const D n = expReduce(axc, bc<D>(0.0), r, rl);
    expKernel(r, rl, ph, pl);
    const D sc = pow2(n);
    const D eh = ph * sc, el = pl * sc;          // exp(|x|), exact scaling
    const D ih = 1.0 / eh;
    const D il = (residual(ih, eh, bc<D>(1.0)) - ih * el) / eh; // 1/exp(|x|)
    D ch, cl;
    twoSum(eh, ih, ch, cl);
    const D small = 0.5 * (ch + (cl + (el + il)));
    return select(isnan(x), x, select(lt(ax, bc<D>(22.0)), small, halfExpLarge(ax)));
}

template<class D>
AETHER_PM_INLINE D vtanh(D x)
{
    const D ax = abs(x);
    const D axc = select(lt(ax, bc<D>(22.0)), ax, bc<D>(0.0));
    D e, el;
    expm1DD(axc + axc, e, el);
    D dh, dl;
    twoSum(bc<D>(2.0), e, dh, dl);
    D th, tl;
    ddDiv(e, el, dh, dl + el, th, tl);
    const D r = select(ge(ax, bc<D>(22.0)), bc<D>(1.0), th + tl);
    return copysign(select(isnan(x), x, r), x);
}

/// log(ax) + ln2, rounded once (asinh/acosh beyond 2^28).
template<class D>
AETHER_PM_INLINE D logPlusLn2(D ax)
{
    D h, l, s, se;
    logDD(ax, h, l);
    twoSum(h, bc<D>(kLn2DdHi), s, se);
    return s + (se + (l + kLn2DdLo));
}

template<class D>
AETHER_PM_INLINE D vasinh(D x)
{
    const D ax = abs(x);
    const auto huge = gt(ax, bc<D>(0x1p28));
    const D a = select(huge, bc<D>(1.0), ax);
    D t, te;
    twoProd(a, a, t, te);
    D oh, ol;
    twoSum(bc<D>(1.0), t, oh, ol);
    D qh, ql;
    ddSqrt(oh, ol + te, qh, ql);
    D dh, dl;
    twoSum(bc<D>(1.0), qh, dh, dl);
    D Qh, Ql;
    ddDiv(t, te, dh, dl + ql, Qh, Ql);
    D wh, wl;
    twoSum(a, Qh, wh, wl);
    D h, l;
    log1pDD(wh, wl + Ql, h, l);
    D r = select(huge, logPlusLn2(ax), h + l);
    r = select(eq(ax, bc<D>(0.0)) | isinf(x) | isnan(x), x, r);
    return copysign(r, x);
}

template<class D>
AETHER_PM_INLINE D vacosh(D x)
{
    const auto huge = gt(x, bc<D>(0x1p28));
    const auto dom = ge(x, bc<D>(1.0)) & ~huge;
    const D t = select(dom, x, bc<D>(2.0)) - 1.0; // exact for 1 <= x <= 2^28
    D tt, tte;
    twoProd(t, t, tt, tte);
    D vh, vl;
    twoSum(t + t, tt, vh, vl);
    D qh, ql;
    ddSqrt(vh, vl + tte, qh, ql);
    D wh, wl;
    twoSum(t, qh, wh, wl);
    D h, l;
    log1pDD(wh, wl + ql, h, l);
    D r = select(huge, logPlusLn2(x), h + l);
    r = select(eq(x, bc<D>(1.0)), bc<D>(0.0), r);
    r = select(lt(x, bc<D>(1.0)), bc<D>(__builtin_nan("")), r);
    return select(isnan(x) | eq(x, bc<D>(__builtin_inf())), x, r);
}

template<class D>
AETHER_PM_INLINE D vatanh(D x)
{
    const D ax = abs(x);
    const D a = select(lt(ax, bc<D>(1.0)), ax, bc<D>(0.5));
    D dh, dl;
    twoSum(bc<D>(1.0), -a, dh, dl);
    D wh, wl;
    ddDiv(a + a, bc<D>(0.0), dh, dl, wh, wl);
    D h, l;
    log1pDD(wh, wl, h, l);
    D r = 0.5 * (h + l);
    r = select(eq(ax, bc<D>(1.0)), bc<D>(__builtin_inf()), r);
    r = select(gt(ax, bc<D>(1.0)), bc<D>(__builtin_nan("")), r);
    r = select(eq(ax, bc<D>(0.0)) | isnan(x), x, r);
    return copysign(r, x);
}

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
