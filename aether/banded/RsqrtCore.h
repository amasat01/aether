// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file RsqrtCore.h
 * @brief `RsqrtCore` / `rsqrtCore` / `rsqrt` / `sqrt_` / `rsqrtCube` — the
 *        reciprocal-square-root family over `Band`.
 *
 * Every primitive here is FP32 error-free-transform arithmetic over
 * `Band`'s own limbs, routed through the same `aether::banded::detail`
 * primitives (`twoProd`, `fast2Sum`, `subRN`/`mulRN`/`addRN`, `fmaRN`)
 * `Band`'s own certified core already uses (`Band.h`'s reciprocal section
 * — `recipRaw`/`recip`).
 *
 * @section seed The device seed
 * The device arm seeds from the SFU intrinsic via inline PTX
 * (`rsqrt.approx.f32`); the host arm seeds from `1.0f / sqrtf(x.hi)`. Both
 * are refined in FP32 by the same Newton schedule (`rsqrtCore`'s body
 * below): the seed is the only host/device divergence in this family.
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

#include <cmath>

namespace aether {
namespace banded {
namespace detail {

// =====================================================================
//  RsqrtCore -- the shared Newton-refined root the whole family reuses
// =====================================================================

/// @brief The refined reciprocal-square-root root plus its square and the
/// Newton residual — the shared state `rsqrt`/`sqrt_`/`rsqrtCube` each apply
/// their own scalar correction to, so the expensive core (seed + refinement +
/// one Band-Newton residual) is paid exactly once per call site regardless of
/// which of the three the caller wants.
struct RsqrtCore {
    BandRaw y;  ///< the refined root, `|dy/y| <~ 2^-45`, unnormalized by design
    BandRaw yy; ///< `y^2` -- formed for the residual, reused by the cube
    float e;    ///< `1 - x*y^2`, about `2^-44` and carrying about `2^-68`
};

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() RsqrtCore rsqrtCore(Band x)
{
    float y0;
#if defined(__CUDA_ARCH__)
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y0) : "f"(x.hi));
#else
    y0 = 1.0f / sqrtf(x.hi);
#endif

    // FP32 refinement of the hardware seed into a 2-limb root.
    float s, se;
    twoProd(y0, y0, s, se);
    const float t = fmaRN(-x.hi, s, 1.0f);
    const float w = fmaRN(-x.hi, se, -mulRN(x.lo, s));
    const float r = addRN(t, w);
    const float h = mulRN(0.5f, r);
    const BandRaw y{ y0, mulRN(y0, h), 0.0f };

    // One Band Newton residual. `yy` is the square the cube reuses.
    const BandRaw yy  = sqrRaw(y);
    const BandRaw xyy = mulRaw(yy, x);
    const float a     = subRN(1.0f, xyy.hi); // exact (Sterbenz)
    const float b     = subRN(a, xyy.lo);
    const float e     = subRN(b, xyy.tail);

    return RsqrtCore{ y, yy, e };
}

// =====================================================================
//  rsqrt -- x^(-1/2): the refined seed plus one Newton step, in correction
//  form, kept as two FP32 limbs (one limb cannot hold the correction
//  precisely enough)
// =====================================================================

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band rsqrt(Band x)
{
    const RsqrtCore k = rsqrtCore(x);

    // `y_new = y*(1 + e/2)`, the correction as two FP32 limbs.
    const float he  = mulRN(0.5f, k.e);
    const float cHi = mulRN(k.y.hi, he);
    const float cLo = addRN(fmaRN(k.y.hi, he, -cHi), mulRN(k.y.lo, he));
    float lo2, t2;
    fast2Sum(k.y.lo, cHi, lo2, t2);
    return normalize(BandRaw{ k.y.hi, lo2, addRN(t2, cLo) });
}

// =====================================================================
//  sqrt_ -- sqrt(x): the root's correction is applied to the product x*y
//  rather than to the root itself, which needs only one final rounding
//  barrier instead of two
// =====================================================================

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sqrt_(Band x)
{
    // Specials: an operand guard (the seed destroys the class before a
    // leading limb exists to guard on, exactly as `recipSeed` does).
    if (bandHiIsSpecial(x.hi)) {
        const bool plusInf = static_cast<std::uint32_t>(floatAsInt(x.hi)) == kBandFp32ExpMask;
        return bandSpecialCarrier(
            plusInf ? x.hi : intAsFloat(static_cast<int>(kBandCanonicalNanBits)));
    }
    // sqrt(+-0) = +-0 (the limb is the answer, so the zero's sign is kept).
    if (x.hi == 0.0f)
        return Band{ x.hi, 0.0f, 0.0f };

    const RsqrtCore k = rsqrtCore(x);

    // The product of x with the uncorrected root, then the scalar
    // correction (1 + e/2) applied to that product.
    const BandRaw P = mulRaw(x, k.y);
    const float he  = mulRN(0.5f, k.e);
    const float cHi = mulRN(P.hi, he);
    const float cLo = addRN(fmaRN(P.hi, he, -cHi), mulRN(P.lo, he));
    float lo2, t2;
    fast2Sum(P.lo, cHi, lo2, t2);
    // P.tail is about 2^-45 here, not 2^-48: k.y is a deliberately
    // unnormalized 2-limb pair, so mulRaw's residual is large.
    return normalize(BandRaw{ P.hi, lo2, addRN(addRN(t2, P.tail), cLo) });
}

// =====================================================================
//  rsqrtCube -- x^(-3/2), fused (never materializes 1/sqrt(x)): the cube of
//  the uncorrected root, reusing rsqrtCore's own `yy = sqrRaw(y)`, plus the
//  scalar correction (1 + 3e/2)
// =====================================================================

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band rsqrtCube(Band x)
{
    const RsqrtCore k = rsqrtCore(x);

    // The cube of the uncorrected root, then the scalar correction
    // (1 + 3e/2). Y = mulRaw(yy, y) reuses the square rsqrtCore formed for
    // the residual, so the root's own correction is never applied and no
    // 1/|u|^3 carrier is ever materialized.
    const BandRaw Y = mulRaw(k.yy, k.y);
    const float c   = mulRN(1.5f, k.e);
    const float cHi = mulRN(Y.hi, c);
    const float cLo = addRN(fmaRN(Y.hi, c, -cHi), mulRN(Y.lo, c));
    float lo2, t2;
    fast2Sum(Y.lo, cHi, lo2, t2);
    // Y.tail is about 2^-45 here, not 2^-48: y is a deliberately
    // unnormalized 2-limb pair, so mulRaw's residual is large. Dropping it
    // costs about 596 ULP, measured.
    return normalize(BandRaw{ Y.hi, lo2, addRN(addRN(t2, Y.tail), cLo) });
}

} // namespace detail
} // namespace banded
} // namespace aether
