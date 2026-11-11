// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandExpLog.h
 * @brief `exp` / `expm1` / `log` / `log1p` / `exp2` / `log2` / `log10` / `pow` over `Band`, plus the 16-entry `log` table
 *        and their admission predicates (`bandExpAdmits`/`bandLogAdmits`/
 *        `bandPowAdmits`).
 *
 * `exp`'s and `log`'s cores consume `fmaRaw`/`fmaRawDom`/`fmaRawS`/
 * `fmaRawSDom` (unnormalized FMA over `BandRaw`), defined once in
 * `aether/banded/detail/FmaRaw.h` (included below) and shared with
 * `BandRound.h`'s `fma`. `aether/banded/detail/ExactSum.h` (`exactSum4`,
 * used by `BandRound.h`'s `round()` and the trig reductions) is included here for them.
 */

#include "aether/banded/Band.h"
#include "aether/banded/detail/ExactSum.h"
#include "aether/banded/detail/FmaRaw.h"
#include "aether/macros.h"

#include <cmath> // host `truncf`/`copysignf` for exp's k = round(x*log2e)

namespace aether {
namespace banded {
namespace detail {

// =========================================================================
//  ddMul / ddAdd / ddFmaDom -- the "double-float" demand rung `exp`/`log`
//  share: plain float-pair helpers, not `Ff2`.
// =========================================================================

/// @brief Double-float multiply: `(a_hi+a_lo) * (b_hi+b_lo) -> (p_hi+p_lo)`.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void ddMul(
    float a_hi, float a_lo, float b_hi, float b_lo, float& p_hi, float& p_lo)
{
    p_hi = mulRN(a_hi, b_hi);
    p_lo = fmaRN(a_hi, b_hi, -p_hi) + fmaRN(a_hi, b_lo, a_lo * b_hi);
}

/// @brief Double-float add: `(a_hi+a_lo) + (b_hi+b_lo) -> (s_hi+s_lo)`.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void ddAdd(
    float a_hi, float a_lo, float b_hi, float b_lo, float& s_hi, float& s_lo)
{
    s_hi     = addRN(a_hi, b_hi);
    float v  = subRN(s_hi, a_hi);
    float sv = subRN(s_hi, v);
    float e1 = subRN(a_hi, sv);
    float e2 = subRN(b_hi, v);
    s_lo     = addRN(addRN(e1, e2), addRN(a_lo, b_lo));
}

/// @brief Double-float fused multiply-add `r = x*p + c`, all DD pairs, for a Horner rung whose ADDEND DOMINATES the
/// product (`|x*p| <= 0.414 |c|`, the claim every Horner site proves for its own argument range): the leading
/// `twoSum` of `ddAdd` is a `fast2Sum`, 9 FP32 ops instead of 12. `exp`'s, `log`'s and the trig cores' mid rungs use it.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void ddFmaDom(float x_hi, float x_lo,
    float p_hi, float p_lo, float c_hi, float c_lo, float& r_hi, float& r_lo)
{
    float t_hi, t_lo;
    ddMul(x_hi, x_lo, p_hi, p_lo, t_hi, t_lo);
    r_hi          = addRN(c_hi, t_hi);
    const float e = subRN(t_hi, subRN(r_hi, c_hi));
    r_lo          = addRN(e, addRN(t_lo, c_lo));
}

// =========================================================================
//  exp
// =========================================================================

/// @brief `ln(2)` as four FP32 limbs, for Cody-Waite reduction. The top
/// three limbs each carry 16 significant bits with the low 8 mantissa bits
/// zeroed, which is what makes `k * CWi` exact in FP32 for every `|k| <=
/// 160` the admitted domain can produce.
inline constexpr std::uint32_t kBandLn2CW1Bits = 0x3F317200u; ///< ~2^-0.53
inline constexpr std::uint32_t kBandLn2CW2Bits = 0x35BFBE00u; ///< ~2^-19.42
inline constexpr std::uint32_t kBandLn2CW3Bits = 0x2D8E7B00u; ///< ~2^-35.85
inline constexpr std::uint32_t kBandLn2CW4Bits = 0x25CD5E4Fu; ///< ~2^-51.32

/// @brief `log2(e)`, for `k = round(x * log2e)`. FP32 is enough — see `exp`.
inline constexpr float kBandLog2e = 1.44269504088896340736f;

/// @brief Where `exp` saturates (FP32 carrier limits — above `128*ln2` the
/// leading limb overflows FP32, below `-150*ln2` the result underflows the
/// smallest subnormal to exactly zero). @see `bandExpAdmits` for the far
/// narrower interval over which the result is certified.
inline constexpr float kBandExpSatHigh = 88.7228391f;  ///<  128 * ln2
inline constexpr float kBandExpSatLow  = -103.972077f; ///< -150 * ln2

/// @brief Upper bound on the base-2 exponent of `e^maxInput`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr int bandExpCeilExp(
    float maxInput)
{
    const float t = maxInput * kBandLog2e;
    const int   i = static_cast<int>(t); // truncates toward zero
    return (t > static_cast<float>(i)) ? i + 1 : i;
}

/// @brief Lower bound on the base-2 exponent of `e^minInput`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr int bandExpFloorExp(
    float minInput)
{
    const float t = minInput * kBandLog2e;
    const int   i = static_cast<int>(t);
    return (t < static_cast<float>(i)) ? i - 1 : i;
}

/// @brief The certified input interval, at the default margin: `[-60.997,
/// +74.167]` — about an eighth of IEEE double's `exp` domain, derived from
/// the carrier envelope (`kBandNominalCeilingExp`/`kBandCarrierFloorExp`/
/// `kBandAdmissionMargin`, `Band.h`), not chosen independently.
inline constexpr float kBandExpMaxCertified
    = static_cast<float>(kBandNominalCeilingExp - kBandAdmissionMargin) * 0.6931471805599453f;
inline constexpr float kBandExpMinCertified
    = static_cast<float>(kBandCarrierFloorExp + kBandAdmissionMargin) * 0.6931471805599453f;

/**
 * @brief Admission for a consumer that calls `exp`, stated in terms of the
 * input interval (what the consumer knows) rather than an output exponent.
 *
 * Both legs required, not symmetric: the upper leg is the FP32 overflow
 * cliff (sharp); the lower leg is the carrier floor (gradual, carrier-
 * resident — no later multiply repairs it). The delivery floor
 * (`bandFloorAdmits`) is deliberately absent — it binds what a consumer
 * stores, not what `exp` computes.
 *
 * @param maxInput largest `x` the consumer evaluates -> the ceiling exposure.
 * @param minInput most negative `x` -> the carrier-floor exposure.
 * @param margin   the certified admission margin (defaults to
 *                 `kBandAdmissionMargin`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandExpAdmits(
    float maxInput, float minInput, int margin = kBandAdmissionMargin)
{
    return bandCeilingAdmits(bandExpCeilExp(maxInput), margin)
        && bandIntermediateAdmits(bandExpFloorExp(minInput), margin);
}

// Two-sided: the guard must agree with the constants it is derived from,
// and must reject on both sides.
static_assert(bandExpAdmits(kBandExpMaxCertified, kBandExpMinCertified),
    "exp's own certified interval must be admitted by its own guard");
static_assert(!bandExpAdmits(80.0f, -40.0f),
    "exp must REJECT an input reaching 80 (e^80 ~ 2^115, at the nominal "
    "ceiling with no margin left)");
static_assert(!bandExpAdmits(40.0f, -70.0f),
    "exp must REJECT an input reaching -70 (e^-70 ~ 2^-101, past the carrier "
    "floor) -- the leg that is gradual, not the one that is sharp");
static_assert(bandExpAdmits(74.0f, -60.0f),
    "exp must ADMIT the interior of its own interval");
static_assert(bandExpAdmits(78.0f, -65.0f, 0) && !bandExpAdmits(78.0f, -65.0f),
    "exp's domain must respond to the margin exactly as the other legs do");
static_assert(kBandExpSatLow < kBandExpMinCertified && kBandExpMaxCertified < kBandExpSatHigh,
    "the SATURATION bounds must lie strictly outside the CERTIFIED ones");

/// @brief `2^(j/32)`, j = 0..31, as three FP32 limbs (+pad), exact to < 2^-72 relative. Device: global memory, one LDG.128.
#if defined(__CUDACC__)
static __device__ const float4 kBandExpTDev[32] = {
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 1.0218972f, -4.81155986e-08f, 9.39270706e-16f, 0.0f },
    { 1.04427373f, 4.83347016e-08f, -1.3652571e-16f, 0.0f },
    { 1.06714046f, -5.93375198e-08f, 1.43046064e-16f, 0.0f },
    { 1.09050775f, -1.30775399e-08f, -2.52512433e-16f, 0.0f },
    { 1.1143868f, -5.43554002e-08f, -1.22816485e-15f, 0.0f },
    { 1.13878858f, 5.38622231e-08f, -1.68722875e-15f, 0.0f },
    { 1.1637249f, -4.05144149e-08f, 2.60336662e-16f, 0.0f },
    { 1.18920708f, 3.79763527e-08f, 1.15004321e-15f, 0.0f },
    { 1.21524739f, -3.26739489e-08f, -1.18734929e-15f, 0.0f },
    { 1.24185777f, 4.4968381e-08f, 4.9066951e-16f, 0.0f },
    { 1.26905096f, 1.41933332e-09f, 2.6679321e-18f, 0.0f },
    { 1.29683959f, -4.01899953e-08f, 1.57969474e-15f, 0.0f },
    { 1.32523668f, -3.49637332e-08f, 4.15501901e-16f, 0.0f },
    { 1.35425556f, -1.01233493e-08f, 2.99054088e-16f, 0.0f },
    { 1.38390994f, -5.87557736e-08f, 1.04251788e-15f, 0.0f },
    { 1.41421354f, 2.4203235e-08f, -7.62806744e-16f, 0.0f },
    { 1.44518077f, 3.32419994e-08f, 8.57940826e-16f, 0.0f },
    { 1.47682619f, -4.50089885e-08f, 1.51947229e-15f, 0.0f },
    { 1.50916445f, -2.49593732e-08f, 5.64488273e-16f, 0.0f },
    { 1.54221082f, 8.07090483e-09f, -1.42546261e-16f, 0.0f },
    { 1.5759809f, -5.66102543e-08f, 1.10008616e-15f, 0.0f },
    { 1.61049032f, 9.83621717e-09f, 2.47071934e-17f, 0.0f },
    { 1.64575553f, -5.12497209e-08f, -7.67390621e-16f, 0.0f },
    { 1.68179286f, -2.47553267e-08f, -5.84143725e-16f, 0.0f },
    { 1.71861935f, -4.8496176e-08f, 1.53579842e-15f, 0.0f },
    { 1.75625217f, -9.23577037e-09f, 2.9601406e-17f, 0.0f },
    { 1.79470909f, -1.14150449e-08f, 2.40272066e-16f, 0.0f },
    { 1.8340081f, -1.1239278e-08f, -1.89213528e-16f, 0.0f },
    { 1.87416768f, -4.66300563e-08f, -2.83272249e-16f, 0.0f },
    { 1.91520655f, 9.84532811e-09f, 3.37889754e-16f, 0.0f },
    { 1.95714414f, -1.70218044e-08f, 8.9607675e-17f, 0.0f }
};
#endif
inline constexpr float kBandExpTHost[128] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    1.0218972f, -4.81155986e-08f, 9.39270706e-16f, 0.0f,
    1.04427373f, 4.83347016e-08f, -1.3652571e-16f, 0.0f,
    1.06714046f, -5.93375198e-08f, 1.43046064e-16f, 0.0f,
    1.09050775f, -1.30775399e-08f, -2.52512433e-16f, 0.0f,
    1.1143868f, -5.43554002e-08f, -1.22816485e-15f, 0.0f,
    1.13878858f, 5.38622231e-08f, -1.68722875e-15f, 0.0f,
    1.1637249f, -4.05144149e-08f, 2.60336662e-16f, 0.0f,
    1.18920708f, 3.79763527e-08f, 1.15004321e-15f, 0.0f,
    1.21524739f, -3.26739489e-08f, -1.18734929e-15f, 0.0f,
    1.24185777f, 4.4968381e-08f, 4.9066951e-16f, 0.0f,
    1.26905096f, 1.41933332e-09f, 2.6679321e-18f, 0.0f,
    1.29683959f, -4.01899953e-08f, 1.57969474e-15f, 0.0f,
    1.32523668f, -3.49637332e-08f, 4.15501901e-16f, 0.0f,
    1.35425556f, -1.01233493e-08f, 2.99054088e-16f, 0.0f,
    1.38390994f, -5.87557736e-08f, 1.04251788e-15f, 0.0f,
    1.41421354f, 2.4203235e-08f, -7.62806744e-16f, 0.0f,
    1.44518077f, 3.32419994e-08f, 8.57940826e-16f, 0.0f,
    1.47682619f, -4.50089885e-08f, 1.51947229e-15f, 0.0f,
    1.50916445f, -2.49593732e-08f, 5.64488273e-16f, 0.0f,
    1.54221082f, 8.07090483e-09f, -1.42546261e-16f, 0.0f,
    1.5759809f, -5.66102543e-08f, 1.10008616e-15f, 0.0f,
    1.61049032f, 9.83621717e-09f, 2.47071934e-17f, 0.0f,
    1.64575553f, -5.12497209e-08f, -7.67390621e-16f, 0.0f,
    1.68179286f, -2.47553267e-08f, -5.84143725e-16f, 0.0f,
    1.71861935f, -4.8496176e-08f, 1.53579842e-15f, 0.0f,
    1.75625217f, -9.23577037e-09f, 2.9601406e-17f, 0.0f,
    1.79470909f, -1.14150449e-08f, 2.40272066e-16f, 0.0f,
    1.8340081f, -1.1239278e-08f, -1.89213528e-16f, 0.0f,
    1.87416768f, -4.66300563e-08f, -2.83272249e-16f, 0.0f,
    1.91520655f, 9.84532811e-09f, 3.37889754e-16f, 0.0f,
    1.95714414f, -1.70218044e-08f, 8.9607675e-17f, 0.0f
};


// the table-exp pieces shared by exp / expm1 / exp2 / expRaw -- one reduction, one core.
/// Reduce x = (k + j/32) ln2 + r, |r| <= ln2/64 (6-limb ln2/32; admitted x, so the exit normalise needs no specials guard).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band expTabReduce(Band x, int& k, int& j, bool& m0)
{
    constexpr float cw1 = 0.0216522217f;
    constexpr float cw2 = 8.62032175e-06f;
    constexpr float cw3 = 7.38873496e-09f;
    constexpr float cw4 = 2.32347475e-12f;
    constexpr float cw5 = 1.71737624e-15f;
    constexpr float cw6 = 7.24702137e-19f;
    constexpr float kN = 32.0f * kBandLog2e;
    const float vf = x.hi * kN;
    const float mf = truncf(vf + copysignf(0.5f, vf));
    const int   m  = static_cast<int>(mf);
    k  = m >> 5;
    j  = m & 31;
    m0 = (m == 0);
    const float p2 = mf * cw2;
    const float p3 = mf * cw3;
    const float p4 = mf * cw4;
    const float p5 = mf * cw5;
    const float p6 = mf * cw6;
    const BandRaw lhs{ x.hi - mf * cw1, x.lo, x.tail };
    const BandRaw rhs{ -p2, -p3, 0.0f };
    BandRawCancel ar = addRaw(lhs, rhs);
    ar = addRawS(BandRaw{ ar.hi, ar.lo, ar.tail }, -p4);
    ar = addRawS(BandRaw{ ar.hi, ar.lo, ar.tail }, -p5);
    ar = addRawS(BandRaw{ ar.hi, ar.lo, ar.tail }, -p6);
    return normalizeSafeAdmitted(ar);
}

/// T_j * ((lead + r) + r^2 p(r)) as a raw 3-limb value: lead = 1 gives e^r * T_j, lead = 0 gives T_j * (e^r - 1).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw expTabCore(Band r, float lead, int j, bool fine = false)
{
    const float rh = r.hi, rl = r.lo;
    // p(r) = 1/2 + r/6 + ... : FP32 rung for n >= 5, double-float below (c2 = 1/2 exact).
    float p = 0.000198412701f;
    p = fmaRN(rh, p, 0.00138888892f);  // c6
    p = fmaRN(rh, p, 0.00833333377f);  // c5
    float ph = p, pl = 0.0f;
    ddFmaDom(rh, rl, ph, pl, 0.0416666679f, -1.24176347e-09f, ph, pl);  // c4
    ddFmaDom(rh, rl, ph, pl, 0.166666672f, -4.96705388e-09f, ph, pl);  // c3
    ddFmaDom(rh, rl, ph, pl, 0.5f, 0.0f, ph, pl);  // c2
    // `fine` (expm1: the result can be 2^-6.5 while the core's w = r'^2 p is 2^-14, so w needs ~2^-50 relative, not the
    // 2^-45 two ddMuls give): r'^2 by sqrRaw and w = r'^2 * p by mulRaw, carried as three limbs.
    float wh, wl, wt = 0.0f;
    if (fine) {
        const BandRaw r2 = sqrRaw(r);
        const BandRaw w3 = mulRaw(r2, BandRaw{ ph, pl, 0.0f });
        wh = w3.hi; wl = w3.lo; wt = w3.tail;
    } else {
        float r2h, r2l;
        ddMul(rh, rl, rh, rl, r2h, r2l);
        ddMul(r2h, r2l, ph, pl, wh, wl);
    }

    // S = (lead + r) + r^2*p as a 3-limb raw value (|lead| >= |r| or lead == 0, so fast2Sum is exact).
    float s, e;
    fast2Sum(lead, rh, s, e);
    float s1, e1;
    fast2Sum(s, wh, s1, e1);
    float a1, a2, b1, b2, c1, c2;
    twoSum(e1, e, a1, a2);
    twoSum(a1, rl, b1, b2);
    twoSum(b1, wl, c1, c2);
    const BandRaw S{ s1, c1, ((a2 + b2) + (c2 + (r.tail + wt))) };

#if defined(__CUDA_ARCH__)
    const float4 tv = __ldg(&kBandExpTDev[j]);
    const float t0 = tv.x, t1 = tv.y, t2 = tv.z;
#else
    const float t0 = kBandExpTHost[4*j], t1 = kBandExpTHost[4*j+1], t2 = kBandExpTHost[4*j+2];
#endif
    return mulRaw(BandRaw{ t0, t1, t2 }, S);
}

/// 2^k scaling of a raw value, split into two normal-range factors (exact).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw scaleRawPow2(BandRaw v, int k)
{
    const int   k1 = k >> 1;
    const int   k2 = k - k1;
    const float s1 = intAsFloat((127 + k1) << 23);
    const float s2 = intAsFloat((127 + k2) << 23);
    return BandRaw{ (v.hi * s1) * s2, (v.lo * s1) * s2, (v.tail * s1) * s2 };
}

/**
 * @brief `e^x` on the banded carrier, certified to 53 bits over the interval
 * `bandExpAdmits` describes ([-60.997, +74.167] at the default margin).
 *
 * @par Algorithm
 *  1. `k = round(x * log2e)` in FP32, half away from zero.
 *  2. `r = x - k*ln2` via the four-limb Cody-Waite constants above — the
 *     leading subtraction is exact by Sterbenz for every `|k| >= 1`.
 *  3. Degree-13 Taylor of `e^r` on `|r| <= ln2/2`, on a three-rung demand
 *     ladder: degrees 13..10 pure FP32, 9..4 double-float (`ddFma`), 3..0
 *     the carrier (`fmaRaw`/`fmaRawS`), one `normalize` at the exit.
 *  4. `e^x = 2^k * P(r)` via `scalePow2f`, exact, split into two normal-range
 *     factors so a saturating-low `k` still scales to the correct subnormal.
 *
 * `c0 = c1 = 1`, `c2 = 1/2` are exact in one float, which is why a Taylor
 * series (rather than a minimax one) is used here: three of the four
 * Band-rung steps become `fmaRawS` instead of the general three-limb
 * `fmaRaw`.
 *
 * @par Host/device: bit-identical — no hardware seed anywhere in this body.
 * Cost: about 285 FP32 operations.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band exp(Band x)
{
    if (!(x.hi >= kBandExpSatLow && x.hi <= kBandExpSatHigh)) {
        if (x.hi != x.hi)
            return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        return (x.hi > 0.0f) ? Band{ intAsFloat(0x7F800000), 0.0f, 0.0f }
                              : Band{ 0.0f, 0.0f, 0.0f };
    }
    int        k = 0, j = 0;
    bool       m0 = false;
    const Band r  = expTabReduce(x, k, j, m0);
    const Band pe = normalize(expTabCore(r, 1.0f, j)); // the one normalization barrier the core owes
    const int   k1 = k >> 1;
    const int   k2 = k - k1;
    const float s1f = intAsFloat((127 + k1) << 23);
    const float s2f = intAsFloat((127 + k2) << 23);
    return scalePow2f(scalePow2f(pe, s1f), s2f);
}

/// @brief `exp(x)` without the exit barrier: the raw core scaled by `2^k` (exact), one raw hop, `BandRawT<1>`.
/// `normalize` of it is `exp(x)` for every input whose scaled result is normal; where the scale reaches the
/// subnormal range the barrier-then-scale order of `exp` rounds the limbs there, so the twin is for chain consumers
/// inside the normal range.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawT<1> expRaw(Band x)
{
    if (!(x.hi >= kBandExpSatLow && x.hi <= kBandExpSatHigh)) {
        if (x.hi != x.hi)
            return BandRawT<1>{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        return (x.hi > 0.0f) ? BandRawT<1>{ intAsFloat(0x7F800000), 0.0f, 0.0f } : BandRawT<1>{ 0.0f, 0.0f, 0.0f };
    }

    int        k = 0, j = 0;
    bool       m0 = false;
    const Band r  = expTabReduce(x, k, j, m0);
    const BandRaw v = scaleRawPow2(expTabCore(r, 1.0f, j), k);
    return BandRawT<1>{ v.hi, v.lo, v.tail };
}

/**
 * @brief `e^x - 1` on the banded carrier, certified to 53 bits over the
 * same input interval as `exp`, and accurate relative to the result all the
 * way down to `x = 0` (no cancellation against the 1).
 *
 * @par Algorithm
 *  `exp`'s table reduction `x = (k + j/32) ln2 + r'`, then `2^k T_j (lead + r' + r'^2 p(r')) - lead` with
 *  `lead = 1` away from `m = 0` and `lead = 0` at it (`r'^2 p` by `sqrRaw` and `mulRaw`: the `- 1` cancels up to
 *  2^6.6 near `|m| = 1`, so the plain double-float product is too coarse there).
 *
 * @par Exact points
 *  `expm1(+-0) = +-0` (sign preserved), `expm1(+inf) = +inf`,
 *  `expm1(-inf) = -1`, `expm1(NaN) = NaN`; at or below `kBandExpSatLow` the
 *  result is exactly `-1`, above `kBandExpSatHigh` it is `+inf`.
 *
 * @par Host/device: bit-identical, no hardware seed anywhere in this body.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band expm1(Band x)
{
    if (!(x.hi >= kBandExpSatLow && x.hi <= kBandExpSatHigh)) {
        if (x.hi != x.hi)
            return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        return (x.hi > 0.0f) ? Band{ intAsFloat(0x7F800000), 0.0f, 0.0f } : Band{ -1.0f, 0.0f, 0.0f };
    }
    if (x.hi == 0.0f)
        return x; // expm1(+-0) = +-0

    // expm1 through exp's table core. x = (k + j/32) ln2 + r', so
    //   expm1(x) = 2^k T_j (lead + r' + r'^2 p(r')) - lead,   lead = 1 (m != 0) or 0 (m == 0: T = 1, k = 0, no cancellation).
    // For m != 0, |x| >= ln2/64 so |expm1| >= 0.0107 and the subtraction of 1 amplifies the core's 2^-70 by <= 2^6.6 (< 2^-63).
    // The core and the constant are both carried raw: addRawS is exact for any order, one normalise at the exit.
    int k, j; bool m0;
    const Band  r    = expTabReduce(x, k, j, m0);
    const float lead = m0 ? 0.0f : 1.0f;
    const BandRaw E  = scaleRawPow2(expTabCore(r, lead, j, true), k);
    if (!(E.hi < intAsFloat(0x7F800000))) // the overflow edge (x just above 128 ln2): 2^k * T overflows in the scaling; the -1 is 2^-128 relative
        return Band{ intAsFloat(0x7F800000), 0.0f, 0.0f };
    return normalizeSafeAdmitted(addRawS(E, -lead));
}

// =========================================================================
//  log
// =========================================================================

/// @brief `sqrt(2)`, the near-one centring point.
inline constexpr float kBandLogSqrt2 = 1.4142135382e+00f;

/// @brief The bucket base of the table index -- 16 buckets of `2^19` in bit
/// space, biased so `m == 1` sits inside a bucket (never on a boundary): the
/// near-one derivation transplanted into a table lookup (`z = m-1` stays
/// exact by Sterbenz in the `t == 1` bucket).
inline constexpr int kBandLogT16Base = 0x3F340000;

/// @brief The bucket multiplier `t_i`, one float so `mul(m, {t,0,0})` stays
/// the cheap Band-by-scalar product.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float bandLogT16Mul(int i)
{
    static constexpr float kMul[16] = {
        1.387709498e+00f, 1.333622813e+00f, 1.280256033e+00f, 1.230996847e+00f,
        1.185388446e+00f, 1.143039346e+00f, 1.103612304e+00f, 1.066814899e+00f,
        1.032392383e+00f, 1.000000000e+00f, 9.415838122e-01f, 8.892320395e-01f,
        8.423969746e-01f, 8.002501130e-01f, 7.621208429e-01f, 7.254095674e-01f,
    };
    return kMul[i];
}

/// @brief `L_i = -log(t_i)` to three limbs, taken on the stored `t_i` — the
/// matched-pair invariant (`t_i` rounded first, `L_i` derived from that
/// rounded value, never from the nominal centre) makes the reduction exact
/// by construction.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float bandLogT16Log(int i)
{
    static constexpr float kLog[48] = {
        -3.276545405e-01f, -4.543616150e-09f, -1.430006947e-16f,
        -2.878991663e-01f, 7.540168490e-09f, 1.845857243e-16f,
        -2.470600903e-01f, 6.637089989e-09f, 1.120728615e-16f,
        -2.078242898e-01f, 3.807896665e-09f, -3.149882266e-17f,
        -1.700705290e-01f, 5.703849837e-09f, -8.785787207e-19f,
        -1.336908042e-01f, -3.197830667e-09f, -8.987325296e-17f,
        -9.858871251e-02f, 4.887696914e-10f, 7.205334569e-18f,
        -6.467747688e-02f, -2.828740575e-09f, 2.763700424e-17f,
        -3.187881038e-02f, -1.385269954e-10f, 2.607234620e-18f,
        0.000000000e+00f, 0.000000000e+00f, 0.000000000e+00f,
        6.019191444e-02f, 5.067707920e-10f, 1.801093305e-17f,
        1.173970625e-01f, 3.288845640e-09f, 8.683722880e-17f,
        1.715039164e-01f, -6.765249250e-09f, -2.168552708e-16f,
        2.228309661e-01f, -7.201700569e-09f, 7.161069322e-17f,
        2.716501355e-01f, 1.382477866e-08f, -2.748613973e-16f,
        3.210188746e-01f, -1.150144602e-08f, 2.579454611e-16f,
    };
    return kLog[i];
}

#if defined(__CUDACC__)
// the 16-entry log table as ONE float4 per bucket {t_i, L0, L1, L2}: one LDG.128 instead of four divergently-indexed constant loads.
static __device__ const float4 kBandLogTDev[16] = {
    { 1.387709498e+00f, -3.276545405e-01f, -4.543616150e-09f, -1.430006947e-16f },
    { 1.333622813e+00f, -2.878991663e-01f, 7.540168490e-09f, 1.845857243e-16f },
    { 1.280256033e+00f, -2.470600903e-01f, 6.637089989e-09f, 1.120728615e-16f },
    { 1.230996847e+00f, -2.078242898e-01f, 3.807896665e-09f, -3.149882266e-17f },
    { 1.185388446e+00f, -1.700705290e-01f, 5.703849837e-09f, -8.785787207e-19f },
    { 1.143039346e+00f, -1.336908042e-01f, -3.197830667e-09f, -8.987325296e-17f },
    { 1.103612304e+00f, -9.858871251e-02f, 4.887696914e-10f, 7.205334569e-18f },
    { 1.066814899e+00f, -6.467747688e-02f, -2.828740575e-09f, 2.763700424e-17f },
    { 1.032392383e+00f, -3.187881038e-02f, -1.385269954e-10f, 2.607234620e-18f },
    { 1.000000000e+00f, 0.000000000e+00f, 0.000000000e+00f, 0.000000000e+00f },
    { 9.415838122e-01f, 6.019191444e-02f, 5.067707920e-10f, 1.801093305e-17f },
    { 8.892320395e-01f, 1.173970625e-01f, 3.288845640e-09f, 8.683722880e-17f },
    { 8.423969746e-01f, 1.715039164e-01f, -6.765249250e-09f, -2.168552708e-16f },
    { 8.002501130e-01f, 2.228309661e-01f, -7.201700569e-09f, 7.161069322e-17f },
    { 7.621208429e-01f, 2.716501355e-01f, 1.382477866e-08f, -2.748613973e-16f },
    { 7.254095674e-01f, 3.210188746e-01f, -1.150144602e-08f, 2.579454611e-16f }
};
#endif
inline constexpr float kBandLogTHost[64] = {
    1.387709498e+00f, -3.276545405e-01f, -4.543616150e-09f, -1.430006947e-16f,
    1.333622813e+00f, -2.878991663e-01f, 7.540168490e-09f, 1.845857243e-16f,
    1.280256033e+00f, -2.470600903e-01f, 6.637089989e-09f, 1.120728615e-16f,
    1.230996847e+00f, -2.078242898e-01f, 3.807896665e-09f, -3.149882266e-17f,
    1.185388446e+00f, -1.700705290e-01f, 5.703849837e-09f, -8.785787207e-19f,
    1.143039346e+00f, -1.336908042e-01f, -3.197830667e-09f, -8.987325296e-17f,
    1.103612304e+00f, -9.858871251e-02f, 4.887696914e-10f, 7.205334569e-18f,
    1.066814899e+00f, -6.467747688e-02f, -2.828740575e-09f, 2.763700424e-17f,
    1.032392383e+00f, -3.187881038e-02f, -1.385269954e-10f, 2.607234620e-18f,
    1.000000000e+00f, 0.000000000e+00f, 0.000000000e+00f, 0.000000000e+00f,
    9.415838122e-01f, 6.019191444e-02f, 5.067707920e-10f, 1.801093305e-17f,
    8.892320395e-01f, 1.173970625e-01f, 3.288845640e-09f, 8.683722880e-17f,
    8.423969746e-01f, 1.715039164e-01f, -6.765249250e-09f, -2.168552708e-16f,
    8.002501130e-01f, 2.228309661e-01f, -7.201700569e-09f, 7.161069322e-17f,
    7.621208429e-01f, 2.716501355e-01f, 1.382477866e-08f, -2.748613973e-16f,
    7.254095674e-01f, 3.210188746e-01f, -1.150144602e-08f, 2.579454611e-16f
};

/// @brief Which bucket `m` falls in. The clamp is real: the top bucket
/// starts at `1.40625` and `m` reaches `sqrt(2)`, so the last 0.008 of the
/// interval indexes 16 and folds into bucket 15 (`|z|` then `2^-5.27`,
/// inside the ladder's `2^-5.000`).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() int bandLogT16Index(float mhi)
{
    const int k = (floatAsInt(mhi) - kBandLogT16Base) >> 19;
    return k > 15 ? 15 : (k < 0 ? 0 : k);
}

/// @brief `log`'s certified relative bound: half an ULP at 53 bits, one leg.
inline constexpr int kBandLogRelBoundExp = -53;

/// @brief Where `log`'s output stops carrying its full relative accuracy:
/// `s = log(x)/2` keeps its `tail` limb ~2^-48 below `hi`, and FP32's
/// smallest normal is `2^-126`, so the tail goes subnormal at `|s| = 2^-78`,
/// i.e. `|log x| = 2^-77`. Derived from the format.
inline constexpr int kBandLogTailNormalExp = -77;

/**
 * @brief Admission for a consumer that calls `log`. Three legs: the
 * argument's ceiling and carrier floor (unchanged from any carrier value),
 * plus `log`'s own leg on the result's carrier floor — `log(x)` is delivered
 * as `2 s q` with `s ~ (x-1)/2`, a factor of the output, so the output
 * inherits its relative error un-attenuated and `bandIntermediateAdmits`
 * binds on it exactly as on any other spine magnitude.
 *
 * @param maxAbsExpArg    exponent of the largest argument magnitude.
 * @param minAbsExpArg    exponent of the smallest argument magnitude.
 * @param minAbsExpResult exponent of the smallest `|log x|` the consumer
 *                        produces -- i.e. how close to 1 its arguments come.
 * @param margin          the certified admission margin (defaults to
 *                        `kBandAdmissionMargin`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandLogAdmits(
    int maxAbsExpArg, int minAbsExpArg, int minAbsExpResult, int margin = kBandAdmissionMargin)
{
    return bandCeilingAdmits(maxAbsExpArg, margin) && bandIntermediateAdmits(minAbsExpArg, margin)
        && bandIntermediateAdmits(minAbsExpResult, margin);
}

// Two-sided.
static_assert(bandLogAdmits(107, -88, -88),
    "log over the carrier's whole envelope, with arguments coming as close "
    "to 1 as the RESULT's own carrier-floor leg permits");
static_assert(!bandLogAdmits(108, -88, -88),
    "REJECT on the argument's CEILING leg -- log adds nothing there, so the "
    "edge must be exactly bandCeilingAdmits's");
static_assert(!bandLogAdmits(107, -89, -88),
    "REJECT on the argument's carrier floor: 2^-88 admitted, 2^-89 not");
static_assert(bandLogAdmits(0, 0, -88) && !bandLogAdmits(0, 0, -89),
    "THE NEAR-ONE LEG, pinned from both sides: a leg on the RESULT, because "
    "log(x) is delivered as 2 s q with s ~ (x-1)/2 -- a FACTOR of the "
    "output, so bandIntermediateAdmits binds on it exactly as on any other "
    "spine magnitude");
static_assert(kBandLogTailNormalExp == -77,
    "the tail-normal knee is derived from the FORMAT: s's tail limb sits "
    "~2^-48 under its hi and FP32's smallest normal is 2^-126, so s goes "
    "tail-subnormal at 2^-78 and |log x| = 2|s| at 2^-77");

/**
 * @brief `log(x)` on the banded carrier, certified to 53 bits over
 * `bandLogAdmits`.
 *
 * @par Algorithm
 *  1. Exact exponent split `x = m * 2^e` (subnormal leading limb pre-scaled
 *     by `2^100` first, exactly).
 *  2. Centre on `sqrt(2)`: `m > sqrt2` halves `m` and bumps `e` (exact).
 *  3. Table reduction (16-entry): `t = t_i` one float, `L = -log(t_i)` its
 *     matched three-limb partner; `p = m*t` lands in `[1-2^-5, 1+2^-5]`,
 *     `z = p-1` exact by Sterbenz.
 *  4. `q(z) = sum_n (-z)^n/(n+1)`, degree 12, on the same three-rung demand
 *     ladder `exp` uses (12..9 FP32, 8..3 double-float via `ddFmaDom`, 2..0 the
 *     carrier), one `normalize` at the exit.
 *  5. `log(m) = L + z*q`; reconstruction `log(x) = e*ln2 + log(m)` via the
 *     four-limb Cody-Waite products, exact, summed by an ordered `fast2Sum` cascade (a plain
 *     `addRaw` would round this sum at `2^-52`, eight binades too coarse).
 *
 * @par Host/device: bit-identical — `log` calls `recip` (not `recipRaw`),
 * which seeds from `recipSeed` (a bit-hack, no hardware SFU) on both arms.
 *
 * @par Exact points
 * `log(1) == +0` bit-exactly, `log(2^k)` is the four-limb Cody-Waite `k*ln2`
 * and nothing else, `log(+-0) = -inf`, `log(x<0)`/`log(NaN)` = NaN,
 * `log(+inf) = +inf`.
 *
 * @par Cost: about 495 total operations, 440 of them FP32 (two normalises
 * dropped: -18 FP32, -10 control).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawT<1> logRaw(Band x) // raw-exit twin of `log`, which is `logRaw` plus the exit barrier
{
    // Guards. `!(x.hi > 0)` is false for a NaN too, so one test opens the
    // whole rejected set and the three exits are then distinguished.
    if (!(x.hi > 0.0f)) {
        if (x.hi != x.hi)
            return BandRawT<1>{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        if (x.hi == 0.0f)
            return BandRawT<1>{ intAsFloat(0xFF800000), 0.0f, 0.0f }; // log(+-0)
        return BandRawT<1>{ intAsFloat(0x7FC00000), 0.0f, 0.0f };     // x < 0
    }
    if (x.hi == intAsFloat(0x7F800000))
        return BandRawT<1>{ x.hi, 0.0f, 0.0f }; // log(+inf) = +inf

    // 1+2. Exact exponent split and sqrt(2) centring, ONE scaling: the mantissa m = xs * 2^-eb is read off
    //      the bits, `up` (m > sqrt2) is decided there, and the three limbs are scaled once by 2^-(eb+up). (A subnormal leading
    //      limb has no meaningful exponent field, so it is pre-scaled by 2^100 first -- exact.)
    const bool sub = floatAsInt(x.hi) < 0x00800000;
    const Band xs  = sub ? scalePow2f(x, intAsFloat((127 + 100) << 23)) : x;
    const int  eb  = ((floatAsInt(xs.hi) >> 23) & 0xFF) - 127;
    const float mh = intAsFloat((floatAsInt(xs.hi) & 0x007FFFFF) | 0x3F800000);
    const int  up  = (mh > kBandLogSqrt2) ? 1 : 0;
    const int  sh  = eb + up;                       // 2^-sh is subnormal for sh >= 127: split into two normal factors (6 FMUL, not 12 + 3)
    const int  h1  = sh >> 1;
    const Band m   = scalePow2f(scalePow2f(xs, intAsFloat((127 - h1) << 23)), intAsFloat((127 - (sh - h1)) << 23));
    const int  e   = eb - (sub ? 100 : 0) + up;

    // 3. Table reduction.
    const int  ti = bandLogT16Index(m.hi);
#if defined(__CUDA_ARCH__)
    const float4 tv = __ldg(&kBandLogTDev[ti]);
    const float tm = tv.x; const Band L{ tv.y, tv.z, tv.w };
#else
    const float tm = kBandLogTHost[4 * ti]; const Band L{ kBandLogTHost[4 * ti + 1], kBandLogTHost[4 * ti + 2], kBandLogTHost[4 * ti + 3] };
#endif
    // the product by the bucket multiplier stays raw -- normalizeSafeAdmitted below is a full renormalisation of {hi-1, lo, tail}.
    const BandRaw mt = mulRawS(m, tm);
    const Band z  = normalizeSafeAdmitted(BandRawCancel{ mt.hi - 1.0f, mt.lo, mt.tail }); // admitted, no specials guard

    // 4. q(z) = sum_n (-z)^n/(n+1), degree 12, on the three-rung ladder.
    //    Coefficients alternate: at step j the addend c_j = 1/(j+1) and the
    //    product is z*q with |z*q|/|c_j| <= |z|*(j+1)/(j+2) <= 2^-5, so the
    //    addend dominates by 32x and no step can lose a leading limb.
    float p = 7.692307979e-02f;
    p       = fmaRN(z.hi, p, -8.333333582e-02f);
    p       = fmaRN(z.hi, p, 9.090909362e-02f);
    p       = fmaRN(z.hi, p, -1.000000015e-01f);
    p       = fmaRN(z.hi, p, 1.111111119e-01f);
    float ph = p, pl = 0.0f;
    ddFmaDom(z.hi, z.lo, ph, pl, -1.250000000e-01f, 0.000000000e+00f, ph, pl);
    ddFmaDom(z.hi, z.lo, ph, pl, 1.428571492e-01f, -6.386212004e-09f, ph, pl);
    ddFmaDom(z.hi, z.lo, ph, pl, -1.666666716e-01f, 4.967053879e-09f, ph, pl);
    ddFmaDom(z.hi, z.lo, ph, pl, 2.000000030e-01f, -2.980232283e-09f, ph, pl);
    ddFmaDom(z.hi, z.lo, ph, pl, -2.500000000e-01f, 0.000000000e+00f, ph, pl);
    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawDom(z, q, BandRaw{ 3.333333433e-01f, -9.934107759e-09f, 2.960594821e-16f });
    q = fmaRawSDom(z, q, -0.5f); // c1 = -1/2, exact in one float
    q = fmaRawSDom(z, q, 1.0f);  // c0 =  1,   exact in one float
    // No normalise: mulRaw below consumes q raw (overlap <= 2.5 ulp per hop, bound unchanged).
    const BandRaw qn = q;

    // log(m) = L + z*q. At most one bit of cancellation (|L|/|log m| <=
    // 2.024 over the table), the same inequality the centring buys the
    // outer reconstruction one level down.
    // lm stays raw too: addRaw at the return is exact for any ordered input
    // and the final normalizeSafe absorbs the cancellation. The relabel from
    // BandRawCancel is deliberate.
    const BandRawCancel lmc = addRaw(mulRaw(z, qn), L);
    const BandRaw       lm  = BandRaw{ lmc.hi, lmc.lo, lmc.tail };

    // 5. Reconstruction: the four e*CWi products are exact in FP32 for
    //    |e| <= 160 and summed exactly (not addRaw -- that would round at
    //    2^-52). The limbs fall 2^16 apart, so |c1| >> |c2| >> |c3| >> |c4| and each running sum's error is a
    //    multiple of the next finer grid and at most half an ulp of the sum it came from: every hop is ordered
    //    (or zero), so `fast2Sum` is exact and cheaper than the order-free `twoSum` of `exactSum4`.
    const float ef = static_cast<float>(e);
    const float c1 = ef * intAsFloat(static_cast<int>(kBandLn2CW1Bits));
    const float c2 = ef * intAsFloat(static_cast<int>(kBandLn2CW2Bits));
    const float c3 = ef * intAsFloat(static_cast<int>(kBandLn2CW3Bits));
    const float c4 = ef * intAsFloat(static_cast<int>(kBandLn2CW4Bits));
    float s1, e1, s2, e2, s3, e3, u1, v1, u2, v2;
    fast2Sum(c3, c4, s1, e1);
    fast2Sum(c2, s1, s2, e2);
    fast2Sum(c1, s2, s3, e3);
    fast2Sum(e3, e2, u1, v1);
    fast2Sum(u1, e1, u2, v2);
    const Band E{ s3, u2, addRN(v2, v1) };
    const BandRawCancel lr = addRaw(E, lm); // exit normalise deferred to the consumer (<= 1 bit cancellation, see above)
    return BandRawT<1>{ lr.hi, lr.lo, lr.tail };
}

/// @brief `log(x)` as a certified carrier: `logRaw` plus the exit barrier. The exit specials guard is dead there
/// (`logRaw`'s guards already returned every special).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log(Band x)
{
    const BandRawT<1> r = logRaw(x);
    if (r.hi - r.hi != 0.0f) // -inf, NaN, +inf: the exact exits of logRaw, already normal
        return Band{ r.hi, 0.0f, 0.0f };
    return normalizeSafeAdmitted(BandRawCancel{ r.hi, r.lo, r.tail });
}

// =========================================================================
//  log1p
// =========================================================================

/// @brief Where `log1p` switches from its series (below) to `log(1 + x)`:
/// `|x| < 2^-19`.
inline constexpr float kBandLog1pSeam = 0x1p-19f;
inline constexpr int   kBandLog1pSeamExp = -19;

// The series stops after the x^3/4 term; the first dropped term, x^4/5
// relative to x, weighs at most 2^(4*seamExp - 2) at the seam, which must sit
// under the 2^-72 the three-limb carrier resolves.
static_assert(4 * kBandLog1pSeamExp - 2 < -72, "log1p's first dropped series term must weigh less than 2^-72");

/**
 * @brief `log(1 + x)` on the banded carrier, accurate relative to the result
 * down to `x = 0` and for `x` as close to `-1` as the carrier resolves.
 *
 * @par Algorithm
 *  `|x| < 2^-19`: `log1p = x * (1 - x/2 + x^2/3 - x^3/4)` on the three-rung
 *  demand ladder (the `x^3` coefficient in one float, `x^2` double-float, the
 *  last two exact singles), one `mulRaw`, one `normalize`. Otherwise
 *  `log(1 + x)`: the sum is exact for `x` in `[-1, -0.5]` and loses only
 *  `x`'s bits below `2^-72` elsewhere, a relative `2^-72/|x|` that is at most
 *  `2^-53` above the seam.
 *
 * @par Certified bound: 1 ULP at 53 bits (`log`'s half ULP plus the input
 * truncation at the seam).
 *
 * @par Exact points
 *  `log1p(+-0) = +-0`, `log1p(-1) = -inf`, `log1p(x < -1)` and `log1p(NaN)`
 *  are NaN, `log1p(+inf) = +inf`.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log1p(Band x)
{
    if (!(x.hi >= -1.0f))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    if (x.hi == intAsFloat(0x7F800000) || x.hi == 0.0f)
        return x;
    if (fabsf(x.hi) < kBandLog1pSeam) {
        float ph = -0.25f, pl = 0.0f;                                                       // -1/4
        ddFmaDom(x.hi, x.lo, ph, pl, 0x1.5555560000000p-2f, -0x1.5555560000000p-27f, ph, pl); // 1/3
        BandRaw q{ ph, pl, 0.0f };
        q = fmaRawSDom(x, q, -0.5f);
        q = fmaRawSDom(x, q, 1.0f);
        return normalize(mulRaw(x, q));
    }
    if (x.hi == -1.0f && x.lo == 0.0f && x.tail == 0.0f)
        return Band{ intAsFloat(0xFF800000), 0.0f, 0.0f };
    const Band u = normalizeSafeAdmitted(addRaw(Band{ 1.0f, 0.0f, 0.0f }, x)); // x finite here
    if (u.hi <= 0.0f) // past the guard only a lower limb can put 1 + x at or below 0
        return Band{ intAsFloat(u.hi < 0.0f ? 0x7FC00000 : 0xFF800000), 0.0f, 0.0f };
    return log(u);
}

// =========================================================================
//  exp2 / log2 / log10 -- exact scalings of exp / log.
// =========================================================================

/// @brief `ln 2`, `log2 e`, `log10 e` as three FP32 limbs each (72 bits; the
/// constant error is under `2^-77`).
inline constexpr Band kBandLn2{ 0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f, -0x1.950d880000000p-54f };
inline constexpr Band kBandLog2e3{ 0x1.7154760000000p+0f, 0x1.4ae0c00000000p-26f, -0x1.e888300000000p-52f };
inline constexpr Band kBandLog10e3{ 0x1.bcb7b20000000p-2f, -0x1.5b235e0000000p-27f, -0x1.cd59540000000p-54f };

static_assert(kBandLn2.hi + kBandLn2.lo == kBandLn2.hi && kBandLn2.lo + kBandLn2.tail == kBandLn2.lo
        && kBandLog2e3.hi + kBandLog2e3.lo == kBandLog2e3.hi && kBandLog2e3.lo + kBandLog2e3.tail == kBandLog2e3.lo
        && kBandLog10e3.hi + kBandLog10e3.lo == kBandLog10e3.hi
        && kBandLog10e3.lo + kBandLog10e3.tail == kBandLog10e3.lo,
    "each three-limb constant must be ordered: every limb below half an ULP of the one above");

/**
 * @brief `2^x`: `exp(x * ln2)`, over `exp`'s certified interval scaled by
 * `1/ln2` ([-88, +107] at the default margin). Integer `x` gives `2^x` to the
 * carrier's resolution; `exp2(+-inf)` and `exp2(NaN)` follow `exp`. Derived
 * bound 1.5 ULP at 53 bits (0.5 for `exp`, 0.5 for the product, 2^-72 for the
 * constant).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band exp2(Band x)
{
    if (x.hi - x.hi != 0.0f) // +-inf, NaN: the product would be NaN
        return (x.hi > 0.0f || x.hi != x.hi) ? x : Band{ 0.0f, 0.0f, 0.0f };
    // exp2 on exp's table core with NO log2e reduction: m = round(32 x), d = x - m/32 exact (|d| <= 1/64),
    // r' = d ln2 (one 72-bit product, |r'| <= ln2/64 as in exp), 2^x = 2^k T_j e^{r'}. Saturation at 2^128 / 2^-150 as exp's.
    if (!(x.hi >= -150.0f && x.hi <= 128.0f))
        return (x.hi > 0.0f) ? Band{ intAsFloat(0x7F800000), 0.0f, 0.0f } : Band{ 0.0f, 0.0f, 0.0f };
    const float vf = x.hi * 32.0f;
    const float mf = rintf(vf); // nearest (never the 0.5-ulp double-rounding overshoot of trunc(vf + 0.5)), so Sterbenz keeps d exact
    const int   m  = static_cast<int>(mf);
    const int   k  = m >> 5;
    const int   j  = m & 31;
    const Band  d  = normalize(BandRaw{ x.hi - mf * 0.03125f, x.lo, x.tail }); // exact: the grid point is a float, canonicalise the pair
    const Band  r  = normalize(mulRaw(d, BandRaw{ kBandLn2.hi, kBandLn2.lo, kBandLn2.tail }));
    const Band  pe = normalize(expTabCore(r, 1.0f, j));
    const int   k1 = k >> 1;
    const int   k2 = k - k1;
    return scalePow2f(scalePow2f(pe, intAsFloat((127 + k1) << 23)), intAsFloat((127 + k2) << 23));
}

/// @brief `log2(x)`: `log(x) * log2e`. Same domain as `log`; `log2(2^k) = k`,
/// `log2(+-0) = -inf`, bound 1.5 ULP at 53 bits.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log2(Band x)
{
    const BandRawT<1> r = logRaw(x); // the raw exit feeds mulRaw directly
    if (r.hi - r.hi != 0.0f) // -inf, NaN, +inf: exact exits of `log`
        return Band{ r.hi, 0.0f, 0.0f };
    return normalize(mulRaw(BandRaw(r), BandRaw{ kBandLog2e3.hi, kBandLog2e3.lo, kBandLog2e3.tail }));
}

/// @brief `log10(x)`: `log(x) * log10e`. Same domain as `log`; `log10(10^k)
/// = k` wherever `10^k` is exact in the carrier, bound 1.5 ULP at 53 bits.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log10(Band x)
{
    const BandRawT<1> r = logRaw(x);
    if (r.hi - r.hi != 0.0f)
        return Band{ r.hi, 0.0f, 0.0f };
    return normalize(mulRaw(BandRaw(r), BandRaw{ kBandLog10e3.hi, kBandLog10e3.lo, kBandLog10e3.tail }));
}

// =========================================================================
//  pow -- `pow = exp(y * log(x))`.
// =========================================================================

/// @brief `pow`'s certified relative bound: half an ULP at 53 bits, one leg.
inline constexpr int kBandPowRelBoundExp = -53;

/**
 * @brief Admission for a consumer that calls `pow`. `pow`'s legs are
 * `log`'s and `exp`'s and one narrower than either -- the near-one leg.
 * `log`'s error stops being purely relative once `s`'s `tail` limb goes
 * subnormal (`|log x| < 2^-77`); `pow` multiplies that same absolute floor
 * by `|y|`, and at the amplification corner `|y| = |t|/|log x|` reaches
 * `2^94` -- 1.0 ULP at `|log x| = 2^-88`, 56 ULP at `2^-94`. Requiring the
 * tail limb normal with the standard margin (`|log x| >= 2^-69`) removes the
 * whole regime -- a genuine narrowing of `pow` relative to `log`, pinned
 * from both sides by the static_asserts below.
 *
 * @param maxT            largest `y*log x` -> exp's ceiling exposure.
 * @param minT            most negative `y*log x` -> exp's floor exposure.
 * @param maxAbsExpBase   exponent of the largest base magnitude.
 * @param minAbsExpBase   exponent of the smallest base magnitude.
 * @param minAbsExpLog    exponent of the smallest `|log x|` -- the near-one leg.
 * @param margin          the certified admission margin (defaults to
 *                        `kBandAdmissionMargin`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandPowAdmits(float maxT,
    float minT, int maxAbsExpBase, int minAbsExpBase, int minAbsExpLog,
    int margin = kBandAdmissionMargin)
{
    return bandLogAdmits(maxAbsExpBase, minAbsExpBase, minAbsExpLog, margin)
        && bandExpAdmits(maxT, minT, margin) && (minAbsExpLog - margin >= kBandLogTailNormalExp);
}

// Two-sided.
static_assert(bandPowAdmits(74.0f, -60.0f, 107, -88, -69),
    "pow at the amplification corner: t spanning exp's certified interval, "
    "a base anywhere in the envelope, bases no closer to 1 than the "
    "tail-normal leg with its margin allows");
static_assert(!bandPowAdmits(80.0f, -60.0f, 107, -88, -69),
    "REJECT on exp's CEILING leg -- pow inherits it unchanged");
static_assert(!bandPowAdmits(74.0f, -70.0f, 107, -88, -69),
    "REJECT on exp's carrier-floor leg, likewise inherited");
static_assert(!bandPowAdmits(74.0f, -60.0f, 108, -88, -69),
    "REJECT on the BASE's ceiling leg, inherited from log");
static_assert(!bandPowAdmits(74.0f, -60.0f, 107, -89, -69),
    "REJECT on the BASE's carrier-floor leg, inherited from log");
static_assert(bandPowAdmits(74.0f, -60.0f, 107, -88, -69) && !bandPowAdmits(74.0f, -60.0f, 107, -88, -70),
    "pow's NEAR-ONE leg must bite one binade tighter than the tail-normal "
    "knee plus the default margin: |log x| >= 2^-69 admitted, 2^-70 rejected");
static_assert(bandLogAdmits(0, 0, -80) && !bandPowAdmits(74.0f, -60.0f, 0, 0, -80),
    "log ADMITS a near-one range pow REJECTS -- the narrowing, asserted "
    "rather than described");
static_assert(bandPowAdmits(74.0f, -60.0f, 107, -88, -77, 0) && !bandPowAdmits(74.0f, -60.0f, 107, -88, -77),
    "the near-one leg must move WITH the margin");

/**
 * @brief `x^y` for `x > 0`, certified to 53 bits over `bandPowAdmits`.
 *
 * Rows that are not the composition (IEEE / C Annex F, zero accuracy cost):
 *  - `y == 0` returns `1` for every `x` (checked first, so it covers
 *    rejected bases too -- `exp(0 * log(x))` is NaN wherever `log(x)` is
 *    infinite).
 *  - `y == 1` returns `x` bit for bit (`exp(log(x))` would not be).
 *  - `x == 1` returns `1` for every `y`, NaN and +-Inf included.
 *  - `x == +-0`: `y > 0` gives `+0` (`-0` for `x == -0` and odd-integer
 *    `y`); `y < 0` gives `+Inf` (`-Inf` for `x == -0` and odd-integer `y`).
 *  - `x == +Inf`: `y > 0` gives `+Inf`, `y < 0` gives `+0`.
 *  - `x == -Inf`: the `x == -0` rows with the magnitudes swapped (`y > 0`
 *    gives `+Inf`, `-Inf` for odd-integer `y`; `y < 0` gives `+0`, `-0` for
 *    odd-integer `y`).
 *  - Everything else outside `x > 0` finite returns NaN: a negative finite
 *    base, NaN (a negative base with an integer exponent has a real
 *    answer, but computing it is `pown`, a different op).
 *
 * @par Cost: about 882 total operations, 764 of them FP32 -- additive in
 * `log` (523) + `exp` (311/294) + `mul` (36) + 12 for the three guards. The
 * finite path pays one extra FP32 compare (`x.hi == 1`); the zero and
 * infinite bases pay only on their own branch.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band pow(Band x, Band y)
{
    if (y.hi == 0.0f && y.lo == 0.0f && y.tail == 0.0f)
        return Band{ 1.0f, 0.0f, 0.0f };
    if (y.hi == 1.0f && y.lo == 0.0f && y.tail == 0.0f)
        return x;
    if (x.hi == 1.0f && x.lo == 0.0f && x.tail == 0.0f)
        return Band{ 1.0f, 0.0f, 0.0f };
    // x <= 0, +inf and NaN in one test: both comparisons are false for a
    // NaN, so the negated conjunction is the whole special set.
    if (!(x.hi > 0.0f && x.hi < intAsFloat(0x7F800000))) {
        const float inf = intAsFloat(0x7F800000);
        if (x.hi == inf) {
            if (y.hi > 0.0f)
                return Band{ inf, 0.0f, 0.0f };
            if (y.hi < 0.0f)
                return Band{ 0.0f, 0.0f, 0.0f };
        } else if (x.hi == 0.0f || x.hi == -inf) {
            // odd-integer y: every limb an integer, an odd count of them odd
            // (a limb at or past 2^24 is an even integer).
            const float lim = intAsFloat(0x4B800000); // 2^24
            const float ah = y.hi < 0.0f ? -y.hi : y.hi;
            const float al = y.lo < 0.0f ? -y.lo : y.lo;
            const float at = y.tail < 0.0f ? -y.tail : y.tail;
            const bool  ih = ah >= lim || (ah == static_cast<float>(static_cast<int>(ah)));
            const bool  il = al >= lim || (al == static_cast<float>(static_cast<int>(al)));
            const bool  it = at >= lim || (at == static_cast<float>(static_cast<int>(at)));
            const int   nh = ah < lim ? static_cast<int>(ah) : 0;
            const int   nl = al < lim ? static_cast<int>(al) : 0;
            const int   nt = at < lim ? static_cast<int>(at) : 0;
            const bool  odd = ih && il && it && (((nh ^ nl ^ nt) & 1) != 0);
            const bool  negz = floatAsInt(x.hi) < 0;
            // -0 and -Inf are each other's reciprocal: same signs, swapped magnitudes.
            const bool zero = x.hi == 0.0f;
            if (y.hi > 0.0f)
                return Band{ (negz && odd) ? (zero ? -0.0f : -inf) : (zero ? 0.0f : inf), 0.0f, 0.0f };
            if (y.hi < 0.0f)
                return Band{ (negz && odd) ? (zero ? -inf : -0.0f) : (zero ? inf : 0.0f), 0.0f, 0.0f };
        }
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    }
    // `logRaw`'s exit is `E + log(m)` with at most one bit of cancellation, so its lower limbs overlap by at most one
    // more bit than a product hop's: inside the first chain hop's bound (`bandChainErrorBound(1)` allows `lo` 1.5 ulp),
    // and the Pow cert rows exercise exactly this path. `mulRaw` is exact for it; the one barrier sits before `exp`.
    return exp(normalize(mulRaw(BandRaw{ y.hi, y.lo, y.tail }, BandRaw(logRaw(x)))));
}

} // namespace detail
} // namespace banded
} // namespace aether
