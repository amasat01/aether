// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandExpLog.h
 * @brief `exp` / `log` / `pow` over `Band`, plus the 16-entry `log` table
 *        and their admission predicates (`bandExpAdmits`/`bandLogAdmits`/
 *        `bandPowAdmits`).
 *
 * `exp`'s and `log`'s cores consume `fmaRaw`/`fmaRawNoCancel`/`fmaRawS`/
 * `fmaRawSNoCancel` (unnormalized FMA over `BandRaw`), defined once in
 * `aether/banded/detail/FmaRaw.h` (included below) and shared with
 * `BandRound.h`'s `fma`. `log`'s reconstruction uses `exactSum4`, defined
 * in `aether/banded/detail/ExactSum.h` and shared with `BandRound.h`'s
 * `round()`.
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
//  ddMul / ddAdd / ddFma -- the "double-float" demand rung `exp`/`log`
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

/// @brief Double-float fused multiply-add: `r = x*p + c`, all DD pairs — the
/// mid-rung Horner step `exp`'s and `log`'s cores both use.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void ddFma(float x_hi, float x_lo,
    float p_hi, float p_lo, float c_hi, float c_lo, float& r_hi, float& r_lo)
{
    float t_hi, t_lo;
    ddMul(x_hi, x_lo, p_hi, p_lo, t_hi, t_lo);
    ddAdd(t_hi, t_lo, c_hi, c_lo, r_hi, r_lo);
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
    // Guards. Also the NaN exit: both comparisons are false for a NaN, so
    // the negated conjunction catches it without a third test.
    if (!(x.hi >= kBandExpSatLow && x.hi <= kBandExpSatHigh)) {
        if (x.hi != x.hi)
            return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        return (x.hi > 0.0f) ? Band{ intAsFloat(0x7F800000), 0.0f, 0.0f }
                              : Band{ 0.0f, 0.0f, 0.0f };
    }

    const float cw1 = intAsFloat(static_cast<int>(kBandLn2CW1Bits));
    const float cw2 = intAsFloat(static_cast<int>(kBandLn2CW2Bits));
    const float cw3 = intAsFloat(static_cast<int>(kBandLn2CW3Bits));
    const float cw4 = intAsFloat(static_cast<int>(kBandLn2CW4Bits));

    // 1. k = round(x * log2e), half away from zero.
    const float vf = x.hi * kBandLog2e;
    const float kf = truncf(vf + copysignf(0.5f, vf));
    const int   k  = static_cast<int>(kf);

    // 2. r = x - k*ln2. Leading limb cancels exactly (Sterbenz); the three
    //    lower products fold in through the certified add.
    const BandRaw lhs{ x.hi - kf * cw1, x.lo, x.tail };
    const BandRaw rhs{ -(kf * cw2), -(kf * cw3), -(kf * cw4) };
    const Band    r = normalizeSafe(addRaw(lhs, rhs));

    // 3. Degree-13 Taylor on the demand ladder. Coefficients 1/k!.
    float p = 1.605904437e-10f;                  // c13
    p       = fmaRN(r.hi, p, 2.087675588e-09f);  // c12
    p       = fmaRN(r.hi, p, 2.505210794e-08f);  // c11
    p       = fmaRN(r.hi, p, 2.755731998e-07f);  // c10

    float ph = p, pl = 0.0f;
    ddFma(r.hi, r.lo, ph, pl, 2.755731884e-06f, 3.793571224e-14f, ph, pl); // c9
    ddFma(r.hi, r.lo, ph, pl, 2.480158764e-05f, -3.406996094e-13f, ph, pl); // c8
    ddFma(r.hi, r.lo, ph, pl, 1.984127011e-04f, -2.725596875e-12f, ph, pl); // c7
    ddFma(r.hi, r.lo, ph, pl, 1.388888923e-03f, -3.363109444e-11f, ph, pl); // c6
    ddFma(r.hi, r.lo, ph, pl, 8.333333768e-03f, -4.346172033e-10f, ph, pl); // c5
    ddFma(r.hi, r.lo, ph, pl, 4.166666791e-02f, -1.241763470e-09f, ph, pl); // c4

    // c3 = 1/6 is not exact in one float, so it uses the full three-limb
    // fmaRaw. c2, c1, c0 are exact in one float, so they use fmaRawS. The
    // NoCancel variants are safe for all four steps: entering step j,
    // |r*q| <= 0.414*c_j, so the sum never falls below 0.586*c_j and no
    // leading limb is lost, for every admitted input.
    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawNoCancel(r, q, BandRaw{ 1.666666716e-01f, -4.967053879e-09f, 1.480297410e-16f });
    q = fmaRawSNoCancel(r, q, 0.5f);
    q = fmaRawSNoCancel(r, q, 1.0f);
    q = fmaRawSNoCancel(r, q, 1.0f);
    const Band pe = normalize(q); // the one normalization barrier the core owes

    // 4. 2^k, exact. Split into two normal-range factors.
    const int   k1 = k >> 1;
    const int   k2 = k - k1;
    const float s1 = intAsFloat((127 + k1) << 23);
    const float s2 = intAsFloat((127 + k2) << 23);
    return scalePow2f(scalePow2f(pe, s1), s2);
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
 *     ladder `exp` uses (12..9 FP32, 8..3 double-float via `ddFma`, 2..0 the
 *     carrier), one `normalize` at the exit.
 *  5. `log(m) = L + z*q`; reconstruction `log(x) = e*ln2 + log(m)` via the
 *     four-limb Cody-Waite products, exact, summed by `exactSum4` (a plain
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
 * @par Cost: about 523 total operations, 458 of them FP32.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band log(Band x)
{
    // Guards. `!(x.hi > 0)` is false for a NaN too, so one test opens the
    // whole rejected set and the three exits are then distinguished.
    if (!(x.hi > 0.0f)) {
        if (x.hi != x.hi)
            return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        if (x.hi == 0.0f)
            return Band{ intAsFloat(0xFF800000), 0.0f, 0.0f }; // log(+-0)
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };     // x < 0
    }
    if (x.hi == intAsFloat(0x7F800000))
        return x; // log(+inf) = +inf

    // 1. Exact exponent split. A subnormal leading limb has no meaningful
    //    exponent field, so pre-scale by 2^100 first -- exact.
    const bool sub = floatAsInt(x.hi) < 0x00800000;
    const Band xs  = sub ? scalePow2f(x, intAsFloat((127 + 100) << 23)) : x;
    const int  eb  = ((floatAsInt(xs.hi) >> 23) & 0xFF) - 127;

    const int h1 = eb >> 1;
    const int h2 = eb - h1;
    Band      m
        = scalePow2f(scalePow2f(xs, intAsFloat((127 - h1) << 23)), intAsFloat((127 - h2) << 23));
    int e = eb - (sub ? 100 : 0);

    // 2. Centre on sqrt(2) -- the near-one derivation. Halving is exact.
    const bool up = (m.hi > kBandLogSqrt2);
    m             = scalePow2f(m, up ? 0.5f : 1.0f);
    e += up ? 1 : 0;

    // 3. Table reduction.
    const int  ti = bandLogT16Index(m.hi);
    const Band mt = mul(m, Band{ bandLogT16Mul(ti), 0.0f, 0.0f });
    const Band z  = normalizeSafe(BandRaw{ mt.hi - 1.0f, mt.lo, mt.tail });
    const Band L{ bandLogT16Log(3 * ti), bandLogT16Log(3 * ti + 1), bandLogT16Log(3 * ti + 2) };

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
    ddFma(z.hi, z.lo, ph, pl, -1.250000000e-01f, 0.000000000e+00f, ph, pl);
    ddFma(z.hi, z.lo, ph, pl, 1.428571492e-01f, -6.386212004e-09f, ph, pl);
    ddFma(z.hi, z.lo, ph, pl, -1.666666716e-01f, 4.967053879e-09f, ph, pl);
    ddFma(z.hi, z.lo, ph, pl, 2.000000030e-01f, -2.980232283e-09f, ph, pl);
    ddFma(z.hi, z.lo, ph, pl, -2.500000000e-01f, 0.000000000e+00f, ph, pl);
    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawNoCancel(z, q, BandRaw{ 3.333333433e-01f, -9.934107759e-09f, 2.960594821e-16f });
    q = fmaRawSNoCancel(z, q, -0.5f); // c1 = -1/2, exact in one float
    q = fmaRawSNoCancel(z, q, 1.0f);  // c0 =  1,   exact in one float
    const Band qn = normalize(q); // the one normalization barrier the core owes

    // log(m) = L + z*q. At most one bit of cancellation (|L|/|log m| <=
    // 2.024 over the table), the same inequality the centring buys the
    // outer reconstruction one level down.
    const Band lm = normalizeSafe(addRaw(mulRaw(z, qn), L));

    // 5. Reconstruction: the four e*CWi products are exact in FP32 for
    //    |e| <= 160, summed by exactSum4 (not addRaw -- that would round at
    //    2^-52).
    const float ef = static_cast<float>(e);
    const Band  E  = exactSum4(ef * intAsFloat(static_cast<int>(kBandLn2CW1Bits)),
        ef * intAsFloat(static_cast<int>(kBandLn2CW2Bits)),
        ef * intAsFloat(static_cast<int>(kBandLn2CW3Bits)),
        ef * intAsFloat(static_cast<int>(kBandLn2CW4Bits)));
    return normalizeSafe(addRaw(E, lm));
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
 * Three rows that are not the composition:
 *  - `y == 0` returns `1` for every `x` (checked first, so it covers
 *    rejected bases too -- `exp(0 * log(x))` is NaN wherever `log(x)` is
 *    infinite).
 *  - `y == 1` returns `x` bit for bit (`exp(log(x))` would not be).
 *  - `x <= 0`, `x == +inf`, `x == NaN` return NaN: a loud reject of the
 *    uncertified domain (a negative base with an integer exponent has a
 *    real answer, but computing it is `pown`, a different op).
 *
 * @par Cost: about 882 total operations, 764 of them FP32 -- additive in
 * `log` (523) + `exp` (311/294) + `mul` (36) + 12 for the three guards.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band pow(Band x, Band y)
{
    if (y.hi == 0.0f && y.lo == 0.0f && y.tail == 0.0f)
        return Band{ 1.0f, 0.0f, 0.0f };
    if (y.hi == 1.0f && y.lo == 0.0f && y.tail == 0.0f)
        return x;
    // x <= 0, +inf and NaN in one test: both comparisons are false for a
    // NaN, so the negated conjunction is the whole rejected set.
    if (!(x.hi > 0.0f && x.hi < intAsFloat(0x7F800000)))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    return exp(mul(y, log(x)));
}

} // namespace detail
} // namespace banded
} // namespace aether
