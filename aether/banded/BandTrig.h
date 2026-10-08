// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandTrig.h
 * @brief `trigReduce` / `sincos` / `sin` / `cos` over `Band`, with explicit
 *        `fma` and no `std::` math in device code.
 *
 * @section renames Renames (algorithm and constants otherwise untouched)
 *  - `rintf` (round-to-nearest-even float->float) has no aether primitive
 *    -- `detail/Fp32.h` only carries `roundToInt32`/`roundToInt64`
 *    (float->int). Replaced by this file's own `roundNearestEven` below
 *    (@see its own doc comment); the underlying rounding instruction is
 *    identical (`cvt.rni.f32.f32`/device `__float2int_rn` vs the host's
 *    default-rounding-mode `__builtin_lrintf` -- both round-to-nearest-even),
 *    so this is a reroute through an existing primitive, not new rounding
 *    behaviour.
 *
 * @section reuse Reuses `BandExpLog.h`'s raw-carrier family
 * `sinCoreRaw`/`cosCoreRaw`/`trigSelectCore` consume `fmaRawNoCancel`/
 * `fmaRawSNoCancel` (`BandRaw`-level fma) and `trigReduce` consumes
 * `exactSum4` (four-term exact-sum cascade), both from `BandExpLog.h`
 * rather than a second copy of the same helpers.
 *
 * @section facadeGap Extends `BandedFacade` (`BandedRealOps.h`)
 * `aether::math::sin/cos` and the `aether::math::sincos` Band overload
 * reach this file's bodies through `BandedFacade<T>::{sin,cos,sincos}`
 * (`BandedRealOps.h`): each new member there is a one-line forward,
 * additive only.
 *
 * @section trigReduceDispatch `trigReduce` has no `aether::math::` dispatch entry
 * It is an internal reduction primitive (its own signature, `Band
 * trigReduce(Band, int&)`, does not fit any `aether::math` unary/binary
 * shape), so it stays `aether::banded::detail`-only. Its own admission is
 * `bandTrigAdmits`, shared with `sin`/`cos`/`sincos` (stated for a
 * consumer that calls one of those, not for `trigReduce` directly), and
 * its accuracy is certified transitively: every `sin`/`cos`/`sincos`
 * corpus row exercises `trigReduce` first, so a reduction defect surfaces
 * there. No separate golden table for raw `(k, r)` is minted.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandExpLog.h"
#include "aether/banded/detail/PiConstants.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

// =========================================================================
//  roundNearestEven -- rintf's replacement. `x` is round-to-nearest-even
//  BOTH times `trigReduce` calls it: once on `vh` (magnitude up to
//  ~2^24 * 2/pi ~ 2^23.4, comfortably inside int32's exact range) and once
//  on a value under 3/2 in magnitude. `roundToInt32` (`detail/Fp32.h`) is
//  round-to-nearest-even float->int32 (device `__float2int_rn` == PTX
//  `cvt.rni.f32.s32`; host `__builtin_lrintf`, default rounding mode); the
//  int32 -> float conversion back is EXACT whenever the rounded integer's
//  own magnitude is under 2^24 (guaranteed here, since `roundToInt32`'s
//  OUTPUT magnitude never exceeds `roundToInt32`'s own INPUT magnitude by
//  more than 1/2, and this file's only two call sites bound that input to
//  well under 2^24). The composition therefore reproduces `rintf` bit for
//  bit, host and device.
// =========================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float roundNearestEven(float x)
{
    return static_cast<float>(roundToInt32(x));
}

// =========================================================================
//  Reduction constants.
// =========================================================================

/// @brief `pi/2` as four Cody-Waite FP32 limbs, exact for `k*CWi` at every
/// representable `|k| <= 2^24` (a `twoProd`-based reduction, not the
/// zeroed-mantissa trick `exp`'s `ln2` limbs use).
// pi/2 Cody-Waite limbs: aether/banded/detail/PiConstants.h

/// @brief `2/pi` as an unevaluated FP32 pair, for `k = round(x * 2/pi)`.
inline constexpr std::uint32_t kBand2OverPiHiBits = 0x3F22F983u;
inline constexpr std::uint32_t kBand2OverPiLoBits = 0x32DC9C88u;

/// @brief Where the trig reduction stops being exact: `k` is carried as a
/// float (so `twoProd(k, CWi)` is exact) AND as an int (quadrant `k & 3`).
/// The float side binds: past `2^24` an integer is no longer representable
/// in FP32. Guarding on `|x| <= 2^24` gives `|k| <= 2^24*2/pi + 1/2 <
/// 2^23.4`, 0.6 binades of slack over the resolution limit.
inline constexpr int   kBandTrigKBits  = 24;
inline constexpr float kBandTrigSatMax = 16777216.0f; ///< 2^24

/// @brief The largest base-2 argument exponent the reduction is exact for.
inline constexpr int kBandTrigMaxArgExp = kBandTrigKBits;

/// @brief The certified argument bound at the DEFAULT margin: `2^16 rad`.
inline constexpr float kBandTrigCertifiedMax = static_cast<float>(
    1u << static_cast<unsigned>(kBandTrigMaxArgExp - kBandAdmissionMargin));

/// @brief The certified ABSOLUTE error of `sin`/`cos`: `2^-68` (the
/// reduction contributes `~2^-70`; this carries a binade of declared slack).
inline constexpr int    kBandTrigAbsBoundExp = -68;
inline constexpr double kBandTrigAbsBound    = 3.3881317890172014e-21; // 2^-68

/// @brief Where the ULP claim starts holding, DERIVED from the absolute
/// one: `2^-68 * 2^53 / 2^-13 = 0.25`, a quarter of the half-ULP budget.
inline constexpr int kBandTrigUlpFloorExp = -13;

/**
 * @brief Admission for a consumer that calls `sin`, `cos` or `sincos` (and,
 * transitively, `trigReduce` -- @see this file's own "trigReduceDispatch"
 * DEVIATION note), stated on the ARGUMENT's largest magnitude exponent.
 *
 * One leg, no floor: the reduction's absolute error is UNIFORM over the
 * whole admitted range (it does not grow with `|x|`, @see `trigReduce`'s
 * own doc comment below) and the output is bounded, so there is no gradual
 * degraded band to guard -- past the bound `k` stops being representable
 * and the guard fires a NaN. The margin buys DECLARATION headroom under a
 * sharp cliff, the same job it does in `bandCeilingAdmits`.
 *
 * @param maxAbsExpArg base-2 exponent of the largest argument magnitude.
 * @param margin       declaration headroom in exponent steps (default
 *                     `kBandAdmissionMargin`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandTrigAdmits(
    int maxAbsExpArg, int margin = kBandAdmissionMargin)
{
    return maxAbsExpArg + margin <= kBandTrigMaxArgExp;
}

// Two-sided: one admitted case and one rejected case per edge.
static_assert(bandTrigAdmits(16),
    "the DEFAULT-margin certified argument bound must be admitted by its "
    "own predicate -- if this fails the constant and the guard have "
    "drifted apart");
static_assert(!bandTrigAdmits(17),
    "one binade past the certified bound must be REJECTED at the default "
    "margin; a guard that only rejects far excursions has no edge");
static_assert(bandTrigAdmits(17, 4) && !bandTrigAdmits(17),
    "trig's domain must respond to the margin exactly as the other legs do");
// The margin below matches a real downstream consumer's worst-case
// unwrapped argument: an unwrapped nutation-precession polynomial over a
// multi-century span reaches 7.1934e5 rad = 2^19.46. `(20, 4)` is the
// declaration that consumer must make; `(17, 4)` covers a shorter-period
// consumer and is admitted too, but is not the widest binding case.
static_assert(bandTrigAdmits(20, 4),
    "the binding consumer declaration must be admitted: a nutation-"
    "precession argument reaching 2^19.46 rad over a century declares "
    "(20, 4). A predicate that rejects it sends a real consumer to CompDD");
static_assert(!bandTrigAdmits(20, 5),
    "(20, 4) must be the WIDEST margin that admits that argument -- "
    "otherwise the arm above understates what the consumer is spending");
static_assert(bandTrigAdmits(17, 4) && 131072.0f < 719343.0f,
    "the narrower (17, 4) declaration, 2^17 = 1.311e5 rad, is 5.5x SHORT of "
    "the 7.1934e5 rad the consumer actually reaches -- a compile-time "
    "fact, not just a paragraph");
static_assert(bandTrigAdmits(kBandTrigMaxArgExp, 0),
    "at zero margin the reduction's own exactness bound must be admitted "
    "-- the margin is headroom under the cliff, not part of the cliff");
static_assert(!bandTrigAdmits(kBandTrigMaxArgExp + 1, 0),
    "past FP32's 24-bit integer resolution `k` is no longer representable "
    "and the reduced argument is wrong by a whole pi/2 -- no margin makes "
    "that safe");
static_assert(kBandTrigCertifiedMax <= kBandTrigSatMax,
    "the CERTIFIED argument bound must lie inside the runtime guard, or a "
    "consumer that declared correctly would still be NaN'd");
static_assert(kBandTrigAbsBoundExp + 53 - kBandTrigUlpFloorExp <= -2,
    "at the ULP floor the absolute error must spend at most a quarter of "
    "the half-ULP budget; it does not, so the two bounds are inconsistent");
static_assert(kBandTrigAbsBoundExp + 53 - (kBandTrigUlpFloorExp - 1) > -2,
    "the ULP floor is one binade higher than the derivation requires -- it "
    "is a choice, and this rung does not ship chosen bounds");

// =========================================================================
//  trigReduce -- `x = k*(pi/2) + r`, `|r| <= pi/4`, `r` exact to ~2^-70
//  ABSOLUTE for every admitted `x`, uniformly.
// =========================================================================

/**
 * @brief `x = k*(pi/2) + r`, `|r| <= pi/4`, `r` exact to `~2^-70` ABSOLUTE
 * for every admitted `x` -- uniformly, not degrading with `|x|`.
 *
 * @par Why the absolute error does not grow with the argument
 * The leading subtraction `x.hi - p1_hi` is EXACT by Sterbenz (`p1_hi`
 * approximates `k*pi/2`, within `pi/4` of `x`, so the two operands sit
 * within a factor of two for every `|k| >= 1`; `k == 0` subtracts against
 * zero). After it, every surviving term sits at magnitude `2^1` or below
 * regardless of how large `x` was, so the carrier's 72 bits buy `2^-71`
 * there rather than `2^-71 * |x|`. The only `|k|`-proportional term is
 * `pi/2`'s own truncation at `2^-103.2`, reaching `2^-79.2` at the domain
 * edge. THIS is what makes a wide domain affordable at all.
 *
 * @par Why `exactSum4` and not `addRaw`
 * `addRaw` folds its residuals with three PLAIN FP32 adds, rounding at
 * `2^-24` of the largest residual -- invisible when the two operands are
 * already banded and fatal here: the four level-0 terms below are all at
 * `2^0.65`, so `addRaw` would round the reduction at `2^-47`, twenty-three
 * binades above what it needs. The terms are grouped BY BINADE and each
 * group summed with a `twoSum` cascade that drops nothing; only the final
 * combination of two properly banded groups goes through `addRaw`.
 *
 * @par `p4_lo` is dropped, and that is a derivation not an omission
 * `|k * CW4| <= 2^-52.3` at the domain edge, so its `twoProd` residual is
 * under `2^-76.3` -- six binades below the `2^-70` the rest of the
 * reduction rounds at.
 *
 * @par TWO-STAGE rounding (a bug this round's own gate caught)
 * `rintf(vh + vc)` rounds TWICE (the add lands on a float grid of spacing
 * `ulp(v)`, `rintf` then lands on the integers), compounding to `0.5 +
 * ulp(v)/2` -- 0.75 for `|v|` in `[2^22, 2^23)`, putting `|r|` at 1.18, half
 * again over the cores' `pi/4` truncation bound. Splitting the integer part
 * off FIRST removes the compounding: `vh - k1` is exact, so the second
 * `roundNearestEven` sees a value under 3/2 and rounds it once. Measured:
 * max|r| 0.788152 -> 0.785398.
 *
 * @param x    the argument.
 * @param kOut the reduction quadrant, `k = round(x * 2/pi)`.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band trigReduce(Band x, int& kOut)
{
    const float cw1 = intAsFloat(static_cast<int>(kBandPiO2CW1Bits));
    const float cw2 = intAsFloat(static_cast<int>(kBandPiO2CW2Bits));
    const float cw3 = intAsFloat(static_cast<int>(kBandPiO2CW3Bits));
    const float cw4 = intAsFloat(static_cast<int>(kBandPiO2CW4Bits));
    const float ch  = intAsFloat(static_cast<int>(kBand2OverPiHiBits));
    const float cl  = intAsFloat(static_cast<int>(kBand2OverPiLoBits));

    // k = round(x * 2/pi), in an unevaluated pair. The three corrections
    // bring the product's relative error from 2^-24 to ~2^-48.
    const float vh = x.hi * ch;
    float       vc = fmaRN(x.hi, ch, -vh);
    vc             = fmaRN(x.hi, cl, vc);
    vc             = fmaRN(x.lo, ch, vc);
    const float k1 = roundNearestEven(vh);
    const float kf = k1 + roundNearestEven((vh - k1) + vc);
    kOut           = static_cast<int>(kf);

    float p1h, p1l, p2h, p2l, p3h, p3l, p4h, p4l;
    twoProd(kf, cw1, p1h, p1l);
    twoProd(kf, cw2, p2h, p2l);
    twoProd(kf, cw3, p3h, p3l);
    twoProd(kf, cw4, p4h, p4l);

    // Level 0 (~2^0.65) and level 1 (~2^-23.4), each summed exactly.
    const Band a = exactSum4(x.hi - p1h, x.lo, -p1l, -p2h);
    const Band b = exactSum4(x.tail, -p2l, -p3h, -(p3l + p4h));
    return normalizeSafe(addRaw(a, b));
}

// =========================================================================
//  sinCoreRaw / cosCoreRaw / trigSelectCore.
//  Demand ladder: FP32 rung (j high), double-float rung (`ddFma`), Band
//  rung (`fmaRawNoCancel`/`fmaRawSNoCancel`), ONE `normalize` at the exit.
// =========================================================================

/// @brief `sin(r)/r` as a polynomial in `u = r*r`, `|r| <= pi/4`. Two Band
/// steps (not three): `c_2` evaluates on the double-float rung, a 2.32x
/// measured margin over the 0.5 ULP contract.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw sinCoreRaw(Band u)
{
    float p = 2.8114573590e-15f;                  // c8
    p       = fmaRN(u.hi, p, -7.6471636098e-13f);  // c7
    p       = fmaRN(u.hi, p, 1.6059044372e-10f);   // c6

    float ph = p, pl = 0.0f;
    ddFma(u.hi, u.lo, ph, pl, -2.5052107944e-08f, -4.4176230446e-16f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, 2.7557318845e-06f, 3.7935712243e-14f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, -1.9841270114e-04f, 2.7255968749e-12f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, 8.3333337680e-03f, -4.3461720334e-10f, ph, pl);

    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawNoCancel(u, q,
        BandRaw{ -1.6666667163e-01f, 4.9670538793e-09f, -1.4802974103e-16f });
    return fmaRawSNoCancel(u, q, 1.0f); // c0 == 1 exactly -> scalar addend
}

/// @brief `cos(r)` as a polynomial in `u = r*r`, `|r| <= pi/4`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw cosCoreRaw(Band u)
{
    float p = -1.5619206969e-16f;                 // d9
    p       = fmaRN(u.hi, p, 4.7794772561e-14f);   // d8
    p       = fmaRN(u.hi, p, -1.1470745361e-11f);  // d7

    float ph = p, pl = 0.0f;
    ddFma(u.hi, u.lo, ph, pl, 2.0876755880e-09f, 1.1082839147e-16f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, -2.7557319981e-07f, 7.5751122091e-15f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, 2.4801587642e-05f, -3.4069960937e-13f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl, -1.3888889225e-03f, 3.3631094437e-11f, ph, pl);

    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawNoCancel(u, q,
        BandRaw{ 4.1666667908e-02f, -1.2417634698e-09f, 3.7007435257e-17f });
    q = fmaRawSNoCancel(u, q, -0.5f); // d1 == -1/2 exactly
    return fmaRawSNoCancel(u, q, 1.0f); // d0 == 1 exactly
}

/**
 * @brief ONE Horner sweep answering `sin(r)` or `cos(r)` -- the
 * coefficient-swap core for a caller wanting exactly one of them (`sin`/
 * `cos` below; `sincos` uses `sinCoreRaw`/`cosCoreRaw` directly instead: a
 * lone `sin`/`cos` is a DIFFERENT body that delivers DIFFERENT BITS from
 * `sincos(x,s,c).s` on the same argument -- both inside the certified
 * bound, neither a re-spelling of the other).
 *
 * @param r       the reduced argument, `|r| <= pi/4`.
 * @param u       `r*r`.
 * @param wantCos select the cosine series (and drop the trailing `* r`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band trigSelectCore(
    Band r, Band u, bool wantCos)
{
    // FP32 rung, j = 9, 8, 7. sin's c_9 is 0 -- its series stops at 8.
    float p = wantCos ? -1.5619206969e-16f : 0.0f;
    p = fmaRN(u.hi, p, wantCos ? 4.7794772561e-14f : 2.8114573590e-15f);
    p = fmaRN(u.hi, p, wantCos ? -1.1470745361e-11f : -7.6471636098e-13f);

    // double-float rung, j = 6, 5, 4, 3.
    float ph = p, pl = 0.0f;
    ddFma(u.hi, u.lo, ph, pl,
        wantCos ? 2.0876755880e-09f : 1.6059044372e-10f,
        wantCos ? 1.1082839147e-16f : -5.3525265116e-18f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl,
        wantCos ? -2.7557319981e-07f : -2.5052107944e-08f,
        wantCos ? 7.5751122091e-15f : -4.4176230446e-16f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl,
        wantCos ? 2.4801587642e-05f : 2.7557318845e-06f,
        wantCos ? -3.4069960937e-13f : 3.7935712243e-14f, ph, pl);
    ddFma(u.hi, u.lo, ph, pl,
        wantCos ? -1.3888889225e-03f : -1.9841270114e-04f,
        wantCos ? 3.3631094437e-11f : 2.7255968749e-12f, ph, pl);

    // Band rung, j = 2, 1, 0.
    BandRaw q{ ph, pl, 0.0f };
    q = fmaRawNoCancel(u, q,
        BandRaw{ wantCos ? 4.1666667908e-02f : 8.3333337680e-03f,
            wantCos ? -1.2417634698e-09f : -4.3461720334e-10f,
            wantCos ? 3.7007435257e-17f : 1.8503718042e-18f });
    q = fmaRawNoCancel(u, q,
        BandRaw{ wantCos ? -5.0000000000e-01f : -1.6666667163e-01f,
            wantCos ? 0.0f : 4.9670538793e-09f,
            wantCos ? 0.0f : -1.4802974103e-16f });
    q = fmaRawSNoCancel(u, q, 1.0f); // c0 == d0 == 1 -> F-S on BOTH branches

    // The cosine series IS the answer; the sine series is `r * P(u)`. One
    // multiply serves both, by selecting the multiplicand.
    const BandRaw m{ wantCos ? 1.0f : r.hi, wantCos ? 0.0f : r.lo,
        wantCos ? 0.0f : r.tail };
    return normalize(mulRaw(m, q));
}

// =========================================================================
//  sincos / sin / cos.
// =========================================================================

/**
 * @brief `sin(x)` and `cos(x)` together -- the primitive, because a
 * quadrant-reduced `sin` needs the cosine core anyway.
 *
 * @par Domain
 *     runtime guard   |x| <= 2^24 = 16777216 rad      -> else NaN, both outputs
 *     certified       |x| <  2^16 = 65536    rad      (default margin 8)
 * Accuracy is UNIFORM over the whole admitted range (the reduction's
 * absolute error does not grow with `|x|`) and then the guard fires --
 * there is no gradual degraded band to document, per `bandTrigAdmits`'s
 * own doc comment above.
 *
 * @par The certified bound has TWO parts
 *     |computed - exact| <= 2^-57 * |exact| + 2^-68        (absolute leg)
 *     |computed - exact| <= 0.5 ULP-at-53                   (|exact| >= 2^-13)
 * `sincos` consumers are structurally exempt from the gap near a zero --
 * `sin` and `cos` cannot both be small, so at least one output always
 * carries the full relative bound.
 *
 * @par The exact points
 * `sin(+-0) == +-0` with the sign, `cos(+-0) == 1` bit-exactly. For every
 * non-zero `x` small enough that `r*r` vanishes, `sin(x) == x` BIT-exactly.
 *
 * @par Host/device: bit-identical
 * No hardware seed, no table, no branch on data except the quadrant
 * select; `roundNearestEven` is round-to-nearest-even on both arms (this
 * file's own doc comment above).
 *
 * @par Cost (isolation census, sm_61) 492 FP32 / 729 total, against 972
 * FP32 for a separate `sin` plus `cos`.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void sincos(Band x, Band& sinOut, Band& cosOut)
{
    // Guards. Both comparisons are false for a NaN, so the negated
    // conjunction catches NaN, both infinities and the out-of-domain finite
    // arguments in one test; `x.hi == 0` is split out because the sign of a
    // zero does not survive `normalize`.
    if (!(x.hi >= -kBandTrigSatMax && x.hi <= kBandTrigSatMax)) {
        const Band nan{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
        sinOut = nan;
        cosOut = nan;
        return;
    }
    if (x.hi == 0.0f) {
        sinOut = Band{ x.hi, 0.0f, 0.0f }; // +-0, sign preserved
        cosOut = Band{ 1.0f, 0.0f, 0.0f };
        return;
    }

    int        k = 0;
    const Band r = trigReduce(x, k);
    const Band u = normalize(sqrRaw(r));

    const Band s = normalize(mulRaw(r, sinCoreRaw(u)));
    const Band c = normalize(cosCoreRaw(u));

    // Quadrant. `k & 3` is the residue for negative `k` too (two's
    // complement):
    //   sin: q=0 +s, q=1 +c, q=2 -s, q=3 -c
    //   cos: q=0 +c, q=1 -s, q=2 -c, q=3 +s
    const bool swap = (k & 1) != 0;
    const Band a    = swap ? c : s;
    const Band b    = swap ? s : c;
    sinOut          = (k & 2) != 0 ? neg(a) : a;
    cosOut          = ((k + 1) & 2) != 0 ? neg(b) : b;
}

/**
 * @brief `sin(x)` -- ONE Horner sweep via `trigSelectCore`, not a `sincos`
 * that throws half away (measured 366 FP32 against 485 for the wrapper
 * form, -24.5%). `sin(x)` is NOT bit-identical to `sincos(x,s,c).s` (both
 * are inside the certified bound, neither a re-spelling of the other --
 * see `trigSelectCore`'s own doc comment). `sin(+-0) == +-0`, `sin(x) ==
 * x` where `u` underflows, and the guard NaNs -- all three come from the
 * shared prologue, not from either core.
 *
 * @par Domain: identical to `sincos`'s (see `bandTrigAdmits`).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sin(Band x)
{
    if (!(x.hi >= -kBandTrigSatMax && x.hi <= kBandTrigSatMax))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    if (x.hi == 0.0f)
        return Band{ x.hi, 0.0f, 0.0f }; // +-0, sign preserved

    int        k = 0;
    const Band r = trigReduce(x, k);
    const Band u = normalize(sqrRaw(r));
    // sin(k*pi/2 + r): k=0 +sin, k=1 +cos, k=2 -sin, k=3 -cos.
    const Band v = trigSelectCore(r, u, (k & 1) != 0);
    return (k & 2) != 0 ? neg(v) : v;
}

/// @brief `cos(x)` -- the same single sweep, quadrant map rotated by one:
/// `cos(k*pi/2 + r)` is `+cos, -sin, -cos, +sin` for `k & 3`. @see sin.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band cos(Band x)
{
    if (!(x.hi >= -kBandTrigSatMax && x.hi <= kBandTrigSatMax))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    if (x.hi == 0.0f)
        return Band{ 1.0f, 0.0f, 0.0f };

    int        k = 0;
    const Band r = trigReduce(x, k);
    const Band u = normalize(sqrRaw(r));
    const Band v = trigSelectCore(r, u, (k & 1) == 0);
    return ((k + 1) & 2) != 0 ? neg(v) : v;
}

} // namespace detail
} // namespace banded
} // namespace aether
