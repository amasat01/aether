// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandRoot.h
 * @brief `cbrt` / `hypot` / the `rsqrt` IEEE-divergence wrapper / the root
 *        family's admission predicates.
 *
 * `sqrt`/`rsqrt`/`rsqrtCube` themselves are not redefined here — they are
 * `aether/banded/RsqrtCore.h`'s `sqrt_`/`rsqrt`/`rsqrtCube`, already
 * correct for everything except the one fix below. This file supplies
 * the two additional op bodies (`cbrt`, `hypot`), the `rsqrt(+Inf)` IEEE
 * fix as a wrapper (never touching `RsqrtCore.h`'s own `rsqrt`), and the
 * family's admission predicates plus two-sided `static_assert`s.
 *
 * @section specials `rsqrt(+Inf)`
 * The hardware seed answers `+0` for a `+Inf` operand, so the leading
 * limb is finite at `rsqrtCore`'s exit and `normalize`'s NaN-propagation
 * guard never fires — the carrier comes back NaN where C99 Annex F asks
 * for `+0`. Because `RsqrtCore.h::rsqrt` is not touched, the fix lives
 * here as `rsqrtIeee` — an operand guard staged exactly like `sqrt_`'s own
 * specials guard (same file, `bandHiIsSpecial`/`bandSpecialCarrier`
 * idiom) — and `BandedRealOps.h`'s `BandedFacade::rsqrt` forwards to this,
 * not to `banded::detail::rsqrt` directly. `rsqrtCube(+Inf)` is not
 * touched: it has no analogous fix.
 *
 * @section cbrt cbrt
 * Seed/refine/fused-correction (`cbrtSeedRaw`/`cbrtRefineF32`/`cbrt`),
 * including the three magic constants (`kBandCbrtSeedMagic`/
 * `kBandThirdF`/`kBandTwoThirdsF`). Total on the admissible carrier, odd
 * (`cbrt(-x) == neg(cbrt(x))` limb for limb), certified bound
 * `0.5 * ulp53(|exact|)`, derived from the dropped `(5/9)E^2` term
 * (`2^-77.8`) plus the scalar correction's own rounding (`2^-62.1`) plus
 * the Band arithmetic floor (`2^-70.0`), summing to `2^-61.9`, 478x under
 * `2^-53`.
 *
 * @section hypot hypot
 * Reuses `sqrt_`, `sqrRaw`, `addRawNoCancel`, `scalePow2f`, `abs` — all
 * already in `aether/banded/Band.h`/`RsqrtCore.h`, so nothing new is
 * needed at the primitive level. Overflow-/underflow-safe via exact
 * power-of-two pre-scaling by the larger operand's own binade (one binade
 * wider here: `hypot`'s admission is stated on `max(|x|,|y|)`, the
 * smaller operand's own additive entry is exempt). Certified bound
 * `2^-65.9` (1.4e-4 ULP), contract 0.5 ULP.
 *
 * @section admits Admission predicates
 * `bandCbrtAdmits` and `bandHypotAdmits` both measure clean (0 max |ULP|
 * over this package's own MPFR corpus, `tests/bandmath/golden_{cbrt,
 * hypot}.h`). `sqrt`/`rsqrt`/`rsqrtCube` have no simple mirror predicate:
 * a first draft mechanically applied `bandCbrtAdmits`'s mirror-predicate
 * technique to `RsqrtCore`'s `y ~ x^(-1/2)` intermediate, and this
 * package's own MPFR probe caught it wrong — the mirror-only window
 * admitted `sqrt_(x)` at `|x`'s exponent`| ~ 107` while the measured error
 * there is 2021 ULP, four orders past any "0.5 ULP" claim. `bandRsqrtAdmits`
 * below is therefore measured-then-derived, not mirror-only: @see its own
 * doc comment for the traced root cause (`rsqrtCore`'s refinement-step
 * `twoProd` residual going FP32-subnormal) and the measured curve.
 * `bandRsqrtCubeAdmits` escapes this cliff by construction — the same
 * probe found it clean up to `|x's exponent| ~ 60` while its own
 * delivery-floor leg already caps admission at `~32` — so its two legs
 * (both at the output's `-3/2` scale) stand as originally derived, no
 * measured correction needed. `ceilHalf`/`floorHalf` below are a
 * scale-1/2-correct equivalent of a `(+2)/3` ceiling-division spelling,
 * verified against every integer parity by hand.
 */

#include "aether/banded/Band.h"
#include "aether/banded/RsqrtCore.h"
#include "aether/macros.h"

#include <cstdint>

namespace aether {
namespace banded {
namespace detail {

// =====================================================================
//  bandFloorAdmits -- the delivery floor. Lives here rather than in
//  `Band.h` because `rsqrtCube` is the first op whose own domain doc
//  names it (the composition it replaces owes `bandIntermediateAdmits`
//  instead; `rsqrtCube` itself owes this one).
// =====================================================================

/// @brief The nominal output-floor exponent. `bandToIEEE`'s store terminal
/// drops any limb whose float exponent field is zero (subnormal); nominal
/// `-56` with the standard 8-binade margin admits `-48`.
inline constexpr int kBandNominalFloorExp = -56;

/// @brief Does a consumer's declared smallest output magnitude stay clear of
/// the store terminal's subnormal-limb drop, with margin? Binds what an op
/// delivers, never an interior multiplicative intermediate (that is
/// `bandIntermediateAdmits`'s job) -- kept separate because the mechanism
/// differs (store-terminal truncation vs. FP32-subnormal carrier collapse).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandFloorAdmits(
    int minOutputExp, int margin = kBandAdmissionMargin)
{
    return minOutputExp - margin >= kBandNominalFloorExp;
}

// =====================================================================
//  Integer ceiling/floor of n/2 -- exact for any sign of n, no floating
//  point (this file's admission predicates are AETHER_DEVICEHOST()
//  constexpr, and device caps at C++20 where std::ceil/floor are not
//  constexpr -- exactly the reason bandCbrtAdmits's own (+2)/3 trick is
//  integer-only). Verified: ceilHalf({5,4,-5,-4}) = {3,2,-2,-2};
//  floorHalf({5,4,-5,-4}) = {2,2,-3,-2} -- matches ceil/floor(n/2.0) at
//  every parity.
// =====================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr int ceilHalf(int n)
{
    return (n >= 0) ? (n + 1) / 2 : -((-n) / 2);
}
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr int floorHalf(int n)
{
    return (n >= 0) ? n / 2 : -((-n + 1) / 2);
}

// =====================================================================
//  bandRsqrtAdmits / bandSqrtAdmits -- measured, not just mirror-derived.
//  A first draft mirrored the argument's own ceiling/floor at
//  rsqrtCore's y~x^(-1/2) scale (the same technique bandCbrtAdmits
//  uses) and admitted up to |x's own exponent| ~= 107 at the standard
//  8-binade margin. This package's own MPFR probe (`gen_corpus_root.py`'s
//  golden rows, host, `sqrt_`/`rsqrtIeee`) found that window wrong: at
//  `e=107` the measured error is **2021 ULP** for `sqrt_`, not the ~0.5-1
//  ULP the mirror-only derivation implied. Root cause, traced by hand: in
//  `rsqrtCore`'s refinement step, `se` (the `twoProd(y0,y0,s,se)`
//  residual) sits at `ulp(s)` scale, `s = y0^2`; for `x`'s own exponent
//  `E`, `y0`'s exponent is `~-E/2`, `s`'s is `~-E`, and `se`'s is
//  `~-E-23` -- FP32-subnormal (below -126) once `E > ~103`, losing
//  mantissa bits exactly where the refinement correction needs them
//  most. Measured (this package's own probe, MPFR truth, both
//  directions -- the same cliff appears mirrored at tiny `x` too,
//  `x.lo`/`x.tail` themselves approaching the carrier's own subnormal-
//  float boundary): `<=1 ULP` for `|E| <= 98`, already `150`/`276` ULP
//  by `|E| = 103`/`107`, `>1e5` ULP by `|E| = 116-128`. This governs
//  both directions identically (unlike the generic ceiling/floor, which
//  are asymmetric), so the predicate is a single symmetric threshold on
//  `|E|` rather than the generic `bandCeilingAdmits`/
//  `bandIntermediateAdmits` pair. `sqrt_`'s own extra intermediate `P =
//  x*y ~ x^(+1/2)` stays strictly milder than this cliff at every
//  admitted `E`, so `bandSqrtAdmits` is the same predicate under its
//  own name.
// =====================================================================

/// @brief `|x's own float exponent|` past which `rsqrtCore`'s refinement
/// residual `se` measurably loses FP32 mantissa bits (see the derivation
/// above). Symmetric: binds `sqrt_`/`rsqrt` on either side of zero, unlike
/// the generic (asymmetric) `kBandNominalCeilingExp`/`kBandCarrierFloorExp`.
inline constexpr int kBandRsqrtCeilingExp = 103;

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandRsqrtAdmits(
    int maxAbsExpArg, int minAbsExpArg, int margin = kBandAdmissionMargin)
{
    const int worstAbsExp = (maxAbsExpArg > -minAbsExpArg) ? maxAbsExpArg : -minAbsExpArg;
    return worstAbsExp + margin <= kBandRsqrtCeilingExp;
}

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandSqrtAdmits(
    int maxAbsExpArg, int minAbsExpArg, int margin = kBandAdmissionMargin)
{
    return bandRsqrtAdmits(maxAbsExpArg, minAbsExpArg, margin);
}

// =====================================================================
//  bandRsqrtCubeAdmits -- two legs, both at the output's own -3/2 scale
//  (Y = x^(-3/2) is what `normalize` actually delivers, and what an FP32
//  leading limb has to hold): (1) the delivery ceiling: `x` tiny
//  (`minAbsExpArg` very negative) makes Y huge (found by this package's
//  own probe: without this leg, x ~ 2^-148 would be let through as
//  "admitted" while Y's own exponent (~2^222) is 94 binades past the
//  FP32 hard ceiling, though `rsqrtCube` correctly answers `+Inf`
//  there); (2) the delivery floor: `x` large (`maxAbsExpArg` large)
//  makes Y tiny, reproduced exactly below for E=32 (`-1.5E >= -48, i.e.
//  E <= 32`). No intermediate leg: `rsqrtCube` never forms `x^1.5` at
//  all, so `bandIntermediateAdmits` never enters.
// =====================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandRsqrtCubeAdmits(
    int maxAbsExpArg, int minAbsExpArg, int margin = kBandAdmissionMargin)
{
    const int outCeilMirror = ceilHalf(-3 * minAbsExpArg);  // Y's ceiling risk (x tiny)
    const int hi              = (outCeilMirror > maxAbsExpArg) ? outCeilMirror : maxAbsExpArg;
    const int minOutputExp   = floorHalf(-3 * maxAbsExpArg); // Y's delivery-floor risk (x large)
    return bandCeilingAdmits(hi, margin) && bandFloorAdmits(minOutputExp, margin);
}

// =====================================================================
//  bandCbrtAdmits -- total on the real line; the widest intermediate is
//  `r^2 = a^(-2/3)`, mirroring the argument's own exponent about zero at
//  two-thirds scale.
// =====================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandCbrtAdmits(
    int maxAbsExpArg, int minAbsExpArg, int margin = kBandAdmissionMargin)
{
    const int hiMirror = (-2 * minAbsExpArg + 2) / 3;
    const int loMirror = -((2 * maxAbsExpArg + 2) / 3);
    const int hi        = (hiMirror > maxAbsExpArg) ? hiMirror : maxAbsExpArg;
    const int lo         = (loMirror < minAbsExpArg) ? loMirror : minAbsExpArg;
    return bandCeilingAdmits(hi, margin) && bandIntermediateAdmits(lo, margin);
}

// =====================================================================
//  bandHypotAdmits -- stated on the larger argument only: the ceiling leg
//  is owed at `E(max)+1` (the answer can exceed the larger operand by
//  half a bit), the carrier-floor leg at `E(max)` (pre-scaling removes
//  the squares' own excursion entirely).
// =====================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandHypotAdmits(
    int maxAbsExpArg, int minAbsExpArg, int margin = kBandAdmissionMargin)
{
    return bandCeilingAdmits(maxAbsExpArg + 1, margin) && bandIntermediateAdmits(minAbsExpArg, margin);
}

// =====================================================================
//  rsqrtIeee -- rsqrt with the `rsqrt(+Inf) == +0` fix (IEEE/C99 Annex
//  F). An operand guard, staged ahead of `rsqrtCore`'s seed rather than
//  patched at the exit; `-Inf` and NaN are left untouched (they fall
//  through to `rsqrt`, which already answers NaN for both) since the fix
//  is scoped to `+Inf` only.
// =====================================================================
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band rsqrtIeee(Band x)
{
    if (bandHiIsSpecial(x.hi)) {
        const bool plusInf = static_cast<std::uint32_t>(floatAsInt(x.hi)) == kBandFp32ExpMask;
        if (plusInf)
            return Band{ 0.0f, 0.0f, 0.0f }; // rsqrt(+Inf) = +0 -- the fix (not x.hi, which is +Inf here).
    }
    return rsqrt(x);
}

// =====================================================================
//  cbrt
// =====================================================================

/// @brief Magic-number seed for `a^(-1/3)`, `|d| <= 0.1358`.
inline constexpr std::uint32_t kBandCbrtSeedMagic = 0x548C6ACBu;
/// @brief `1/3` rounded to one float -- the Newton-refinement coefficient.
inline constexpr float kBandThirdF = 3.3333333433e-01f;
/// @brief `2/3` rounded to one float -- the scalar-correction coefficient.
inline constexpr float kBandTwoThirdsF = 6.6666668653e-01f;

[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float cbrtSeedRaw(float a)
{
    const std::uint32_t b = static_cast<std::uint32_t>(floatAsInt(a));
    return intAsFloat(static_cast<int>(kBandCbrtSeedMagic - b / 3u));
}

/// @brief One FP32 Newton step for `a^(-1/3)` in correction form.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float cbrtRefineF32(float a, float r)
{
    float w = mulRN(a, r);
    w       = mulRN(w, r);
    w       = mulRN(w, r);
    const float e = subRN(1.0f, w);
    return mulRN(r, fmaRN(e, kBandThirdF, 1.0f));
}

/// @brief `x^(1/3)`, total on the admissible carriers, exactly odd. @see
/// the file header's "cbrt" section for the derived bound (0.5 ULP).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band cbrt(Band x)
{
    const int  sgn = floatAsInt(x.hi) & static_cast<int>(0x80000000u);
    const Band a   = applySignMask(x, sgn); // |x|, three XORs

    // Zero, NaN and +inf are their own cube roots, and are exactly the
    // arguments the iteration cannot start from. `!(a.hi > 0)` catches the
    // zero and the NaN in one test.
    if (!(a.hi > 0.0f) || a.hi == intAsFloat(0x7F800000))
        return x;

    float r = cbrtSeedRaw(a.hi);
    r       = cbrtRefineF32(a.hi, r);
    r       = cbrtRefineF32(a.hi, r);
    r       = cbrtRefineF32(a.hi, r);
    r       = cbrtRefineF32(a.hi, r);

    // The fifth refinement, kept as an unevaluated pair.
    float s, se;
    twoProd(r, r, s, se); // r^2 = s + se, exact
    float c, ce;
    twoProd(r, s, c, ce); // r^3 = c + ce + r*se
    const float t0 = fmaRN(-a.hi, c, 1.0f);
    const float w0 = fmaRN(a.hi, ce, mulRN(a.lo, c));
    const float w1 = fmaRN(mulRN(a.hi, r), se, w0);
    const float e  = subRN(t0, w1); // e = 1 - a r^3, about 2^-22.4
    const BandRaw y{ r, mulRN(r, mulRN(e, kBandThirdF)), 0.0f };

    // At exponent +1/3, `P = a y^2` is the residual's own subexpression
    // and the answer, and the correction `(1 + 2E/3)` lands on the
    // product.
    const BandRaw yy = sqrRaw(y);
    const BandRaw P  = mulRaw(a, yy);
    const BandRaw Pr = mulRaw(P, y);
    const float a1   = subRN(1.0f, Pr.hi); // exact (Sterbenz)
    const float a2   = subRN(a1, Pr.lo);
    const float E    = subRN(a2, Pr.tail);

    const float he  = mulRN(kBandTwoThirdsF, E);
    const float cHi = mulRN(P.hi, he);
    const float cLo = addRN(fmaRN(P.hi, he, -cHi), mulRN(P.lo, he));
    float lo2, t2;
    fast2Sum(P.lo, cHi, lo2, t2);
    // `P.tail` is large here for `sqrt_`'s reason -- `y` is a deliberately
    // unnormalized two-limb pair -- so it is folded in, not dropped.
    const Band m = normalize(BandRaw{ P.hi, lo2, addRN(addRN(t2, P.tail), cLo) });
    return applySignMask(m, sgn);
}

// =====================================================================
//  hypot
// =====================================================================

/// @brief `hypot(x, y) = sqrt(x^2 + y^2)`, overflow- and underflow-safe.
/// @see the file header's "hypot" section for the derived bound (0.5 ULP).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band hypot(Band x, Band y)
{
    const float kInf = intAsFloat(0x7F800000);

    Band ax = abs(x);
    Band ay = abs(y);
    // Order by magnitude first, so the scaling exponent, the addend order
    // and the zero test are all symmetric in the two arguments by
    // construction.
    if (ax.hi < ay.hi) {
        const Band t = ax;
        ax           = ay;
        ay           = t;
    }

    // Infinity wins over a NaN (IEEE-754 hypot); the swap above put an
    // infinite operand in `ax` unless the other one is a NaN, so both
    // tests are needed.
    if (ax.hi == kInf || ay.hi == kInf)
        return Band{ kInf, 0.0f, 0.0f };
    if (!(ax.hi - ax.hi == 0.0f) || !(ay.hi - ay.hi == 0.0f))
        return Band{ intAsFloat(0x7FC00000), 0.0f, 0.0f };
    // `ay.hi == 0` also covers `hypot(0,0) == +0`: `ax` is then `+0` too.
    if (ay.hi == 0.0f)
        return ax;

    // Exact power-of-two pre-scaling by the larger magnitude's binade.
    const int   e  = ((floatAsInt(ax.hi) >> 23) & 0xFF) - 127;
    const int   h1 = (-e) >> 1;
    const float f1 = intAsFloat((127 + h1) << 23);
    const float f2 = intAsFloat((127 + (-e - h1)) << 23);
    const Band  xs = scalePow2f(scalePow2f(ax, f1), f2);
    const Band  ys = scalePow2f(scalePow2f(ay, f1), f2);

    // No-cancellation argument: both addends are squares, hence
    // non-negative, so the sum cannot cancel.
    const Band s = normalize(addRawNoCancel(sqrRaw(xs), sqrRaw(ys)));

    // Undo the scaling on the root, so the exponent halves: `2^e`, split.
    const int g1 = e >> 1;
    return scalePow2f(scalePow2f(sqrt_(s), intAsFloat((127 + g1) << 23)), intAsFloat((127 + (e - g1)) << 23));
}

// =====================================================================
//  Two-sided static_asserts, following the same pattern as `bandExpAdmits`.
// =====================================================================
static_assert(bandRsqrtAdmits(0, 0) && bandRsqrtAdmits(60, -60),
    "bandRsqrtAdmits must admit the well-conditioned interior");
static_assert(!bandRsqrtAdmits(0, -170) && !bandRsqrtAdmits(220, 0),
    "bandRsqrtAdmits must reject an operand whose y-mirror leaves the carrier");
static_assert(bandSqrtAdmits(0, 0) == bandRsqrtAdmits(0, 0),
    "bandSqrtAdmits is documented as bandRsqrtAdmits under its own name");
static_assert(bandRsqrtAdmits(95, 95) && !bandRsqrtAdmits(96, 96),
    "bandRsqrtAdmits's edge must sit exactly at the MEASURED se-subnormal "
    "cliff (kBandRsqrtCeilingExp - kBandAdmissionMargin = 95), not a "
    "mirror-only derivation's looser one (~107, measured 2021 ULP there)");
static_assert(bandRsqrtAdmits(0, -95) && !bandRsqrtAdmits(0, -96),
    "the cliff is SYMMETRIC: a tiny operand hits it at the same |exponent| a huge one does");

static_assert(bandRsqrtCubeAdmits(0, 0) && bandRsqrtCubeAdmits(32, -32),
    "bandRsqrtCubeAdmits must admit the well-conditioned interior (E<=32)");
static_assert(!bandRsqrtCubeAdmits(33, -32),
    "bandRsqrtCubeAdmits must reject past the worked delivery-floor edge (E=33)");
static_assert(!bandRsqrtCubeAdmits(0, -148),
    "bandRsqrtCubeAdmits must reject a tiny operand whose -3/2-scale delivery ceiling "
    "overflows FP32 (the labelling gap this package's own probe found -- x ~ 2^-148 "
    "answers +Inf, 94 binades past the hard ceiling, if this leg is missing)");

static_assert(bandCbrtAdmits(0, 0) && bandCbrtAdmits(80, -80),
    "bandCbrtAdmits must admit the well-conditioned interior");
static_assert(bandCbrtAdmits(0, -88) && !bandCbrtAdmits(0, -89),
    "bandCbrtAdmits's floor mirror must sit exactly at the carrier's own edge");

static_assert(bandHypotAdmits(106, -88), "bandHypotAdmits must admit the well-conditioned interior");
static_assert(!bandHypotAdmits(107, -88),
    "bandHypotAdmits must reject past bandCeilingAdmits(maxArg+1)'s own edge");

} // namespace detail
} // namespace banded
} // namespace aether
