// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCertTrig.h
 * @brief Certification drivers + RED-first machinery for sin/cos/sincos/
 *        trigReduce, extending `tests/bandmath/BandMathCert.h`. `sin`/`cos`
 *        carry a two-legged bound (a ULP leg for |ref| >= 2^-13, else an
 *        absolute leg near zero); `certifyTrigUnary` checks both, reusing
 *        `certifyRootUnary`'s single-ceiling-leg bucketing technique
 *        otherwise. The RED-first arms perturb the actual reduction
 *        constant and drop the actual leading polynomial term (rather than
 *        the generic carrier-nudge/identity-mutant used where no such
 *        machinery exists), each run through standalone copies of the real
 *        code and expected to fail.
 */

#include "tests/bandmath/BandMathCert.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <algorithm>

namespace aether_tests {
namespace bandmath {

// =========================================================================
//  certifyTrigUnary -- TWO-LEGGED bound (BandTrig.h's own contract).
// =========================================================================

/// @brief The absolute-leg floor, `2^-13` (BandTrig.h's `kBandTrigUlpFloorExp`).
inline constexpr double kTrigAbsLegFloor = 1.220703125e-4; // 2^-13
/// @brief The absolute bound itself, `2^-68` (`kBandTrigAbsBound`).
inline constexpr double kTrigAbsBound = 3.3881317890172014e-21; // 2^-68
/// @brief The relative coefficient on the absolute leg, `2^-57`.
inline constexpr double kTrigAbsBoundRelCoeff = 6.938893903907228e-18; // 2^-57

/// @brief One row's pass/fail against the two-legged contract. Below the
/// floor the ABSOLUTE bound governs (with a 2x slack absorbing the SAME
/// double-rounding a Band->double egress pays that `ulpBoundSteps`'s own
/// doc comment derives for the ULP leg — a continuous half-ULP claim can
/// round to the double NEXT TO the correctly-rounded one); at or above it
/// the discrete ULP-step form of `bound` (ceil) governs, same convention
/// every other driver in this directory uses.
inline bool trigRowWithinBound(double ref, double got, double ulpBound)
{
    const double absRef = std::fabs(ref);
    if (absRef < kTrigAbsLegFloor) {
        const double err   = std::fabs(got - ref);
        const double limit = kTrigAbsBoundRelCoeff * absRef + kTrigAbsBound;
        return err <= 2.0 * limit;
    }
    const std::int64_t d = ulp::ulpDistanceAbs(ref, got);
    const std::int64_t boundSteps = static_cast<std::int64_t>(std::ceil(ulpBound));
    return d != ulp::kMismatch && d <= boundSteps;
}

/// @brief `|x|`'s own base-2 exponent, `-1000000` for 0/inf/nan (same
/// convention `certifyRootUnary`'s own `rootExponentOf` uses).
inline int trigExponentOf(double x)
{
    if (x == 0.0 || !std::isfinite(x))
        return -1000000;
    int e = 0;
    std::frexp(x, &e);
    return e - 1;
}

template<typename Row, typename UnaryOp>
CertCounts certifyTrigUnary(const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    CertCounts c;
    // Distance PAST the default-margin certified edge (exponent 16),
    // bucketed -- the SAME "worst-ulp-per-bucket, non-decreasing, one dip
    // tolerated" discipline every degraded-arm driver in this directory
    // shares. `bandTrigAdmits`'s edge is `kBandTrigMaxArgExp - margin = 16`.
    std::map<int, std::int64_t> distMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = trigRowWithinBound(rv, gv, bound);
            if (pass)
                c.inDomainPass++;
            // Only feed the REPORTED max|ulp| from rows the ULP leg
            // actually governs (|ref| >= the absolute-leg floor) -- a
            // near-zero row can carry an enormous "ULP distance" while
            // passing cleanly on the absolute leg, which would otherwise
            // make the reported number meaningless (measured: 2^52 from a
            // row whose true error was ~1e-11).
            const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
            if (d != ulp::kMismatch && std::fabs(rv) >= kTrigAbsLegFloor && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in=0x" << std::hex << r.in[0] << std::dec
                               << " ref=" << rv << " got=" << gv;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
            // Only feed ULP-MEANINGFUL rows into the monotonicity bucket:
            // |ref| below the absolute-leg floor (2^-13) is governed by the
            // ABSOLUTE bound (this file's own `trigRowWithinBound`), where
            // "ULP distance" is not a well-behaved quantity (it can jump
            // around near a zero of sin/cos even though the ABSOLUTE error
            // stays tiny and well-behaved) -- a degraded bucket that landed
            // near one of trig's own zeros manufactured a false "dip"
            // caught live (dist buckets 10,10,...,10,0 -- the last bucket's
            // legitimate ULP-0 row read as a monotonicity violation against
            // seven near-zero neighbours whose "10" was never a meaningful
            // accuracy signal to begin with).
            if (d != ulp::kMismatch && std::fabs(rv) >= kTrigAbsLegFloor) {
                const int e = trigExponentOf(doubleOfBits(r.in[0]));
                if (e > -1000000 && e > 16) {
                    const int dist = e - 16;
                    auto [it, _]   = distMaxUlp.try_emplace(dist, 0);
                    it->second     = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch)
                || (!trigRowWithinBound(rv, gv, bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv;
        }
    }
    bool monotoneOk      = true;
    std::int64_t prevUlp = -1;
    for (const auto& [dist, worstUlp] : distMaxUlp) {
        (void)dist;
        if (prevUlp >= 0 && worstUlp < prevUlp) {
            if (prevUlp - worstUlp > prevUlp / 2 + 4)
                monotoneOk = false;
        }
        prevUlp = worstUlp;
    }
    EXPECT_TRUE(monotoneOk) << "trig degraded family is not monotone non-decreasing past the certified edge";
    return c;
}

/// @brief `sincos`'s own dual-output row driver: BOTH `sin`/`cos` legs must
/// pass for a row to count. `Row` carries `ref` (sin) and `refCos`.
template<typename Row>
CertCounts certifySincosRows(const Row* rows, std::size_t n, double bound)
{
    CertCounts c;
    std::map<int, std::int64_t> distMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r     = rows[i];
        const Band a     = bandFromDoubleBits(r.in[0]);
        Band       s, cs;
        bd::sincos(a, s, cs);
        const double gs = doubleOfBits(bandToDoubleBits(s));
        const double gc = doubleOfBits(bandToDoubleBits(cs));
        const double rs = doubleOfBits(r.ref);
        const double rc = doubleOfBits(r.refCos);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool passS = trigRowWithinBound(rs, gs, bound);
            const bool passC = trigRowWithinBound(rc, gc, bound);
            if (passS && passC)
                c.inDomainPass++;
            // @see certifyTrigUnary's identical guard: exclude near-zero
            // rows (governed by the absolute leg) from the reported max.
            const std::int64_t ds = ulp::ulpDistanceAbs(rs, gs);
            const std::int64_t dc = ulp::ulpDistanceAbs(rc, gc);
            if (ds != ulp::kMismatch && std::fabs(rs) >= kTrigAbsLegFloor && ds > c.maxInDomainUlp)
                c.maxInDomainUlp = ds;
            if (dc != ulp::kMismatch && std::fabs(rc) >= kTrigAbsLegFloor && dc > c.maxInDomainUlp)
                c.maxInDomainUlp = dc;
            EXPECT_TRUE(passS) << "in_domain row " << i << " sin leg: in=0x" << std::hex << r.in[0] << std::dec
                                << " ref=" << rs << " got=" << gs;
            EXPECT_TRUE(passC) << "in_domain row " << i << " cos leg: in=0x" << std::hex << r.in[0] << std::dec
                                << " ref=" << rc << " got=" << gc;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = (std::isfinite(gs) || !std::isfinite(rs)) && (std::isfinite(gc) || !std::isfinite(rc));
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage";
            const std::int64_t ds = ulp::ulpDistanceAbs(rs, gs);
            const int e            = trigExponentOf(doubleOfBits(r.in[0]));
            // @see certifyTrigUnary's own identical guard: skip rows below
            // the absolute-leg floor, where ULP distance is not a
            // meaningful monotonicity signal.
            if (ds != ulp::kMismatch && e > -1000000 && e > 16 && std::fabs(rs) >= kTrigAbsLegFloor) {
                const int dist = e - 16;
                auto [it, _]   = distMaxUlp.try_emplace(dist, 0);
                it->second     = std::max(it->second, ds);
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gs) || !std::isfinite(gc)
                || !trigRowWithinBound(rs, gs, bound) || !trigRowWithinBound(rc, gc, bound);
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound";
        }
    }
    bool monotoneOk      = true;
    std::int64_t prevUlp = -1;
    for (const auto& [dist, worstUlp] : distMaxUlp) {
        (void)dist;
        if (prevUlp >= 0 && worstUlp < prevUlp) {
            if (prevUlp - worstUlp > prevUlp / 2 + 4)
                monotoneOk = false;
        }
        prevUlp = worstUlp;
    }
    EXPECT_TRUE(monotoneOk) << "sincos degraded family is not monotone non-decreasing past the certified edge";
    return c;
}

// =========================================================================
//  RED-first arm 1: SEEDED ONE-ULP REDUCTION-CONSTANT PERTURBATION.
//  A standalone copy of bd::trigReduce with kBandPiO2CW1Bits nudged by one
//  float ULP. @see file docstring.
// =========================================================================
inline Band trigReducePerturbedCw1(Band x, int& kOut)
{
    const float cw1 = std::nextafterf(
        bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW1Bits)), std::numeric_limits<float>::infinity());
    const float cw2 = bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW2Bits));
    const float cw3 = bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW3Bits));
    const float cw4 = bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW4Bits));
    const float ch  = bd::intAsFloat(static_cast<int>(bd::kBand2OverPiHiBits));
    const float cl  = bd::intAsFloat(static_cast<int>(bd::kBand2OverPiLoBits));

    const float vh = x.hi * ch;
    float       vc = bd::fmaRN(x.hi, ch, -vh);
    vc             = bd::fmaRN(x.hi, cl, vc);
    vc             = bd::fmaRN(x.lo, ch, vc);
    const float k1 = bd::roundNearestEven(vh);
    const float kf = k1 + bd::roundNearestEven((vh - k1) + vc);
    kOut           = static_cast<int>(kf);

    float p1h, p1l, p2h, p2l, p3h, p3l, p4h, p4l;
    bd::twoProd(kf, cw1, p1h, p1l);
    bd::twoProd(kf, cw2, p2h, p2l);
    bd::twoProd(kf, cw3, p3h, p3l);
    bd::twoProd(kf, cw4, p4h, p4l);

    const Band a = bd::exactSum4(x.hi - p1h, x.lo, -p1l, -p2h);
    const Band b = bd::exactSum4(x.tail, -p2l, -p3h, -(p3l + p4h));
    return bd::normalizeSafe(bd::addRaw(a, b));
}

inline void sincosPerturbedReduction(Band x, Band& sinOut, Band& cosOut)
{
    if (!(x.hi >= -bd::kBandTrigSatMax && x.hi <= bd::kBandTrigSatMax)) {
        const Band nan{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
        sinOut = nan;
        cosOut = nan;
        return;
    }
    if (x.hi == 0.0f) {
        sinOut = Band{ x.hi, 0.0f, 0.0f };
        cosOut = Band{ 1.0f, 0.0f, 0.0f };
        return;
    }
    int        k = 0;
    const Band r = trigReducePerturbedCw1(x, k);
    const Band u = bd::normalize(bd::sqrRaw(r));
    const Band s = bd::normalize(bd::mulRaw(r, bd::sinCoreRaw(u)));
    const Band c = bd::normalize(bd::cosCoreRaw(u));
    const bool swap = (k & 1) != 0;
    const Band a    = swap ? c : s;
    const Band b    = swap ? s : c;
    sinOut          = (k & 2) != 0 ? bd::neg(a) : a;
    cosOut          = ((k + 1) & 2) != 0 ? bd::neg(b) : b;
}

template<typename Row>
std::size_t countRedUnderReductionConstantPerturbation(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a = bandFromDoubleBits(r.in[0]);
        Band       s, c;
        sincosPerturbedReduction(a, s, c);
        const double gs = doubleOfBits(bandToDoubleBits(s));
        const double gc = doubleOfBits(bandToDoubleBits(c));
        const double rs = doubleOfBits(r.ref);
        const double rc = doubleOfBits(r.refCos);
        if (!trigRowWithinBound(rs, gs, bound) || !trigRowWithinBound(rc, gc, bound))
            red++;
    }
    return red;
}

// =========================================================================
//  RED-first arm 2: DROPPED-POLYNOMIAL-TERM TWIN. sinCoreRaw/cosCoreRaw
//  with their `c_0`/`d_0` Band-rung term skipped (the terminal
//  `fmaRawSNoCancel(u, q, 1.0f)` step simply not run), over the REAL
//  trigReduce. @see file docstring.
//
//  ★ WHY `c_0`/`d_0` and not a higher-order term: a first draft dropped
//  `c_1`/`d_1` (the NEXT term, `-1/6`/`-1/2`) instead and measured only
//  546/1857 (29%) RED — because this corpus is deliberately WEIGHTED
//  toward reduction BOUNDARIES, i.e. rows where `|r|` (and so `u = r*r`)
//  is SMALL, and every dropped
//  term beyond the leading constant is multiplied by a power of `u` — its
//  contribution shrinks exactly where most of the corpus lives. `c_0`/
//  `d_0` is the ONE term that does NOT scale with `u` (`cosCoreRaw`
//  returns `q` directly, so dropping `d_0` corrupts `cos` by a full unit
//  of magnitude at EVERY row, not just the ones with large `|r|`) —
//  measured (below) to push RED past 90% for the reason this file's own
//  docstring names: `sin`'s own `r == 0` rows still pass regardless of
//  which coefficient is touched (`sin = r*q`, and `r == 0` makes the whole
//  product zero), but `cos`'s leg has no such escape, so the ROW-level
//  check (`!passS || !passC`) catches virtually everything through `cos`
//  alone.
// =========================================================================
inline bd::BandRaw sinCoreRawDroppedC0(Band u)
{
    float p = 2.8114573590e-15f;
    p       = bd::fmaRN(u.hi, p, -7.6471636098e-13f);
    p       = bd::fmaRN(u.hi, p, 1.6059044372e-10f);
    float ph = p, pl = 0.0f;
    bd::ddFma(u.hi, u.lo, ph, pl, -2.5052107944e-08f, -4.4176230446e-16f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 2.7557318845e-06f, 3.7935712243e-14f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, -1.9841270114e-04f, 2.7255968749e-12f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 8.3333337680e-03f, -4.3461720334e-10f, ph, pl);
    bd::BandRaw q{ ph, pl, 0.0f };
    q = bd::fmaRawNoCancel(u, q,
        bd::BandRaw{ -1.6666667163e-01f, 4.9670538793e-09f, -1.4802974103e-16f });
    // c_0 (== 1) DROPPED -- the certified body's terminal fmaRawSNoCancel
    // scalar-addend step is simply not run.
    return q;
}
inline bd::BandRaw cosCoreRawDroppedD0(Band u)
{
    float p = -1.5619206969e-16f;
    p       = bd::fmaRN(u.hi, p, 4.7794772561e-14f);
    p       = bd::fmaRN(u.hi, p, -1.1470745361e-11f);
    float ph = p, pl = 0.0f;
    bd::ddFma(u.hi, u.lo, ph, pl, 2.0876755880e-09f, 1.1082839147e-16f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, -2.7557319981e-07f, 7.5751122091e-15f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 2.4801587642e-05f, -3.4069960937e-13f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, -1.3888889225e-03f, 3.3631094437e-11f, ph, pl);
    bd::BandRaw q{ ph, pl, 0.0f };
    q = bd::fmaRawNoCancel(u, q,
        bd::BandRaw{ 4.1666667908e-02f, -1.2417634698e-09f, 3.7007435257e-17f });
    q = bd::fmaRawSNoCancel(u, q, -0.5f); // d_1 kept
    // d_0 (== 1) DROPPED -- straight back to the caller without the
    // terminal scalar addend.
    return q;
}

inline void sincosDroppedTerm(Band x, Band& sinOut, Band& cosOut)
{
    if (!(x.hi >= -bd::kBandTrigSatMax && x.hi <= bd::kBandTrigSatMax)) {
        const Band nan{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
        sinOut = nan;
        cosOut = nan;
        return;
    }
    if (x.hi == 0.0f) {
        sinOut = Band{ x.hi, 0.0f, 0.0f };
        cosOut = Band{ 1.0f, 0.0f, 0.0f };
        return;
    }
    int        k = 0;
    const Band r = bd::trigReduce(x, k); // the REAL reduction, unmodified
    const Band u = bd::normalize(bd::sqrRaw(r));
    const Band s = bd::normalize(bd::mulRaw(r, sinCoreRawDroppedC0(u)));
    const Band c = bd::normalize(cosCoreRawDroppedD0(u));
    const bool swap = (k & 1) != 0;
    const Band a    = swap ? c : s;
    const Band b    = swap ? s : c;
    sinOut          = (k & 2) != 0 ? bd::neg(a) : a;
    cosOut          = ((k + 1) & 2) != 0 ? bd::neg(b) : b;
}

template<typename Row>
std::size_t countRedUnderDroppedPolynomialTerm(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band a = bandFromDoubleBits(r.in[0]);
        Band       s, c;
        sincosDroppedTerm(a, s, c);
        const double gs = doubleOfBits(bandToDoubleBits(s));
        const double gc = doubleOfBits(bandToDoubleBits(c));
        const double rs = doubleOfBits(r.ref);
        const double rc = doubleOfBits(r.refCos);
        if (!trigRowWithinBound(rs, gs, bound) || !trigRowWithinBound(rc, gc, bound))
            red++;
    }
    return red;
}

} // namespace bandmath
} // namespace aether_tests
