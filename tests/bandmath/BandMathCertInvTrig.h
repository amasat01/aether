// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCertInvTrig.h
 * @brief Certification drivers + RED-first machinery for atan/atan2/asin/
 *        acos, extending `tests/bandmath/BandMathCert.h`. These ops share
 *        `sqrt`/`rsqrt`/`cbrt`/`hypot`'s single-leg ULP-53 contract, but
 *        each op's degraded band mixes multiple independent mechanisms
 *        into one exponent bucket, so reusing `certifyRootUnary`/`Binary`
 *        fires spuriously; `certifyInvTrigDomainUnary`/`Binary` below drop
 *        the monotonicity claim instead. `asin`/`acos` additionally enforce
 *        `bandAsinAdmits`'s hard, no-margin [-1,1] domain. RED-first arms
 *        perturb the real reduction-table entry or drop the real leading
 *        polynomial term (matching `sin`/`cos`'s own technique), run
 *        through standalone copies of the real code.
 */

#include "tests/bandmath/BandMathCertRoot.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace aether_tests {
namespace bandmath {

// =========================================================================
//  certifyInvTrigDomainUnary -- asin/acos's own driver. @see file docstring
//  "asinacos" section: same per-row three-way contract certifyRootUnary
//  uses, WITHOUT the exponent-distance monotonicity claim (there is no
//  degradation curve for a hard, no-margin domain to be monotone about).
// =========================================================================
template<typename Row, typename UnaryOp>
CertCounts certifyInvTrigDomainUnary(const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    CertCounts c;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) && (d <= static_cast<std::int64_t>(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": ref=" << rv << " got=" << gv << " ulp=" << d
                               << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            // The ONE narrow exception this driver exists for (@see file
            // docstring): bandFromIEEE's ingest floor rounds a too-tiny
            // double to exact 0, which happens to land within `bound` of
            // the true (equally tiny) reference by coincidence, not by a
            // curve -- weak claim only, stays finite.
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    return c;
}

// =========================================================================
//  certifyInvTrigDomainBinary -- atan2's own driver, same reasoning as
//  certifyInvTrigDomainUnary above (@see file docstring "asinacos"
//  section, extended to atan2): MEASURED (this package's own probe),
//  atan2's degraded band mixes MULTIPLE independent mechanisms bucketed
//  by ONE exponent key (bandAtan2Admits's own ceiling-margin leg, its
//  divisor-ceiling leg, its intermediate-floor leg AND its output-floor
//  leg all land degraded rows in the SAME `max(operand exponent)` bucket
//  at different distances) -- no SINGLE smooth curve for
//  certifyRootBinary's monotonicity check to be monotone about. Same
//  three-way per-row contract, no monotonicity claim.
// =========================================================================
template<typename Row, typename BinaryOp>
CertCounts certifyInvTrigDomainBinary(const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    CertCounts c;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band b    = bandFromDoubleBits(r.in[1]);
        const Band got  = op(a, b);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) && (d <= static_cast<std::int64_t>(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": ref=" << rv << " got=" << gv << " ulp=" << d
                               << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > static_cast<std::int64_t>(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    return c;
}

// =========================================================================
//  RED-first arm 1a: MATCHED-PAIR TABLE-ENTRY PERTURBATION (atan/atan2).
//  A standalone copy of `AtanSector`'s table + `atanSector` + `atan2`'s own
//  reduction, with sector 4's stored `t.hi` nudged by one float ULP while
//  `phi` is left as the (now WRONG) angle for the unperturbed `t` — the
//  matched-pair invariant `BandInvTrig.h`'s own docstring calls load-
//  bearing, deliberately broken. Everything else is `atan2`'s own
//  unmodified reduction/reconstruction, copied verbatim (`AtanSector.h`/
//  `BandInvTrig.h`, do-not-touch, are never edited).
// =========================================================================
inline bd::AtanSector atanSectorPerturbedT(float b, float a)
{
    // Sector 4's own two stored constants (AtanSector.h), t.hi nudged by
    // one float ULP toward +inf; phi is copied UNCHANGED -- no longer
    // `atan` of the perturbed t.
    const bool s1 = b > a * 8.7488666177e-02f;
    const bool s2 = b > a * 2.6794919372e-01f;
    const bool s3 = b > a * 4.6630766988e-01f;
    const bool s4 = b > a * 7.0020753145e-01f;
    const float perturbedT4Hi = std::nextafterf(8.3909964561e-01f, std::numeric_limits<float>::infinity());
    const Band t{ s4 ? perturbedT4Hi
                     : (s3 ? 5.7735025883e-01f : (s2 ? 3.6397022009e-01f : (s1 ? 1.7632697523e-01f : 0.0f))),
        s4 ? -1.4437343765e-08f
           : (s3 ? 1.0362416702e-08f : (s2 ? 1.4177243379e-08f : (s1 ? 5.4820628037e-09f : 0.0f))),
        s4 ? -2.4673516535e-16f
           : (s3 ? -4.1063892585e-16f : (s2 ? 2.4185138514e-16f : (s1 ? -1.1295152588e-16f : 0.0f))) };
    const Band phi{ s4 ? 6.9813168049e-01f
                       : (s3 ? 5.2359879017e-01f : (s2 ? 3.4906584024e-01f : (s1 ? 1.7453292012e-01f : 0.0f))),
        s4 ? 2.0309144588e-08f
           : (s3 ? -1.4570463058e-08f : (s2 ? 1.0154572294e-08f : (s1 ? 5.0772861471e-09f : 0.0f))),
        s4 ? 8.1670630901e-16f
           : (s3 ? -2.7564868795e-16f : (s2 ? 4.0835315450e-16f : (s1 ? 2.0417657725e-16f : 0.0f))) };
    return bd::AtanSector{ t, phi };
}

inline Band atan2PerturbedSector(Band y, Band x)
{
    const Band  pio2 = bd::bandPiO2();
    const int   kAbs = static_cast<int>(0x7FFFFFFFu);
    const float ayh  = bd::intAsFloat(bd::floatAsInt(y.hi) & kAbs);
    const float axh  = bd::intAsFloat(bd::floatAsInt(x.hi) & kAbs);
    const float kInf = bd::intAsFloat(0x7F800000);
    const bool  yNeg = bd::floatAsInt(y.hi) < 0;
    const bool  xNeg = bd::floatAsInt(x.hi) < 0;
    const Band  zeroB{ 0.0f, 0.0f, 0.0f };

    Band spec   = zeroB;
    bool isSpec = true;
    if (y.hi != y.hi || x.hi != x.hi)
        return Band{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
    else if (ayh == kInf)
        spec = (axh == kInf) ? (xNeg ? bd::add(pio2, bd::scalePow2f(pio2, 0.5f)) : bd::scalePow2f(pio2, 0.5f)) : pio2;
    else if (axh == kInf || y.hi == 0.0f)
        spec = xNeg ? bd::scalePow2f(pio2, 2.0f) : zeroB;
    else if (x.hi == 0.0f)
        spec = pio2;
    else
        isSpec = false;
    if (isSpec)
        return yNeg ? bd::neg(spec) : spec;

    const Band ay   = bd::abs(y);
    const Band axb  = bd::abs(x);
    const bool swap = ayh > axh;
    const Band b    = swap ? axb : ay;
    const Band a    = swap ? ay : axb;

    const bd::AtanSector sec = atanSectorPerturbedT(b.hi, a.hi); // PERTURBED table copy
    const Band num = bd::normalize(bd::fmaRaw(bd::neg(a), sec.t, b));
    const Band den = bd::normalize(bd::fmaRaw(b, sec.t, a));

    const Band t = bd::div(num, den);
    const Band u = bd::normalize(bd::sqrRaw(t));
    const Band s = bd::normalize(bd::mulRaw(t, bd::atanCoreRaw(u)));
    const Band A = bd::add(sec.phi, s);

    const Band B = bd::scalePow2f(pio2, swap ? 1.0f : (xNeg ? 2.0f : 0.0f));
    const Band r = bd::add(B, (swap != xNeg) ? bd::neg(A) : A);
    return yNeg ? bd::neg(r) : r;
}

inline std::uint32_t bitsOf(float v) { return std::bit_cast<std::uint32_t>(v); }

template<typename Row>
std::size_t countRedUnderMatchedPairPerturbationAtan2(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band y   = bandFromDoubleBits(r.in[0]);
        const Band x   = bandFromDoubleBits(r.in[1]);
        const Band got = atan2PerturbedSector(y, x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderMatchedPairPerturbationAtan(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = atan2PerturbedSector(x, Band{ 1.0f, 0.0f, 0.0f });
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

// =========================================================================
//  RED-first arm 1b: asin/acos's own shared-circle constant, perturbed.
//  DEVIATION: asin/acos consume no AtanSector table, so this arm nudges
//  `bandPiO2()`'s CW1 limb instead, standing in for the "table entry"/
//  "reduction constant" the RED-first technique above expects.
// =========================================================================
inline Band bandPiO2PerturbedCw1()
{
    const float cw1 = std::nextafterf(
        bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW1Bits)), std::numeric_limits<float>::infinity());
    return Band{ cw1, bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW2Bits)),
        bd::intAsFloat(static_cast<int>(bd::kBandPiO2CW3Bits)) };
}

inline Band asinPerturbedPiO2(Band x)
{
    bool       high = false, bad = false;
    const Band v = bd::asinReducedRaw(bd::abs(x), high, bad);
    if (bad)
        return v;
    const Band m = high ? bd::sub(bandPiO2PerturbedCw1(), bd::scalePow2f(v, 2.0f)) : v;
    return (bd::floatAsInt(x.hi) < 0) ? bd::neg(m) : m;
}

inline Band acosPerturbedPiO2(Band x)
{
    bool       high = false, bad = false;
    const Band v    = bd::asinReducedRaw(bd::abs(x), high, bad);
    if (bad)
        return v;
    const Band pio2 = bandPiO2PerturbedCw1();
    const Band m    = high ? bd::scalePow2f(v, 2.0f) : bd::sub(pio2, v);
    return (bd::floatAsInt(x.hi) < 0) ? bd::sub(bd::scalePow2f(pio2, 2.0f), m) : m;
}

template<typename Row>
std::size_t countRedUnderMatchedPairPerturbationAsin(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = asinPerturbedPiO2(x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderMatchedPairPerturbationAcos(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = acosPerturbedPiO2(x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

// =========================================================================
//  RED-first arm 2: DROPPED-POLYNOMIAL-TERM TWIN. `atanCoreRaw`/
//  `asinCoreRaw` with their `c_0` (== 1, the ONE term that does not scale
//  with `u`) terminal step skipped, over the REAL unmodified reduction,
//  matching the RED-first technique described above.
// =========================================================================
inline bd::BandRaw atanCoreRawDroppedC0(Band u)
{
    float p = -5.2631579340e-02f;
    p       = bd::fmaRN(u.hi, p, 5.8823529631e-02f);
    p       = bd::fmaRN(u.hi, p, -6.6666670144e-02f);
    p       = bd::fmaRN(u.hi, p, 7.6923079789e-02f);
    p       = bd::fmaRN(u.hi, p, -9.0909093618e-02f);
    float ph = p, pl = 0.0f;
    bd::ddFma(u.hi, u.lo, ph, pl, 1.1111111194e-01f, -8.2784229471e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, -1.4285714924e-01f, 6.3862120037e-09f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 2.0000000298e-01f, -2.9802322832e-09f, ph, pl);
    bd::BandRaw q{ ph, pl, 0.0f };
    q = bd::fmaRawNoCancel(u, q, bd::BandRaw{ -3.3333334327e-01f, 9.9341077586e-09f, -2.9605948206e-16f });
    // c0 (== 1) DROPPED -- the certified body's terminal fmaRawSNoCancel
    // scalar-addend step is simply not run.
    return q;
}

inline bd::BandRaw asinCoreRawDroppedC0(Band u)
{
    float p = 1.6816094285e-03f;
    p       = bd::fmaRN(u.hi, p, 1.7680810997e-03f);
    p       = bd::fmaRN(u.hi, p, 1.8622264033e-03f);
    p       = bd::fmaRN(u.hi, p, 1.9650335889e-03f);
    p       = bd::fmaRN(u.hi, p, 2.0776609890e-03f);
    p       = bd::fmaRN(u.hi, p, 2.2014740389e-03f);
    p       = bd::fmaRN(u.hi, p, 2.3380918428e-03f);
    p       = bd::fmaRN(u.hi, p, 2.4894487578e-03f);
    p       = bd::fmaRN(u.hi, p, 2.6578705292e-03f);
    p       = bd::fmaRN(u.hi, p, 2.8461783659e-03f);
    p       = bd::fmaRN(u.hi, p, 3.0578216538e-03f);
    p       = bd::fmaRN(u.hi, p, 3.2970595639e-03f);
    p       = bd::fmaRN(u.hi, p, 3.5692052916e-03f);
    p       = bd::fmaRN(u.hi, p, 3.8809645921e-03f);
    p       = bd::fmaRN(u.hi, p, 4.2409072630e-03f);
    p       = bd::fmaRN(u.hi, p, 4.6601435170e-03f);
    float ph = p, pl = 0.0f;
    bd::ddFma(u.hi, u.lo, ph, pl, 5.1533095539e-03f, 1.2845828568e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 5.7400376536e-03f, 1.7246714473e-11f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 6.4472104423e-03f, -1.3038516100e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 7.3125259951e-03f, -1.2147685635e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 8.3903353661e-03f, 4.4348694161e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 9.7616091371e-03f, 3.9213582381e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 1.1551801115e-02f, -2.1913472426e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 1.3964843936e-02f, -1.8626451770e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 1.7352763563e-02f, 8.5968238084e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 2.2372158244e-02f, 8.4665691125e-10f, ph, pl);
    bd::ddFma(u.hi, u.lo, ph, pl, 3.0381944031e-02f, 4.1392114736e-10f, ph, pl);
    bd::BandRaw q{ ph, pl, 0.0f };
    q = bd::fmaRawNoCancel(u, q, bd::BandRaw{ 4.4642858207e-02f, -1.0643687043e-09f, 4.7580987242e-17f });
    q = bd::fmaRawNoCancel(u, q, bd::BandRaw{ 7.5000002980e-02f, -2.9802322832e-09f, 4.4408921647e-17f });
    q = bd::fmaRawNoCancel(u, q, bd::BandRaw{ 1.6666667163e-01f, -4.9670538793e-09f, 1.4802974103e-16f });
    // c0 (== 1) DROPPED.
    return q;
}

inline Band atan2DroppedTerm(Band y, Band x)
{
    const Band  pio2 = bd::bandPiO2();
    const int   kAbs = static_cast<int>(0x7FFFFFFFu);
    const float ayh  = bd::intAsFloat(bd::floatAsInt(y.hi) & kAbs);
    const float axh  = bd::intAsFloat(bd::floatAsInt(x.hi) & kAbs);
    const float kInf = bd::intAsFloat(0x7F800000);
    const bool  yNeg = bd::floatAsInt(y.hi) < 0;
    const bool  xNeg = bd::floatAsInt(x.hi) < 0;
    const Band  zeroB{ 0.0f, 0.0f, 0.0f };

    Band spec   = zeroB;
    bool isSpec = true;
    if (y.hi != y.hi || x.hi != x.hi)
        return Band{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
    else if (ayh == kInf)
        spec = (axh == kInf) ? (xNeg ? bd::add(pio2, bd::scalePow2f(pio2, 0.5f)) : bd::scalePow2f(pio2, 0.5f)) : pio2;
    else if (axh == kInf || y.hi == 0.0f)
        spec = xNeg ? bd::scalePow2f(pio2, 2.0f) : zeroB;
    else if (x.hi == 0.0f)
        spec = pio2;
    else
        isSpec = false;
    if (isSpec)
        return yNeg ? bd::neg(spec) : spec;

    const Band ay   = bd::abs(y);
    const Band axb  = bd::abs(x);
    const bool swap = ayh > axh;
    const Band b    = swap ? axb : ay;
    const Band a    = swap ? ay : axb;

    const bd::AtanSector sec = bd::atanSector(b.hi, a.hi); // the REAL, unmodified table
    const Band num = bd::normalize(bd::fmaRaw(bd::neg(a), sec.t, b));
    const Band den = bd::normalize(bd::fmaRaw(b, sec.t, a));

    const Band t = bd::div(num, den);
    const Band u = bd::normalize(bd::sqrRaw(t));
    const Band s = bd::normalize(bd::mulRaw(t, atanCoreRawDroppedC0(u))); // DROPPED core
    const Band A = bd::add(sec.phi, s);

    const Band B = bd::scalePow2f(pio2, swap ? 1.0f : (xNeg ? 2.0f : 0.0f));
    const Band r = bd::add(B, (swap != xNeg) ? bd::neg(A) : A);
    return yNeg ? bd::neg(r) : r;
}

template<typename Row>
std::size_t countRedUnderDroppedPolynomialTermAtan2(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band y   = bandFromDoubleBits(r.in[0]);
        const Band x   = bandFromDoubleBits(r.in[1]);
        const Band got = atan2DroppedTerm(y, x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderDroppedPolynomialTermAtan(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = atan2DroppedTerm(x, Band{ 1.0f, 0.0f, 0.0f });
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

inline Band asinDroppedTerm(Band x)
{
    bool       high = false, bad = false;
    Band       t, u;
    const Band ax = bd::abs(x);
    high          = ax.hi > 0.5f;
    if (high) {
        const Band w = bd::scalePow2f(bd::sub(Band{ 1.0f, 0.0f, 0.0f }, ax), 0.5f);
        if (!(w.hi >= 0.0f)) {
            bad = true;
        } else {
            t = bd::sqrt_(w);
            u = w;
        }
    } else {
        t = ax;
        u = bd::normalize(bd::sqrRaw(ax));
    }
    if (bad)
        return Band{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
    const Band v = bd::normalize(bd::mulRaw(t, asinCoreRawDroppedC0(u))); // DROPPED core
    const Band m = high ? bd::sub(bd::bandPiO2(), bd::scalePow2f(v, 2.0f)) : v;
    return (bd::floatAsInt(x.hi) < 0) ? bd::neg(m) : m;
}

inline Band acosDroppedTerm(Band x)
{
    bool       high = false, bad = false;
    Band       t, u;
    const Band ax = bd::abs(x);
    high          = ax.hi > 0.5f;
    if (high) {
        const Band w = bd::scalePow2f(bd::sub(Band{ 1.0f, 0.0f, 0.0f }, ax), 0.5f);
        if (!(w.hi >= 0.0f)) {
            bad = true;
        } else {
            t = bd::sqrt_(w);
            u = w;
        }
    } else {
        t = ax;
        u = bd::normalize(bd::sqrRaw(ax));
    }
    if (bad)
        return Band{ bd::intAsFloat(0x7FC00000), 0.0f, 0.0f };
    const Band v    = bd::normalize(bd::mulRaw(t, asinCoreRawDroppedC0(u))); // DROPPED core
    const Band pio2 = bd::bandPiO2();
    const Band m    = high ? bd::scalePow2f(v, 2.0f) : bd::sub(pio2, v);
    return (bd::floatAsInt(x.hi) < 0) ? bd::sub(bd::scalePow2f(pio2, 2.0f), m) : m;
}

template<typename Row>
std::size_t countRedUnderDroppedPolynomialTermAsin(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = asinDroppedTerm(x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

template<typename Row>
std::size_t countRedUnderDroppedPolynomialTermAcos(const Row* rows, std::size_t n, double bound)
{
    std::size_t red = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        if (r.label != 0)
            continue;
        const Band x    = bandFromDoubleBits(r.in[0]);
        const Band got  = acosDroppedTerm(x);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);
        if (d == ulp::kMismatch || d > static_cast<std::int64_t>(bound))
            red++;
    }
    return red;
}

} // namespace bandmath
} // namespace aether_tests
