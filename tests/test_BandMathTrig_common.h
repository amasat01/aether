// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathTrig_common.h
 * @brief `BandMathTrig` suite — shared by
 *        `test_BandMathTrig.cpp` and `test_BandMathTrig.cu` (both TUs
 *        simply `#include` this file, the same twin convention — @see
 *        `test_BandMathCert_common.h`).
 *
 * Registers the certification + RED-first rows for `sin`/`cos`/`sincos`
 * (`aether::banded::detail::{sin,cos,sincos}`, `aether/banded/BandTrig.h`)
 * against the MPFR golden tables `tools/bandmath/gen_corpus_trig.py` mints
 * (`tests/bandmath/golden_{sin,cos,sincos}.h`). Certification driver
 * (`certifyTrigUnary`/`certifySincosRows`) and the LITERAL RED-first arms
 * (`countRedUnderReductionConstantPerturbation`/
 * `countRedUnderDroppedPolynomialTerm`) live in the NEW add-on header
 * `tests/bandmath/BandMathCertTrig.h` — `BandMathCert.h` itself is
 * do-not-touch and its own docstring sanctions this exact extension route.
 *
 * The RED-first arms run against the `sincos` corpus ONLY (not separately
 * for `sin`/`cos`): `trigReduce` and the two shared cores
 * (`sinCoreRaw`/`cosCoreRaw`) are the SAME machinery `sin`/`cos`/`sincos`
 * all consume (`sincos` calls them directly; `sin`/`cos` call the
 * coefficient-swap `trigSelectCore` instead, a DIFFERENT body over the SAME
 * reduction — @see `BandTrig.h`'s own doc comment), and the `sincos` golden
 * table carries BOTH legs (`ref`=sin, `refCos`=cos) per row, so one RED-first
 * pass over it exercises the reduction-constant/dropped-term defect against
 * both outputs at once.
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCertTrig.h"
#include "tests/bandmath/golden_sin.h"
#include "tests/bandmath/golden_cos.h"
#include "tests/bandmath/golden_sincos.h"

#include <cstdio>
#include <limits>
#include <cmath>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_sin    = ::aether_tests::bandmath_golden::sin;
namespace g_cos     = ::aether_tests::bandmath_golden::cos;
namespace g_sincos = ::aether_tests::bandmath_golden::sincos;

void printTrigCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathTrig %-6s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp), c.degradedPass,
        c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  sin -- certified over bandTrigAdmits, two-legged bound (0.5 ULP /
//  absolute 2^-68 below |ref|=2^-13). Uses trigSelectCore, NOT sincos's
//  own sinCoreRaw (@see BandTrig.h's own doc comment: different bodies).
// =========================================================================
TEST(BandMathTrig, SinCertifiesAdmitted)
{
    auto c = certifyTrigUnary(g_sin::kRows, g_sin::kCorpusSize, g_sin::kUlpBound, [](Band a) { return bd::sin(a); });
    printTrigCounts("sin", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  cos -- certified over bandTrigAdmits, same two-legged bound.
// =========================================================================
TEST(BandMathTrig, CosCertifiesAdmitted)
{
    auto c = certifyTrigUnary(g_cos::kRows, g_cos::kCorpusSize, g_cos::kUlpBound, [](Band a) { return bd::cos(a); });
    printTrigCounts("cos", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  sincos -- certified over the SAME domain, both outputs.
// =========================================================================
TEST(BandMathTrig, SincosCertifiesAdmitted)
{
    auto c = certifySincosRows(g_sincos::kRows, g_sincos::kCorpusSize, g_sincos::kUlpBound);
    printTrigCounts("sincos", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  RED-first (literal mechanism -- @see
//  BandMathCertTrig.h's own file docstring). Run against the sincos
//  corpus, which shares trigReduce/sinCoreRaw/cosCoreRaw with sin/cos.
// =========================================================================
TEST(BandMathTrig, RedFirstReductionConstantPerturbation)
{
    const std::size_t red = countRedUnderReductionConstantPerturbation(
        g_sincos::kRows, g_sincos::kCorpusSize, g_sincos::kUlpBound);
    std::fprintf(stderr, "BandMathTrig RED-first(reduction-constant-perturbation): %zu/%zu rows RED\n", red,
        g_sincos::kInDomainCount);
    EXPECT_GT(red, g_sincos::kInDomainCount * 9 / 10);
}
TEST(BandMathTrig, RedFirstDroppedPolynomialTerm)
{
    const std::size_t red
        = countRedUnderDroppedPolynomialTerm(g_sincos::kRows, g_sincos::kCorpusSize, g_sincos::kUlpBound);
    std::fprintf(
        stderr, "BandMathTrig RED-first(dropped-polynomial-term): %zu/%zu rows RED\n", red, g_sincos::kInDomainCount);
    EXPECT_GT(red, g_sincos::kInDomainCount * 9 / 10);
}

// =========================================================================
//  Exact points (BandTrig.h's own doc comment: sin(+-0)==+-0 WITH the
//  sign, cos(+-0)==1 bit-exactly, sin(x)==x where u underflows, the guard
//  NaNs both outputs) -- asserted DIRECTLY, not just via the corpus's own
//  weak 3-way contract (mirrors BandMathRoot's own
//  RsqrtPlusInfIsPlusZero test for the SAME reason: a labelled exact point
//  deserves its own row).
// =========================================================================
TEST(BandMathTrig, ExactPointsAtZeroAndTheGuard)
{
    const Band posZero{ 0.0f, 0.0f, 0.0f };
    const Band negZero{ -0.0f, 0.0f, 0.0f };

    const Band sp = bd::sin(posZero);
    EXPECT_EQ(sp.hi, 0.0f);
    EXPECT_FALSE(std::signbit(sp.hi));
    const Band sn = bd::sin(negZero);
    EXPECT_EQ(sn.hi, 0.0f);
    EXPECT_TRUE(std::signbit(sn.hi)) << "sin(-0) must preserve the sign";

    EXPECT_EQ(bd::cos(posZero).hi, 1.0f);
    EXPECT_EQ(bd::cos(negZero).hi, 1.0f);

    Band s, c;
    bd::sincos(posZero, s, c);
    EXPECT_EQ(s.hi, 0.0f);
    EXPECT_FALSE(std::signbit(s.hi));
    EXPECT_EQ(c.hi, 1.0f);
    bd::sincos(negZero, s, c);
    EXPECT_EQ(s.hi, 0.0f);
    EXPECT_TRUE(std::signbit(s.hi));
    EXPECT_EQ(c.hi, 1.0f);

    // The runtime guard: |x| <= 2^24 admitted (even though far past the
    // DEFAULT-margin certified edge), |x| > 2^24 -> NaN, both outputs.
    const Band atGuard{ 16777216.0f, 0.0f, 0.0f }; // exactly 2^24
    EXPECT_TRUE(std::isfinite(bd::sin(atGuard).hi)) << "the guard is <=, not <";
    EXPECT_TRUE(std::isfinite(bd::cos(atGuard).hi));

    const Band pastGuard{ std::nextafterf(16777216.0f, std::numeric_limits<float>::infinity()), 0.0f, 0.0f };
    EXPECT_TRUE(std::isnan(bd::sin(pastGuard).hi));
    EXPECT_TRUE(std::isnan(bd::cos(pastGuard).hi));
    Band gs, gc;
    bd::sincos(pastGuard, gs, gc);
    EXPECT_TRUE(std::isnan(gs.hi));
    EXPECT_TRUE(std::isnan(gc.hi));
}

// =========================================================================
//  The worked consumer example, evaluated directly (not just enumerated
//  into the corpus): BandTrig.h's own bandTrigAdmits doc comment -- a
//  Mars-M2 nutation-precession argument, declaring (20, 4), reaching
//  7.1934e5 rad = 2^19.46 at T=+-1 century. This reduces and evaluates at
//  that magnitude rather than only asserting the inequality.
// =========================================================================
TEST(BandMathTrig, MarsM2NutationArgumentIsAdmittedAndAccurate)
{
    ASSERT_TRUE(bd::bandTrigAdmits(20, 4)) << "the binding consumer declaration must be admitted";

    const double kM2Const = 192.93;
    const double kM2Rate  = 41215163.19675;
    const double kDeg2Rad = 1.74532925199432957692e-2;

    double worstUlp = 0.0;
    for (int i = -100; i <= 100; i += 5) {
        const double T   = static_cast<double>(i) * 0.01;
        const double arg = (kM2Const + kM2Rate * T) * kDeg2Rad;
        ASSERT_LE(std::fabs(arg), static_cast<double>(bd::kBandTrigSatMax));

        const Band   x  = bandFromDoubleBits(doubleBitsOf(arg));
        Band         s, c;
        bd::sincos(x, s, c);
        const double gs = doubleOfBits(bandToDoubleBits(s));
        const double gc = doubleOfBits(bandToDoubleBits(c));
        const double rs = std::sin(arg);
        const double rc = std::cos(arg);
        EXPECT_TRUE(trigRowWithinBound(rs, gs, g_sincos::kUlpBound)) << "T=" << T << " arg=" << arg;
        EXPECT_TRUE(trigRowWithinBound(rc, gc, g_sincos::kUlpBound)) << "T=" << T << " arg=" << arg;
        const std::int64_t ds = ulp::ulpDistanceAbs(rs, gs);
        const std::int64_t dc = ulp::ulpDistanceAbs(rc, gc);
        if (ds != ulp::kMismatch)
            worstUlp = std::max(worstUlp, static_cast<double>(ds));
        if (dc != ulp::kMismatch)
            worstUlp = std::max(worstUlp, static_cast<double>(dc));
    }
    std::fprintf(stderr, "BandMathTrig Mars-M2 worked example: worst |ulp| over T in [-1,1] century = %.3f\n",
        worstUlp);
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
