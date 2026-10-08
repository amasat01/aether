// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathInvTrig_common.h
 * @brief `BandMathInvTrig` suite — shared by `test_BandMathInvTrig.cpp` and
 *        `test_BandMathInvTrig.cu` (mirrors `test_BandMathRoot_common.h`'s
 *        own twin convention).
 *
 * Registers the certification + RED-first rows for the FOUR ops: `atan`,
 * `atan2`, `asin`, `acos`. Golden tables minted by
 * `tools/bandmath/gen_corpus_invtrig.py` (MPFR @ 256 bits, native
 * ATAN/ASIN/ACOS/ATAN2 FuncIds). Every op's
 * own certification uses `BandMathCertInvTrig.h`'s own
 * `certifyInvTrigDomainUnary`/`Binary` (same single-leg ULP-53 contract,
 * same `Row` layout `BandMathCertRoot.h`'s own drivers use, but with NO
 * monotonicity claim — @see that file's own "asinacos" section: every op
 * here mixes multiple independent degradation mechanisms into one
 * exponent-keyed bucket, so there is no single smooth curve to be
 * monotone about). RED-first via table-entry/reduction-constant
 * perturbation + dropped-polynomial-term twin (`BandMathCertInvTrig.h`),
 * PLUS the generic carrier-perturbation/identity-mutant arms
 * (`BandMathCert.h`, reused verbatim — free, and an extra non-vacuity
 * check on top of the literal arms).
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCertInvTrig.h"
#include "tests/bandmath/golden_atan.h"
#include "tests/bandmath/golden_atan2.h"
#include "tests/bandmath/golden_asin.h"
#include "tests/bandmath/golden_acos.h"

#include <cstdio>
#include <cmath>
#include <limits>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_atan  = ::aether_tests::bandmath_golden::atan;
namespace g_atan2 = ::aether_tests::bandmath_golden::atan2;
namespace g_asin  = ::aether_tests::bandmath_golden::asin;
namespace g_acos  = ::aether_tests::bandmath_golden::acos;

void printInvTrigCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathInvTrig %-8s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp),
        c.degradedPass, c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  atan -- TIER-BIT host<->device (no sqrt_/hardware-seeded call anywhere
//  in atan2's body -- @see BandInvTrig.h's own header docstring, "asin's
//  two branches" section; the twin file asserts this directly).
// =========================================================================
TEST(BandMathInvTrig, AtanCertifiesAdmitted)
{
    auto c = certifyInvTrigDomainUnary(g_atan::kRows, g_atan::kCorpusSize, g_atan::kUlpBound, [](Band a) { return bd::atan(a); });
    printInvTrigCounts("atan", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathInvTrig, AtanRedFirstMatchedPairPerturbation)
{
    // Threshold is deliberately looser than the generic arms' n/10: a
    // SINGLE sector entry's own `t.hi` only affects rows whose (b,a) ratio
    // actually lands in THAT sector (roughly a factor-2 window around
    // |x|==1 for atan(x)=atan2(x,1)) -- unlike a shared reduction constant
    // (trig's own kBandPiO2CW1Bits), which every row consumes. The corpus
    // is weighted toward that window (`uniform 0.71 1.42`/`-1.42 -0.71`)
    // so a meaningful fraction still goes RED.
    const std::size_t red = countRedUnderMatchedPairPerturbationAtan(g_atan::kRows, g_atan::kCorpusSize, g_atan::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan RED-first(matched-pair-perturbation): %zu/%zu rows RED\n", red,
        g_atan::kInDomainCount);
    EXPECT_GT(red, g_atan::kInDomainCount / 20);
}
TEST(BandMathInvTrig, AtanRedFirstDroppedPolynomialTerm)
{
    // 8/10, not the generic arms' 9/10: `atan`'s corpus deliberately
    // samples down to |x| ~ 1e-40 (the underflow-identity boundary, @see
    // `asin`'s own "asin(x)==x" exact-point test) -- for a row whose
    // REDUCED argument `t` underflows to exact FP32 zero, `mulRaw(t=0, q)`
    // is `0` regardless of which coefficient `q` carries, so the correct
    // and dropped-term cores agree exactly. Same benign non-100% pattern
    // `BandMathCertTrig.h`'s own docstring documents for `sin`'s `r==0`
    // rows -- measured here: 87% RED.
    const std::size_t red = countRedUnderDroppedPolynomialTermAtan(g_atan::kRows, g_atan::kCorpusSize, g_atan::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan RED-first(dropped-polynomial-term): %zu/%zu rows RED\n", red,
        g_atan::kInDomainCount);
    EXPECT_GT(red, g_atan::kInDomainCount * 8 / 10);
}
TEST(BandMathInvTrig, AtanRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_atan::kRows, g_atan::kCorpusSize, g_atan::kUlpBound, [](Band a) { return bd::atan(a); });
    std::fprintf(stderr, "BandMathInvTrig atan RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_atan::kInDomainCount);
    EXPECT_GT(red, g_atan::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, AtanRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_atan::kRows, g_atan::kCorpusSize, g_atan::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_atan::kInDomainCount);
    EXPECT_GT(red, g_atan::kInDomainCount / 10);
}

// =========================================================================
//  atan2 -- TIER-BIT host<->device. Built on AtanSector.h's lookup table.
// =========================================================================
TEST(BandMathInvTrig, Atan2CertifiesAdmitted)
{
    auto c = certifyInvTrigDomainBinary(
        g_atan2::kRows, g_atan2::kCorpusSize, g_atan2::kUlpBound, [](Band y, Band x) { return bd::atan2(y, x); });
    printInvTrigCounts("atan2", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathInvTrig, Atan2RedFirstMatchedPairPerturbation)
{
    // @see AtanRedFirstMatchedPairPerturbation's own comment: one sector
    // entry only touches rows whose ratio lands in that sector.
    const std::size_t red = countRedUnderMatchedPairPerturbationAtan2(g_atan2::kRows, g_atan2::kCorpusSize, g_atan2::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan2 RED-first(matched-pair-perturbation): %zu/%zu rows RED\n", red,
        g_atan2::kInDomainCount);
    EXPECT_GT(red, g_atan2::kInDomainCount / 20);
}
TEST(BandMathInvTrig, Atan2RedFirstDroppedPolynomialTerm)
{
    // 8/10, not the generic arms' 9/10: @see AtanRedFirstDroppedPolynomialTerm's
    // own comment -- same underflow-collapse mechanism (t==0 exactly makes
    // mulRaw(t,q) agree regardless of q), and atan2's own corpus samples
    // operand ratios down to 1e-110 by construction (divisor-ceiling edge
    // probes). Measured here: 81% RED.
    const std::size_t red = countRedUnderDroppedPolynomialTermAtan2(g_atan2::kRows, g_atan2::kCorpusSize, g_atan2::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan2 RED-first(dropped-polynomial-term): %zu/%zu rows RED\n", red,
        g_atan2::kInDomainCount);
    EXPECT_GT(red, g_atan2::kInDomainCount * 8 / 10);
}
TEST(BandMathInvTrig, Atan2RedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_atan2::kRows, g_atan2::kCorpusSize, g_atan2::kUlpBound, [](Band y, Band x) { return bd::atan2(y, x); });
    std::fprintf(stderr, "BandMathInvTrig atan2 RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_atan2::kInDomainCount);
    EXPECT_GT(red, g_atan2::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, Atan2RedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantBinary(g_atan2::kRows, g_atan2::kCorpusSize, g_atan2::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig atan2 RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_atan2::kInDomainCount);
    EXPECT_GT(red, g_atan2::kInDomainCount / 10);
}
// Sixteen sign/zero/inf special rows, direct value assertions (not just via
// the corpus's own WEAK 3-way contract) -- the C99/IEEE-754 table
// BandInvTrig.h::atan2's own doc comment enumerates.
TEST(BandMathInvTrig, Atan2SpecialsTableMatchesStandard)
{
    const float kInf = std::numeric_limits<float>::infinity();
    const Band  z{ 0.0f, 0.0f, 0.0f }, nz{ -0.0f, 0.0f, 0.0f };
    const Band  one{ 1.0f, 0.0f, 0.0f }, none{ -1.0f, 0.0f, 0.0f };
    const Band  pinf{ kInf, 0.0f, 0.0f }, ninf{ -kInf, 0.0f, 0.0f };
    auto        hi = [](Band b) { return doubleOfBits(bandToDoubleBits(b)); };
    const double pi = 3.14159265358979323846;

    EXPECT_DOUBLE_EQ(hi(bd::atan2(z, one)), 0.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(nz, one)), -0.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(z, none)), pi);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(nz, none)), -pi);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(one, z)), pi / 2.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(none, z)), -pi / 2.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(pinf, one)), pi / 2.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(ninf, one)), -pi / 2.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(pinf, pinf)), pi / 4.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(ninf, pinf)), -pi / 4.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(pinf, ninf)), 3.0 * pi / 4.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(ninf, ninf)), -3.0 * pi / 4.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(one, pinf)), 0.0);
    EXPECT_DOUBLE_EQ(hi(bd::atan2(one, ninf)), pi);
    const Band nan{ std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f };
    EXPECT_TRUE(std::isnan(hi(bd::atan2(nan, one))));
    EXPECT_TRUE(std::isnan(hi(bd::atan2(one, nan))));
}

// =========================================================================
//  asin -- LOW branch (|x|<=0.5, no sqrt_ call) TIER-BIT; HIGH branch
//  (|x|>0.5, uses RsqrtCore.h::sqrt_) TIER-TOL, inherited from the root
//  family's own established host/device seed divergence -- @see
//  BandInvTrig.h's own header docstring.
// =========================================================================
TEST(BandMathInvTrig, AsinCertifiesAdmitted)
{
    auto c = certifyInvTrigDomainUnary(g_asin::kRows, g_asin::kCorpusSize, g_asin::kUlpBound, [](Band a) { return bd::asin(a); });
    printInvTrigCounts("asin", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathInvTrig, AsinRedFirstMatchedPairPerturbation)
{
    const std::size_t red = countRedUnderMatchedPairPerturbationAsin(g_asin::kRows, g_asin::kCorpusSize, g_asin::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig asin RED-first(shared-circle-constant-perturbation): %zu/%zu rows RED\n", red,
        g_asin::kInDomainCount);
    EXPECT_GT(red, g_asin::kInDomainCount / 10);
}
TEST(BandMathInvTrig, AsinRedFirstDroppedPolynomialTerm)
{
    const std::size_t red = countRedUnderDroppedPolynomialTermAsin(g_asin::kRows, g_asin::kCorpusSize, g_asin::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig asin RED-first(dropped-polynomial-term): %zu/%zu rows RED\n", red,
        g_asin::kInDomainCount);
    EXPECT_GT(red, g_asin::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, AsinRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_asin::kRows, g_asin::kCorpusSize, g_asin::kUlpBound, [](Band a) { return bd::asin(a); });
    std::fprintf(stderr, "BandMathInvTrig asin RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_asin::kInDomainCount);
    EXPECT_GT(red, g_asin::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, AsinRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_asin::kRows, g_asin::kCorpusSize, g_asin::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig asin RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_asin::kInDomainCount);
    EXPECT_GT(red, g_asin::kInDomainCount / 10);
}
TEST(BandMathInvTrig, AsinExactPointsBitExact)
{
    // asin's own DESIGN-cited exact points: +-0, +-1 -> +-pi/2, and the
    // underflow identity asin(x)==x for x*x underflowing FP32.
    auto hi = [](Band b) { return doubleOfBits(bandToDoubleBits(b)); };
    EXPECT_DOUBLE_EQ(hi(bd::asin(Band{ 0.0f, 0.0f, 0.0f })), 0.0);
    EXPECT_DOUBLE_EQ(hi(bd::asin(Band{ -0.0f, 0.0f, 0.0f })), -0.0);
    EXPECT_DOUBLE_EQ(hi(bd::asin(Band{ 1.0f, 0.0f, 0.0f })), 1.5707963267948966);
    EXPECT_DOUBLE_EQ(hi(bd::asin(Band{ -1.0f, 0.0f, 0.0f })), -1.5707963267948966);
    const Band tiny{ 1.0e-30f, 0.0f, 0.0f };
    const Band got = bd::asin(tiny);
    EXPECT_EQ(got.hi, tiny.hi);
    EXPECT_EQ(got.lo, tiny.lo);
    EXPECT_EQ(got.tail, tiny.tail);
    // Out-of-domain: NaN, unconditionally, no margin.
    EXPECT_TRUE(std::isnan(hi(bd::asin(Band{ 2.0f, 0.0f, 0.0f }))));
}

// =========================================================================
//  acos -- same TIER split as asin (shares asinReducedRaw).
// =========================================================================
TEST(BandMathInvTrig, AcosCertifiesAdmitted)
{
    auto c = certifyInvTrigDomainUnary(g_acos::kRows, g_acos::kCorpusSize, g_acos::kUlpBound, [](Band a) { return bd::acos(a); });
    printInvTrigCounts("acos", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathInvTrig, AcosRedFirstMatchedPairPerturbation)
{
    const std::size_t red = countRedUnderMatchedPairPerturbationAcos(g_acos::kRows, g_acos::kCorpusSize, g_acos::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig acos RED-first(shared-circle-constant-perturbation): %zu/%zu rows RED\n", red,
        g_acos::kInDomainCount);
    EXPECT_GT(red, g_acos::kInDomainCount / 10);
}
TEST(BandMathInvTrig, AcosRedFirstDroppedPolynomialTerm)
{
    const std::size_t red = countRedUnderDroppedPolynomialTermAcos(g_acos::kRows, g_acos::kCorpusSize, g_acos::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig acos RED-first(dropped-polynomial-term): %zu/%zu rows RED\n", red,
        g_acos::kInDomainCount);
    EXPECT_GT(red, g_acos::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, AcosRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_acos::kRows, g_acos::kCorpusSize, g_acos::kUlpBound, [](Band a) { return bd::acos(a); });
    std::fprintf(stderr, "BandMathInvTrig acos RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_acos::kInDomainCount);
    EXPECT_GT(red, g_acos::kInDomainCount * 9 / 10);
}
TEST(BandMathInvTrig, AcosRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_acos::kRows, g_acos::kCorpusSize, g_acos::kUlpBound);
    std::fprintf(stderr, "BandMathInvTrig acos RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_acos::kInDomainCount);
    EXPECT_GT(red, g_acos::kInDomainCount / 10);
}
TEST(BandMathInvTrig, AcosExactPointsBitExact)
{
    auto hi = [](Band b) { return doubleOfBits(bandToDoubleBits(b)); };
    EXPECT_DOUBLE_EQ(hi(bd::acos(Band{ 1.0f, 0.0f, 0.0f })), 0.0);
    EXPECT_DOUBLE_EQ(hi(bd::acos(Band{ -1.0f, 0.0f, 0.0f })), 3.14159265358979323846);
    EXPECT_DOUBLE_EQ(hi(bd::acos(Band{ 0.0f, 0.0f, 0.0f })), 1.5707963267948966);
    EXPECT_DOUBLE_EQ(hi(bd::acos(Band{ -0.0f, 0.0f, 0.0f })), 1.5707963267948966);
    EXPECT_TRUE(std::isnan(hi(bd::acos(Band{ 2.0f, 0.0f, 0.0f }))));
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
