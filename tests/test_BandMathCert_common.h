// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathCert_common.h
 * @brief `BandMathCert` suite — shared by `test_BandMathCert.cpp` and
 *        `test_BandMathCert.cu` (both TUs simply `#include` this file, the
 *        established twin convention — @see `test_Ff2Cert.cpp`/`.cu`).
 *
 * Registers the certification + RED-first rows for the EIGHT live proof ops
 * (abs, copysign, fmax, fmin, add, sub, mul, fma). `fma` reuses
 * `aether/banded/BandRound.h::fma` (over the hoisted `fmaRaw` family) and
 * `BandMathCertRound.h`'s `certifySpineAdmittedTernary` (mirroring
 * `BandMathCertRoot.h`'s precedent) rather than a driver from this file,
 * since `BandMathCert.h`'s own drivers are unary/binary only.
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCert.h"
#include "tests/bandmath/BandMathCertRound.h"
#include "tests/bandmath/golden_abs.h"
#include "tests/bandmath/golden_copysign.h"
#include "tests/bandmath/golden_fmax.h"
#include "tests/bandmath/golden_fmin.h"
#include "tests/bandmath/golden_add.h"
#include "tests/bandmath/golden_sub.h"
#include "tests/bandmath/golden_mul.h"
#include "tests/bandmath/golden_fma.h"

#include <cstdio>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_abs      = ::aether_tests::bandmath_golden::abs;
namespace g_copysign = ::aether_tests::bandmath_golden::copysign;
namespace g_fmax     = ::aether_tests::bandmath_golden::fmax;
namespace g_fmin     = ::aether_tests::bandmath_golden::fmin;
namespace g_add      = ::aether_tests::bandmath_golden::add;
namespace g_sub      = ::aether_tests::bandmath_golden::sub;
namespace g_mul      = ::aether_tests::bandmath_golden::mul;
namespace g_fma      = ::aether_tests::bandmath_golden::fma;

void printCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathCert %-9s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp),
        c.degradedPass, c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  abs — exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathCert, AbsCertifiesInDomain)
{
    auto c = certifyExactTotalUnary(g_abs::kRows, g_abs::kCorpusSize, g_abs::kUlpBound,
        [](Band a) { return bd::abs(a); });
    printCounts("abs", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, AbsRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_abs::kRows, g_abs::kCorpusSize, g_abs::kUlpBound, [](Band a) { return bd::abs(a); });
    std::fprintf(stderr, "BandMathCert abs RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_abs::kInDomainCount);
    EXPECT_GT(red, g_abs::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, AbsRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_abs::kRows, g_abs::kCorpusSize, g_abs::kUlpBound);
    std::fprintf(
        stderr, "BandMathCert abs RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_abs::kInDomainCount);
    EXPECT_GT(red, g_abs::kInDomainCount / 10);
}

// =========================================================================
//  copysign — exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathCert, CopysignCertifiesInDomain)
{
    auto c = certifyExactTotalBinary(g_copysign::kRows, g_copysign::kCorpusSize, g_copysign::kUlpBound,
        [](Band a, Band b) { return bd::copysign(a, b); });
    printCounts("copysign", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, CopysignRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(g_copysign::kRows,
        g_copysign::kCorpusSize, g_copysign::kUlpBound, [](Band a, Band b) { return bd::copysign(a, b); });
    std::fprintf(stderr, "BandMathCert copysign RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_copysign::kInDomainCount);
    EXPECT_GT(red, g_copysign::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, CopysignRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantBinary(
        g_copysign::kRows, g_copysign::kCorpusSize, g_copysign::kUlpBound);
    std::fprintf(stderr, "BandMathCert copysign RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_copysign::kInDomainCount);
    EXPECT_GT(red, g_copysign::kInDomainCount / 10);
}

// =========================================================================
//  fmax — exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathCert, FmaxCertifiesInDomain)
{
    auto c = certifyExactTotalBinary(g_fmax::kRows, g_fmax::kCorpusSize, g_fmax::kUlpBound,
        [](Band a, Band b) { return bd::fmax(a, b); });
    printCounts("fmax", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, FmaxRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_fmax::kRows, g_fmax::kCorpusSize, g_fmax::kUlpBound, [](Band a, Band b) { return bd::fmax(a, b); });
    std::fprintf(stderr, "BandMathCert fmax RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_fmax::kInDomainCount);
    EXPECT_GT(red, g_fmax::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, FmaxRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_fmax::kRows, g_fmax::kCorpusSize, g_fmax::kUlpBound);
    std::fprintf(stderr, "BandMathCert fmax RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_fmax::kInDomainCount);
    EXPECT_GT(red, g_fmax::kInDomainCount / 10);
}

// =========================================================================
//  fmin — exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathCert, FminCertifiesInDomain)
{
    auto c = certifyExactTotalBinary(g_fmin::kRows, g_fmin::kCorpusSize, g_fmin::kUlpBound,
        [](Band a, Band b) { return bd::fmin(a, b); });
    printCounts("fmin", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, FminRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_fmin::kRows, g_fmin::kCorpusSize, g_fmin::kUlpBound, [](Band a, Band b) { return bd::fmin(a, b); });
    std::fprintf(stderr, "BandMathCert fmin RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_fmin::kInDomainCount);
    EXPECT_GT(red, g_fmin::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, FminRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_fmin::kRows, g_fmin::kCorpusSize, g_fmin::kUlpBound);
    std::fprintf(stderr, "BandMathCert fmin RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_fmin::kInDomainCount);
    EXPECT_GT(red, g_fmin::kInDomainCount / 10);
}

// =========================================================================
//  add/sub/mul — spine-admitted, bound 2 ULP (ruling c derivation, see
//  tools/bandmath/gen_corpus.py's OPS table).
// =========================================================================
TEST(BandMathCert, AddCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedBinary(
        g_add::kRows, g_add::kCorpusSize, g_add::kUlpBound, [](Band a, Band b) { return bd::add(a, b); });
    printCounts("add", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, AddRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_add::kRows, g_add::kCorpusSize, g_add::kUlpBound, [](Band a, Band b) { return bd::add(a, b); });
    std::fprintf(stderr, "BandMathCert add RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_add::kInDomainCount);
    EXPECT_GT(red, g_add::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, AddRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_add::kRows, g_add::kCorpusSize, g_add::kUlpBound);
    std::fprintf(stderr, "BandMathCert add RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_add::kInDomainCount);
    EXPECT_GT(red, g_add::kInDomainCount / 10);
}

TEST(BandMathCert, SubCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedBinary(
        g_sub::kRows, g_sub::kCorpusSize, g_sub::kUlpBound, [](Band a, Band b) { return bd::sub(a, b); });
    printCounts("sub", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, SubRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_sub::kRows, g_sub::kCorpusSize, g_sub::kUlpBound, [](Band a, Band b) { return bd::sub(a, b); });
    std::fprintf(stderr, "BandMathCert sub RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_sub::kInDomainCount);
    EXPECT_GT(red, g_sub::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, SubRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_sub::kRows, g_sub::kCorpusSize, g_sub::kUlpBound);
    std::fprintf(stderr, "BandMathCert sub RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_sub::kInDomainCount);
    EXPECT_GT(red, g_sub::kInDomainCount / 10);
}

TEST(BandMathCert, MulCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedBinary(
        g_mul::kRows, g_mul::kCorpusSize, g_mul::kUlpBound, [](Band a, Band b) { return bd::mul(a, b); });
    printCounts("mul", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, MulRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_mul::kRows, g_mul::kCorpusSize, g_mul::kUlpBound, [](Band a, Band b) { return bd::mul(a, b); });
    std::fprintf(stderr, "BandMathCert mul RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_mul::kInDomainCount);
    EXPECT_GT(red, g_mul::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, MulRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_mul::kRows, g_mul::kCorpusSize, g_mul::kUlpBound);
    std::fprintf(stderr, "BandMathCert mul RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_mul::kInDomainCount);
    EXPECT_GT(red, g_mul::kInDomainCount / 10);
}

// =========================================================================
//  fma -- spine-admitted, bound 2 ULP. Ternary, so the driver comes from
//  BandMathCertRound.h, not this file.
// =========================================================================
TEST(BandMathCert, FmaCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedTernary(
        g_fma::kRows, g_fma::kCorpusSize, g_fma::kUlpBound, [](Band a, Band b, Band c3) { return bd::fma(a, b, c3); });
    printCounts("fma", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathCert, FmaRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationTernary(
        g_fma::kRows, g_fma::kCorpusSize, g_fma::kUlpBound, [](Band a, Band b, Band c3) { return bd::fma(a, b, c3); });
    std::fprintf(stderr, "BandMathCert fma RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_fma::kInDomainCount);
    EXPECT_GT(red, g_fma::kInDomainCount * 9 / 10);
}
TEST(BandMathCert, FmaRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantTernary(g_fma::kRows, g_fma::kCorpusSize, g_fma::kUlpBound);
    std::fprintf(stderr, "BandMathCert fma RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_fma::kInDomainCount);
    EXPECT_GT(red, g_fma::kInDomainCount / 10);
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
