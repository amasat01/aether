// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathRoot_common.h
 * @brief `BandMathRoot` suite — shared by `test_BandMathRoot.cpp`
 *        and `test_BandMathRoot.cu` (mirrors `test_BandMathCert_common.h`'s
 *        own twin convention).
 *
 * Registers the certification + RED-first rows for the FIVE ops: `sqrt`,
 * `rsqrt`, `rsqrtCube`, `cbrt`, `hypot`. Golden tables minted by
 * `tools/bandmath/gen_corpus_root.py` (MPFR @ 256 bits, except
 * `rsqrtCube`'s own `decimal`-based minter — see that script's module
 * docstring). RED-first arms, adapted exactly as
 * `test_BandMathCert_common.h`'s own docstring records for the SAME reason
 * (these are FP32 primitives, not table/reduction-constant-driven
 * transcendentals): arm 1 (`countRedUnderCarrierPerturbation*`)
 * = the "seeded one-ULP reduction-constant perturbation" analog (a one-FP32-
 * ULP carrier-level defect on the LEADING limb); arm 2
 * (`countRedUnderIdentityMutant*`) = the "dropped-polynomial-term"/"dropped-
 * Newton-step" analog (the WRONG "forgot to run the Newton refinement at
 * all" implementation, `return a;`).
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCertRoot.h"
#include "tests/bandmath/golden_sqrt.h"
#include "tests/bandmath/golden_rsqrt.h"
#include "tests/bandmath/golden_rsqrtCube.h"
#include "tests/bandmath/golden_cbrt.h"
#include "tests/bandmath/golden_hypot.h"

#include <cstdio>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_sqrt      = ::aether_tests::bandmath_golden::sqrt;
namespace g_rsqrt      = ::aether_tests::bandmath_golden::rsqrt;
namespace g_rsqrtCube = ::aether_tests::bandmath_golden::rsqrtCube;
namespace g_cbrt      = ::aether_tests::bandmath_golden::cbrt;
namespace g_hypot     = ::aether_tests::bandmath_golden::hypot;

void printRootCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathRoot %-10s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp),
        c.degradedPass, c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  sqrt -- TOL tier (device rsqrt.approx.f32 seed vs host 1/sqrtf).
//  Contract 1 ULP, MEASURED (BandRoot.h's own kBandRsqrtCeilingExp doc).
// =========================================================================
TEST(BandMathRoot, SqrtCertifiesTolAdmitted)
{
    auto c = certifyRootUnary(
        g_sqrt::kRows, g_sqrt::kCorpusSize, g_sqrt::kUlpBound, [](Band a) { return bd::sqrt_(a); });
    printRootCounts("sqrt", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRoot, SqrtRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_sqrt::kRows, g_sqrt::kCorpusSize, g_sqrt::kUlpBound, [](Band a) { return bd::sqrt_(a); });
    std::fprintf(stderr, "BandMathRoot sqrt RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_sqrt::kInDomainCount);
    EXPECT_GT(red, g_sqrt::kInDomainCount * 9 / 10);
}
TEST(BandMathRoot, SqrtRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_sqrt::kRows, g_sqrt::kCorpusSize, g_sqrt::kUlpBound);
    std::fprintf(
        stderr, "BandMathRoot sqrt RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_sqrt::kInDomainCount);
    EXPECT_GT(red, g_sqrt::kInDomainCount / 10);
}

// =========================================================================
//  rsqrt -- TOL tier. Contract 1 ULP. Routes through BandRoot.h's
//  rsqrtIeee (fixes rsqrt(+Inf)=+0 over RsqrtCore.h::rsqrt).
// =========================================================================
TEST(BandMathRoot, RsqrtCertifiesTolAdmitted)
{
    auto c = certifyRootUnary(
        g_rsqrt::kRows, g_rsqrt::kCorpusSize, g_rsqrt::kUlpBound, [](Band a) { return bd::rsqrtIeee(a); });
    printRootCounts("rsqrt", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRoot, RsqrtRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_rsqrt::kRows, g_rsqrt::kCorpusSize, g_rsqrt::kUlpBound, [](Band a) { return bd::rsqrtIeee(a); });
    std::fprintf(stderr, "BandMathRoot rsqrt RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_rsqrt::kInDomainCount);
    EXPECT_GT(red, g_rsqrt::kInDomainCount * 9 / 10);
}
TEST(BandMathRoot, RsqrtRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_rsqrt::kRows, g_rsqrt::kCorpusSize, g_rsqrt::kUlpBound);
    std::fprintf(
        stderr, "BandMathRoot rsqrt RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_rsqrt::kInDomainCount);
    EXPECT_GT(red, g_rsqrt::kInDomainCount / 10);
}
TEST(BandMathRoot, RsqrtPlusInfIsPlusZero)
{
    // The rsqrt(+Inf)=+0 fix, asserted directly (not just
    // via the corpus's specials rows, which use the WEAK 3-way contract).
    const Band posInf = Band{ std::numeric_limits<float>::infinity(), 0.0f, 0.0f };
    const Band got     = bd::rsqrtIeee(posInf);
    EXPECT_EQ(got.hi, 0.0f);
    EXPECT_FALSE(std::signbit(got.hi));
    EXPECT_EQ(got.lo, 0.0f);
    EXPECT_EQ(got.tail, 0.0f);
}

// =========================================================================
//  rsqrtCube -- TOL tier (shares RsqrtCore's seed). Contract 0.5 ULP,
//  MEASURED clean throughout bandRsqrtCubeAdmits's own window.
// =========================================================================
TEST(BandMathRoot, RsqrtCubeCertifiesTolAdmitted)
{
    auto c = certifyRootUnary(g_rsqrtCube::kRows, g_rsqrtCube::kCorpusSize, g_rsqrtCube::kUlpBound,
        [](Band a) { return bd::rsqrtCube(a); });
    printRootCounts("rsqrtCube", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRoot, RsqrtCubeRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(g_rsqrtCube::kRows, g_rsqrtCube::kCorpusSize,
        g_rsqrtCube::kUlpBound, [](Band a) { return bd::rsqrtCube(a); });
    std::fprintf(stderr, "BandMathRoot rsqrtCube RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_rsqrtCube::kInDomainCount);
    EXPECT_GT(red, g_rsqrtCube::kInDomainCount * 9 / 10);
}
TEST(BandMathRoot, RsqrtCubeRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_rsqrtCube::kRows, g_rsqrtCube::kCorpusSize, g_rsqrtCube::kUlpBound);
    std::fprintf(stderr, "BandMathRoot rsqrtCube RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        g_rsqrtCube::kInDomainCount);
    EXPECT_GT(red, g_rsqrtCube::kInDomainCount / 10);
}

// =========================================================================
//  cbrt -- BIT tier (no hardware seed, bit-identical by construction).
//  Contract 0.5 ULP, following the reference derivation verbatim.
// =========================================================================
TEST(BandMathRoot, CbrtCertifiesTolAdmitted)
{
    auto c = certifyRootUnary(
        g_cbrt::kRows, g_cbrt::kCorpusSize, g_cbrt::kUlpBound, [](Band a) { return bd::cbrt(a); });
    printRootCounts("cbrt", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRoot, CbrtRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_cbrt::kRows, g_cbrt::kCorpusSize, g_cbrt::kUlpBound, [](Band a) { return bd::cbrt(a); });
    std::fprintf(stderr, "BandMathRoot cbrt RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_cbrt::kInDomainCount);
    EXPECT_GT(red, g_cbrt::kInDomainCount * 9 / 10);
}
TEST(BandMathRoot, CbrtRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_cbrt::kRows, g_cbrt::kCorpusSize, g_cbrt::kUlpBound);
    std::fprintf(
        stderr, "BandMathRoot cbrt RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_cbrt::kInDomainCount);
    EXPECT_GT(red, g_cbrt::kInDomainCount / 10);
}

// =========================================================================
//  hypot -- TOL tier (inherits from sqrt_ internally). Contract 0.5 ULP,
//  following the reference derivation verbatim.
// =========================================================================
TEST(BandMathRoot, HypotCertifiesTolAdmitted)
{
    auto c = certifyRootBinary(g_hypot::kRows, g_hypot::kCorpusSize, g_hypot::kUlpBound,
        [](Band a, Band b) { return bd::hypot(a, b); });
    printRootCounts("hypot", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.degradedPass, c.degradedTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRoot, HypotRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_hypot::kRows, g_hypot::kCorpusSize, g_hypot::kUlpBound, [](Band a, Band b) { return bd::hypot(a, b); });
    std::fprintf(stderr, "BandMathRoot hypot RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_hypot::kInDomainCount);
    EXPECT_GT(red, g_hypot::kInDomainCount * 9 / 10);
}
TEST(BandMathRoot, HypotRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_hypot::kRows, g_hypot::kCorpusSize, g_hypot::kUlpBound);
    std::fprintf(
        stderr, "BandMathRoot hypot RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_hypot::kInDomainCount);
    EXPECT_GT(red, g_hypot::kInDomainCount / 10);
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
