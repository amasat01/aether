// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathExpLog_common.h
 * @brief `BandMathExpLog` suite — shared by
 *        `test_BandMathExpLog.cpp` and `test_BandMathExpLog.cu` (both TUs
 *        simply `#include` this file, the same twin convention — @see
 *        `test_BandMathCert_common.h`).
 *
 * Registers the certification + RED-first rows for `exp`/`log`/`pow`
 * (`aether::banded::detail::exp/log/pow`, `aether/banded/BandExpLog.h`)
 * against the MPFR golden tables `tools/bandmath/gen_corpus_explog.py`
 * mints (`tests/bandmath/golden_{exp,log,pow}.h`). Certification drivers
 * (`certifyExpLike`/`certifyLogLike`/`certifyPowLike`) live in the NEW
 * add-on header `tests/bandmath/BandMathCertTranscendental.h` — `BandMathCert.h`
 * itself is do-not-touch and its own docstring sanctions this exact
 * extension route. RED-first arms reuse `BandMathCert.h`'s existing
 * generic templates unchanged (no new machinery needed there).
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCertTranscendental.h"
#include "tests/bandmath/golden_exp.h"
#include "tests/bandmath/golden_expm1.h"
#include "tests/bandmath/golden_log.h"
#include "tests/bandmath/golden_acosh.h"
#include "tests/bandmath/golden_asinh.h"
#include "tests/bandmath/golden_atanh.h"
#include "tests/bandmath/golden_cosh.h"
#include "tests/bandmath/golden_exp2.h"
#include "tests/bandmath/golden_log1p.h"
#include "tests/bandmath/golden_log10.h"
#include "tests/bandmath/golden_log2.h"
#include "tests/bandmath/golden_sinh.h"
#include "tests/bandmath/golden_tan.h"
#include "tests/bandmath/golden_tanh.h"
#include "tests/bandmath/golden_pow.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_exp = ::aether_tests::bandmath_golden::exp;
namespace g_expm1 = ::aether_tests::bandmath_golden::expm1;
namespace g_log = ::aether_tests::bandmath_golden::log;
namespace g_tan = ::aether_tests::bandmath_golden::tan;
namespace g_asinh = ::aether_tests::bandmath_golden::asinh;
namespace g_acosh = ::aether_tests::bandmath_golden::acosh;
namespace g_atanh = ::aether_tests::bandmath_golden::atanh;
namespace g_tanh = ::aether_tests::bandmath_golden::tanh;
namespace g_sinh = ::aether_tests::bandmath_golden::sinh;
namespace g_cosh = ::aether_tests::bandmath_golden::cosh;
namespace g_exp2 = ::aether_tests::bandmath_golden::exp2;
namespace g_log2 = ::aether_tests::bandmath_golden::log2;
namespace g_log10 = ::aether_tests::bandmath_golden::log10;
namespace g_log1p = ::aether_tests::bandmath_golden::log1p;
namespace g_pow = ::aether_tests::bandmath_golden::pow;

void printExpLogCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathExpLog %-4s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp), c.degradedPass,
        c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  exp — certified over bandExpAdmits, bound 0.5 ULP (kExpUlpBound).
// =========================================================================
TEST(BandMathExpLog, ExpCertifiesAdmitted)
{
    auto c = certifyExpLike(g_exp::kRows, g_exp::kCorpusSize, g_exp::kUlpBound, aether::banded::detail::kBandExpSatLow,
        aether::banded::detail::kBandExpSatHigh, [](Band a) { return bd::exp(a); });
    printExpLogCounts("exp", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, ExpRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_exp::kRows, g_exp::kCorpusSize, g_exp::kUlpBound, [](Band a) { return bd::exp(a); });
    std::fprintf(
        stderr, "BandMathExpLog exp RED-first(carrier-perturbation): %zu/%zu rows RED\n", red, g_exp::kInDomainCount);
    EXPECT_GT(red, g_exp::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, ExpRedFirstIdentityMutant)
{
    // "dropped polynomial term" analog: bandmath's RED-first identity twin
    // is arity-generic ("return a unchanged") — for a transcendental this IS
    // the dropped-everything degenerate case (the whole reduction+core+
    // reconstruction chain missing), a strictly STRONGER defect than
    // dropping one polynomial term, and still the same "required piece of
    // the computation is missing" analog the file docstring names.
    const std::size_t red = countRedUnderIdentityMutantUnary(g_exp::kRows, g_exp::kCorpusSize, g_exp::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog exp RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_exp::kInDomainCount);
    EXPECT_GT(red, g_exp::kInDomainCount * 9 / 10);
}

// =========================================================================
//  expm1 — same domain and bound as exp (kBandExpSat*), 0.5 ULP.
// =========================================================================
TEST(BandMathExpLog, Expm1CertifiesAdmitted)
{
    auto c = certifyExpLike(g_expm1::kRows, g_expm1::kCorpusSize, g_expm1::kUlpBound,
        aether::banded::detail::kBandExpSatLow, aether::banded::detail::kBandExpSatHigh,
        [](Band a) { return bd::expm1(a); });
    printExpLogCounts("expm1", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, Expm1RedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_expm1::kRows, g_expm1::kCorpusSize, g_expm1::kUlpBound, [](Band a) { return bd::expm1(a); });
    std::fprintf(stderr, "BandMathExpLog expm1 RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_expm1::kInDomainCount);
    EXPECT_GT(red, g_expm1::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, Expm1RedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_expm1::kRows, g_expm1::kCorpusSize, g_expm1::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog expm1 RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_expm1::kInDomainCount);
    EXPECT_GT(red, g_expm1::kInDomainCount / 10);
}

// =========================================================================
//  log — certified over bandLogAdmits, bound 0.5 ULP (kLogUlpBound).
// =========================================================================
TEST(BandMathExpLog, LogCertifiesAdmitted)
{
    auto c = certifyLogLike(g_log::kRows, g_log::kCorpusSize, g_log::kUlpBound, [](Band a) { return bd::log(a); });
    printExpLogCounts("log", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, LogRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_log::kRows, g_log::kCorpusSize, g_log::kUlpBound, [](Band a) { return bd::log(a); });
    std::fprintf(
        stderr, "BandMathExpLog log RED-first(carrier-perturbation): %zu/%zu rows RED\n", red, g_log::kInDomainCount);
    EXPECT_GT(red, g_log::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, LogRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_log::kRows, g_log::kCorpusSize, g_log::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog log RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_log::kInDomainCount);
    EXPECT_GT(red, g_log::kInDomainCount / 10);
}

// =========================================================================
//  log1p — 1 ULP bound (kLog1pUlpBound), no saturation edges.
// =========================================================================
TEST(BandMathExpLog, Log1pCertifiesAdmitted)
{
    auto c = certifyExpLike(g_log1p::kRows, g_log1p::kCorpusSize, g_log1p::kUlpBound, -3.0e38f, 3.0e38f,
        [](Band a) { return bd::log1p(a); });
    printExpLogCounts("log1p", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, Log1pRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_log1p::kRows, g_log1p::kCorpusSize, g_log1p::kUlpBound, [](Band a) { return bd::log1p(a); });
    std::fprintf(stderr, "BandMathExpLog log1p RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_log1p::kInDomainCount);
    EXPECT_GT(red, g_log1p::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, Log1pRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_log1p::kRows, g_log1p::kCorpusSize, g_log1p::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog log1p RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_log1p::kInDomainCount);
    EXPECT_GT(red, g_log1p::kInDomainCount / 10);
}

// =========================================================================
//  tan -- sin/cos, 1.5 ULP where |cos x| >= 2^-13 (degraded rows below it,
//  none minted within 2^-30 of a pole).
// =========================================================================
TEST(BandMathExpLog, TanCertifiesAdmitted)
{
    auto c = certifyExpLike(g_tan::kRows, g_tan::kCorpusSize, g_tan::kUlpBound, -3.0e38f, 3.0e38f,
        [](Band a) { return bd::tan(a); });
    printExpLogCounts("tan", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, TanRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_tan::kRows, g_tan::kCorpusSize, g_tan::kUlpBound, [](Band a) { return bd::tan(a); });
    std::fprintf(stderr, "BandMathExpLog tan RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_tan::kInDomainCount);
    EXPECT_GT(red, g_tan::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, TanRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_tan::kRows, g_tan::kCorpusSize, g_tan::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog tan RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_tan::kInDomainCount);
    EXPECT_GT(red, g_tan::kInDomainCount / 10);
}

// =========================================================================
//  asinh / acosh / atanh -- identities over log / log1p, 2 ULP bound.
// =========================================================================
TEST(BandMathExpLog, AsinhCertifiesAdmitted)
{
    auto c = certifyExpLike(g_asinh::kRows, g_asinh::kCorpusSize, g_asinh::kUlpBound, -3.0e38f, 3.0e38f,
        [](Band a) { return bd::asinh(a); });
    printExpLogCounts("asinh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, AsinhRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_asinh::kRows, g_asinh::kCorpusSize, g_asinh::kUlpBound, [](Band a) { return bd::asinh(a); });
    std::fprintf(stderr, "BandMathExpLog asinh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_asinh::kInDomainCount);
    EXPECT_GT(red, g_asinh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, AsinhRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_asinh::kRows, g_asinh::kCorpusSize, g_asinh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog asinh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_asinh::kInDomainCount);
    EXPECT_GT(red, g_asinh::kInDomainCount / 10);
}

TEST(BandMathExpLog, AcoshCertifiesAdmitted)
{
    auto c = certifyExpLike(g_acosh::kRows, g_acosh::kCorpusSize, g_acosh::kUlpBound, -3.0e38f, 3.0e38f,
        [](Band a) { return bd::acosh(a); });
    printExpLogCounts("acosh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, AcoshRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_acosh::kRows, g_acosh::kCorpusSize, g_acosh::kUlpBound, [](Band a) { return bd::acosh(a); });
    std::fprintf(stderr, "BandMathExpLog acosh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_acosh::kInDomainCount);
    EXPECT_GT(red, g_acosh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, AcoshRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_acosh::kRows, g_acosh::kCorpusSize, g_acosh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog acosh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_acosh::kInDomainCount);
    EXPECT_GT(red, g_acosh::kInDomainCount / 10);
}

TEST(BandMathExpLog, AtanhCertifiesAdmitted)
{
    auto c = certifyExpLike(g_atanh::kRows, g_atanh::kCorpusSize, g_atanh::kUlpBound, -3.0e38f, 3.0e38f,
        [](Band a) { return bd::atanh(a); });
    printExpLogCounts("atanh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, AtanhRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_atanh::kRows, g_atanh::kCorpusSize, g_atanh::kUlpBound, [](Band a) { return bd::atanh(a); });
    std::fprintf(stderr, "BandMathExpLog atanh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_atanh::kInDomainCount);
    EXPECT_GT(red, g_atanh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, AtanhRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_atanh::kRows, g_atanh::kCorpusSize, g_atanh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog atanh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_atanh::kInDomainCount);
    EXPECT_GT(red, g_atanh::kInDomainCount / 10);
}

// =========================================================================
//  tanh / sinh / cosh -- compositions over expm1, 2 ULP bound.
// =========================================================================
TEST(BandMathExpLog, TanhCertifiesAdmitted)
{
    auto c = certifyExpLike(g_tanh::kRows, g_tanh::kCorpusSize, g_tanh::kUlpBound, -88.7228391f, 88.7228391f,
        [](Band a) { return bd::tanh(a); });
    printExpLogCounts("tanh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, TanhRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_tanh::kRows, g_tanh::kCorpusSize, g_tanh::kUlpBound, [](Band a) { return bd::tanh(a); });
    std::fprintf(stderr, "BandMathExpLog tanh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_tanh::kInDomainCount);
    EXPECT_GT(red, g_tanh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, TanhRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_tanh::kRows, g_tanh::kCorpusSize, g_tanh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog tanh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_tanh::kInDomainCount);
    EXPECT_GT(red, g_tanh::kInDomainCount / 10);
}

TEST(BandMathExpLog, SinhCertifiesAdmitted)
{
    auto c = certifyExpLike(g_sinh::kRows, g_sinh::kCorpusSize, g_sinh::kUlpBound, -88.7228391f, 88.7228391f,
        [](Band a) { return bd::sinh(a); });
    printExpLogCounts("sinh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, SinhRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_sinh::kRows, g_sinh::kCorpusSize, g_sinh::kUlpBound, [](Band a) { return bd::sinh(a); });
    std::fprintf(stderr, "BandMathExpLog sinh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_sinh::kInDomainCount);
    EXPECT_GT(red, g_sinh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, SinhRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_sinh::kRows, g_sinh::kCorpusSize, g_sinh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog sinh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_sinh::kInDomainCount);
    EXPECT_GT(red, g_sinh::kInDomainCount / 10);
}

TEST(BandMathExpLog, CoshCertifiesAdmitted)
{
    auto c = certifyExpLike(g_cosh::kRows, g_cosh::kCorpusSize, g_cosh::kUlpBound, -88.7228391f, 88.7228391f,
        [](Band a) { return bd::cosh(a); });
    printExpLogCounts("cosh", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, CoshRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_cosh::kRows, g_cosh::kCorpusSize, g_cosh::kUlpBound, [](Band a) { return bd::cosh(a); });
    std::fprintf(stderr, "BandMathExpLog cosh RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_cosh::kInDomainCount);
    EXPECT_GT(red, g_cosh::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, CoshRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_cosh::kRows, g_cosh::kCorpusSize, g_cosh::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog cosh RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_cosh::kInDomainCount);
    EXPECT_GT(red, g_cosh::kInDomainCount / 10);
}

// =========================================================================
//  exp2 / log2 / log10 -- exact scalings of exp / log, 1.5 ULP bound.
// =========================================================================
TEST(BandMathExpLog, Exp2CertifiesAdmitted)
{
    auto c = certifyExpLike(g_exp2::kRows, g_exp2::kCorpusSize, g_exp2::kUlpBound, -150.0f, 128.0f,
        [](Band a) { return bd::exp2(a); });
    printExpLogCounts("exp2", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, Exp2RedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_exp2::kRows, g_exp2::kCorpusSize, g_exp2::kUlpBound, [](Band a) { return bd::exp2(a); });
    std::fprintf(stderr, "BandMathExpLog exp2 RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_exp2::kInDomainCount);
    EXPECT_GT(red, g_exp2::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, Exp2RedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_exp2::kRows, g_exp2::kCorpusSize, g_exp2::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog exp2 RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_exp2::kInDomainCount);
    EXPECT_GT(red, g_exp2::kInDomainCount / 10);
}

TEST(BandMathExpLog, Log2CertifiesAdmitted)
{
    auto c = certifyLogLike(g_log2::kRows, g_log2::kCorpusSize, g_log2::kUlpBound, [](Band a) { return bd::log2(a); });
    printExpLogCounts("log2", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, Log2RedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_log2::kRows, g_log2::kCorpusSize, g_log2::kUlpBound, [](Band a) { return bd::log2(a); });
    std::fprintf(stderr, "BandMathExpLog log2 RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_log2::kInDomainCount);
    EXPECT_GT(red, g_log2::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, Log2RedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_log2::kRows, g_log2::kCorpusSize, g_log2::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog log2 RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_log2::kInDomainCount);
    EXPECT_GT(red, g_log2::kInDomainCount / 10);
}

TEST(BandMathExpLog, Log10CertifiesAdmitted)
{
    auto c = certifyLogLike(g_log10::kRows, g_log10::kCorpusSize, g_log10::kUlpBound, [](Band a) { return bd::log10(a); });
    printExpLogCounts("log10", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, Log10RedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_log10::kRows, g_log10::kCorpusSize, g_log10::kUlpBound, [](Band a) { return bd::log10(a); });
    std::fprintf(stderr, "BandMathExpLog log10 RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_log10::kInDomainCount);
    EXPECT_GT(red, g_log10::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, Log10RedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantUnary(g_log10::kRows, g_log10::kCorpusSize, g_log10::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog log10 RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_log10::kInDomainCount);
    EXPECT_GT(red, g_log10::kInDomainCount / 10);
}

// The scaled ops land exactly on the integers they are defined to hit.
TEST(BandMathExpLog, ScaledOpsHitExactIntegerPoints)
{
    auto bitsOf = [](double d) {
        std::uint64_t u;
        std::memcpy(&u, &d, sizeof u);
        return u;
    };
    auto egress = [&](Band b) { return bandToDoubleBits(b); };
    for (int k = -126; k <= 127; ++k) {
        EXPECT_EQ(egress(bd::exp2(bandFromDoubleBits(bitsOf(static_cast<double>(k))))), bitsOf(std::ldexp(1.0, k)))
            << "exp2(" << k << ")";
        EXPECT_EQ(egress(bd::log2(bandFromDoubleBits(bitsOf(std::ldexp(1.0, k))))), bitsOf(static_cast<double>(k)))
            << "log2(2^" << k << ")";
    }
    for (int k = 0; k <= 22; ++k)
        EXPECT_EQ(egress(bd::log10(bandFromDoubleBits(bitsOf(std::pow(10.0, k))))), bitsOf(static_cast<double>(k)))
            << "log10(10^" << k << ")";
}

// =========================================================================
//  pow — certified over bandPowAdmits, bound 0.5 ULP (kPowUlpBound).
// =========================================================================
TEST(BandMathExpLog, PowCertifiesAdmitted)
{
    auto c = certifyPowLike(
        g_pow::kRows, g_pow::kCorpusSize, g_pow::kUlpBound, [](Band a, Band b) { return bd::pow(a, b); });
    printExpLogCounts("pow", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathExpLog, PowRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_pow::kRows, g_pow::kCorpusSize, g_pow::kUlpBound, [](Band a, Band b) { return bd::pow(a, b); });
    std::fprintf(
        stderr, "BandMathExpLog pow RED-first(carrier-perturbation): %zu/%zu rows RED\n", red, g_pow::kInDomainCount);
    EXPECT_GT(red, g_pow::kInDomainCount * 9 / 10);
}
TEST(BandMathExpLog, PowRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantBinary(g_pow::kRows, g_pow::kCorpusSize, g_pow::kUlpBound);
    std::fprintf(
        stderr, "BandMathExpLog pow RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_pow::kInDomainCount);
    EXPECT_GT(red, g_pow::kInDomainCount / 10);
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
