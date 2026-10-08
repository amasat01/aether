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
#include "tests/bandmath/golden_log.h"
#include "tests/bandmath/golden_pow.h"

#include <cstdio>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_exp = ::aether_tests::bandmath_golden::exp;
namespace g_log = ::aether_tests::bandmath_golden::log;
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
