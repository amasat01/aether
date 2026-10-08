// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandMathRound_common.h
 * @brief `BandMathRound` suite — shared by
 *        `test_BandMathRound.cpp` and `test_BandMathRound.cu` (mirrors
 *        `test_BandMathRoot_common.h`'s own twin convention).
 *
 * Registers the certification + RED-first rows for this package's SIX ops
 * (`floor`, `ceil`, `round`, `trunc`, `fdim`, `fmod`) plus `fma`'s
 * previously-unwired golden table — wired here AND in
 * `test_BandMathCert_common.h`, @see that file's own updated docstring.
 * Golden tables minted by
 * `tools/bandmath/gen_corpus_round.py` (native double arithmetic — @see
 * that script's module docstring for why MPFR is not needed for any op in
 * this package, `fmod` included).
 *
 * Driver choice per op (`BandMathCertRound.h`'s own docstring has the full
 * reasoning):
 *   - `floor`/`ceil`/`round`/`trunc`: `certifyExactTotalUnary`, bound 0.
 *   - `fdim`: `certifySpineAdmittedBinary`, bound 2 (inherits `sub`'s
 *     spine bound — NOT the exact-total class, @see `BandRound.h`'s own
 *     "admission" section).
 *   - `fmod`: `certifyExactTotalBinary`'s SHAPE, bound 0 (EXACT, but via
 *     `bandFmodAdmits`'s envelope legs, not the generic one — every
 *     double-sourced row here has span <= 53 <= `kBandFmodMaxSpan`, so the
 *     envelope-only labelling this driver applies is exactly right).
 *   - `fma`: `certifySpineAdmittedTernary` (new, `BandMathCertRound.h`),
 *     bound 2.
 *
 * RED-first arms, adapted exactly as
 * `test_BandMathCert_common.h`'s own docstring records: arm 1
 * (`countRedUnderCarrierPerturbation*`) = the "seeded one-ULP reduction-
 * constant perturbation" analog; arm 2 (`countRedUnderIdentityMutant*`) =
 * the "dropped-polynomial-term" analog (the WRONG "forgot to run the op"
 * implementation, `return a;`).
 *
 * `fmod`'s EXACTNESS-ON-SPAN-<=72 claim is a THEOREM about the operands'
 * limb SPAN, a property no double-bit-pattern row
 * can ever exercise past span 53 (a double IS 53 bits). It is certified
 * separately below, over HAND-BUILT `Band` literals whose span is
 * controlled directly (`FmodSpanLadderIsBitExactWithinTheDeclaredContract`),
 * against a reference computed OUTSIDE `bandFmodStep` entirely — exact
 * rational arithmetic (Python `fractions.Fraction`, not this op's own
 * cascade) decomposed into the same three-limb float32 form, so the
 * comparison does not trust the very machinery it is certifying.
 */

#include <gtest/gtest.h>

#include "tests/bandmath/BandMathCertRound.h"
#include "tests/bandmath/golden_floor.h"
#include "tests/bandmath/golden_ceil.h"
#include "tests/bandmath/golden_round.h"
#include "tests/bandmath/golden_trunc.h"
#include "tests/bandmath/golden_fdim.h"
#include "tests/bandmath/golden_fmod.h"
#include "tests/bandmath/golden_fma.h"

#include <cmath>
#include <cstdio>

namespace aether_tests {
namespace bandmath {
namespace {

namespace g_floor = ::aether_tests::bandmath_golden::floor;
namespace g_ceil  = ::aether_tests::bandmath_golden::ceil;
namespace g_round = ::aether_tests::bandmath_golden::round;
namespace g_trunc = ::aether_tests::bandmath_golden::trunc;
namespace g_fdim  = ::aether_tests::bandmath_golden::fdim;
namespace g_fmod  = ::aether_tests::bandmath_golden::fmod;

void printRoundCounts(const char* op, const CertCounts& c)
{
    std::fprintf(stderr,
        "BandMathRound %-8s in_domain %zu/%zu (max|ulp|=%lld) degraded %zu/%zu rejected %zu/%zu\n",
        op, c.inDomainPass, c.inDomainTotal, static_cast<long long>(c.maxInDomainUlp),
        c.degradedPass, c.degradedTotal, c.rejectedPass, c.rejectedTotal);
}

// =========================================================================
//  floor -- exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathRound, FloorCertifiesInDomain)
{
    auto c = certifyExactTotalUnary(
        g_floor::kRows, g_floor::kCorpusSize, g_floor::kUlpBound, [](Band a) { return bd::floor(a); });
    printRoundCounts("floor", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, FloorRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_floor::kRows, g_floor::kCorpusSize, g_floor::kUlpBound, [](Band a) { return bd::floor(a); });
    std::fprintf(stderr, "BandMathRound floor RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_floor::kInDomainCount);
    EXPECT_GT(red, g_floor::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, FloorRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_floor::kRows, g_floor::kCorpusSize, g_floor::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound floor RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_floor::kInDomainCount);
    EXPECT_GT(red, g_floor::kInDomainCount / 10);
}

// =========================================================================
//  ceil -- exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathRound, CeilCertifiesInDomain)
{
    auto c = certifyExactTotalUnary(
        g_ceil::kRows, g_ceil::kCorpusSize, g_ceil::kUlpBound, [](Band a) { return bd::ceil(a); });
    printRoundCounts("ceil", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, CeilRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_ceil::kRows, g_ceil::kCorpusSize, g_ceil::kUlpBound, [](Band a) { return bd::ceil(a); });
    std::fprintf(stderr, "BandMathRound ceil RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_ceil::kInDomainCount);
    EXPECT_GT(red, g_ceil::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, CeilRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_ceil::kRows, g_ceil::kCorpusSize, g_ceil::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound ceil RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_ceil::kInDomainCount);
    EXPECT_GT(red, g_ceil::kInDomainCount / 10);
}

// =========================================================================
//  round -- exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathRound, RoundCertifiesInDomain)
{
    auto c = certifyExactTotalUnary(
        g_round::kRows, g_round::kCorpusSize, g_round::kUlpBound, [](Band a) { return bd::round(a); });
    printRoundCounts("round", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, RoundRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_round::kRows, g_round::kCorpusSize, g_round::kUlpBound, [](Band a) { return bd::round(a); });
    std::fprintf(stderr, "BandMathRound round RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_round::kInDomainCount);
    EXPECT_GT(red, g_round::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, RoundRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_round::kRows, g_round::kCorpusSize, g_round::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound round RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_round::kInDomainCount);
    EXPECT_GT(red, g_round::kInDomainCount / 10);
}

// =========================================================================
//  trunc -- NEW op. Exact/total, bound 0 ULP.
// =========================================================================
TEST(BandMathRound, TruncCertifiesInDomain)
{
    auto c = certifyExactTotalUnary(
        g_trunc::kRows, g_trunc::kCorpusSize, g_trunc::kUlpBound, [](Band a) { return bd::trunc(a); });
    printRoundCounts("trunc", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, TruncRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationUnary(
        g_trunc::kRows, g_trunc::kCorpusSize, g_trunc::kUlpBound, [](Band a) { return bd::trunc(a); });
    std::fprintf(stderr, "BandMathRound trunc RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_trunc::kInDomainCount);
    EXPECT_GT(red, g_trunc::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, TruncRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantUnary(g_trunc::kRows, g_trunc::kCorpusSize, g_trunc::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound trunc RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_trunc::kInDomainCount);
    EXPECT_GT(red, g_trunc::kInDomainCount / 10);
}
TEST(BandMathRound, TruncMatchesFloorForNonNegativeAndNegatesFloorOfAbsOtherwise)
{
    // The correctness argument BandRound.h::trunc's own docstring makes,
    // asserted directly rather than only via the corpus's aggregate bound.
    EXPECT_FLOAT_EQ(bd::trunc(Band{ 2.75f, 0.0f, 0.0f }).hi, 2.0f);
    EXPECT_FLOAT_EQ(bd::trunc(Band{ -2.75f, 0.0f, 0.0f }).hi, -2.0f);
    const Band negHalf = bd::trunc(Band{ -0.5f, 0.0f, 0.0f });
    EXPECT_EQ(negHalf.hi, 0.0f);
    EXPECT_TRUE(std::signbit(negHalf.hi)); // trunc(-0.5) == -0
    const Band negZero = bd::trunc(Band{ -0.0f, 0.0f, 0.0f });
    EXPECT_EQ(negZero.hi, 0.0f);
    EXPECT_TRUE(std::signbit(negZero.hi)); // trunc(-0) == -0
}

// =========================================================================
//  fdim -- inherits sub's 2-ULP spine bound (NOT exact/total, @see
//  BandRound.h's own "admission" section). Already-landed body (Band.h);
//  this package supplies the golden table + dispatch wiring.
// =========================================================================
TEST(BandMathRound, FdimCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedBinary(
        g_fdim::kRows, g_fdim::kCorpusSize, g_fdim::kUlpBound, [](Band a, Band b) { return bd::fdim(a, b); });
    printRoundCounts("fdim", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, FdimRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_fdim::kRows, g_fdim::kCorpusSize, g_fdim::kUlpBound, [](Band a, Band b) { return bd::fdim(a, b); });
    std::fprintf(stderr, "BandMathRound fdim RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_fdim::kInDomainCount);
    EXPECT_GT(red, g_fdim::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, FdimRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_fdim::kRows, g_fdim::kCorpusSize, g_fdim::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound fdim RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_fdim::kInDomainCount);
    EXPECT_GT(red, g_fdim::kInDomainCount / 10);
}

// =========================================================================
//  fmod -- EXACT (0 ULP), envelope-labelled via bandFmodAdmits (every
//  double-sourced row has span <= 53, so the SPAN leg never fires here —
//  @see the dedicated span-ladder test below for that leg).
// =========================================================================
TEST(BandMathRound, FmodCertifiesInDomain)
{
    auto c = certifyExactTotalBinary(
        g_fmod::kRows, g_fmod::kCorpusSize, g_fmod::kUlpBound, [](Band a, Band b) { return bd::fmod(a, b); });
    printRoundCounts("fmod", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, FmodRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationBinary(
        g_fmod::kRows, g_fmod::kCorpusSize, g_fmod::kUlpBound, [](Band a, Band b) { return bd::fmod(a, b); });
    std::fprintf(stderr, "BandMathRound fmod RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        g_fmod::kInDomainCount);
    EXPECT_GT(red, g_fmod::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, FmodRedFirstIdentityMutant)
{
    const std::size_t red
        = countRedUnderIdentityMutantBinary(g_fmod::kRows, g_fmod::kCorpusSize, g_fmod::kUlpBound);
    std::fprintf(
        stderr, "BandMathRound fmod RED-first(identity-mutant): %zu/%zu rows RED\n", red, g_fmod::kInDomainCount);
    EXPECT_GT(red, g_fmod::kInDomainCount / 10);
}

// =========================================================================
//  fmod's SPAN leg — `kBandFmodMaxSpan` (72), certified over HAND-BUILT
//  `Band` literals (no double-bit-pattern row can express span > 53).
//
//  Construction (computed independently in Python, `fractions.Fraction`
//  exact rational arithmetic — NOT via bandFmodStep, so the comparison
//  does not trust the machinery it certifies): `y = 1 + 2^-20` (S(y)=21),
//  `x = 2^20 + (0b101 * 2^-22) + 2^(21-S)` for the target overall span
//  `S(x) = S` (the low spike sits far enough below `y`'s own leading bit
//  that the coarse reduction path runs at least once — `d = E(x)-E(y) =
//  20 > kBandFmodChunk = 20`... — before the close-range fixup delivers
//  the final remainder). Each `(x, y, expected)` triple below is the
//  EXACT `x mod y`, decomposed into three round-to-nearest float32 limbs
//  (lossless for every triple: the Python minter's own `resid == 0`
//  check, re-verified by hand for this file).
// =========================================================================
namespace fmod_span {
struct SpanRow {
    int span;
    Band x;
    Band y;
    Band expected;
    bool exactRequired; // false past kBandFmodMaxSpan: reported, not required
};

// clang-format off
inline const SpanRow kLadder[] = {
    // span, x{hi,lo,tail}, y{hi,lo,tail}, expected{hi,lo,tail}, exactRequired
    {  8, Band{ 0x1.02p20f,  0x1.4p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.fc0068p-1f, 0.0f, 0.0f }, true },
    { 16, Band{ 0x1.0002p20f, 0x1.4p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.fffc68p-1f, 0.0f, 0.0f }, true },
    { 24, Band{ 0x1.000002p20f, 0x1.4p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.000120p-3f, 0.0f, 0.0f }, true },
    { 32, Band{ 0x1.0p20f, 0x1.00ap-11f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.0120p-11f, 0.0f, 0.0f }, true },
    { 40, Band{ 0x1.0p20f, 0x1.ap-19f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.1p-18f, 0.0f, 0.0f }, true },
    { 48, Band{ 0x1.0p20f, 0x1.42p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.21p-19f, 0.0f, 0.0f }, true },
    { 53, Band{ 0x1.0p20f, 0x1.401p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2008p-19f, 0.0f, 0.0f }, true },
    { 60, Band{ 0x1.0p20f, 0x1.40002p-20f, 0.0f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.20001p-19f, 0.0f, 0.0f }, true },
    { 66, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-45f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-45f, 0.0f }, true },
    { 70, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-49f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-49f, 0.0f }, true },
    { 72, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-51f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-51f, 0.0f }, true },
    { 74, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-53f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-53f, 0.0f }, false },
    { 76, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-55f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-55f, 0.0f }, false },
    { 80, Band{ 0x1.0p20f, 0x1.4p-20f, 0x1.0p-59f }, Band{ 0x1.00001p0f, 0.0f, 0.0f }, Band{ 0x1.2p-19f, 0x1.0p-59f, 0.0f }, false },
};
// clang-format on
} // namespace fmod_span

TEST(BandMathRound, FmodSpanLadderIsBitExactWithinTheDeclaredContract)
{
    std::size_t exactWithinContract = 0, totalWithinContract = 0;
    std::size_t exactBeyondContract = 0, totalBeyondContract = 0;
    for (const auto& row : fmod_span::kLadder) {
        const Band got = bd::fmod(row.x, row.y);
        const bool bitExact
            = (got.hi == row.expected.hi) && (got.lo == row.expected.lo) && (got.tail == row.expected.tail);
        if (row.exactRequired) {
            totalWithinContract++;
            if (bitExact)
                exactWithinContract++;
            EXPECT_TRUE(bitExact) << "span=" << row.span << " (<= kBandFmodMaxSpan=72, MUST be bit-exact): got=("
                                   << got.hi << "," << got.lo << "," << got.tail << ") expected=("
                                   << row.expected.hi << "," << row.expected.lo << "," << row.expected.tail << ")";
        } else {
            totalBeyondContract++;
            if (bitExact)
                exactBeyondContract++;
            // Past kBandFmodMaxSpan the contract is DEGRADATION, not a
            // guarantee -- reported, not required (the reference
            // measurement puts the first inexact row at span 75, this
            // package does not re-derive that number, only exercises it).
            EXPECT_TRUE(std::isfinite(got.hi)) << "span=" << row.span << " must stay finite even past the "
                                                   "declared exactness edge";
        }
    }
    std::fprintf(stderr,
        "BandMathRound fmod span-ladder: bit-exact %zu/%zu within contract (span<=72), "
        "%zu/%zu still exact beyond it (informational, not required)\n",
        exactWithinContract, totalWithinContract, exactBeyondContract, totalBeyondContract);
    EXPECT_EQ(exactWithinContract, totalWithinContract);
}

// =========================================================================
//  fma -- shares its golden table (509/24/8) with test_BandMathCert_common.h,
//  wired here too. Spine-admitted, bound 2 ULP.
// =========================================================================
TEST(BandMathRound, FmaCertifiesSpineAdmitted)
{
    auto c = certifySpineAdmittedTernary(::aether_tests::bandmath_golden::fma::kRows,
        ::aether_tests::bandmath_golden::fma::kCorpusSize, ::aether_tests::bandmath_golden::fma::kUlpBound,
        [](Band a, Band b, Band c3) { return bd::fma(a, b, c3); });
    printRoundCounts("fma", c);
    EXPECT_EQ(c.inDomainPass, c.inDomainTotal);
    EXPECT_EQ(c.rejectedPass, c.rejectedTotal);
}
TEST(BandMathRound, FmaRedFirstCarrierPerturbation)
{
    const std::size_t red = countRedUnderCarrierPerturbationTernary(::aether_tests::bandmath_golden::fma::kRows,
        ::aether_tests::bandmath_golden::fma::kCorpusSize, ::aether_tests::bandmath_golden::fma::kUlpBound,
        [](Band a, Band b, Band c3) { return bd::fma(a, b, c3); });
    std::fprintf(stderr, "BandMathRound fma RED-first(carrier-perturbation): %zu/%zu rows RED\n", red,
        ::aether_tests::bandmath_golden::fma::kInDomainCount);
    EXPECT_GT(red, ::aether_tests::bandmath_golden::fma::kInDomainCount * 9 / 10);
}
TEST(BandMathRound, FmaRedFirstIdentityMutant)
{
    const std::size_t red = countRedUnderIdentityMutantTernary(::aether_tests::bandmath_golden::fma::kRows,
        ::aether_tests::bandmath_golden::fma::kCorpusSize, ::aether_tests::bandmath_golden::fma::kUlpBound);
    std::fprintf(stderr, "BandMathRound fma RED-first(identity-mutant): %zu/%zu rows RED\n", red,
        ::aether_tests::bandmath_golden::fma::kInDomainCount);
    EXPECT_GT(red, ::aether_tests::bandmath_golden::fma::kInDomainCount / 10);
}

} // namespace
} // namespace bandmath
} // namespace aether_tests
