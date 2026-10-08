// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_Ff2Cert_common.h
 * @brief `Ff2` (w=45 rung, df64) DD-oracle certification harness.
 *
 * Sibling of `test_Ff1Cert_common.h` — @see that file's docstring for the
 * shared shape (host/cuda split, the ingest swap).
 *
 * Two rows have no subject and are not reproduced as vacuous passes, same
 * accounting as `Ff1Cert`: no aether counterpart exists for an
 * `IsSoftWorking<Ff2>`-style trait, and no Band-typed recurrence primitive
 * exists for a chain-length sweep to call. `WidthTagAndClassification`
 * below keeps the WorkingCarrier-conformance + width-tag claim only, same
 * shape as `Ff1Cert`'s reduction.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"

#include "tests/banded/carrier_egress.h"
#include "tests/banded/cert_harness.h"
#include "tests/banded/minimal_mode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace aether_tests {
namespace Ff2CertTest {

using aether::banded::BandedReal;
using aether::banded::Ff1;
using aether::banded::Ff2;

namespace ddref = aether_tests::cert::ddref;
using aether_tests::cert::dfBits;
using aether_tests::cert::effBits;

// ─────────────────────────────────────────────────────────────────────────
//  Ingest + metrics
// ─────────────────────────────────────────────────────────────────────────

/// @brief `double` -> `Ff2`, via the UNCHECKED `Band` ingest
/// (`detail::bandFromIEEE`), NOT `BandedReal::fromDouble`. @see
/// `Ff1Cert`'s identical "swap"/ingest note (deep-underflow corpora here
/// reach ~1e-40, below `BandedReal`'s tier-1 storage window).
inline Ff2 ff2FromDouble(double v)
{
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const aether::banded::Band b = aether::banded::detail::bandFromIEEE(
        static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32));
    return b;
}

/// @brief `Ff2` -> `double`, the EGRESS half of the same swap: through the
/// CARRIER egress (`tests/banded/carrier_egress.h` -> `detail::bandToIEEE`),
/// NOT `Ff2::toDouble()`. `Ff2::toDouble()` is the STORAGE terminal
/// (`toBandedReal()` -> `detail::cell8FromBand`) and its precondition is
/// `cell8BandIsStorable`, which the deep-underflow corpus below violates by
/// construction — it aborts a debug build and silently mis-encodes a release
/// one. This reads the carrier back as a bare IEEE bit copy with no codec
/// attached. @see `carrier_egress.h`.
inline double ff2ToDouble(Ff2 v) { return aether_tests::cert::carrierToDouble(v.toBand()); }

/// @brief The EXACT df value a carrier holds (`hi + lo` is exact in double).
static inline ddref::DD ff2ExactDf(Ff2 v)
{
    return ddref::DD{ static_cast<double>(v.hi), static_cast<double>(v.lo) };
}

/// @brief Effective bits of the carrier's exact df value vs a df-matched
/// oracle.
static inline double algoBits(Ff2 got, ddref::DD oracle) { return dfBits(ff2ExactDf(got), oracle); }

// ─────────────────────────────────────────────────────────────────────────
//  Certification fixture.
// ─────────────────────────────────────────────────────────────────────────
class Ff2Cert : public ::testing::Test {
protected:
    static double mag(std::mt19937_64& rng, double lo, double hi)
    {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::bernoulli_distribution sign(0.5);
        double e = lo + u01(rng) * (hi - lo);
        return (sign(rng) ? 1.0 : -1.0) * std::pow(10.0, e);
    }
    static int samples() { return aether_tests::isMinimalMode() ? 20000 : 150000; }
};

// -------------------------------------------------------------------------
//  1. Per-primitive effective bits >= 45 (material band, df-matched inputs).
//     fma is asserted only on SAME-SIGN draws (product/addend cannot
//     cancel); the catastrophic-cancel case is CancellationOnTerminal (3b).
// -------------------------------------------------------------------------
TEST_F(Ff2Cert, PerPrimitiveEffectiveBits45)
{
    std::mt19937_64 rng(0x0FF2C0DEu);
    const int N = samples();
    double w_add = 100, w_sub = 100, w_mul = 100, w_div = 100, w_sqrt = 100, w_rsqrt = 100,
           w_fma = 100;
    double m_add = 0, m_mul = 0, m_div = 0, m_sqrt = 0;
    for (int i = 0; i < N; i++) {
        Ff2 A = ff2FromDouble(mag(rng, -4, 4));
        Ff2 B = ff2FromDouble(mag(rng, -4, 4));
        Ff2 C = ff2FromDouble(mag(rng, -4, 4));
        ddref::DD dA = ff2ExactDf(A), dB = ff2ExactDf(B), dC = ff2ExactDf(C);
        w_add = std::min(w_add, algoBits(A + B, ddref::add(dA, dB)));
        w_sub = std::min(w_sub, algoBits(A - B, ddref::sub(dA, dB)));
        w_mul = std::min(w_mul, algoBits(A * B, ddref::mul(dA, dB)));
        w_div = std::min(w_div, algoBits(A / B, ddref::divv(dA, dB)));
        m_add += algoBits(A + B, ddref::add(dA, dB));
        m_mul += algoBits(A * B, ddref::mul(dA, dB));
        m_div += algoBits(A / B, ddref::divv(dA, dB));
        ddref::DD prod = ddref::mul(dA, dB);
        const bool sameSign = (ddref::toDouble(prod) >= 0.0) == (ddref::toDouble(dC) >= 0.0);
        if (sameSign)
            w_fma = std::min(w_fma, algoBits(aether::banded::fma(A, B, C), ddref::add(prod, dC)));
        Ff2 P        = ff2FromDouble(std::fabs(mag(rng, -4, 4)));
        ddref::DD dP = ff2ExactDf(P);
        w_sqrt  = std::min(w_sqrt, algoBits(aether::banded::sqrt(P), ddref::sqrtv(dP)));
        m_sqrt += algoBits(aether::banded::sqrt(P), ddref::sqrtv(dP));
        w_rsqrt = std::min(w_rsqrt,
            algoBits(aether::banded::rsqrt(P), ddref::divv(ddref::dd(1.0), ddref::sqrtv(dP))));
    }
    std::printf(
        "[PerPrimitive] N=%d worst bits: add=%.2f sub=%.2f mul=%.2f fma*=%.2f "
        "div=%.2f sqrt=%.2f rsqrt=%.2f (means add=%.2f mul=%.2f div=%.2f "
        "sqrt=%.2f)\n",
        N, w_add, w_sub, w_mul, w_fma, w_div, w_sqrt, w_rsqrt, m_add / N, m_mul / N, m_div / N,
        m_sqrt / N);
    EXPECT_GE(w_add, 45.0);
    EXPECT_GE(w_sub, 45.0);
    EXPECT_GE(w_mul, 45.0) << "df64 mul is the tightest primitive (~45.1 worst)";
    EXPECT_GE(w_div, 45.0);
    EXPECT_GE(w_sqrt, 45.0);
    EXPECT_GE(w_rsqrt, 45.0);
    EXPECT_GE(w_fma, 45.0) << "non-cancelling fma";
}

// -------------------------------------------------------------------------
//  2. Adversarial magnitude corners (F0): the material 2-slot band holds
//     >= 45; the deep-underflow band is CHARACTERISED, not asserted.
// -------------------------------------------------------------------------
TEST_F(Ff2Cert, AdversarialCornersF0)
{
    std::mt19937_64 rng(0x0FF2C0DEu ^ 1u);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int N  = samples() / 2;
    double w_mat = 100;
    for (int i = 0; i < N; i++) {
        Ff2 A = ff2FromDouble(mag(rng, -6, 3.85)); // 3.85 -> ~7e3, F0's large corner
        Ff2 B = ff2FromDouble(mag(rng, -6, 3.85));
        ddref::DD dA = ff2ExactDf(A), dB = ff2ExactDf(B);
        w_mat = std::min(w_mat, algoBits(A * B, ddref::mul(dA, dB)));
        w_mat = std::min(w_mat, algoBits(A + B, ddref::add(dA, dB)));
    }
    {
        Ff2 big = ff2FromDouble(7079.0);
        w_mat   = std::min(w_mat, algoBits(big * big, ddref::mul(ff2ExactDf(big), ff2ExactDf(big))));
    }
    double w_deep = 100, mean_deep = 0;
    int nd = 0;
    for (int i = 0; i < N; i++) {
        double v = std::pow(10.0, -40.0 + u01(rng) * 12.0);
        Ff2 x    = ff2FromDouble(v);
        double b = effBits(ff2ToDouble(x), v);
        w_deep = std::min(w_deep, b);
        mean_deep += b;
        nd++;
    }
    std::printf(
        "[F0Corners] material band results in [~1e-12, 5e7] (incl. 7e3 corner) "
        "worst mul/add=%.2f bits (floor 45); deep-underflow round-trip "
        "[1e-40,1e-28] worst=%.2f mean=%.2f bits (CHARACTERISED, FTZ-benign per "
        "F0 -- immaterial to the terminal, not asserted)\n",
        w_mat, w_deep, mean_deep / nd);
    EXPECT_GE(w_mat, 45.0) << "2-slot full precision must hold across F0's material band";
}

// -------------------------------------------------------------------------
//  3. Cancellation-on-terminal (constructed). add/sub: the ORDER-FREE
//     algorithm captures input cancellation exactly -> still >= 45. fma: the
//     df64 product representation floors it under catastrophic product/
//     addend cancellation -> the ABSOLUTE error stays at the operand-scale
//     df floor (asserted); the relative-bit conditioning loss is reported.
// -------------------------------------------------------------------------
TEST_F(Ff2Cert, CancellationOnTerminal)
{
    std::mt19937_64 rng(0x0FF2C0DEu ^ 2u);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    double w_addsub = 100;
    for (int k = 1; k <= 40; k++) {
        double base = std::pow(10.0, -2.0 + u01(rng) * 5.0);
        double eps  = base * std::pow(2.0, -static_cast<double>(k));
        Ff2 A = ff2FromDouble(base + eps);
        Ff2 B = ff2FromDouble(-base);
        ddref::DD dA = ff2ExactDf(A), dB = ff2ExactDf(B);
        w_addsub = std::min(w_addsub, algoBits(A + B, ddref::add(dA, dB)));
    }
    EXPECT_GE(w_addsub, 45.0) << "order-free add is cancellation-robust vs the exact df-input sum";

    double worst_rel = 100, worst_abs_bits = 100;
    for (int k = 1; k <= 40; k++) {
        Ff2 A = ff2FromDouble(std::pow(10.0, -2.0 + u01(rng) * 4.0));
        Ff2 B = ff2FromDouble(std::pow(10.0, -2.0 + u01(rng) * 4.0));
        ddref::DD prod = ddref::mul(ff2ExactDf(A), ff2ExactDf(B));
        double gap = std::fabs(ddref::toDouble(prod)) * std::pow(2.0, -static_cast<double>(k));
        Ff2 C          = ff2FromDouble(-ddref::toDouble(prod) + gap);
        ddref::DD fmaO = ddref::add(prod, ff2ExactDf(C));
        Ff2 got        = aether::banded::fma(A, B, C);
        double scale   = std::max(std::fabs(ddref::toDouble(prod)), std::fabs(C.toDouble()));
        double absErr  = std::fabs(got.toDouble() - ddref::toDouble(fmaO));
        double absBits = absErr == 0.0 ? 60.0 : -std::log2(absErr / scale);
        worst_abs_bits = std::min(worst_abs_bits, absBits);
        worst_rel      = std::min(worst_rel, algoBits(got, fmaO));
    }
    std::printf(
        "[Cancellation] add/sub worst(vs df sum)=%.2f bits (floor 45); fma "
        "catastrophic-cancel worst relative=%.2f bits (df64 product floor, "
        "inherent) but absolute error stays %.2f bits below operand scale "
        "(floor 44)\n",
        w_addsub, worst_rel, worst_abs_bits);
    EXPECT_GE(worst_abs_bits, 44.0)
        << "fma must not add error beyond the df64 product floor of its inputs";
}

// -------------------------------------------------------------------------
//  4. Width-tag + WorkingCarrier classification.
//     @see the file docstring: an `IsSoftWorking<Ff2>`-style trait and a
//     `materialize<Bits>` round-trip have no subject here.
// -------------------------------------------------------------------------
TEST_F(Ff2Cert, WidthTagAndClassification)
{
    static_assert(aether::banded::WorkingCarrier<Ff2>,
        "Ff2 must conform to the WorkingCarrier concept");
    static_assert(Ff2::certifiedBits == 45, "Ff2 carries the w=45 rung width tag");
    static_assert(Ff1::certifiedBits < Ff2::certifiedBits, "Ff1 (24) is narrower than Ff2 (45)");

    std::printf("[WidthTag] WorkingCarrier<Ff2>=1 certifiedBits(Ff2)=45 (> Ff1 "
                "24); materialize<Bits> round-trip DEFERRED\n");
    SUCCEED();
}

// -------------------------------------------------------------------------
//  5. Store round-trip: `double -> Ff2 -> double` preserves >= 45 bits.
// -------------------------------------------------------------------------
TEST_F(Ff2Cert, StoreRoundTrip)
{
    std::mt19937_64 rng(0x0FF2C0DEu ^ 3u);
    const int N  = samples() / 3;
    double worst = 100;
    for (int i = 0; i < N; i++) {
        double v = mag(rng, -20, 20);
        Ff2 x    = ff2FromDouble(v);
        double rt = x.toDouble();
        worst     = std::min(worst, effBits(rt, v));
    }
    std::printf("[RoundTrip] double->Ff2->double on [1e-20,1e20] worst=%.2f "
                "bits (floor 45)\n",
        worst);
    EXPECT_GE(worst, 45.0);
}

} // namespace Ff2CertTest
} // namespace aether_tests
