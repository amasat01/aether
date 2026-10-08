// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_Ff1Cert_common.h
 * @brief `Ff1` (w=24 rung) DD-oracle certification harness.
 *
 * Shared by `test_Ff1Cert.cu` (CUDA mode: these host certs + a device-parity
 * kernel) and `test_Ff1Cert.cpp` (CPP_MODE: these host certs only) — aether's
 * suite globs `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE (mirrors
 * `test_BandCell8`'s own split), so exactly ONE TU registers the `TEST_F`s
 * below per mode. `Ff1` is `AETHER_DEVICEHOST()`, so every cert here runs
 * identically on the host in both modes; the device path is exercised by the
 * `.cu`'s device-parity kernel.
 *
 * Two rows have no subject and are not reproduced as vacuous passes: a
 * `WorkingCarrier`-typed recurrence terminal does not exist for `Ff1` (no
 * `Recurrence.h` primitive is Band-typed), so a chain-length recurrence
 * sweep has nothing to call; the `WidthTagAndClassification` row below
 * keeps the WorkingCarrier-conformance + width-tag claim only.
 *
 * @section swap The ingest swap
 * Every operand is built via the unchecked `Band` ingest and converted to
 * `Ff1` (through `Ff2`'s certified IEEE decode) rather than through
 * `BandedReal::fromDouble` (tier-1 HOST ingest, a narrower domain). @see
 * `ff1FromDouble` below.
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
namespace Ff1CertTest {

using aether::banded::BandedReal;
using aether::banded::Ff1;
using aether::banded::Ff2;

namespace ddref = aether_tests::cert::ddref;
using aether_tests::cert::dfBits;
using aether_tests::cert::effBits;

// ─────────────────────────────────────────────────────────────────────────
//  Ingest + metrics
// ─────────────────────────────────────────────────────────────────────────

/// @brief `double` -> `Ff1`, via the UNCHECKED `Band` ingest
/// (`detail::bandFromIEEE`), NOT `BandedReal::fromDouble` — the DD-oracle
/// corpora below deliberately probe magnitudes (e.g. `AdversarialCornersF0`'s
/// deep-underflow arm, ~1e-44) far outside `BandedReal`'s tier-1 STORAGE
/// window `[2^-94, 2^126)`. This is a lossless bit-carry with no
/// storage-window semantics attached (Ff1/Ff2 are chain-resident working
/// carriers, never storage). @see the file docstring's "swap" section.
inline Ff1 ff1FromDouble(double v)
{
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const aether::banded::Band b = aether::banded::detail::bandFromIEEE(
        static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32));
    return b;
}

/// @brief `Ff1` -> `double`, the EGRESS half of the same swap: through the
/// CARRIER egress (`tests/banded/carrier_egress.h` -> `detail::bandToIEEE`),
/// NOT `Ff1::toDouble()`. `Ff1::toDouble()` is the STORAGE terminal
/// (`toFf2().toBandedReal()` -> `detail::cell8FromBand`) and its precondition
/// is `cell8BandIsStorable`, which the deep-underflow corpus below violates by
/// construction — it aborts a debug build and silently mis-encodes a release
/// one. This reads the carrier back as a bare IEEE bit copy with no codec
/// attached. @see `carrier_egress.h`.
inline double ff1ToDouble(Ff1 v) { return aether_tests::cert::carrierToDouble(v.toFf2().toBand()); }

/// @brief The EXACT value a carrier holds (a single binary32 -> exact double).
static inline ddref::DD ff1ExactDf(Ff1 v) { return ddref::DD{ static_cast<double>(v.v), 0.0 }; }

/// @brief Effective bits of the carrier's exact value vs a matched oracle
/// (isolates the OPERATION's rounding error).
static inline double algoBits(Ff1 got, ddref::DD oracle) { return dfBits(ff1ExactDf(got), oracle); }

// ─────────────────────────────────────────────────────────────────────────
//  Certification fixture. Random helpers follow aether's own house idiom:
//  a local `std::mt19937_64` per test, @see test_BandCell8_common.h.
// ─────────────────────────────────────────────────────────────────────────
class Ff1Cert : public ::testing::Test {
protected:
    /// @brief Signed magnitude `10^[lo,hi]`.
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
//  1. Per-primitive effective bits: a single correctly-rounded binary32 op
//     carries >= 24 bits. div/sqrt/fma single ops too; rsqrt is two RN ops
//     (~1 ulp -> ~23 bits, characterised). fma holds >= 24 EVEN under
//     cancellation (one RN op rounds the exact result once).
// -------------------------------------------------------------------------
TEST_F(Ff1Cert, PerPrimitiveEffectiveBits24)
{
    std::mt19937_64 rng(0x0FF1C0DEu);
    const int N = samples();
    double w_add = 100, w_sub = 100, w_mul = 100, w_div = 100, w_sqrt = 100, w_rsqrt = 100,
           w_fma = 100;
    double m_add = 0, m_mul = 0, m_div = 0, m_sqrt = 0;
    for (int i = 0; i < N; i++) {
        Ff1 A = ff1FromDouble(mag(rng, -4, 4));
        Ff1 B = ff1FromDouble(mag(rng, -4, 4));
        Ff1 C = ff1FromDouble(mag(rng, -4, 4));
        ddref::DD dA = ff1ExactDf(A), dB = ff1ExactDf(B), dC = ff1ExactDf(C);
        w_add = std::min(w_add, algoBits(A + B, ddref::add(dA, dB)));
        w_sub = std::min(w_sub, algoBits(A - B, ddref::sub(dA, dB)));
        w_mul = std::min(w_mul, algoBits(A * B, ddref::mul(dA, dB)));
        w_div = std::min(w_div, algoBits(A / B, ddref::divv(dA, dB)));
        m_add += algoBits(A + B, ddref::add(dA, dB));
        m_mul += algoBits(A * B, ddref::mul(dA, dB));
        m_div += algoBits(A / B, ddref::divv(dA, dB));
        w_fma = std::min(
            w_fma, algoBits(aether::banded::fma(A, B, C), ddref::add(ddref::mul(dA, dB), dC)));
        Ff1 P        = ff1FromDouble(std::fabs(mag(rng, -4, 4)));
        ddref::DD dP = ff1ExactDf(P);
        w_sqrt  = std::min(w_sqrt, algoBits(aether::banded::sqrt(P), ddref::sqrtv(dP)));
        m_sqrt += algoBits(aether::banded::sqrt(P), ddref::sqrtv(dP));
        w_rsqrt = std::min(w_rsqrt,
            algoBits(aether::banded::rsqrt(P), ddref::divv(ddref::dd(1.0), ddref::sqrtv(dP))));
    }
    std::printf(
        "[PerPrimitive] N=%d worst bits: add=%.2f sub=%.2f mul=%.2f fma=%.2f "
        "div=%.2f sqrt=%.2f rsqrt=%.2f (means add=%.2f mul=%.2f div=%.2f "
        "sqrt=%.2f)\n",
        N, w_add, w_sub, w_mul, w_fma, w_div, w_sqrt, w_rsqrt, m_add / N, m_mul / N, m_div / N,
        m_sqrt / N);
    EXPECT_GE(w_add, 24.0);
    EXPECT_GE(w_sub, 24.0);
    EXPECT_GE(w_mul, 24.0);
    EXPECT_GE(w_div, 24.0);
    EXPECT_GE(w_sqrt, 24.0);
    EXPECT_GE(w_fma, 24.0) << "single RN fma is >= 24 even under cancellation";
    EXPECT_GE(w_rsqrt, 23.0) << "rsqrt = 1/sqrt is two RN ops (~1 ulp)";
}

// -------------------------------------------------------------------------
//  2. Adversarial magnitude corners: the single-float normal band holds
//     >= 24; the deep-underflow band (denormals) is CHARACTERISED, not
//     asserted.
// -------------------------------------------------------------------------
TEST_F(Ff1Cert, AdversarialCornersF0)
{
    std::mt19937_64 rng(0x0FF1C0DEu ^ 1u);
    const int N  = samples() / 2;
    double w_mat = 100;
    for (int i = 0; i < N; i++) {
        Ff1 A = ff1FromDouble(mag(rng, -12, 12));
        Ff1 B = ff1FromDouble(mag(rng, -12, 12));
        ddref::DD dA = ff1ExactDf(A), dB = ff1ExactDf(B);
        w_mat = std::min(w_mat, algoBits(A * B, ddref::mul(dA, dB)));
        w_mat = std::min(w_mat, algoBits(A + B, ddref::add(dA, dB)));
    }
    double w_deep = 100, mean_deep = 0;
    int nd = 0;
    for (int i = 0; i < N; i++) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        double v = std::pow(10.0, -44.0 + u01(rng) * 5.0);
        Ff1 x    = ff1FromDouble(v);
        double b = effBits(ff1ToDouble(x), v);
        w_deep = std::min(w_deep, b);
        mean_deep += b;
        nd++;
    }
    std::printf(
        "[F0Corners] normal band [1e-12,1e12] worst mul/add=%.2f bits (floor "
        "24); deep-underflow round-trip [1e-44,1e-39] worst=%.2f mean=%.2f bits "
        "(CHARACTERISED, denormal -- immaterial to the terminal, not asserted)\n",
        w_mat, w_deep, mean_deep / nd);
    EXPECT_GE(w_mat, 24.0) << "single-float full precision must hold across the normal band";
}

// -------------------------------------------------------------------------
//  3. Cancellation-on-terminal: a single RN op rounds the exact result once,
//     so add/sub/fma stay >= 24 EVEN under constructed catastrophic
//     cancellation.
// -------------------------------------------------------------------------
TEST_F(Ff1Cert, CancellationOnTerminal)
{
    std::mt19937_64 rng(0x0FF1C0DEu ^ 2u);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double w_addsub = 100;
    for (int k = 1; k <= 40; k++) {
        double base = std::pow(10.0, -2.0 + u01(rng) * 5.0);
        double eps  = base * std::pow(2.0, -static_cast<double>(k));
        Ff1 A = ff1FromDouble(base + eps);
        Ff1 B = ff1FromDouble(-base);
        ddref::DD dA = ff1ExactDf(A), dB = ff1ExactDf(B);
        w_addsub = std::min(w_addsub, algoBits(A + B, ddref::add(dA, dB)));
    }
    double w_fma = 100;
    for (int k = 1; k <= 40; k++) {
        Ff1 A = ff1FromDouble(std::pow(10.0, -2.0 + u01(rng) * 4.0));
        Ff1 B = ff1FromDouble(std::pow(10.0, -2.0 + u01(rng) * 4.0));
        ddref::DD prod = ddref::mul(ff1ExactDf(A), ff1ExactDf(B));
        double gap = std::fabs(ddref::toDouble(prod)) * std::pow(2.0, -static_cast<double>(k));
        Ff1 C          = ff1FromDouble(-ddref::toDouble(prod) + gap);
        ddref::DD fmaO = ddref::add(prod, ff1ExactDf(C));
        w_fma          = std::min(w_fma, algoBits(aether::banded::fma(A, B, C), fmaO));
    }
    std::printf(
        "[Cancellation] add/sub worst(vs df sum)=%.2f bits (floor 24); fma "
        "catastrophic-cancel worst=%.2f bits (single RN op, floor 24 -- no "
        "df-product floor on a 1-limb carrier)\n",
        w_addsub, w_fma);
    EXPECT_GE(w_addsub, 24.0) << "single-float add is cancellation-robust";
    EXPECT_GE(w_fma, 24.0) << "single RN fma rounds the exact result once -- >= 24 under cancel";
}

// -------------------------------------------------------------------------
//  4. Width-tag + WorkingCarrier classification.
//     @see the file docstring: a `materialize<Bits>` demand-terminal
//     round-trip has no subject here (no Band-domain demand mechanism is
//     wired for this rung yet), so this row keeps the conformance +
//     width-tag half only.
// -------------------------------------------------------------------------
TEST_F(Ff1Cert, WidthTagAndClassification)
{
    static_assert(aether::banded::WorkingCarrier<Ff1>,
        "Ff1 must conform to the WorkingCarrier concept");
    static_assert(Ff1::certifiedBits == 24, "Ff1 carries the FP32-tail 24-bit width tag");
    static_assert(Ff1::certifiedBits < Ff2::certifiedBits, "Ff1 (24) is narrower than Ff2 (45)");

    // What remains is the conformance + width-tag half, fully testable from
    // Ff1.h/Ff2.h alone.
    std::printf("[WidthTag] WorkingCarrier<Ff1>=1 certifiedBits(Ff1)=24 (< Ff2 "
                "45); materialize<Bits> round-trip DEFERRED (width wall not "
                "yet wired for this rung)\n");
    SUCCEED();
}

// -------------------------------------------------------------------------
//  5. Store round-trip: `double -> Ff1 -> double` preserves >= 24 bits.
// -------------------------------------------------------------------------
TEST_F(Ff1Cert, StoreRoundTrip)
{
    std::mt19937_64 rng(0x0FF1C0DEu ^ 3u);
    const int N  = samples() / 3;
    double worst = 100;
    for (int i = 0; i < N; i++) {
        double v = mag(rng, -20, 20);
        Ff1 x    = ff1FromDouble(v);
        double rt = x.toDouble();
        worst     = std::min(worst, effBits(rt, v));
    }
    std::printf("[RoundTrip] double->Ff1->double on [1e-20,1e20] worst=%.2f "
                "bits (floor 24)\n",
        worst);
    EXPECT_GE(worst, 24.0);
}

} // namespace Ff1CertTest
} // namespace aether_tests
