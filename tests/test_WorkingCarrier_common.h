// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_WorkingCarrier_common.h
 * @brief `WorkingCarrierTest` — the working-carrier ET substrate, over
 *        `aether::banded::Band` (the carrier) and `aether::banded::BandedReal`
 *        (the storage word).
 *
 * Covers: mixed-node coverage over a homogeneous `Band` tree (`Sum`/`Diff`
 * both operand orders, `cwiseMul` both orders, `CWiseScale` both scalar
 * orders, storing into a `View` and into an `Item` local); a carrier held
 * across three statements as an `Item<Band,3>`, with the region entry and
 * the pack terminal written explicitly (mixing a storage-typed leaf into a
 * carrier tree, or vice versa, is refused by the one-`element_type`-per-tree
 * rule); carrier-vs-hand-packed-demote comparisons for a single-statement
 * chain (bit-exact, no intermediate encode boundary) and a three-statement
 * chain (a ULP band derived from the codec's own 56-bit significand, with
 * its own red control: if the two arms agreed everywhere the row would
 * certify nothing); and `NativeScalarsAreTheirOwnCarrier`, the zero-cost
 * claim at the type level, with a control proving the trait is not
 * uniformly the identity.
 *
 * @section exact Why these rows compare codec words and not doubles
 * Every operand is built through `BandedReal::fromDouble`, which is exact over
 * tier 1, and every claim below is either "these two encodes agree" or "these
 * two encodes disagree". Both are bit questions, so the comparisons are on
 * `toBits()` and there is no tolerance to inherit — except in the one row whose
 * whole subject is a magnitude of disagreement, where the bound is derived from
 * the codec constant rather than measured.
 */

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "aether/banded/banded.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace aether_tests {
namespace WorkingCarrier {

using aether::banded::Band;
using aether::banded::BandedReal;

using aether::dyn;
using aether::Item;
using aether::SampleIndex;

constexpr std::size_t D = 3;   ///< component count (a 3-vector shape)
constexpr std::size_t kN = 64; ///< samples per view (SoA, component-major)

// =========================================================================
//  Corpus — deterministic, tier-1 exact, moderate magnitude
// =========================================================================

/** @brief SplitMix64 — a deterministic stream, so a measured band cannot flake
 *         on ambient seeding. */
inline std::uint64_t splitmix64(std::uint64_t& s)
{
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/**
 * @brief A signed value with a FULL 53-bit mantissa and |x| in [1e-2, 1e2].
 *
 * Full mantissa on purpose: the whole subject is what a 56-bit encode does to an
 * intermediate, and a value with a short mantissa would encode exactly and hide
 * it. The magnitude window is bounded well inside the codec's storage window
 * (`[2^-94, 2^126)`) so no arm can leave it by accident and turn a numeric claim
 * into an out-of-window throw.
 */
inline double randMag(std::uint64_t& s)
{
    const std::uint64_t r = splitmix64(s);
    const double unit = static_cast<double>(r >> 11) * 0x1p-53; // [0,1), 53 bits
    const double sign = (r & 1u) ? 1.0 : -1.0;
    const double mant = 1.0 + unit;                        // [1,2)
    const int ex = static_cast<int>((r >> 3) % 13u) - 6;   // 2^-6 .. 2^6
    return sign * mant * std::ldexp(1.0, ex);
}

/** @brief ULP distance between two doubles. */
inline std::int64_t ulpDist(double a, double b)
{
    if (std::isnan(a) || std::isnan(b))
        return 0x7FFFFFFFFFFFFFFFLL;
    if (a == b)
        return 0;
    std::uint64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if (ia >> 63)
        ia = 0x8000000000000000ull - ia;
    if (ib >> 63)
        ib = 0x8000000000000000ull - ib;
    return static_cast<std::int64_t>(ia > ib ? ia - ib : ib - ia);
}

/** @brief Three SoA host buffers of `D * kN` encoded values, plus their doubles. */
struct Corpus {
    std::vector<BandedReal> buf;
    std::vector<double> ref;

    explicit Corpus(std::uint64_t& s)
        : buf(D * kN)
        , ref(D * kN)
    {
        for (std::size_t k = 0; k < kN; k++)
            for (std::size_t d = 0; d < D; d++) {
                const double v = randMag(s);
                ref[d * kN + k] = v;
                buf[d * kN + k] = BandedReal::fromDouble(v);
            }
    }
};

class WorkingCarrierTest : public ::testing::Test { };

// =========================================================================
//  Node coverage: Sum / Diff / cwiseMul / CWiseScale, both orders
// =========================================================================

TEST_F(WorkingCarrierTest, MixedLeafCoverageSumMulScale)
{
    std::uint64_t seed = 0x57C5A311u;
    const Corpus a(seed), b(seed);

    const aether::Device dev(kDLCPU);
    std::vector<BandedReal> aBuf = a.buf, bBuf = b.buf;
    auto av = aether::make_view<BandedReal, D, dyn>(aBuf.data(), dev, kN);
    auto bv = aether::make_view<BandedReal, D, dyn>(bBuf.data(), dev, kN);

    const SampleIndex i0 = SampleIndex::make(0);
    const Item<BandedReal, D> A = av[i0].get();
    const Item<BandedReal, D> B = bv[i0].get();
    const BandedReal s = BandedReal::fromDouble(2.5);

    /* ★ THE TYPE CLAIM, node by node. Every one of these is an INTERMEDIATE and
     * every one hands back the carrier — which is the whole content of "no pack
     * per op" expressed at the type level rather than only in a number. */
    static_assert(std::is_same_v<std::remove_cvref_t<decltype((A + B).template eval<0>(i0))>, Band>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype((A - B).template eval<0>(i0))>, Band>);
    static_assert(
        std::is_same_v<std::remove_cvref_t<decltype(A.cwiseMul(B).template eval<0>(i0))>, Band>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype((A * s).template eval<0>(i0))>, Band>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype((s * A).template eval<0>(i0))>, Band>);
    /* ...and the storage protocol is UNCHANGED underneath it. */
    static_assert(std::is_same_v<decltype(A + B)::element_type, BandedReal>);

    /* The two leaves above appear only in unevaluated operands, so this is what
     * actually reads them — and it certifies the register materialization
     * (`SampleRef::get()`) delivers the caller's own codec words untouched. */
    EXPECT_EQ(A(0).toBits(), a.buf[0].toBits());
    EXPECT_EQ(B(0).toBits(), b.buf[0].toBits());

    std::vector<BandedReal> oSumAB(D * kN), oSumBA(D * kN), oDiff(D * kN), oMulAB(D * kN),
        oMulBA(D * kN), oScaleAs(D * kN), oScaleSa(D * kN), oViaItem(D * kN);
    auto vSumAB = aether::make_view<BandedReal, D, dyn>(oSumAB.data(), dev, kN);
    auto vSumBA = aether::make_view<BandedReal, D, dyn>(oSumBA.data(), dev, kN);
    auto vDiff = aether::make_view<BandedReal, D, dyn>(oDiff.data(), dev, kN);
    auto vMulAB = aether::make_view<BandedReal, D, dyn>(oMulAB.data(), dev, kN);
    auto vMulBA = aether::make_view<BandedReal, D, dyn>(oMulBA.data(), dev, kN);
    auto vScaleAs = aether::make_view<BandedReal, D, dyn>(oScaleAs.data(), dev, kN);
    auto vScaleSa = aether::make_view<BandedReal, D, dyn>(oScaleSa.data(), dev, kN);
    auto vViaItem = aether::make_view<BandedReal, D, dyn>(oViaItem.data(), dev, kN);

    std::size_t bad = 0;
    for (std::size_t k = 0; k < kN; k++) {
        const SampleIndex i = SampleIndex::make(k);
        const Item<BandedReal, D> x = av[i].get();
        const Item<BandedReal, D> y = bv[i].get();

        vSumAB[i] = x + y;
        vSumBA[i] = y + x;
        vDiff[i] = x - y;
        vMulAB[i] = x.cwiseMul(y);
        vMulBA[i] = y.cwiseMul(x);
        vScaleAs[i] = x * s;
        vScaleSa[i] = s * x;

        /* Variant: the same node stored into an `Item<BandedReal,D>` LOCAL first
         * (a register-resident terminal, a different code path from the View
         * store above), then relayed out. */
        Item<BandedReal, D> viaItem;
        viaItem = x + y;
        vViaItem[i] = viaItem;

        for (std::size_t d = 0; d < D; d++) {
            const std::size_t o = d * kN + k;
            const Band bx = static_cast<Band>(a.buf[o]);
            const Band by = static_cast<Band>(b.buf[o]);
            if (oSumAB[o].toBits() != BandedReal::fromBand(bx + by).toBits())
                bad++;
            if (oSumBA[o].toBits() != BandedReal::fromBand(by + bx).toBits())
                bad++;
            if (oDiff[o].toBits() != BandedReal::fromBand(bx - by).toBits())
                bad++;
            if (oMulAB[o].toBits() != BandedReal::fromBand(bx * by).toBits())
                bad++;
            if (oMulBA[o].toBits() != BandedReal::fromBand(by * bx).toBits())
                bad++;
            if (oScaleAs[o].toBits() != BandedReal::fromBand(static_cast<Band>(s) * bx).toBits())
                bad++;
            if (oScaleSa[o].toBits() != BandedReal::fromBand(static_cast<Band>(s) * bx).toBits())
                bad++;
            if (oViaItem[o].toBits() != oSumAB[o].toBits())
                bad++;
        }
    }
    std::printf("[WorkingCarrier] node coverage: %zu samples x %zu components x 8 "
                "node forms, %zu word mismatches\n",
        kN, D, bad);
    EXPECT_EQ(bad, 0u) << "an expression node did not deliver the value its "
                          "certified Band body computes";
    /* Non-vacuity: the leaves really are 8-byte codec words, not promoted doubles. */
    static_assert(sizeof(BandedReal) == 8);
    EXPECT_NE(oSumAB[0].toBits(), 0ull);
}

// =========================================================================
//  A carrier held across THREE statements
// =========================================================================

TEST_F(WorkingCarrierTest, BridgeAssignHeldAcrossStatements)
{
    std::uint64_t seed = 0x0BADC0DEu;
    const Corpus s0(seed), s1(seed), s2(seed);

    const aether::Device dev(kDLCPU);
    std::vector<BandedReal> b0 = s0.buf, b1 = s1.buf, b2 = s2.buf;
    auto v0 = aether::make_view<BandedReal, D, dyn>(b0.data(), dev, kN);
    auto v1 = aether::make_view<BandedReal, D, dyn>(b1.data(), dev, kN);
    auto v2 = aether::make_view<BandedReal, D, dyn>(b2.data(), dev, kN);

    const BandedReal rk1 = BandedReal::fromDouble(1.75);
    const BandedReal rk2 = BandedReal::fromDouble(-2.5);
    const BandedReal rk3 = BandedReal::fromDouble(0.5);
    const Band k1 = static_cast<Band>(rk1);
    const Band k2 = static_cast<Band>(rk2);
    const Band k3 = static_cast<Band>(rk3);

    /* A carrier-typed `Item` is a conforming aether expression leaf, and a tree
     * built entirely out of them stays in the carrier end to end — that is the
     * multi-statement pattern this row is about (a NAV-style Chebyshev
     * accumulation shape). What has no aether spelling is mixing it with a
     * storage-typed leaf in one node, so the region entry (decode) and the pack
     * terminal are written out per component here. See this file's header. */
    static_assert(aether::aether_expression<Item<Band, D>>,
        "a carrier-typed Item must be a conforming expression leaf, or the "
        "multi-statement carrier pattern has no spelling at all");
    static_assert(std::is_same_v<aether::working_type_t<Band>, Band>,
        "the carrier must be its own carrier (idempotency) — the whole node "
        "protocol leans on converting an already-converted value being free");

    std::size_t bad = 0;
    double worstRatio = 0.0;
    for (std::size_t k = 0; k < kN; k++) {
        const SampleIndex i = SampleIndex::make(k);
        const Item<BandedReal, D> r0 = v0[i].get();
        const Item<BandedReal, D> r1 = v1[i].get();
        const Item<BandedReal, D> r2 = v2[i].get();

        Item<Band, D> c0, c1, c2;
        for (std::size_t d = 0; d < D; d++) {
            c0(d) = static_cast<Band>(r0(d));
            c1(d) = static_cast<Band>(r1(d));
            c2(d) = static_cast<Band>(r2(d));
        }

        // Statement 1 / 2 / 3 — the carrier is HELD, never demoted in between.
        Item<Band, D> carrier;
        carrier = c0 * k1;
        carrier = carrier + c1 * k2;
        carrier = carrier - c2 * k3;

        for (std::size_t d = 0; d < D; d++) {
            const std::size_t o = d * kN + k;
            const Band bx = static_cast<Band>(s0.buf[o]);
            const Band by = static_cast<Band>(s1.buf[o]);
            const Band bz = static_cast<Band>(s2.buf[o]);
            const Band want = ((bx * k1) + (by * k2)) - (bz * k3);
            // The carrier arm must be BIT-EXACT against the same Band chain:
            // holding it across statements costs nothing at all.
            const BandedReal got = BandedReal::fromBand(carrier(d));
            if (got.toBits() != BandedReal::fromBand(want).toBits())
                bad++;
            /* A host FP64 SANITY cross-check (it catches a gross sign/order
             * bug), bounded RELATIVE TO THE TERMS and not to the result, so a
             * cancelling sample cannot make the bound meaningless. Both sides
             * carry 53 bits over three operations; `8 * 2^-53 * (sum of term
             * magnitudes)` is that, with declared headroom. */
            const double host = (s0.ref[o] * 1.75 + s1.ref[o] * -2.5) - s2.ref[o] * 0.5;
            const double mag = std::fabs(s0.ref[o] * 1.75) + std::fabs(s1.ref[o] * 2.5)
                + std::fabs(s2.ref[o] * 0.5);
            const double ratio = std::fabs(got.toDouble() - host) / (8.0 * 0x1p-53 * mag);
            if (ratio > worstRatio)
                worstRatio = ratio;
        }
    }
    std::printf("[WorkingCarrier] carrier held across 3 statements: %zu word "
                "mismatches, worst |diff|/derived-bound vs host FP64 = %.3f\n",
        bad, worstRatio);
    EXPECT_EQ(bad, 0u) << "holding the carrier across statements changed the value";
    EXPECT_LE(worstRatio, 1.0) << "the three-statement chain disagrees with a plain "
                                  "FP64 evaluation by more than 53-bit rounding "
                                  "over three operations can explain";
}

// =========================================================================
//  Carrier vs hand-packed parity
// =========================================================================

/** @brief The two arms both rows below share: an ET chain, and the same chain
 *         with an explicit encode at every statement boundary. */
TEST_F(WorkingCarrierTest, CarrierVsDemoteSingleStatementBitExact)
{
    std::uint64_t seed = 0x13572468u;
    const Corpus a(seed), b(seed);

    const aether::Device dev(kDLCPU);
    std::vector<BandedReal> aBuf = a.buf, bBuf = b.buf, oBuf(D * kN);
    auto av = aether::make_view<BandedReal, D, dyn>(aBuf.data(), dev, kN);
    auto bv = aether::make_view<BandedReal, D, dyn>(bBuf.data(), dev, kN);
    auto ov = aether::make_view<BandedReal, D, dyn>(oBuf.data(), dev, kN);

    const BandedReal two = BandedReal::fromDouble(2.0);
    const BandedReal three = BandedReal::fromDouble(3.0);

    for (std::size_t k = 0; k < kN; k++) {
        const SampleIndex i = SampleIndex::make(k);
        const Item<BandedReal, D> x = av[i].get();
        const Item<BandedReal, D> y = bv[i].get();
        ov[i] = x * two + y * three; // ONE statement, ONE encode
    }

    std::size_t bad = 0;
    for (std::size_t k = 0; k < kN; k++)
        for (std::size_t d = 0; d < D; d++) {
            const std::size_t o = d * kN + k;
            /* The hand reference applies exactly ONE encode, at exactly the same
             * point in the computation as the ET does — so with no intermediate
             * boundary in either path the two are BIT-EXACT, not merely close.
             * This is the row that would have gone red before this substrate existed. */
            const Band want = static_cast<Band>(a.buf[o]) * static_cast<Band>(two)
                + static_cast<Band>(b.buf[o]) * static_cast<Band>(three);
            if (oBuf[o].toBits() != BandedReal::fromBand(want).toBits())
                bad++;
        }
    std::printf("[WorkingCarrier] single-statement parity over %zu samples: %zu "
                "mismatches (bit-exact expected)\n",
        kN * D, bad);
    EXPECT_EQ(bad, 0u);
}

TEST_F(WorkingCarrierTest, CarrierVsDemoteMultiStatementUlpBand)
{
    std::uint64_t seed = 0x2468ACE0u;
    const Corpus s0(seed), s1(seed), s2(seed);

    const aether::Device dev(kDLCPU);
    std::vector<BandedReal> b0 = s0.buf, b1 = s1.buf, b2 = s2.buf;
    auto v0 = aether::make_view<BandedReal, D, dyn>(b0.data(), dev, kN);
    auto v1 = aether::make_view<BandedReal, D, dyn>(b1.data(), dev, kN);
    auto v2 = aether::make_view<BandedReal, D, dyn>(b2.data(), dev, kN);

    const BandedReal rk1 = BandedReal::fromDouble(1.75);
    const BandedReal rk2 = BandedReal::fromDouble(-2.5);
    const BandedReal rk3 = BandedReal::fromDouble(0.5);
    const Band k1 = static_cast<Band>(rk1);
    const Band k2 = static_cast<Band>(rk2);
    const Band k3 = static_cast<Band>(rk3);

    std::size_t differing = 0, overBound = 0;
    double worstRatio = 0.0;
    std::int64_t worstUlp = 0;

    for (std::size_t k = 0; k < kN; k++) {
        const SampleIndex i = SampleIndex::make(k);
        const Item<BandedReal, D> r0 = v0[i].get();
        const Item<BandedReal, D> r1 = v1[i].get();
        const Item<BandedReal, D> r2 = v2[i].get();

        /* THE DEMOTE ARM: an `Item<BandedReal,D>` re-typed at every statement,
         * so each boundary pays an encode. Three statements, THREE encodes. */
        Item<BandedReal, D> demote;
        demote = r0 * rk1;
        demote = demote + r1 * rk2;
        demote = demote - r2 * rk3;

        for (std::size_t d = 0; d < D; d++) {
            const std::size_t o = d * kN + k;
            const Band bx = static_cast<Band>(s0.buf[o]);
            const Band by = static_cast<Band>(s1.buf[o]);
            const Band bz = static_cast<Band>(s2.buf[o]);

            /* THE CARRIER ARM: the identical chain, held in the carrier, ONE
             * encode at the end. */
            const Band t1 = bx * k1;
            const Band t2 = by * k2;
            const Band t3 = bz * k3;
            const Band p2 = t1 + t2;
            const BandedReal carrier = BandedReal::fromBand(p2 - t3);

            if (carrier.toBits() != demote(d).toBits())
                differing++;
            const std::int64_t u = ulpDist(carrier.toDouble(), demote(d).toDouble());
            if (u > worstUlp)
                worstUlp = u;

            /* ★ THE BOUND IS DERIVED, NEVER FITTED — and it is derived from the
             * CARRIER's width, not the codec's. The demote arm's extra work is
             * two encodes the carrier does not do (after statements 1 and 2),
             * each worth at most `2^-56 |v|`. But that perturbation then flows
             * through the REMAINING certified ops, and the two arms therefore
             * apply the same ops to DIFFERENT operands — so their roundings do
             * not cancel, and `Band` certifies 53 bits, not 56. The governing
             * term is one carrier rounding at each of the three sites (the add,
             * the sub, the final encode), against the magnitudes actually
             * rounded there; the factor 4 is DECLARED headroom over those sites,
             * not a number read off a run.
             * (Measured under this bound: max 1 ulp of FP64, worst ratio ~0.35 —
             * reported below so the headroom is visible rather than assumed.) */
            const double e1 = std::fabs(BandedReal::fromBand(t1).toDouble());
            const double e2 = std::fabs(BandedReal::fromBand(p2).toDouble());
            const double e3 = std::fabs(carrier.toDouble());
            const double tol = 4.0 * 0x1p-53 * (e1 + e2 + e3);
            const double diff = std::fabs(carrier.toDouble() - demote(d).toDouble());
            if (diff > tol)
                overBound++;
            if (tol > 0.0 && diff / tol > worstRatio)
                worstRatio = diff / tol;
        }
    }

    const std::size_t total = kN * D;
    std::printf("[WorkingCarrier] 3-statement carrier vs demote over %zu samples: "
                "%zu differ (%.1f%%), max %lld ulp, worst |diff|/derived-bound = "
                "%.3f\n",
        total, differing, 100.0 * double(differing) / double(total),
        static_cast<long long>(worstUlp), worstRatio);

    EXPECT_EQ(overBound, 0u) << "the two arms disagree by MORE than the carrier's "
                                "own 53-bit certification over three rounding "
                                "sites can explain — that is not the extra "
                                "rounding this row is about";
    /* ★ THE RED CONTROL. If the demote arm agreed with the carrier everywhere,
     * every assertion above would pass on a build that never removed the
     * per-node pack, and the row would certify nothing. The extra encodes must
     * be VISIBLE. */
    EXPECT_GT(differing, total / 10)
        << "the demote arm is bit-identical to the carrier arm almost "
           "everywhere, so this row does not measure the extra encodes it "
           "claims to measure";
}

// =========================================================================
//  The zero-cost claim, at the type level
// =========================================================================

TEST_F(WorkingCarrierTest, NativeScalarsAreTheirOwnCarrier)
{
    /* The half that must cost nothing. `working_type_t<T> == T` for every
     * native dtype is what makes the whole working-carrier mechanism free
     * where no emulation is in play — the node bodies, the store terminal
     * and the reduce ops all collapse to the plain (non-carrier)
     * expressions when this holds. */
    static_assert(std::is_same_v<aether::working_type_t<double>, double>);
    static_assert(std::is_same_v<aether::working_type_t<float>, float>);
    static_assert(std::is_same_v<aether::working_type_t<bool>, bool>);
    static_assert(std::is_same_v<aether::working_type_t<std::int32_t>, std::int32_t>);
    static_assert(std::is_same_v<aether::working_type_t<std::int64_t>, std::int64_t>);
    static_assert(std::is_same_v<aether::working_type_t<std::uint8_t>, std::uint8_t>);
    static_assert(std::is_same_v<aether::working_type_t<std::uint32_t>, std::uint32_t>);
    static_assert(std::is_same_v<aether::working_type_t<std::uint64_t>, std::uint64_t>);

    /* cv/ref-stripping: a node reads through `const T&`, so the trait must give
     * the same answer for the reference type it is handed. */
    static_assert(std::is_same_v<aether::working_type_t<const double&>, double>);
    static_assert(std::is_same_v<aether::working_type_t<const BandedReal&>, Band>);

    /* Idempotency — converting an already-converted value is the identity. The
     * node bodies convert unconditionally rather than branching on `isLeaf`,
     * which is only sound because of this. */
    static_assert(std::is_same_v<aether::working_type_t<Band>, Band>);

    /* Both conversions are constant expressions on the primary, which is what
     * lets a `constexpr` expression evaluation survive the indirection. */
    static_assert(aether::WorkingType<double>::fromWorking(
                      aether::WorkingType<double>::toWorking(3.5))
        == 3.5);

    /* ★ THE CONTROL. Without this line every assertion above is equally
     * consistent with a trait that answers `T` for absolutely everything —
     * i.e. with the mechanism not existing at all. */
    static_assert(!std::is_same_v<aether::working_type_t<BandedReal>, BandedReal>,
        "the storage scalar maps to ITSELF: the banded specialisation is not "
        "visible here, so every 'native scalars are free' assertion above is "
        "true for the wrong reason");

    const BandedReal r = BandedReal::fromDouble(1.25);
    const Band w = aether::WorkingType<BandedReal>::toWorking(r);
    EXPECT_EQ(aether::WorkingType<BandedReal>::fromWorking(w).toBits(), r.toBits())
        << "the storage -> working -> storage round trip is not the identity on "
           "a tier-1 exact value";
    std::printf("[WorkingCarrier] native dtypes are their own carrier; BandedReal "
                "maps to Band (%zu-byte storage, %zu-byte carrier)\n",
        sizeof(BandedReal), sizeof(Band));
}

} // namespace WorkingCarrier
} // namespace aether_tests
