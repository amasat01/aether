// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once
/**
 * @file test_BandedReal_common.h
 * @brief `BandedReal` -- the banded STORAGE VALUE TYPE: layout, round trip,
 *        injectivity, ordering, the reserved code points, `numeric_limits`
 *        and the `aether::math` facade routing.
 *
 * Every corpus, every tolerance and every derived bound below is enumerated,
 * never sampled. Four rows have no subject in aether today and are NOT
 * reproduced as vacuous passes — they stay `deferred-with-addon` and are
 * named here so the omission is
 * visible rather than silent:
 *   * `FacadePowRoutesToTheCertifiedBandedBody` — the facade entry for banded
 *     `pow` routes correctly, but the certified body it defers to does not
 *     exist yet, so a routing row would be asserting against a
 *     static_assert.
 *   * `TheLadderSelectsTheBandedRungAndLeavesTheSdOneAlone` — the demand
 *     `Ladder` and the `Ff1`/`Ff2` rungs exist, but its inertness half names
 *     a SoftDouble surface aether will never carry.
 *   * `TheArrayBiasOfTheValueTypeIsPinnedToZero` — aether has no per-array bias
 *     declaration mechanism yet.
 *   * `ExpressionAlgebraReducesInTheWorkingCarrier` — covered in two places
 *     instead: `ExpressionAlgebraOverBandedRealLeavesComputesInBandAndPacksOnce`
 *     below (the aether-side measurement of the same claim, with a codec-word
 *     value witness) and the `WorkingCarrierTest` battery in
 *     `tests/test_WorkingCarrier_common.h`.
 *
 * WHY THIS FILE IS SEPARATE FROM BandCell8Common.h / BandCell8ArrayCommon.h
 * -------------------------------------------------------------------------
 * Those two certify the CODEC (one `Band` in, one word out) and the STORAGE
 * PLUMBING (bias, texel, bulk terminals). Everything here is about the type a
 * CONSUMER declares: whether `BandedReal` is a numeric scalar in the sense a
 * generic `ComponentT` needs -- a total order that agrees with the value, an
 * equality that says `-0 == +0` where the codec's own bitwise one says
 * otherwise, limits that a step controller can read, and a facade that routes
 * `abs`/`min`/`max`/`copysign`/`pow` to the certified banded bodies instead of
 * to `std::` in native FP64.
 *
 * WHAT THE ROWS HERE ARE ALLOWED TO ASSERT
 * ----------------------------------------
 *  - The `double` round trip through the VALUE TYPE is claimed BIT-EXACT over
 *    tier 1, so the rows assert equality and never a tolerance. (The codec-level
 *    twin is `BandCell8Array.Tier1BulkRoundTripIsBitExactAcrossTheWholeWindow`;
 *    this one goes through `BandedReal::fromDouble`/`toDouble`, which is a
 *    different entry point with the same terminal underneath.)
 *  - The facade rows assert EQUIVALENCE to the certified `detail::banded::`
 *    bodies, bit for bit on the carrier. They are ROUTING claims, not accuracy
 *    claims: the accuracy of those bodies is `BandSpecialsCert`'s subject and is
 *    not re-litigated here.
 *  - Every derived bound -- `epsilon()`, `max()`, the envelope -- is RE-DERIVED
 *    IN THIS FILE from the `kBandCell8*` constants and compared against what
 *    the header computed. Two independent derivations of the same number, so a
 *    transcription in either one is visible. Derived, never calibrated.
 *  - The ordering row carries a RED CONTROL (`RawWordOrderIsNotValueOrder`): a
 *    raw `uint64_t` word comparison MUST disagree with value order on an
 *    enumerated set. Without it, "the order agrees with the value" would also
 *    pass on an implementation that just compared words, and the whole
 *    sign-magnitude transform would be untested. That row is NOT the same as
 *    `BandCell8Array.TexelWordOrderIsPinnedOnDevice`, which is an ENDIANNESS
 *    row about texel halves -- do not conflate them.
 *
 * ⚠ CONTRACT DOMAIN. `BandedReal` is defined at ARRAY BIAS 0 and its ordering
 * is certified over TIER 1 plus the reserved code points (±Inf, NaN) and both
 * zeros. Escape-tier words are deliberately OUT of contract, and the row
 * `EscapeTierWordOrderIsOutsideTheOrderingContract` demonstrates that the
 * exclusion is honest rather than merely convenient.
 */

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/backend/cuda/Launch.h>

#include "aether/banded/banded.h"

#include "tests/banded/minimal_mode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <type_traits>
#include <vector>

namespace aether_tests {
namespace BandedRealTest {

using aether::banded::Band;
using aether::banded::BandCell8;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;
namespace fm = aether::math;
using NL = std::numeric_limits<BandedReal>;

// =========================================================================
//  Compile-time surface (mirrors the header's own asserts from OUTSIDE it,
//  so a header that lost them is still caught here)
// =========================================================================

static_assert(sizeof(BandedReal) == 8 && alignof(BandedReal) == 8,
    "BandedReal must stay one 8-byte, 8-aligned memory transaction");
static_assert(sizeof(BandedReal) == sizeof(BandCell8),
    "the wrapper must cost NOTHING over the codec word it wraps");
static_assert(std::is_trivially_copyable_v<BandedReal>
        && std::is_standard_layout_v<BandedReal>
        && std::is_trivially_default_constructible_v<BandedReal>,
    "BandedReal must stay a POD: SoA buffers and cudaMemcpy "
    "all depend on it");

/* Explicit-egress-only, pinned at the type surface. An implicit
 * `operator double()` makes every `aether::math` entry point resolve
 * silently on the host through `std::`, in native FP64, while the device arm
 * static_asserts `alwaysFalse` on the identical call. The compile-fail arm
 * in tests/compile_fail/check_bandedreal_typing_rejected.sh pins the live
 * spelling; this row pins the trait. */
static_assert(!std::is_convertible_v<BandedReal, double>,
    "BR_EGRESS - BandedReal converts implicitly to double: the whole "
    "aether::math facade would resolve through std:: in native FP64 on the host "
    "and hard-error on the device");
static_assert(!std::is_convertible_v<double, BandedReal>,
    "BR_INGRESS - a double converts IMPLICITLY to BandedReal; ingest "
    "must be explicit and host-only so no `double` reaches device code by "
    "accident");
/* CONTROL for the two rows above: the sanctioned conversions ARE implicit, so
 * the trait machinery is proven live rather than uniformly answering false. */
static_assert(std::is_convertible_v<BandedReal, Band>,
    "CTRL BR_DEMOTE - the sanctioned region-entry demote to the working "
    "carrier must stay implicit, or the two BR_EGRESS/BR_INGRESS rows above "
    "prove nothing");
static_assert(std::is_convertible_v<Band, BandedReal>,
    "CTRL BR_PACK - the assignment/pack terminal must stay implicit");
static_assert(std::is_constructible_v<BandedReal, double>,
    "CTRL BR_EXPLICIT - the double ingest must still EXIST, explicitly");

/* The facade returns the WORKING carrier for every banded entry point -- a
 * chain therefore stays unpacked and pays no encode/decode per link. */
/* `fabs` has no alias here — `abs` is the single spelling — and `pow` is
 * omitted: naming it in a `decltype` would instantiate the deferring
 * static_assert and break this TU, which is exactly the loud failure that
 * member is written to produce. */
static_assert(std::is_same_v<decltype(fm::abs(std::declval<BandedReal>())), Band>);
static_assert(std::is_same_v<
    decltype(fm::fmax(std::declval<BandedReal>(), std::declval<BandedReal>())), Band>);
static_assert(std::is_same_v<
    decltype(fm::min(std::declval<BandedReal>(), std::declval<Band>())), Band>);
static_assert(std::is_same_v<
    decltype(fm::copysign(std::declval<Band>(), std::declval<BandedReal>())), Band>);
static_assert(std::is_same_v<decltype(std::declval<BandedReal>()
                                      / std::declval<BandedReal>()), Band>);
/* INERTNESS CONTROL: the shipped NATIVE resolutions are exactly what they were.
 * The `MathDispatch.h` scaffold edit (the binary scaffold's new
 * `requires(!IsBandedFamily<T>)` clause) is reached by EVERY `aether::math`
 * entry point in the library, so this is where a regression in it would surface
 * first — and it is a REAL control, because that clause is on the same
 * templates these five rows resolve through. Aether carries no SoftDouble
 * type, so the native pair IS the control here. */
static_assert(std::is_same_v<decltype(fm::abs(std::declval<double>())), double>);
static_assert(std::is_same_v<decltype(fm::abs(std::declval<float>())), float>);
static_assert(std::is_same_v<
    decltype(fm::fmax(std::declval<double>(), std::declval<double>())), double>);
static_assert(std::is_same_v<
    decltype(fm::fmin(std::declval<float>(), std::declval<float>())), float>);
static_assert(std::is_same_v<
    decltype(fm::copysign(std::declval<double>(), std::declval<double>())), double>);
static_assert(std::is_same_v<
    decltype(fm::pow(std::declval<double>(), std::declval<double>())), double>);

// =========================================================================
//  Helpers
// =========================================================================

inline uint32_t fbits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

/// @brief Bit-for-bit carrier equality -- the right comparison for a ROUTING
/// claim, where "the same body ran" is the whole assertion.
inline bool bandBitsEqual(Band a, Band b)
{
    return fbits(a.hi) == fbits(b.hi) && fbits(a.lo) == fbits(b.lo)
        && fbits(a.tail) == fbits(b.tail);
}

/// @brief The mantissa patterns, chosen so every limb boundary of the 24+23+9
/// split is exercised. Same set as the codec battery uses, so a failure here
/// can be lined up against one there.
inline const std::vector<uint64_t>& mantissaPatterns()
{
    static const std::vector<uint64_t> m = {
        0x0000000000000ull, // 1.0 -- no lower limbs at all
        0x8000000000000ull, // 1.5
        0x0000000000001ull, // one bit in the deepest tail position
        0xFFFFFFFFFFFFFull, // all 52 fraction bits
        0x0000001000000ull, // a bit at the lo/tail boundary
        0x8000001000001ull, // one bit in each of the three limbs
        0x5555555555555ull,
        0xAAAAAAAAAAAAAull,
    };
    return m;
}

inline double buildDouble(int e, uint64_t mant52, bool neg)
{
    const uint64_t bits = (neg ? (1ull << 63) : 0ull)
        | (static_cast<uint64_t>(e + 1023) << 52) | mant52;
    double x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

/// @brief Every tier-1 binade, every mantissa pattern, both signs. ENUMERATED
/// over the window rather than sampled -- 220 binades x 8 patterns x 2 signs.
inline const std::vector<double>& windowCorpus()
{
    static const std::vector<double> c = [] {
        std::vector<double> v;
        for (int e = bd::kBandCell8FloorExp; e <= bd::kBandCell8CeilingExp; e++)
            for (uint64_t m : mantissaPatterns()) {
                v.push_back(buildDouble(e, m, false));
                v.push_back(buildDouble(e, m, true));
            }
        return v;
    }();
    return c;
}

/// @brief A smaller ENUMERATION (every binade, two patterns, both signs) for
/// the all-pairs rows, whose cost is quadratic.
inline const std::vector<double>& pairCorpus()
{
    static const std::vector<double> c = [] {
        std::vector<double> v;
        for (int e = bd::kBandCell8FloorExp; e <= bd::kBandCell8CeilingExp; e++) {
            v.push_back(buildDouble(e, 0x0000000000000ull, false));
            v.push_back(buildDouble(e, 0x0000000000000ull, true));
            v.push_back(buildDouble(e, 0xFFFFFFFFFFFFFull, false));
            v.push_back(buildDouble(e, 0xFFFFFFFFFFFFFull, true));
        }
        return v;
    }();
    return c;
}

class BandedRealCert : public ::testing::Test {
};

// =========================================================================
//  1. LAYOUT
// =========================================================================
TEST_F(BandedRealCert, LayoutIsOneEightByteWordThatCostsNothingOverTheCodec)
{
    EXPECT_EQ(sizeof(BandedReal), sizeof(BandCell8));
    EXPECT_EQ(sizeof(BandedReal), 8u);
    EXPECT_EQ(alignof(BandedReal), 8u);
    /* The wrapper must be byte-transparent over the word: a BandedReal
     * reinterpreted as a BandCell8 IS the same eight bytes, which is what the
     * seam memcpy contract and the texel punning both assume. */
    const BandedReal r = BandedReal::fromBits(0x0123456789ABCDEFull);
    uint64_t raw       = 0;
    std::memcpy(&raw, &r, sizeof(raw));
    EXPECT_EQ(raw, 0x0123456789ABCDEFull);
    EXPECT_EQ(r.toBits(), 0x0123456789ABCDEFull);
    EXPECT_EQ(r.cell().w, 0x0123456789ABCDEFull);
}

TEST_F(BandedRealCert, DefaultConstructedValueIsTheCanonicalPositiveZeroWord)
{
    /* The DELIBERATE choice behind a value-initialised element: `BandedReal()`
     * VALUE-initialises, and value-initialisation ZERO-initialises a class whose
     * default constructor is `= default` rather than user-provided. So an
     * unfilled buffer of them reads back as canonical +0, not as an
     * indeterminate word — observable, not academic, since any container that
     * compares a requested fill against the default before deciding whether to
     * fill at all is asking exactly this question. */
    const BandedReal d = BandedReal();
    EXPECT_EQ(d.toBits(), 0ull);
    EXPECT_TRUE(d.isZero());
    const std::vector<BandedReal> viaContainer(4);
    for (const BandedReal& e : viaContainer)
        EXPECT_EQ(e.toBits(), 0ull);
}

// =========================================================================
//  2. ROUND TRIP, INJECTIVITY, ORDERING
// =========================================================================
TEST_F(BandedRealCert, RoundTripThroughTheValueTypeIsBitExactOverTheWholeWindow)
{
    std::size_t bad = 0;
    for (double x : windowCorpus()) {
        const BandedReal r = BandedReal::fromDouble(x);
        if (r.toDouble() != x)
            bad++;
    }
    std::printf("[BandedReal] tier-1 value-type round trip: %zu rows, %zu "
                "mismatches\n",
        windowCorpus().size(), bad);
    EXPECT_GT(windowCorpus().size(), 3000u);
    EXPECT_EQ(bad, 0u) << "the value type's own double round trip is claimed "
                          "BIT-EXACT over tier 1; a tolerance here would be an "
                          "admission that the claim is not the claim";
}

TEST_F(BandedRealCert, DistinctDoublesInWindowMapToDistinctWords)
{
    /* INJECTIVITY, enumerated over the window's binades rather than sampled.
     * A round-trip row alone cannot see a collision: two inputs could both
     * come back correct through a decode that happened to be a left inverse on
     * the corpus while the encode was not injective on it. */
    std::set<double> distinctIn;
    std::set<uint64_t> words;
    for (double x : windowCorpus()) {
        distinctIn.insert(x);
        words.insert(BandedReal::fromDouble(x).toBits());
    }
    std::printf("[BandedReal] injectivity: %zu distinct doubles -> %zu "
                "distinct words\n",
        distinctIn.size(), words.size());
    EXPECT_EQ(words.size(), distinctIn.size());
}

TEST_F(BandedRealCert, ComparisonAgreesWithTheDecodedValueOverTheWholeWindow)
{
    const auto& c = pairCorpus();
    std::vector<BandedReal> enc;
    enc.reserve(c.size());
    for (double x : c)
        enc.push_back(BandedReal::fromDouble(x));

    std::size_t pairs = 0, bad = 0;
    for (std::size_t i = 0; i < c.size(); i++)
        for (std::size_t j = 0; j < c.size(); j++) {
            pairs++;
            if ((enc[i] < enc[j]) != (c[i] < c[j]))
                bad++;
            else if ((enc[i] == enc[j]) != (c[i] == c[j]))
                bad++;
            else if ((enc[i] <= enc[j]) != (c[i] <= c[j]))
                bad++;
            else if ((enc[i] > enc[j]) != (c[i] > c[j]))
                bad++;
            else if ((enc[i] >= enc[j]) != (c[i] >= c[j]))
                bad++;
            else if ((enc[i] != enc[j]) != (c[i] != c[j]))
                bad++;
        }
    std::printf("[BandedReal] ordering: %zu ordered pairs over %zu enumerated "
                "values, %zu disagreements with the decoded value\n",
        pairs, c.size(), bad);
    EXPECT_GT(pairs, 700000u);
    EXPECT_EQ(bad, 0u);
}

TEST_F(BandedRealCert, RawWordOrderIsNotValueOrder)
{
    /* ★ THE RED CONTROL for the ordering row above. If the raw 64-bit word
     * order happened to agree with value order everywhere, then a comparison
     * implemented as a bare word compare would pass that row, and the
     * sign-magnitude transform `orderKey_` would be certified by nothing. It
     * does not agree: for NEGATIVE values the word order runs BACKWARDS, and
     * every negative pair is a disagreement. Counted, not asserted in prose.
     *
     * NOTE this is NOT `BandCell8Array.TexelWordOrderIsPinnedOnDevice`, which
     * is an ENDIANNESS claim about which texel half is which. Different
     * subject, different failure. */
    const auto& c = pairCorpus();
    std::vector<BandedReal> enc;
    enc.reserve(c.size());
    for (double x : c)
        enc.push_back(BandedReal::fromDouble(x));

    std::size_t disagreements = 0, negPairs = 0;
    for (std::size_t i = 0; i < c.size(); i++)
        for (std::size_t j = 0; j < c.size(); j++) {
            const bool rawLess = enc[i].toBits() < enc[j].toBits();
            const bool valLess = c[i] < c[j];
            if (rawLess != valLess)
                disagreements++;
            if (c[i] < 0.0 && c[j] < 0.0)
                negPairs++;
        }
    std::printf("[BandedReal] RED control: raw word order disagrees with value "
                "order on %zu of %zu pairs (%zu negative-negative pairs)\n",
        disagreements, c.size() * c.size(), negPairs);
    EXPECT_GT(disagreements, 0u)
        << "raw word order agrees with value order everywhere on this corpus, "
           "so the ordering row above would pass on a bare word comparison and "
           "certifies nothing";
    /* Sharper: EVERY strictly-ordered negative pair must disagree, because the
     * word order is exactly reversed there. */
    EXPECT_GT(negPairs, 0u);
    EXPECT_GE(disagreements, negPairs - c.size());
}

// =========================================================================
//  3. THE SPECIAL VALUES (at the VALUE level)
// =========================================================================
TEST_F(BandedRealCert, SignedZerosCompareEqualYetKeepDistinctWords)
{
    /* The single sharpest reason `BandedReal` is not a `using` alias for
     * `BandCell8`: the codec's `operator==` is a BIT comparison and answers
     * `+0 != -0` deliberately (`BandCell8::operator==`), which is right for
     * "is this buffer still the fill value" and wrong for arithmetic. */
    const BandedReal pz = BandedReal::fromBits(0ull);
    const BandedReal nz = BandedReal::fromBits(0x8000000000000000ull);
    EXPECT_NE(pz.toBits(), nz.toBits());
    EXPECT_TRUE(pz.cell() != nz.cell()); // the STORAGE answer, unchanged
    EXPECT_TRUE(pz == nz);               // the NUMERIC answer
    EXPECT_FALSE(pz != nz);
    EXPECT_FALSE(pz < nz);
    EXPECT_FALSE(nz < pz);
    EXPECT_TRUE(pz <= nz);
    EXPECT_TRUE(nz >= pz);
    EXPECT_TRUE(pz.isZero() && nz.isZero());
    /* and both order below every positive and above every negative */
    const BandedReal one = BandedReal::fromDouble(1.0);
    const BandedReal mone = BandedReal::fromDouble(-1.0);
    EXPECT_TRUE(nz < one);
    EXPECT_TRUE(pz < one);
    EXPECT_TRUE(mone < nz);
    EXPECT_TRUE(mone < pz);
}

TEST_F(BandedRealCert, SubnormalDoublesFlushToCanonicalZeroSignDestroying)
{
    /* A known-answer row, recorded rather than gated as a numerical
     * property -- what is gated is that the documented behaviour stays
     * the documented behaviour, so a silent change to it is visible.
     *
     * `cell8FromIEEE` returns `BandCell8{0}` for every
     * `bexp == 0` input. That covers ±0 AND every double subnormal, and it is
     * SIGN-DESTROYING: a NEGATIVE subnormal comes back `+0`, not `-0`. The
     * nearest existing row, `BandIngest.KnownAnswerVectorsForTheSubnormal
     * LeadingLimb`, is about a `Band` LIMB and is a different subject. */
    const double vectors[] = {
        std::numeric_limits<double>::denorm_min(),  //  2^-1074
        -std::numeric_limits<double>::denorm_min(), // -2^-1074
        std::nextafter(0.0, 1.0),
        std::ldexp(1.0, -1074),
        std::ldexp(1.0, -1023),
        -std::ldexp(1.0, -1023),
        std::ldexp(1.0, -1050),
        0.0,
        -0.0,
    };
    for (double v : vectors) {
        const BandedReal r = BandedReal::fromDouble(v);
        EXPECT_EQ(r.toBits(), 0ull)
            << "subnormal " << v << " did not flush to the CANONICAL ZERO word";
        EXPECT_EQ(r.toDouble(), 0.0);
        /* the sign is destroyed, not preserved -- stated so a change is loud */
        EXPECT_FALSE(std::signbit(r.toDouble()))
            << "a negative subnormal came back as -0; this flush is documented as "
               "sign-destroying and something changed";
    }
}

TEST_F(BandedRealCert, TheReservedCodePointsClassifyAndOrderAsTheyShould)
{
    const BandedReal pinf = NL::infinity();
    const BandedReal ninf = NL::neg_infinity();
    const BandedReal nan  = NL::quiet_NaN();

    EXPECT_EQ(pinf.toBits(), bd::kBandCell8PosInfWord);
    EXPECT_EQ(ninf.toBits(), bd::kBandCell8NegInfWord);
    EXPECT_EQ(nan.toBits(), bd::kBandCell8NanWord);

    EXPECT_TRUE(pinf.isInf() && ninf.isInf());
    EXPECT_FALSE(pinf.isNan() || ninf.isNan());
    EXPECT_TRUE(nan.isNan());
    EXPECT_FALSE(nan.isInf());

    /* NaN compares FALSE against everything, itself included -- and `!=` is
     * therefore TRUE, which is the one relation that inverts. */
    EXPECT_FALSE(nan == nan);
    EXPECT_TRUE(nan != nan);
    EXPECT_FALSE(nan < pinf);
    EXPECT_FALSE(nan > pinf);
    EXPECT_FALSE(nan <= nan);
    EXPECT_FALSE(nan >= nan);

    /* The infinities bracket the ENTIRE representable window, enumerated. */
    std::size_t bad = 0;
    for (double x : pairCorpus()) {
        const BandedReal r = BandedReal::fromDouble(x);
        if (!(r < pinf) || !(ninf < r))
            bad++;
    }
    EXPECT_EQ(bad, 0u) << "some in-window value did not order strictly between "
                          "-Inf and +Inf";
    EXPECT_TRUE(ninf < pinf);
    EXPECT_TRUE(NL::lowest() < pinf && ninf < NL::lowest());
    EXPECT_TRUE(NL::max() < pinf && ninf < NL::max());
}

TEST_F(BandedRealCert, EscapeTierWordOrderIsOutsideTheOrderingContract)
{
    /* HONESTY ROW. The type's ordering is certified over TIER 1 at bias 0 plus
     * the reserved code points. This row shows the exclusion is REAL rather
     * than defensive drafting: a DOWNWARD escape word stores its extended
     * exponent as a magnitude, so a SMALLER value gets a LARGER bit pattern and
     * the order inverts. Recorded so nobody later "fixes" the contract by
     * widening it without changing the transform. */
    const double tiny1 = std::ldexp(1.0, -200); // below tier 1's floor
    const double tiny2 = std::ldexp(1.0, -300); // further below
    const BandCell8 c1  = bd::cell8FromDouble(tiny1, 0, /*useEscape=*/true);
    const BandCell8 c2  = bd::cell8FromDouble(tiny2, 0, /*useEscape=*/true);
    ASSERT_TRUE(bd::cell8IsEscape(c1) && bd::cell8IsEscape(c2));
    ASSERT_LT(tiny2, tiny1);
    const BandedReal r1 = BandedReal::fromCell(c1);
    const BandedReal r2 = BandedReal::fromCell(c2);
    /* the VALUES say r2 < r1; the type's order does NOT, and that is the
     * documented contract boundary */
    EXPECT_FALSE(r2 < r1) << "the escape tier now orders correctly -- either "
                             "the encoding changed or the contract can be "
                             "widened; either way this row and BandedReal.h's "
                             "contract note must be revisited together";
}

// =========================================================================
//  4. numeric_limits -- DERIVED HERE AGAIN, INDEPENDENTLY
// =========================================================================
TEST_F(BandedRealCert, LimitsAreDerivedFromTheCodecConstantsAndNotTranscribed)
{
    /* The derivation, restated from the `kBandCell8*` names so that the header
     * and the test are two independent computations of the same numbers:
     *
     *   digits    = kBandCell8Depth                                 = 56
     *   epsilon   = 2^(1 - digits)                                  = 2^-55
     *   envelope  = [kBandCell8FloorExp + margin,
     *                kBandCell8CeilingExp - margin]  (margin = 8)   = [-86, 117]
     *   min()     = 2^-86, continuation zero
     *   max()     = the largest binary32 at 2^117 (mantissa all ones)
     *               plus a full 32-bit continuation
     *   lowest()  = -max()
     */
    const int digits = bd::kBandCell8Depth;
    const int margin = bd::kBandCell8AdmissionMargin;
    const int topExp = bd::kBandCell8CeilingExp - margin;
    const int botExp = bd::kBandCell8FloorExp + margin;

    EXPECT_EQ(NL::digits, digits);
    EXPECT_EQ(NL::radix, 2);
    EXPECT_EQ(digits, 56);
    EXPECT_EQ(margin, 8);
    EXPECT_EQ(topExp, 117);
    EXPECT_EQ(botExp, -86);

    auto pow2Word = [](int e) {
        return static_cast<uint64_t>(static_cast<uint32_t>(e + 127) << 23) << 32;
    };
    const uint64_t maxWord
        = (static_cast<uint64_t>((static_cast<uint32_t>(topExp + 127) << 23)
               | 0x7FFFFFu)
              << 32)
        | 0xFFFFFFFFull;

    EXPECT_EQ(NL::epsilon().toBits(), pow2Word(1 - digits));
    EXPECT_EQ(NL::min().toBits(), pow2Word(botExp));
    EXPECT_EQ(NL::denorm_min().toBits(), NL::min().toBits());
    EXPECT_EQ(NL::round_error().toBits(), pow2Word(-1));
    EXPECT_EQ(NL::max().toBits(), maxWord);
    EXPECT_EQ(NL::lowest().toBits(), maxWord | 0x8000000000000000ull);

    /* the standard exponent members follow the SAME envelope, not the codec
     * window -- a specialization whose max() means "admissible" and whose
     * max_exponent means "representable" would contradict itself */
    EXPECT_EQ(NL::max_exponent, topExp + 1);
    EXPECT_EQ(NL::min_exponent, botExp + 1);

    /* flags a consumer branches on */
    EXPECT_TRUE(NL::is_specialized);
    EXPECT_TRUE(NL::is_signed);
    EXPECT_FALSE(NL::is_integer);
    EXPECT_FALSE(NL::is_exact);
    EXPECT_FALSE(NL::is_iec559);
    EXPECT_TRUE(NL::is_bounded);
    EXPECT_TRUE(NL::has_infinity);
    EXPECT_TRUE(NL::has_quiet_NaN);
    EXPECT_FALSE(NL::has_signaling_NaN);
    EXPECT_EQ(NL::has_denorm, std::denorm_absent);
}

TEST_F(BandedRealCert, LimitsMaxIsTheMarginRespectingAdmissibleTopAndTheCodecTopIsNot)
{
    /* The known-answer row for the type-level margin-respecting top.
     *
     * `numeric_limits<BandedReal>::max()` is the TYPE-LEVEL MARGIN-RESPECTING
     * admissible top -- not the codec's own tier-1 top. The reason is
     * mechanical rather than stylistic: `bandCell8Admits` applies
     * `kBandCell8AdmissionMargin = 8` binades at each end, so a consumer
     * declaring the codec's own ceiling is REFUSED. A `max()` returning a value
     * the array's own admission guard rejects is a trap. */
    const int margin = bd::kBandCell8AdmissionMargin;
    const int topExp = bd::kBandCell8CeilingExp - margin;
    const int botExp = bd::kBandCell8FloorExp + margin;

    EXPECT_TRUE(bd::bandCell8Admits(topExp, botExp));
    EXPECT_FALSE(bd::bandCell8Admits(topExp + 1, botExp));
    EXPECT_FALSE(bd::bandCell8Admits(topExp, botExp - 1));
    /* the codec's own window is NOT an admissible declaration */
    EXPECT_FALSE(
        bd::bandCell8Admits(bd::kBandCell8CeilingExp, bd::kBandCell8FloorExp));

    /* max() decodes to a finite, normalized carrier whose exponent IS the
     * declared top */
    const Band bmax = static_cast<Band>(NL::max());
    EXPECT_TRUE(std::isfinite(bmax.hi));
    EXPECT_GT(bmax.hi, 0.0f);
    int e = 0;
    std::frexp(bmax.hi, &e);
    EXPECT_EQ(e - 1, topExp) << "max()'s leading limb does not sit at the "
                                "declared envelope top";
    /* ...and it is normalized: each limb inside one ulp of the previous */
    EXPECT_LE(std::fabs(bmax.lo), std::ldexp(1.0f, topExp - 23));
    EXPECT_LE(std::fabs(bmax.tail), std::ldexp(1.0f, topExp - 46));

    /* the codec top is strictly larger, and is NAMED rather than hidden */
    const BandedReal codecTop
        = BandedReal::fromBits(aether::banded::kBandedRealCodecTopWord);
    EXPECT_TRUE(NL::max() < codecTop);
    EXPECT_EQ(aether::banded::kBandedRealMaxExp, topExp);
    EXPECT_EQ(aether::banded::kBandedRealMinExp, botExp);
    EXPECT_EQ(aether::banded::kBandedRealCodecCeilingExp, bd::kBandCell8CeilingExp);

    /* The capping identity: an "unbounded" step-size sentinel clamped at
     * max() and one clamped at the codec top agree on every step size a
     * shipped configuration domain can reach. A 2^117-second step is not a
     * reachable step size, so the two sentinels are indistinguishable where
     * it matters. */
    const double shippedSteps[] = { 1e-9, 1e-3, 1.0, 60.0, 3600.0, 86400.0,
        86400.0 * 365.25, 86400.0 * 365.25 * 1000.0, std::ldexp(1.0, 60) };
    for (double s : shippedSteps) {
        const BandedReal r = BandedReal::fromDouble(s);
        EXPECT_TRUE(r < NL::max());
        EXPECT_TRUE(r < codecTop);
    }
}

// =========================================================================
//  5. FACADE ROUTING -- equivalence to the certified banded bodies
// =========================================================================
TEST_F(BandedRealCert, FacadeUnaryEntryPointsRouteToTheCertifiedBandedBodies)
{
    std::size_t rows = 0, bad = 0;
    for (double x : pairCorpus()) {
        const BandedReal r = BandedReal::fromDouble(x);
        const Band b       = static_cast<Band>(r);
        rows++;
        /* The facade carries the single spelling `abs`, so there is no
         * second name to route. */
        if (!bandBitsEqual(fm::abs(r), bd::abs(b)))
            bad++;
        if (!bandBitsEqual(fm::abs(b), bd::abs(b)))
            bad++;
    }
    std::printf("[BandedReal] facade unary: %zu rows, %zu route mismatches\n",
        rows, bad);
    EXPECT_GT(rows, 800u);
    EXPECT_EQ(bad, 0u);
}

TEST_F(BandedRealCert, FacadeBinaryEntryPointsRouteToTheCertifiedBandedBodies)
{
    const auto& c = pairCorpus();
    std::size_t rows = 0, bad = 0;
    for (std::size_t i = 0; i + 1 < c.size(); i++) {
        const BandedReal ra = BandedReal::fromDouble(c[i]);
        const BandedReal rb = BandedReal::fromDouble(c[i + 1]);
        const Band a = static_cast<Band>(ra), b = static_cast<Band>(rb);
        rows++;
        if (!bandBitsEqual(fm::fmax(ra, rb), bd::fmax(a, b)))
            bad++;
        if (!bandBitsEqual(fm::min(ra, rb), bd::fmin(a, b)))
            bad++;
        if (!bandBitsEqual(fm::fmax(ra, rb), bd::fmax(a, b)))
            bad++;
        if (!bandBitsEqual(fm::fmin(ra, rb), bd::fmin(a, b)))
            bad++;
        if (!bandBitsEqual(fm::copysign(ra, rb), bd::copysign(a, b)))
            bad++;
        /* MIXED operands: storage x working, both orders. This is the shape a
         * chain actually produces -- one packed leaf against one live carrier. */
        if (!bandBitsEqual(fm::fmax(ra, b), bd::fmax(a, b)))
            bad++;
        if (!bandBitsEqual(fm::min(a, rb), bd::fmin(a, b)))
            bad++;
        /* pure working-carrier calls: RED today before the routing landed */
        if (!bandBitsEqual(fm::fmax(a, b), bd::fmax(a, b)))
            bad++;
    }
    std::printf("[BandedReal] facade binary: %zu rows, %zu route mismatches\n",
        rows, bad);
    EXPECT_GT(rows, 800u);
    EXPECT_EQ(bad, 0u);
}

TEST_F(BandedRealCert, ArithmeticOperatorsRouteToTheCertifiedBandedBodies)
{
    /* The namespace-scope operator set. `Band`'s own operators are HIDDEN
     * FRIENDS, so a conversion cannot reach them --
     * these operators are the only thing that makes `BandedReal / BandedReal`
     * resolve at all, which is what the whole vector expression algebra bottoms
     * out in. Every one of them RETURNS the working carrier. */
    const auto& c = pairCorpus();
    std::size_t rows = 0, bad = 0, packed = 0, unstorable = 0;
    for (std::size_t i = 0; i + 1 < c.size(); i++) {
        const BandedReal ra = BandedReal::fromDouble(c[i]);
        const BandedReal rb = BandedReal::fromDouble(c[i + 1]);
        const Band a = static_cast<Band>(ra), b = static_cast<Band>(rb);
        rows++;
        if (!bandBitsEqual(ra + rb, bd::add(a, b)))
            bad++;
        if (!bandBitsEqual(ra - rb, bd::sub(a, b)))
            bad++;
        if (!bandBitsEqual(ra * rb, bd::mul(a, b)))
            bad++;
        if (!bandBitsEqual(ra / rb, bd::div(a, b)))
            bad++;
        if (!bandBitsEqual(-ra, bd::neg(a)))
            bad++;
        /* mixed storage x working, both orders */
        if (!bandBitsEqual(ra * b, bd::mul(a, b)))
            bad++;
        if (!bandBitsEqual(a - rb, bd::sub(a, b)))
            bad++;

        /* The compound forms pack exactly once, at the destination.
         *
         * ★ GUARDED, and the guard is the CONTRACT rather than a convenience.
         * A pack is `cell8FromBand`, whose precondition is
         * `cell8BandIsStorable` -- a AETHER_DEBUG_MODE build asserts it, and it is
         * right to. This corpus walks consecutive binade entries, so one pair in
         * four is a near-total CANCELLATION (`-M_e` against `+2^(e+1)` leaves
         * `2^(e-52)`), which at the bottom of the window lands 52 binades BELOW
         * the storage floor. Storing that is a consumer error, not a library
         * one, so the row must not manufacture it: it asks the shipped predicate
         * and counts both answers. Measured on this corpus: the unstorable arm
         * is non-empty, which is exactly why the guard cannot be dropped. */
        const Band sum = bd::add(a, b);
        if (!bd::cell8BandIsStorable(sum)) {
            unstorable++;
        } else {
            packed++;
            BandedReal acc = ra;
            acc += rb;
            if (!bandBitsEqual(static_cast<Band>(acc),
                    static_cast<Band>(BandedReal(sum))))
                bad++;
        }
    }
    std::printf("[BandedReal] operators: %zu rows, %zu mismatches; compound "
                "assignment exercised on %zu rows, %zu skipped as unstorable "
                "sums (cancellation below the storage floor)\n",
        rows, bad, packed, unstorable);
    EXPECT_GT(rows, 800u);
    EXPECT_GT(packed, 600u) << "the compound-assignment (pack) leg was skipped "
                               "almost everywhere, so it certifies nothing";
    EXPECT_GT(unstorable, 0u)
        << "no pair in this corpus produces an unstorable sum, so the guard "
           "above is inert and the row would pass without it -- which is not "
           "what was measured when it was written";
    EXPECT_EQ(bad, 0u);
}

// =========================================================================
//  6. THE EXPRESSION-TEMPLATE SEAM -- records a MEASURED answer rather than
//     asserting a desired one
// =========================================================================
TEST_F(BandedRealCert, ExpressionAlgebraOverBandedRealLeavesComputesInBandAndPacksOnce)
{
    /* THE STORAGE/VIEW SEAM. `View<BandedReal, ...>` and `Item<BandedReal,
     * ...>` need no change to aether's core: the chunk holds 8-byte codec
     * words and the assignment proxy writes a chain back into a `BandedReal`
     * slot.
     *
     * An expression tree carries one `element_type`, so an intermediate node
     * packing at every step (`a - b` computed in `Band` via the operators in
     * `BandedRealOps.h`, then encoded on the way out of `eval()`) would cost
     * one encode/decode per link. Instead, `aether::WorkingType<BandedReal>`
     * (declared at the end of `aether/banded/BandedReal.h`) maps the storage
     * scalar to `Band`, `Expression` publishes `working_type`, and every
     * node's `eval()` returns it — so a chain packs exactly once, at the
     * assignment terminal.
     *
     * This row is the measurement of that. It is two claims, and the second
     * is the load-bearing one: a type-level claim (an intermediate node
     * really does hand back `Band`), and a VALUE claim over inputs on which
     * packing at every node vs. packing once would disagree — asserting
     * BOTH equality with the once-packed hand reference AND inequality with
     * the per-node one, because without the inequality leg the row would
     * pass just as happily on either behaviour.
     *
     * The `element_type` protocol is asserted below too: the storage scalar
     * is still what two operands are matched on and what an assignment
     * target is checked against. So is the reduction contract — a reduce
     * folds in the carrier and returns the storage type, one encode at the
     * end; a `Band`-returning reduce is future work if a consumer ever
     * needs one. */
    constexpr std::size_t D = 3;
    constexpr std::size_t n = 2;

    std::vector<BandedReal> errBuf(D * n), desBuf(D * n), nxtBuf(D * n),
        nowBuf(D * n);
    for (std::size_t k = 0; k < n; k++)
        for (std::size_t d = 0; d < D; d++) {
            // SoA layout: component-major, `[d * n + k]`.
            nxtBuf[d * n + k] = BandedReal::fromDouble(-2.0 - double(d));
            nowBuf[d * n + k] = BandedReal::fromDouble(1.0 + double(d));
            errBuf[d * n + k] = BandedReal::fromDouble(1e-6 * (1.0 + double(d)));
            desBuf[d * n + k] = BandedReal::fromDouble(1.0);
        }

    const aether::Device dev(kDLCPU);
    auto err = aether::make_view<BandedReal, D, aether::dyn>(errBuf.data(), dev, n);
    auto des = aether::make_view<BandedReal, D, aether::dyn>(desBuf.data(), dev, n);
    auto nxt = aether::make_view<BandedReal, D, aether::dyn>(nxtBuf.data(), dev, n);
    auto now = aether::make_view<BandedReal, D, aether::dyn>(nowBuf.data(), dev, n);

    const aether::SampleIndex i = aether::SampleIndex::make(0);
    const aether::Item<BandedReal, D> e = err[i].get();
    const aether::Item<BandedReal, D> w = now[i].get();
    const aether::Item<BandedReal, D> x = nxt[i].get();

    // =====================================================================
    //  1. THE TYPE-LEVEL MEASUREMENT
    // =====================================================================

    /* The trait itself, and the two shapes the doctrine is about: a single
     * arithmetic node, and a three-op chain over it. Both hand back the WORKING
     * carrier. `remove_cvref_t` because `eval()` returns by value here but a
     * LEAF returns a reference — the claim is about the carrier, not the value
     * category. */
    static_assert(std::is_same_v<aether::working_type_t<BandedReal>, Band>,
        "BandedReal must map to the Band working carrier -- without this "
        "specialisation the identity primary answers BandedReal and every node "
        "silently encodes again");
    static_assert(
        std::is_same_v<std::remove_cvref_t<decltype((w + x).template eval<0>(i))>, Band>,
        "an intermediate Sum node still evaluates in the STORAGE type: the "
        "working-carrier mapping is not reaching aether's expression nodes");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(
                                     (((w + x) - e) * BandedReal::fromDouble(1.0))
                                         .template eval<0>(i))>,
                      Band>,
        "a three-op chain still evaluates in the STORAGE type");

    /* The half that did NOT move, asserted so a later change cannot quietly
     * take it: `element_type` is still the storage scalar (it is what every
     * operand-match and assignment-target static_assert keys on), and a
     * REDUCTION still returns it. */
    static_assert(std::is_same_v<decltype(w + x)::element_type, BandedReal>,
        "element_type must stay the STORAGE scalar -- the assignment terminal "
        "and every operand match are keyed on it");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(e.maxNorm())>, BandedReal>,
        "a reduction must still RETURN the storage type: fold in the "
        "carrier, ONE encode at the end");

    const auto ratio = e.maxNorm();
    EXPECT_NEAR(ratio.toDouble(), 3e-6, 1e-15);

    /* The VALUES are unaffected by where the packs land, which is the half that
     * matters for a consumer: max(|-(2+d)|) over d = 0..2 is 4. */
    const BandedReal desired = x.maxNorm();
    EXPECT_NEAR(desired.toDouble(), 4.0, 1e-12);

    // =====================================================================
    //  2. THE VALUE WITNESS -- inputs on which the two policies DISAGREE
    // =====================================================================

    /* HOW THE WITNESS WAS CHOSEN (the arithmetic, not a search).
     * The codec resolves a value with float exponent `ee` down to 2^(ee-55); the
     * carrier keeps ~70 bits of limb. So a pack is visible exactly where an
     * intermediate carries information below 2^(ee-55) -- and a following
     * subtraction promotes it to the leading bit.
     *   a = 1, b = 3*2^-57, s = 2^55, chain ((a + b) - a) * s.
     *   packing (a + b) rounds 0.75 ulp of the continuation to 1 ulp (RNE),
     *   so the per-node path loses b's true size: (1 + 2^-55) - 1 = 2^-55,
     *   times s = 1.0 exactly.
     *   the carrier keeps b exactly: 3*2^-57 * 2^55 = 0.75 exactly.
     * Both endpoints are exact tier-1 values, so the comparison is on CODEC
     * WORDS and never a tolerance; 0.75 vs 1.0 is a 1/3 relative gap, which no
     * rounding policy could produce by accident. */
    const BandedReal wA = BandedReal::fromDouble(1.0);
    const BandedReal wB = BandedReal::fromDouble(3.0 * std::ldexp(1.0, -57));
    const BandedReal wS = BandedReal::fromDouble(std::ldexp(1.0, 55));

    // ONCE-packed hand reference: the whole chain in Band, a single encode.
    const BandedReal wantOnce = BandedReal::fromBand(((wA + wB) - wA) * wS);
    // PER-NODE-packed hand reference: an encode at every node (the old policy).
    const BandedReal node1       = wA + wB;
    const BandedReal node2       = node1 - wA;
    const BandedReal wantPerNode = node2 * wS;

    EXPECT_DOUBLE_EQ(wantOnce.toDouble(), 0.75);
    EXPECT_DOUBLE_EQ(wantPerNode.toDouble(), 1.0);
    /* THE NON-VACUITY CONTROL FOR THE WITNESS ITSELF. If the two references
     * agreed, every assertion below would pass on either policy and this row
     * would certify nothing. */
    EXPECT_NE(wantOnce.toBits(), wantPerNode.toBits())
        << "the witness does not discriminate: the once-packed and per-node "
           "references are the same codec word, so the row is inert";

    std::vector<BandedReal> waBuf(D * n, wA), wbBuf(D * n, wB), woBuf(D * n);
    auto wav = aether::make_view<BandedReal, D, aether::dyn>(waBuf.data(), dev, n);
    auto wbv = aether::make_view<BandedReal, D, aether::dyn>(wbBuf.data(), dev, n);
    auto wov = aether::make_view<BandedReal, D, aether::dyn>(woBuf.data(), dev, n);
    const aether::Item<BandedReal, D> A = wav[i].get();
    const aether::Item<BandedReal, D> B = wbv[i].get();
    wov[i] = ((A + B) - A) * wS;

    std::printf("[BandedReal/ET] witness ((1 + 3*2^-57) - 1) * 2^55: chain=%.17g "
                "once-packed=%.17g per-node-packed=%.17g\n",
        woBuf[0].toDouble(), wantOnce.toDouble(), wantPerNode.toDouble());
    for (std::size_t d = 0; d < D; d++) {
        EXPECT_EQ(woBuf[d * n + 0].toBits(), wantOnce.toBits())
            << "component " << d
            << ": the expression chain did not deliver the ONCE-packed value";
        EXPECT_NE(woBuf[d * n + 0].toBits(), wantPerNode.toBits())
            << "component " << d
            << ": the expression chain still packs at every node";
    }

    // =====================================================================
    //  3. THE ASSIGNMENT TERMINAL (unchanged claim, kept)
    // =====================================================================

    /* A two-node chain read from two views and written back into a third. The
     * arithmetic runs in `Band` and the store encodes -- now the ONLY encode.
     * Tier-1 exact operands, so the comparison is EQUALITY and not a tolerance. */
    std::vector<BandedReal> outBuf(D * n);
    auto out = aether::make_view<BandedReal, D, aether::dyn>(outBuf.data(), dev, n);
    out[i]   = w + x;
    for (std::size_t d = 0; d < D; d++) {
        const double expect = (1.0 + double(d)) + (-2.0 - double(d));
        EXPECT_EQ(outBuf[d * n + 0].toDouble(), expect) << "component " << d;
    }

    /* Non-vacuity for the whole row: the leaves really are 8-byte codec words
     * in the caller's own buffer, not a promoted `double` somewhere. */
    static_assert(sizeof(decltype(errBuf)::value_type) == 8);
    EXPECT_NE(errBuf[0].toBits(), 0ull);
    (void)des;
}

} // namespace BandedRealTest
} // namespace aether_tests
