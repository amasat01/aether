// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandSpecials_common.h
 * @brief Enumerated corpus of IEEE-754 special-value behavior for the
 *        `Band` carrier and the `BandCell8` codec: infinities and NaNs
 *        propagate through add/sub/mul/div/recip/abs/neg with the correct
 *        sign (IEEE 754-2019 clauses 6.1-7.3), a special is representable
 *        in a `BandCell8` at reserved code points no finite double can
 *        reach and round-trips through every codec path, and egress
 *        delivers the special it was handed rather than laundering it to a
 *        finite value. Every row is constructed from a named operand
 *        class, not sampled.
 */

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/backend/cuda/Launch.h>

#include "aether/banded/banded.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace aether_tests {
namespace BandSpecialsTest {

using aether::banded::Band;
using aether::banded::BandCell8;
namespace bd = aether::banded::detail;

// =========================================================================
//  Bit-level helpers -- specials are BIT patterns, so nothing here compares
//  with `==` (which is false for every NaN and blind to a zero's sign).
// =========================================================================

inline std::uint32_t bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
inline float fromBits(std::uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
inline std::uint64_t bits64(double d)
{
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}
inline double fromBits64(std::uint64_t u)
{
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

/// @brief The ingest terminal, spelled the way a consumer reaches it.
inline Band bandOf(double x)
{
    const std::uint64_t u = bits64(x);
    return bd::bandFromIEEE(
        static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32));
}

/// @brief The egress terminal, spelled the way a consumer reaches it.
inline double doubleOf(Band b)
{
    std::uint32_t lo = 0, hi = 0;
    bd::bandToIEEE(b, lo, hi);
    return fromBits64((static_cast<std::uint64_t>(hi) << 32) | lo);
}

// =========================================================================
//  The operand classes, and the classifier the IEEE table is written in
// =========================================================================

enum class Cls {
    NegInf,
    NegFinite,
    NegZero,
    PosZero,
    PosFinite,
    PosInf,
    NaN,
};

inline const char* clsName(Cls c)
{
    switch (c) {
        case Cls::NegInf: return "-Inf";
        case Cls::NegFinite: return "-finite";
        case Cls::NegZero: return "-0";
        case Cls::PosZero: return "+0";
        case Cls::PosFinite: return "+finite";
        case Cls::PosInf: return "+Inf";
        case Cls::NaN: return "NaN";
    }
    return "?";
}

/// @brief Classify the value a CARRIER holds, from its leading limb's bits.
inline Cls classify(float v)
{
    const std::uint32_t u = bits(v);
    const bool neg        = (u & 0x80000000u) != 0u;
    const std::uint32_t e = u & 0x7F800000u;
    const std::uint32_t m = u & 0x007FFFFFu;
    if (e == 0x7F800000u)
        return m ? Cls::NaN : (neg ? Cls::NegInf : Cls::PosInf);
    if (e == 0u && m == 0u)
        return neg ? Cls::NegZero : Cls::PosZero;
    return neg ? Cls::NegFinite : Cls::PosFinite;
}

inline Cls classify(double v)
{
    const std::uint64_t u = bits64(v);
    const bool neg         = (u >> 63) != 0u;
    const std::uint64_t e  = u & 0x7FF0000000000000ull;
    const std::uint64_t m  = u & 0x000FFFFFFFFFFFFFull;
    if (e == 0x7FF0000000000000ull)
        return m ? Cls::NaN : (neg ? Cls::NegInf : Cls::PosInf);
    if (e == 0u && m == 0u)
        return neg ? Cls::NegZero : Cls::PosZero;
    return neg ? Cls::NegFinite : Cls::PosFinite;
}

inline Cls classifyBand(Band b) { return classify(b.hi); }

// =========================================================================
//  The enumerated operand set
// =========================================================================

struct Operand {
    const char* name;
    double value;
    Cls cls;
};

inline double posInf() { return std::numeric_limits<double>::infinity(); }
inline double negInf() { return -std::numeric_limits<double>::infinity(); }

/// @brief The NaN PAYLOAD variants, enumerated rather than sampled.
inline std::vector<Operand> nanPayloadVariants()
{
    std::vector<Operand> v;
    auto add = [&v](const char* n, std::uint64_t u) {
        v.push_back(Operand{ n, fromBits64(u), Cls::NaN });
    };
    add("qNaN canonical (+)", 0x7FF8000000000000ull);
    add("qNaN canonical (-)", 0xFFF8000000000000ull);
    add("qNaN min payload", 0x7FF8000000000001ull);
    add("qNaN max payload", 0x7FFFFFFFFFFFFFFFull);
    add("sNaN min payload", 0x7FF0000000000001ull);
    add("sNaN max payload", 0x7FF7FFFFFFFFFFFFull);
    add("sNaN max payload (-)", 0xFFF7FFFFFFFFFFFFull);
    add("qNaN mixed payload", 0x7FFA5A5A5A5A5A5Aull);
    return v;
}

/// @brief The full enumerated operand set: the specials, the signed zeros,
/// the storage tier's two edges, and ordinary finite representatives.
inline std::vector<Operand> operands()
{
    std::vector<Operand> v{
        { "-Inf", negInf(), Cls::NegInf },
        { "+Inf", posInf(), Cls::PosInf },
        { "-0", -0.0, Cls::NegZero },
        { "+0", 0.0, Cls::PosZero },
        { "+0.9", 0.9, Cls::PosFinite },
        { "-2.0", -2.0, Cls::NegFinite },
        { "+1.0", 1.0, Cls::PosFinite },
        { "+2^-94 (tier floor)", 0x1p-94, Cls::PosFinite },
        { "-2^-94 (tier floor)", -0x1p-94, Cls::NegFinite },
        { "+2^125 (tier ceiling)", 0x1p125, Cls::PosFinite },
        { "-2^125 (tier ceiling)", -0x1p125, Cls::NegFinite },
        { "+2^-126 (fp32 subnormal edge)", 0x1p-126, Cls::PosFinite },
    };
    for (const Operand& o : nanPayloadVariants())
        v.push_back(o);
    return v;
}

// =========================================================================
//  The IEEE-754 special-value tables, written out as the reference values
// =========================================================================

struct Rule {
    Cls a;
    Cls b;
    Cls want;
    const char* cite;
};

/// IEEE 754-2019 clause 6.1 (infinity), 6.3 (sign), 7.2 (invalid -> qNaN).
inline const std::vector<Rule>& kMulTable()
{
    static const std::vector<Rule> t{
        { Cls::PosFinite, Cls::PosInf, Cls::PosInf, "6.1 finite x +Inf" },
        { Cls::PosInf, Cls::PosFinite, Cls::PosInf, "6.1 +Inf x finite" },
        { Cls::NegFinite, Cls::PosInf, Cls::NegInf, "6.3 sign of the product" },
        { Cls::PosFinite, Cls::NegInf, Cls::NegInf, "6.3 sign of the product" },
        { Cls::NegFinite, Cls::NegInf, Cls::PosInf, "6.3 sign of the product" },
        { Cls::PosInf, Cls::PosInf, Cls::PosInf, "6.1 Inf x Inf" },
        { Cls::PosInf, Cls::NegInf, Cls::NegInf, "6.1/6.3 Inf x Inf" },
        { Cls::NegInf, Cls::NegInf, Cls::PosInf, "6.1/6.3 Inf x Inf" },
        { Cls::PosZero, Cls::PosInf, Cls::NaN, "7.2 0 x Inf is invalid" },
        { Cls::NegZero, Cls::PosInf, Cls::NaN, "7.2 0 x Inf is invalid" },
        { Cls::PosZero, Cls::NegInf, Cls::NaN, "7.2 0 x Inf is invalid" },
        { Cls::PosInf, Cls::PosZero, Cls::NaN, "7.2 0 x Inf is invalid" },
    };
    return t;
}

inline const std::vector<Rule>& kAddTable()
{
    static const std::vector<Rule> t{
        { Cls::PosInf, Cls::PosFinite, Cls::PosInf, "6.1 Inf + finite" },
        { Cls::PosFinite, Cls::PosInf, Cls::PosInf, "6.1 finite + Inf" },
        { Cls::NegInf, Cls::PosFinite, Cls::NegInf, "6.1 -Inf + finite" },
        { Cls::PosFinite, Cls::NegInf, Cls::NegInf, "6.1 finite + -Inf" },
        { Cls::PosInf, Cls::PosInf, Cls::PosInf, "6.1 Inf + Inf" },
        { Cls::NegInf, Cls::NegInf, Cls::NegInf, "6.1 -Inf + -Inf" },
        { Cls::PosInf, Cls::NegInf, Cls::NaN, "7.2 Inf + -Inf is invalid" },
        { Cls::NegInf, Cls::PosInf, Cls::NaN, "7.2 -Inf + Inf is invalid" },
        { Cls::PosInf, Cls::PosZero, Cls::PosInf, "6.1 Inf + 0" },
    };
    return t;
}

inline const std::vector<Rule>& kDivTable()
{
    static const std::vector<Rule> t{
        { Cls::PosFinite, Cls::PosZero, Cls::PosInf, "7.3 finite / +0" },
        { Cls::PosFinite, Cls::NegZero, Cls::NegInf, "7.3/6.3 finite / -0" },
        { Cls::NegFinite, Cls::PosZero, Cls::NegInf, "7.3/6.3 finite / +0" },
        { Cls::NegFinite, Cls::NegZero, Cls::PosInf, "7.3/6.3 finite / -0" },
        { Cls::PosZero, Cls::PosZero, Cls::NaN, "7.2 0/0 is invalid" },
        { Cls::NegZero, Cls::PosZero, Cls::NaN, "7.2 0/0 is invalid" },
        { Cls::PosFinite, Cls::PosInf, Cls::PosZero, "6.1 finite / Inf" },
        { Cls::PosFinite, Cls::NegInf, Cls::NegZero, "6.1/6.3 finite / -Inf" },
        { Cls::NegFinite, Cls::PosInf, Cls::NegZero, "6.1/6.3 finite / Inf" },
        { Cls::PosInf, Cls::PosFinite, Cls::PosInf, "6.1 Inf / finite" },
        { Cls::PosInf, Cls::NegFinite, Cls::NegInf, "6.1/6.3 Inf / finite" },
        { Cls::PosInf, Cls::PosZero, Cls::PosInf, "6.1 Inf / 0" },
        { Cls::PosInf, Cls::PosInf, Cls::NaN, "7.2 Inf/Inf is invalid" },
        { Cls::NegInf, Cls::PosInf, Cls::NaN, "7.2 Inf/Inf is invalid" },
    };
    return t;
}

/// @brief Pick one enumerated representative of a class.
inline double representative(Cls c)
{
    switch (c) {
        case Cls::NegInf: return negInf();
        case Cls::PosInf: return posInf();
        case Cls::NegZero: return -0.0;
        case Cls::PosZero: return 0.0;
        case Cls::PosFinite: return 0.9;
        case Cls::NegFinite: return -2.0;
        case Cls::NaN: return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.0;
}

// =========================================================================
//  Fixture
// =========================================================================

class BandSpecialsCert : public ::testing::Test {
};

// -------------------------------------------------------------------------
//  1. The laundering multiply.
// -------------------------------------------------------------------------
TEST_F(BandSpecialsCert, MultiplyPreservesInfinitiesWithTheCorrectSign)
{
    for (const Rule& r : kMulTable()) {
        const Band a   = bandOf(representative(r.a));
        const Band b   = bandOf(representative(r.b));
        const Band got = bd::mul(a, b);
        EXPECT_EQ(static_cast<int>(classifyBand(got)), static_cast<int>(r.want))
            << "mul(" << clsName(r.a) << ", " << clsName(r.b) << ") = "
            << clsName(classifyBand(got)) << ", IEEE 754-2019 " << r.cite
            << " requires " << clsName(r.want);
    }
}

// -------------------------------------------------------------------------
//  2. The release misencode -- a special must survive the cell8 codec.
// -------------------------------------------------------------------------
TEST_F(BandSpecialsCert, Cell8IngestDoesNotMisencodeInfinitiesOrNans)
{
    struct Row {
        const char* name;
        double x;
        Cls want;
    };
    const Row rows[] = {
        { "+Inf", posInf(), Cls::PosInf },
        { "-Inf", negInf(), Cls::NegInf },
        { "qNaN", std::numeric_limits<double>::quiet_NaN(), Cls::NaN },
    };
    for (const Row& row : rows) {
        for (const bool useEscape : { false, true }) {
            const std::uint64_t u = bits64(row.x);
            const BandCell8 w
                = bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
                    static_cast<std::uint32_t>(u >> 32), 0, useEscape);
            const Band back = bd::bandFromCell8(w);
            EXPECT_EQ(
                static_cast<int>(classifyBand(back)), static_cast<int>(row.want))
                << "cell8FromIEEE(" << row.name
                << ", useEscape=" << (useEscape ? "true" : "false")
                << ") -> 0x" << std::hex << static_cast<unsigned long long>(w.w)
                << std::dec << " decoded as " << clsName(classifyBand(back));
        }
    }
}

// -------------------------------------------------------------------------
//  3. The `0.9 * pow(0, -1/9)` chain.
// -------------------------------------------------------------------------
/**
 * Only the multiply half of the chain is exercised: `aether::math::pow`
 * routes to `BandedFacade<T>::pow`, but the body is not implemented yet, so
 * `aether::banded::detail::pow` does not exist to call directly.
 */
TEST_F(BandSpecialsCert, TheS3PowChainMultipliesToAPositiveInfinity)
{
    const double p = std::pow(0.0, -1.0 / 9.0);
    ASSERT_EQ(static_cast<int>(classify(p)), static_cast<int>(Cls::PosInf))
        << "the reproduction's premise is that IEEE pow(0, -1/9) is +Inf";

    const Band prod = bd::mul(bandOf(0.9), bandOf(p));
    EXPECT_EQ(static_cast<int>(classifyBand(prod)), static_cast<int>(Cls::PosInf))
        << "0.9 * pow(0, -1/9) must be +Inf in the carrier";
    EXPECT_EQ(static_cast<int>(classify(doubleOf(prod))),
        static_cast<int>(Cls::PosInf))
        << "and the store terminal must deliver it, not launder it into a "
           "finite value";
    // The `bd::pow(0, -1/9)` half is not covered here: banded pow's body is
    // not implemented yet (facade routing only).
}

// -------------------------------------------------------------------------
//  4. The rest of the IEEE-754 arithmetic table.
// -------------------------------------------------------------------------
TEST_F(BandSpecialsCert, AddAndSubtractFollowTheIeeeSpecialTable)
{
    for (const Rule& r : kAddTable()) {
        const Band a = bandOf(representative(r.a));
        const Band b = bandOf(representative(r.b));
        EXPECT_EQ(static_cast<int>(classifyBand(bd::add(a, b))),
            static_cast<int>(r.want))
            << "add(" << clsName(r.a) << ", " << clsName(r.b)
            << ") violates IEEE 754-2019 " << r.cite;
        const Band nb = bandOf(-representative(r.b));
        EXPECT_EQ(static_cast<int>(classifyBand(bd::sub(a, nb))),
            static_cast<int>(r.want))
            << "sub(" << clsName(r.a) << ", -(" << clsName(r.b)
            << ")) must equal add(" << clsName(r.a) << ", " << clsName(r.b)
            << "), IEEE 754-2019 " << r.cite;
    }
    EXPECT_EQ(static_cast<int>(classifyBand(bd::sub(bandOf(posInf()), bandOf(posInf())))),
        static_cast<int>(Cls::NaN))
        << "Inf - Inf is the invalid operation of clause 7.2";
}

/// Does not cover the trailing `sqrt_` special-value rows (needs
/// `RsqrtCore`, not yet implemented); the div/recip table below is
/// complete.
TEST_F(BandSpecialsCert, DivideAndReciprocalFollowTheIeeeSpecialTable)
{
    for (const Rule& r : kDivTable()) {
        const Band a = bandOf(representative(r.a));
        const Band b = bandOf(representative(r.b));
        EXPECT_EQ(static_cast<int>(classifyBand(bd::div(a, b))),
            static_cast<int>(r.want))
            << "div(" << clsName(r.a) << ", " << clsName(r.b)
            << ") violates IEEE 754-2019 " << r.cite;
    }
    struct R {
        const char* name;
        double b;
        Cls want;
    };
    const R rows[] = {
        { "1/+0", 0.0, Cls::PosInf },
        { "1/-0", -0.0, Cls::NegInf },
        { "1/+Inf", posInf(), Cls::PosZero },
        { "1/-Inf", negInf(), Cls::NegZero },
        { "1/NaN", std::numeric_limits<double>::quiet_NaN(), Cls::NaN },
    };
    for (const R& row : rows)
        EXPECT_EQ(static_cast<int>(classifyBand(bd::recip(bandOf(row.b)))),
            static_cast<int>(row.want))
            << "recip: " << row.name << " must be " << clsName(row.want);
    // sqrt_'s clause-5.4.1 special rows are not covered here (bd::sqrt_
    // needs RsqrtCore, not yet implemented).
}

// -------------------------------------------------------------------------
//  5. NaN propagation, over EVERY enumerated payload variant and every op.
// -------------------------------------------------------------------------
/// Does not cover mul3/sumSq3 (composite primitives) or sqrt_ (needs
/// RsqrtCore); the 9 ops in the case array below are.
TEST_F(BandSpecialsCert, NaNPropagatesThroughEveryGatedOpAndEveryPayloadVariant)
{
    const Band finite = bandOf(0.9);
    std::uint64_t rows = 0;
    for (const Operand& o : nanPayloadVariants()) {
        const Band n = bandOf(o.value);
        ASSERT_EQ(static_cast<int>(classifyBand(n)), static_cast<int>(Cls::NaN))
            << "the ingest lost the NaN-ness of '" << o.name << "'";
        struct Case {
            const char* name;
            Band got;
        };
        const Case cases[] = {
            { "mul(NaN, x)", bd::mul(n, finite) },
            { "mul(x, NaN)", bd::mul(finite, n) },
            { "add(NaN, x)", bd::add(n, finite) },
            { "sub(x, NaN)", bd::sub(finite, n) },
            { "div(NaN, x)", bd::div(n, finite) },
            { "div(x, NaN)", bd::div(finite, n) },
            { "recip(NaN)", bd::recip(n) },
            { "neg(NaN)", bd::neg(n) },
            { "abs(NaN)", bd::abs(n) },
        };
        for (const Case& c : cases) {
            rows++;
            EXPECT_EQ(static_cast<int>(classifyBand(c.got)),
                static_cast<int>(Cls::NaN))
                << c.name << " with payload '" << o.name
                << "' did not propagate the NaN (IEEE 754-2019 clause 6.2)";
        }
    }
    // 9 cases x 8 payload variants = 72 rows.
    EXPECT_GT(rows, 50u) << "the payload x op enumeration collapsed";
    std::printf("[BandSpecials] NaN propagation: %llu enumerated rows over %zu "
                "payload variants\n",
        static_cast<unsigned long long>(rows), nanPayloadVariants().size());
}

// -------------------------------------------------------------------------
//  6. The carrier a special comes back in is CANONICAL.
// -------------------------------------------------------------------------
/// Does not cover fma/mul3/sumSq3/exactSum3/exactSum4 (composite
/// primitives) or sqrt (needs RsqrtCore); the 9 ops swept below are.
TEST_F(BandSpecialsCert, EveryGatedOpReturnsACanonicalSpecialCarrier)
{
    const double specials[] = { posInf(), negInf(),
        std::numeric_limits<double>::quiet_NaN(), 0.0, -0.0 };
    const double others[] = { 0.9, -2.0, 0.0, -0.0, posInf(), negInf(),
        std::numeric_limits<double>::quiet_NaN(), 0x1p-94, 0x1p125 };

    std::uint64_t rows = 0, dirty = 0;
    auto check = [&](const char* what, Band v) {
        rows++;
        if (!std::isfinite(v.hi) && !(v.lo == 0.0f && v.tail == 0.0f)) {
            dirty++;
            ADD_FAILURE() << what
                          << " returned a special with a CONTAMINATED lower "
                             "limb (lo=0x"
                          << std::hex << bits(v.lo) << ", tail=0x"
                          << bits(v.tail) << std::dec
                          << "). It reads correctly and poisons the next op.";
        }
        if (std::isnan(v.hi)) {
            EXPECT_EQ(bits(v.hi) & 0x7FFFFFFFu, 0x7FC00000u)
                << what << " returned a NaN leading limb with a payload -- "
                           "every NaN this family produces is the one "
                           "canonical quiet NaN";
        }
    };

    for (double s : specials) {
        for (double o : others) {
            const Band a = bandOf(s), b = bandOf(o);
            check("mul", bd::mul(a, b));
            check("mul(swapped)", bd::mul(b, a));
            check("add", bd::add(a, b));
            check("sub", bd::sub(a, b));
            check("div", bd::div(a, b));
            check("div(swapped)", bd::div(b, a));
        }
        check("recip", bd::recip(bandOf(s)));
        check("neg", bd::neg(bandOf(s)));
        check("abs", bd::abs(bandOf(s)));
    }
    EXPECT_EQ(dirty, 0u);
    // 5 specials x 9 others x 6 + 5 x 3 = 285 rows.
    EXPECT_GT(rows, 200u) << "the operand cross-product collapsed";
    std::printf("[BandSpecials] canonical-carrier sweep: %llu rows, %llu "
                "contaminated\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(dirty));
}

// -------------------------------------------------------------------------
//  7. Ingest and egress totality.
// -------------------------------------------------------------------------
TEST_F(BandSpecialsCert, IngestCarriesEverySpecialAndTheSignOfAZero)
{
    for (const Operand& o : operands()) {
        const Band b = bandOf(o.value);
        EXPECT_EQ(static_cast<int>(classifyBand(b)), static_cast<int>(o.cls))
            << "bandFromIEEE lost the class of '" << o.name << "'";
    }
    EXPECT_EQ(static_cast<int>(classifyBand(bandOf(fromBits64(0x8000000000000001ull)))),
        static_cast<int>(Cls::NegZero))
        << "a negative double subnormal must flush to -0, not to +0";
    EXPECT_EQ(static_cast<int>(classifyBand(bandOf(fromBits64(0x0000000000000001ull)))),
        static_cast<int>(Cls::PosZero));
}

TEST_F(BandSpecialsCert, EgressDeliversInfinitiesNansAndSignedZerosFaithfully)
{
    for (const Operand& o : operands()) {
        const double got = doubleOf(bandOf(o.value));
        EXPECT_EQ(static_cast<int>(classify(got)), static_cast<int>(o.cls))
            << "bandToIEEE laundered '" << o.name << "' into class '"
            << clsName(classify(got)) << "'";
    }
    EXPECT_EQ(static_cast<int>(classify(doubleOf(bd::mul(bandOf(0.9), bandOf(posInf()))))),
        static_cast<int>(Cls::PosInf));
    EXPECT_EQ(static_cast<int>(classify(doubleOf(bd::mul(bandOf(-2.0), bandOf(posInf()))))),
        static_cast<int>(Cls::NegInf));
    const double n = doubleOf(bandOf(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(std::isnan(n));
    EXPECT_EQ(bits64(n), 0x7FF8000000000000ull)
        << "the store terminal must deliver ONE canonical quiet NaN";
}

// =========================================================================
//  The cell8 specials encoding
// =========================================================================

TEST_F(BandSpecialsCert, ReservedCodePointsAreUnreachableByAnyFiniteDouble)
{
    static const std::uint64_t kMant[6] = {
        0x0000000000000ull,
        0xFFFFFFFFFFFFFull,
        0x8000000000000ull, 0x0000000000001ull, 0x7FFFFFFFFFFFFull,
        0xA5A5A5A5A5A5Aull,
    };
    int maxExtUp = -1, maxExtDown = -1;
    std::uint64_t rows = 0, reserved = 0;
    for (int bexp = 1; bexp <= 0x7FE; bexp++) {
        for (std::uint64_t m : kMant) {
            for (std::uint32_t sign : { 0u, 0x80000000u }) {
                const std::uint64_t u = (static_cast<std::uint64_t>(sign) << 32)
                    | (static_cast<std::uint64_t>(bexp) << 52) | m;
                const BandCell8 w = bd::cell8FromIEEE(
                    static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32), 0,
                    true);
                rows++;
                if (bd::cell8IsSpecial(w)) {
                    reserved++;
                    ADD_FAILURE()
                        << "the finite double 0x" << std::hex << u
                        << " encoded INTO the reserved special region (word 0x"
                        << static_cast<unsigned long long>(w.w) << std::dec
                        << ") -- the code points are not reserved after all";
                }
                if (bd::cell8IsEscape(w) && w.w != 0u) {
                    const std::uint32_t w1 = static_cast<std::uint32_t>(w.w >> 32);
                    const int ext = static_cast<int>((w1 >> 12) & 0x7FFu);
                    if (((w1 >> 23) & 0xFFu) == 0xFFu)
                        maxExtUp = (ext > maxExtUp) ? ext : maxExtUp;
                    else
                        maxExtDown = (ext > maxExtDown) ? ext : maxExtDown;
                }
            }
        }
    }
    std::printf("[BandSpecials] reserved code points: %llu enumerated finite "
                "doubles, %llu landed in the region; largest ext written -- "
                "upward %d, downward %d; reserved ext is %d\n",
        static_cast<unsigned long long>(rows),
        static_cast<unsigned long long>(reserved), maxExtUp, maxExtDown,
        bd::kBandCell8SpecialExt);
    EXPECT_EQ(reserved, 0u);
    EXPECT_LE(maxExtUp, bd::kBandCell8MaxIngestExp - bd::kBandCell8EscapeUpBase)
        << "an ingest wrote a larger extended exponent than the derivation "
           "says it can -- the reserved point's clearance is computed from "
           "that bound, so it moves with it";
    EXPECT_LT(maxExtUp, bd::kBandCell8SpecialExt);
    EXPECT_GT(maxExtUp, 0) << "no row exercised the upward escape tag at all";
}

/// Does not cover item (d), the 16-byte `BandCell` sibling check
/// (`bd::cellFromCell8`): `BandCell` has no `aether` counterpart (see
/// `aether/banded/BandCell8.h`). Items (a)-(c) and the intercept-ordering
/// loop below are covered in full.
TEST_F(BandSpecialsCert, Cell8SpecialsRoundTripThroughEveryCodecPath)
{
    struct S {
        const char* name;
        double x;
        std::uint64_t word;
        Cls cls;
    };
    const S rows[] = {
        { "+Inf", posInf(), bd::kBandCell8PosInfWord, Cls::PosInf },
        { "-Inf", negInf(), bd::kBandCell8NegInfWord, Cls::NegInf },
        { "NaN", std::numeric_limits<double>::quiet_NaN(),
            bd::kBandCell8NanWord, Cls::NaN },
    };
    for (const S& s : rows) {
        const std::uint64_t u = bits64(s.x);
        // (a) the IEEE-words ingest terminal, both escape settings.
        for (const bool useEscape : { false, true }) {
            const BandCell8 w = bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
                static_cast<std::uint32_t>(u >> 32), 0, useEscape);
            EXPECT_EQ(w.w, s.word) << s.name << ": wrong code point";
            EXPECT_TRUE(bd::cell8IsSpecial(w));
            EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8(w))),
                static_cast<int>(s.cls));
            EXPECT_EQ(static_cast<int>(classify(bd::floatFromCell8(w))),
                static_cast<int>(s.cls))
                << s.name << ": the FP32 tier disagrees with the full decode";
            EXPECT_EQ(static_cast<int>(classify(
                          bd::doubleFromCell8(w, 0, useEscape))),
                static_cast<int>(s.cls))
                << s.name << ": the host egress terminal lost it";
        }
        // (b) the carrier encode terminal.
        const BandCell8 fromBand = bd::cell8FromBand(bandOf(s.x));
        EXPECT_EQ(fromBand.w, s.word)
            << s.name << ": cell8FromBand chose a different code point than "
                         "cell8FromIEEE -- one format, two encoders";
        EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8(fromBand))),
            static_cast<int>(s.cls));
        // (c) the `BandCell8::band()` member and the checked decode.
        EXPECT_EQ(static_cast<int>(classifyBand(fromBand.band())),
            static_cast<int>(s.cls));
        bool escaped = false;
        EXPECT_EQ(static_cast<int>(classifyBand(
                      bd::bandFromCell8Checked(fromBand, escaped))),
            static_cast<int>(s.cls));
        EXPECT_TRUE(escaped) << "a special word shares the escape tag byte, so "
                                "it must answer TRUE to cell8IsEscape -- that "
                                "is the disambiguation cell8IsSpecial exists "
                                "for, and a reader must test it FIRST";
        // Item (d), the 16-byte BandCell sibling, is not covered (see above).
    }
    EXPECT_EQ(bd::cell8FromDouble(posInf(), 0, false).w, bd::kBandCell8PosInfWord);
    EXPECT_EQ(bd::cell8FromDouble(negInf(), 0, true).w, bd::kBandCell8NegInfWord);

    for (const bool useEscape : { false, true }) {
        for (double x : { posInf(), negInf() }) {
            const std::uint64_t u = bits64(x);
            EXPECT_TRUE(bd::cell8IsSpecial(bd::cell8FromIEEE(
                static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32), 0,
                useEscape)))
                << "an infinity did not take the specials intercept at "
                   "useEscape=" << useEscape;
        }
        for (const Operand& o : nanPayloadVariants()) {
            const std::uint64_t u = bits64(o.value);
            EXPECT_EQ(bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
                          static_cast<std::uint32_t>(u >> 32), 0, useEscape)
                          .w,
                bd::kBandCell8NanWord)
                << "NaN variant '" << o.name
                << "' did not take the specials intercept at useEscape="
                << useEscape;
        }
    }
}

TEST_F(BandSpecialsCert, Cell8SpecialsSurviveEveryRebiasTheTagSpaceAdmits)
{
    for (int bias = bd::kBandCell8RebiasLowSafe;
         bias <= bd::kBandCell8RebiasHighSafe; bias++) {
        const BandCell8 w = bd::cell8FromBandRebias(bandOf(posInf()), bias);
        EXPECT_EQ(w.w, bd::kBandCell8PosInfWord)
            << "a special word was rebiased at arrayBias " << bias;
        EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8Rebias(w, bias))),
            static_cast<int>(Cls::PosInf))
            << "the special did not survive the rebiased decode at " << bias;
    }
    const int unsafeBias = bd::kBandCell8RebiasHighSafe + 1;
    const Band finite    = bandOf(0x1p-4);
    const BandCell8 fw   = bd::cell8FromBandRebias(finite, unsafeBias);
    const Band back      = bd::bandFromCell8Rebias(fw, unsafeBias);
    EXPECT_EQ(bits(back.hi), bits(finite.hi))
        << "outside the safe rebias range a finite value must still round-trip "
           "bit-exactly";
}

TEST_F(BandSpecialsCert, EveryNaNPayloadVariantEncodesToTheOneCanonicalCodePoint)
{
    for (const Operand& o : nanPayloadVariants()) {
        const std::uint64_t u = bits64(o.value);
        const BandCell8 w = bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
            static_cast<std::uint32_t>(u >> 32), 0, true);
        EXPECT_EQ(w.w, bd::kBandCell8NanWord)
            << "NaN variant '" << o.name << "' encoded to 0x" << std::hex
            << static_cast<unsigned long long>(w.w) << std::dec
            << " rather than the one canonical code point";
        EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8(w))),
            static_cast<int>(Cls::NaN));
    }
    for (std::uint32_t payload : { 0x001u, 0x800u, 0xFFFu }) {
        const BandCell8 w{ static_cast<std::uint64_t>(
                               bd::kBandCell8SpecialTag | payload)
            << 32 };
        EXPECT_TRUE(bd::cell8IsSpecial(w));
        EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8(w))),
            static_cast<int>(Cls::NaN))
            << "reserved word with significand 0x" << std::hex << payload
            << std::dec << " must decode as a NaN";
    }
    const BandCell8 lowOnly{ (static_cast<std::uint64_t>(bd::kBandCell8SpecialTag)
                                 << 32)
        | 1u };
    EXPECT_EQ(static_cast<int>(classifyBand(bd::bandFromCell8(lowOnly))),
        static_cast<int>(Cls::NaN));
}

TEST_F(BandSpecialsCert, Cell8IsSpecialSeparatesSpecialsFromEveryFiniteWord)
{
    std::uint64_t tier1 = 0, escapes = 0, specials = 0;
    for (int e = bd::kBandCell8FloorExp; e <= bd::kBandCell8CeilingExp; e++) {
        for (double m : { 1.0, 1.5, 1.9999999 }) {
            for (double sg : { 1.0, -1.0 }) {
                const BandCell8 w = bd::cell8FromBand(bandOf(sg * m * std::ldexp(1.0, e)));
                tier1++;
                ASSERT_FALSE(bd::cell8IsSpecial(w))
                    << "a tier-1 word at 2^" << e << " answered cell8IsSpecial";
            }
        }
    }
    for (int e = -1000; e <= 1000; e += 7) {
        if (e >= bd::kBandCell8FloorExp && e <= bd::kBandCell8CeilingExp)
            continue;
        for (double sg : { 1.0, -1.0 }) {
            const double x  = sg * 1.5 * std::ldexp(1.0, e);
            const std::uint64_t u = bits64(x);
            const BandCell8 w = bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
                static_cast<std::uint32_t>(u >> 32), 0, true);
            escapes++;
            ASSERT_TRUE(bd::cell8IsEscape(w));
            ASSERT_FALSE(bd::cell8IsSpecial(w))
                << "a finite escape word at 2^" << e << " answered "
                                                        "cell8IsSpecial";
        }
    }
    for (std::uint64_t w : { bd::kBandCell8PosInfWord, bd::kBandCell8NegInfWord,
             bd::kBandCell8NanWord }) {
        specials++;
        EXPECT_TRUE(bd::cell8IsSpecial(BandCell8{ w }));
        EXPECT_TRUE(bd::cell8IsEscape(BandCell8{ w }))
            << "specials live in the escape tag space by construction";
    }
    EXPECT_FALSE(bd::cell8IsSpecial(BandCell8{ 0 }));
    EXPECT_TRUE(bd::cell8IsEscape(BandCell8{ 0 }));
    std::printf("[BandSpecials] cell8IsSpecial separation: %llu tier-1 words, "
                "%llu finite escape words, %llu reserved words\n",
        static_cast<unsigned long long>(tier1),
        static_cast<unsigned long long>(escapes),
        static_cast<unsigned long long>(specials));
    EXPECT_GT(tier1, 400u);
    EXPECT_GT(escapes, 100u);
}

/// Does not cover the trailing `sqrt_` exactness row (needs RsqrtCore, not
/// yet implemented); the mul/add/sub/div sweep and exact-value checks below
/// are complete.
TEST_F(BandSpecialsCert, FiniteOperandsAreNeverDivertedIntoTheSpecialsPath)
{
    static const double kFinite[] = { 0.9, -2.0, 1.0, 3.0, 0x1p-94, -0x1p-94,
        0x1p125, -0x1p125, 0x1p-126, 0x1p-40, -0x1p60 };
    std::uint64_t rows = 0;
    for (double a : kFinite) {
        for (double b : kFinite) {
            const Band x = bandOf(a), y = bandOf(b);
            rows++;
            EXPECT_FALSE(std::isnan(bd::mul(x, y).hi))
                << "mul(" << a << ", " << b << ") produced a NaN";
            EXPECT_FALSE(std::isnan(bd::add(x, y).hi));
            EXPECT_FALSE(std::isnan(bd::sub(x, y).hi));
            EXPECT_FALSE(std::isnan(bd::div(x, y).hi))
                << "div(" << a << ", " << b << ") produced a NaN";
        }
        EXPECT_TRUE(std::isfinite(bd::recip(bandOf(a)).hi))
            << "recip(" << a << ") left the finite domain";
    }
    EXPECT_EQ(doubleOf(bd::mul(bandOf(3.0), bandOf(0.5))), 1.5);
    EXPECT_EQ(doubleOf(bd::add(bandOf(1.0), bandOf(2.0))), 3.0);
    EXPECT_EQ(doubleOf(bd::div(bandOf(1.0), bandOf(4.0))), 0.25);
    EXPECT_EQ(doubleOf(bd::recip(bandOf(8.0))), 0.125);
    // sqrt_ needs RsqrtCore (not yet implemented), so its exactness at 4.0
    // is not checked here.
    EXPECT_GT(rows, 100u);
}

// -------------------------------------------------------------------------
//  Falsifiability: the corpus catches deliberately broken mechanisms.
// -------------------------------------------------------------------------
namespace specialsdefect {

/// S1 -- `normalize` without the specials guard: its `mul` launders
/// `finite x Inf` into a NaN carrier.
inline Band normalizeNoGuard(bd::BandRaw r)
{
    float h1, l1;
    bd::fast2Sum(r.hi, r.lo, h1, l1);
    float lc, lb;
    bd::fast2Sum(l1, r.tail, lc, lb);
    float h, l;
    bd::fast2Sum(h1, lc, h, l);
    return Band{ h, l, lb };
}
inline Band mulNoGuard(Band a, Band b) { return normalizeNoGuard(bd::mulRaw(a, b)); }

/// S2 -- the specials ENCODE with the sign point swapped: `-Inf` is written
/// to the `+Inf` code point.
inline BandCell8 cell8SpecialFromIEEESignSwapped(std::uint32_t lo, std::uint32_t hi)
{
    const std::uint32_t mant_nz = (hi & 0x000FFFFFu) | lo;
    if (mant_nz != 0u)
        return BandCell8{ bd::kBandCell8NanWord };
    return BandCell8{ bd::kBandCell8PosInfWord }; // sign dropped
}

/// S3 -- specials DECODE turned off: the tier-1 reinterpretation.
inline Band bandFromCell8DecodeOff(BandCell8 c)
{
    const std::uint32_t w1 = static_cast<std::uint32_t>(c.w >> 32);
    const std::uint32_t w0 = static_cast<std::uint32_t>(c.w);
    const std::uint32_t se = w1 & 0xFF800000u;
    const std::uint32_t eL = se - (23u << 23);
    const std::uint32_t eT = se - (32u << 23);
    const std::uint32_t L  = w0 >> bd::kBandCell8TailBits;
    const std::uint32_t T  = w0 & ((1u << bd::kBandCell8TailBits) - 1u);
    Band b;
    b.hi   = fromBits(w1);
    b.lo   = fromBits(eL | L) - fromBits(eL);
    b.tail = fromBits(eT | T) - fromBits(eT);
    return b;
}

} // namespace specialsdefect

TEST_F(BandSpecialsCert, SeededDefectsAreCaughtByThisCorpus)
{
    std::uint64_t s1 = 0;
    for (const Rule& r : kMulTable()) {
        const Band got = specialsdefect::mulNoGuard(
            bandOf(representative(r.a)), bandOf(representative(r.b)));
        if (static_cast<int>(classifyBand(got)) != static_cast<int>(r.want))
            s1++;
    }
    EXPECT_GT(s1, 0u)
        << "S1 (mul with NO specials guard) satisfied the IEEE multiply table "
           "on every enumerated row -- then no row in this corpus can tell a "
           "guarded multiply from an unguarded one";

    std::uint64_t s2 = 0;
    for (double x : { posInf(), negInf() }) {
        const std::uint64_t u  = bits64(x);
        const BandCell8 w = specialsdefect::cell8SpecialFromIEEESignSwapped(
            static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32));
        const BandCell8 good = bd::cell8FromIEEE(static_cast<std::uint32_t>(u),
            static_cast<std::uint32_t>(u >> 32), 0, true);
        if (w.w != good.w
            || static_cast<int>(classifyBand(bd::bandFromCell8(w)))
                != static_cast<int>(classify(x)))
            s2++;
    }
    EXPECT_GT(s2, 0u)
        << "S2 (the -Inf code point written as +Inf) was indistinguishable "
           "from the shipped encoder";

    std::uint64_t s3 = 0;
    for (std::uint64_t w : { bd::kBandCell8PosInfWord, bd::kBandCell8NegInfWord,
             bd::kBandCell8NanWord }) {
        const Band bad  = specialsdefect::bandFromCell8DecodeOff(BandCell8{ w });
        const Band good = bd::bandFromCell8(BandCell8{ w });
        if (bits(bad.hi) != bits(good.hi))
            s3++;
    }
    EXPECT_GT(s3, 0u)
        << "S3 (specials decode turned off) returned the same carrier as the "
           "shipped decode on every reserved word";

    std::printf("[BandSpecials] seeded defects: S1 (no guard on mul) caught on "
                "%llu of %zu table rows; S2 (swapped Inf sign point) caught: "
                "%s; S3 (specials decode off) caught on %llu of 3 words\n",
        static_cast<unsigned long long>(s1), kMulTable().size(),
        s2 ? "yes" : "NO", static_cast<unsigned long long>(s3));
}

// Transcendental specials (exp/log/sin/cos/asin/acos/cbrt/rsqrt/rsqrtCube/
// pow) are not covered by this file; see disposition_manifest.tsv.

} // namespace BandSpecialsTest
} // namespace aether_tests
