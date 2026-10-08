// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tests/test_BandBits.cpp — `Band::fromBits`/`Band::toBits` round-trip,
// bitwise. Paired with test_BandBits.cu (host row IDENTICAL; the .cu ALSO
// carries a device row — house convention, see tests/test_Item.cpp/.cu).
//
// Two identities, both bitwise (`fromBits`/`toBits` are a bare per-limb
// reinterpret, no rounding, no normalization, no admission check — so both
// hold BY CONSTRUCTION, and this test PINS that, it does not merely hope
// for it):
//   (1) toBits(fromBits(h, l, t)) == (h, l, t)      — starting from bits.
//   (2) fromBits(toBits(b)) == b, bitwise            — starting from a Band.
//
// Corpus: the special-value set — the SAME enumeration
// `tests/test_BandSpecials_common.h`'s `BandSpecialsTest::operands()` uses
// (infinities, signed zeros, tier-floor/ceiling finite representatives, the
// same 8 enumerated NaN payload variants), reproduced LOCALLY rather than
// `#include`-d from that file: it carries live `TEST_F` bodies
// (`BandSpecialsCert`), and `#include`-ing it from a second translation
// unit is an ODR violation (confirmed by the linker — every `test_info_`
// duplicate-defined) — plus 64 random finite triples, seeded via
// `aether::random` (`aether::random::Generator`) so the run is
// reproducible.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "aether/banded/banded.h"

namespace aether_tests {
namespace BandBitsTest {

using aether::banded::Band;
namespace bd = aether::banded::detail;

namespace {

double doubleFromBits64(std::uint64_t u)
{
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

std::uint64_t bits64OfDouble(double d)
{
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

/// @brief The ingest terminal, spelled the way a consumer reaches it —
/// mirrors `test_BandSpecials_common.h`'s own `bandOf()`.
Band bandOfDouble(double x)
{
    const std::uint64_t u = bits64OfDouble(x);
    return bd::bandFromIEEE(static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32));
}

struct Operand {
    const char* name;
    double value;
};

/// @brief The special-value corpus, aligned with
/// `test_BandSpecials_common.h`'s own `operands()` (same values, same
/// names) — see this file's header docstring for why it is reproduced here
/// rather than shared via `#include`.
std::vector<Operand> p11aSpecialsCorpus()
{
    std::vector<Operand> v{
        { "-Inf", -std::numeric_limits<double>::infinity() },
        { "+Inf", std::numeric_limits<double>::infinity() },
        { "-0", -0.0 },
        { "+0", 0.0 },
        { "+0.9", 0.9 },
        { "-2.0", -2.0 },
        { "+1.0", 1.0 },
        { "+2^-94 (tier floor)", 0x1p-94 },
        { "-2^-94 (tier floor)", -0x1p-94 },
        { "+2^125 (tier ceiling)", 0x1p125 },
        { "-2^125 (tier ceiling)", -0x1p125 },
        { "+2^-126 (fp32 subnormal edge)", 0x1p-126 },
    };
    auto addNaN = [&v](const char* n, std::uint64_t bits) { v.push_back({ n, doubleFromBits64(bits) }); };
    addNaN("qNaN canonical (+)", 0x7FF8000000000000ull);
    addNaN("qNaN canonical (-)", 0xFFF8000000000000ull);
    addNaN("qNaN min payload", 0x7FF8000000000001ull);
    addNaN("qNaN max payload", 0x7FFFFFFFFFFFFFFFull);
    addNaN("sNaN min payload", 0x7FF0000000000001ull);
    addNaN("sNaN max payload", 0x7FF7FFFFFFFFFFFFull);
    addNaN("sNaN max payload (-)", 0xFFF7FFFFFFFFFFFFull);
    addNaN("qNaN mixed payload", 0x7FFA5A5A5A5A5A5Aull);
    return v;
}

/** @brief Both round-trip identities for one `Band`, asserted bitwise. */
void expectRoundTripsBitwise(Band b, const char* label)
{
    std::uint32_t h = 0, l = 0, t = 0;
    b.toBits(h, l, t);

    const Band viaFromBits = Band::fromBits(h, l, t);

    // (1) toBits(fromBits(h,l,t)) == (h,l,t)
    std::uint32_t h2 = 0, l2 = 0, t2 = 0;
    viaFromBits.toBits(h2, l2, t2);
    EXPECT_EQ(h, h2) << label << ": toBits(fromBits(...)) hi diverged";
    EXPECT_EQ(l, l2) << label << ": toBits(fromBits(...)) lo diverged";
    EXPECT_EQ(t, t2) << label << ": toBits(fromBits(...)) tail diverged";

    // (2) fromBits(toBits(b)) == b, bitwise (never value-compared — a NaN
    // limb must compare EQUAL here even though NaN != NaN by value).
    EXPECT_EQ(std::bit_cast<std::uint32_t>(viaFromBits.hi), std::bit_cast<std::uint32_t>(b.hi))
        << label << ": fromBits(toBits(b)) hi diverged from b";
    EXPECT_EQ(std::bit_cast<std::uint32_t>(viaFromBits.lo), std::bit_cast<std::uint32_t>(b.lo))
        << label << ": fromBits(toBits(b)) lo diverged from b";
    EXPECT_EQ(std::bit_cast<std::uint32_t>(viaFromBits.tail), std::bit_cast<std::uint32_t>(b.tail))
        << label << ": fromBits(toBits(b)) tail diverged from b";
}

} // namespace

class BandBitsRoundTripTest : public ::testing::Test { };

TEST_F(BandBitsRoundTripTest, P11aSpecialsRoundTripBitwise)
{
    const std::vector<Operand> ops = p11aSpecialsCorpus();
    ASSERT_GT(ops.size(), 0u) << "the specials corpus collapsed";
    for (const Operand& op : ops)
        expectRoundTripsBitwise(bandOfDouble(op.value), op.name);
}

TEST_F(BandBitsRoundTripTest, SixtyFourRandomFiniteTriplesRoundTripBitwise)
{
    // Seeded via aether::random: reproducible across runs. Range is
    // wide but bounded — every draw stays finite by construction (a uniform
    // draw over a finite interval never produces +-Inf/NaN), so this is a
    // genuine "finite triples" corpus, not merely "usually finite".
    constexpr std::uint64_t kSeed = 0xBA4DB175ull; // 'BAND BITS', hex-ish mnemonic
    const aether::random::Generator rng(kSeed);
    const auto leaf = rng.uniform<float, 3>(-1.0e6f, 1.0e6f);

    for (std::size_t idx = 0; idx < 64; ++idx) {
        const aether::SampleIndex i = aether::SampleIndex::make(idx);
        const float hi = leaf.get<0>(i);
        const float lo = leaf.get<1>(i);
        const float tail = leaf.get<2>(i);
        ASSERT_TRUE(std::isfinite(hi) && std::isfinite(lo) && std::isfinite(tail))
            << "draw " << idx << " was not finite — corpus precondition violated";

        char label[64];
        std::snprintf(label, sizeof(label), "random triple #%zu", idx);
        expectRoundTripsBitwise(Band(hi, lo, tail), label);
    }
}

} // namespace BandBitsTest
} // namespace aether_tests
