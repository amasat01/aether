// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The CROSS-MODE anchor for the banded codec and its certified arithmetic
// core, HOST arm.
//
// WHAT THIS ROW IS FOR, AND WHY THE OTHER BATTERIES DO NOT COVER IT.
// `test_BandCell8_common.h` certifies the codec against an exact 128-bit
// reconstruction and an independent probe codec; `test_BandedReal_common.h`
// certifies the value type against the format's own constants. Both are strong,
// and neither can answer "does the DEVICE arm compute the same bits", because
// each arm can only ever ask its own compiler. The committed golden
// (`tests/banded/golden_band.h`, md5-fenced, minted by
// `tools/banded/mint_golden.sh`) turns that into a RUN: this file evaluates the
// rows on the HOST and `test_BandGolden.cu` evaluates the SAME rows inside a
// `__global__` kernel, both under the name `BandGoldenTest.CrossModeBitExact`.
//
// ★ It is also the only instrument that survives a change to BOTH arms. Two arms
// that move together are invisible to a host-versus-device comparison and loud
// here, because the committed numbers do not move when the code does.
//
// The comparison is EXACT — `EXPECT_EQ` on bit patterns, never a tolerance —
// and it is decidable by construction rather than by luck: every error-free
// transform under `aether/banded/detail/Fp32.h` pins its own rounding at both
// ends (inline PTX on the device, register barriers on the host), so the results
// are independent of `-fmad`/`-ffp-contract`.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "aether/banded/banded.h"

#include "tests/banded/golden_band.h"

namespace aether_tests {
namespace {

using aether::banded::Band;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;

std::uint32_t fbits(float f)
{
    std::uint32_t b = 0;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

double dfrom(std::uint64_t b)
{
    double x = 0.0;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

/** @brief Compare one carrier against a golden `{hi, lo, tail}` triple. */
void expectLimbs(const char* what, std::size_t row, Band got, const std::uint32_t (&want)[3])
{
    EXPECT_EQ(fbits(got.hi), want[0]) << what << " hi, row " << row;
    EXPECT_EQ(fbits(got.lo), want[1]) << what << " lo, row " << row;
    EXPECT_EQ(fbits(got.tail), want[2]) << what << " tail, row " << row;
}

} // namespace

TEST(BandGoldenTest, CrossModeBitExact)
{
    ASSERT_EQ(std::size(golden::kBandCodecRows), golden::kBandCodecRowCount);
    ASSERT_EQ(std::size(golden::kBandChainRows), golden::kBandChainRowCount);
    ASSERT_GT(golden::kBandCodecRowCount, 100u)
        << "the golden codec corpus collapsed — a shrunken corpus certifies "
           "whatever is left, which is the failure mode a committed golden "
           "exists to prevent";
    ASSERT_GT(golden::kBandChainRowCount, 100u);

    // --- CODEC: one double in, the stored word and the decoded limbs out. ---
    for (std::size_t i = 0; i < golden::kBandCodecRowCount; ++i) {
        const golden::BandCodecRow& r = golden::kBandCodecRows[i];
        const BandedReal enc          = BandedReal::fromDouble(dfrom(r.inBits));
        EXPECT_EQ(enc.toBits(), r.word) << "codec word, row " << i;
        const Band dec = static_cast<Band>(enc);
        EXPECT_EQ(fbits(dec.hi), r.hi) << "decoded hi, row " << i;
        EXPECT_EQ(fbits(dec.lo), r.lo) << "decoded lo, row " << i;
        EXPECT_EQ(fbits(dec.tail), r.tail) << "decoded tail, row " << i;
    }

    // --- ARITHMETIC: two stored words in, six results out, as raw limbs. ---
    for (std::size_t i = 0; i < golden::kBandChainRowCount; ++i) {
        const golden::BandChainRow& r = golden::kBandChainRows[i];
        const Band a = static_cast<Band>(BandedReal::fromBits(r.aWord));
        const Band b = static_cast<Band>(BandedReal::fromBits(r.bWord));
        expectLimbs("add", i, bd::add(a, b), r.add);
        expectLimbs("sub", i, bd::sub(a, b), r.sub);
        expectLimbs("mul", i, bd::mul(a, b), r.mul);
        expectLimbs("div", i, bd::div(a, b), r.div);
        expectLimbs("chain", i, bd::div(bd::mul(bd::sub(a, b), a), b), r.chain);
        expectLimbs("facade", i,
            bd::copysign(bd::fmax(bd::abs(a), bd::abs(b)), b), r.facade);
    }

    // NON-VACUITY: the corpus must actually contain the reserved code points and
    // the tier edges it claims to. A golden over an all-finite, all-ordinary
    // corpus would pass on a codec that had lost its specials handling entirely.
    std::size_t specials = 0, zeros = 0;
    for (std::size_t i = 0; i < golden::kBandCodecRowCount; ++i) {
        const BandedReal r = BandedReal::fromBits(golden::kBandCodecRows[i].word);
        if (r.isNan() || r.isInf())
            ++specials;
        if (r.isZero())
            ++zeros;
    }
    EXPECT_GE(specials, 6u) << "the golden codec corpus carries no reserved code "
                               "points, so nothing here certifies the specials path";
    EXPECT_GE(zeros, 4u) << "the golden codec corpus carries no zeros (both signs "
                            "and the subnormal flush)";
}

} // namespace aether_tests
