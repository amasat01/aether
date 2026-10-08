// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tests/test_BandBits.cu — `Band::fromBits`/`Band::toBits` round-trip,
// bitwise. Host rows IDENTICAL to test_BandBits.cpp (house convention, see
// tests/test_Item.cpp/.cu); this file ADDS a device row — the SAME two
// identities, exercised through an actual on-device `fromBits`/`toBits`
// call rather than the host-compiled branch.
//
// Corpus: the special-value set, reproduced LOCALLY (not `#include`-d from
// tests/test_BandSpecials_common.h — that file carries live `TEST_F`
// bodies and a second inclusion is an ODR violation; see
// test_BandBits.cpp's own header docstring for the full note) plus 64
// random finite triples, seeded via `aether::random`.

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

// ---------------------------------------------------------------------
// Device row: the SAME fromBits/toBits round trip,
// evaluated ON DEVICE via an actual kernel — pins that the `__CUDA_ARCH__`
// branch (`__float_as_uint`/`__uint_as_float`) round-trips identically to
// the host `std::bit_cast` branch above.
// ---------------------------------------------------------------------

#ifndef AETHER_CPP_MODE

using ScalarU32Strided = aether::View<std::uint32_t, aether::extents<aether::dyn>, aether::layout_stride>;

AETHER_KERNEL()
void bandBitsDeviceRoundTripKernel(
    ScalarU32Strided hiIn, ScalarU32Strided loIn, ScalarU32Strided tailIn,
    ScalarU32Strided hiOut, ScalarU32Strided loOut, ScalarU32Strided tailOut)
{
    const aether::SampleIndex i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= hiIn.samples())
        return;
    const Band b = Band::fromBits(hiIn(i.global()), loIn(i.global()), tailIn(i.global()));
    std::uint32_t h = 0, l = 0, t = 0;
    b.toBits(h, l, t);
    hiOut(i.global()) = h;
    loOut(i.global()) = l;
    tailOut(i.global()) = t;
}

TEST_F(BandBitsRoundTripTest, DeviceRoundTripMatchesHostBitwise)
{
    const std::vector<Operand> ops = p11aSpecialsCorpus();
    const std::size_t n = ops.size();
    ASSERT_GT(n, 0u) << "the specials corpus collapsed";

    aether::Array<std::uint32_t> hi(n), lo(n), tail(n);
    aether::Array<std::uint32_t> hiOut(n), loOut(n), tailOut(n);

    auto hv = hi.hostView();
    auto lv = lo.hostView();
    auto tv = tail.hostView();
    for (std::size_t idx = 0; idx < n; ++idx) {
        std::uint32_t h = 0, l = 0, t = 0;
        bandOfDouble(ops[idx].value).toBits(h, l, t);
        hv(idx) = h;
        lv(idx) = l;
        tv(idx) = t;
    }

    hi.upload();
    lo.upload();
    tail.upload();

    const int threads = 128;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));
    bandBitsDeviceRoundTripKernel<<<blocks, threads>>>(
        hi.deviceView(), lo.deviceView(), tail.deviceView(), hiOut.deviceView(), loOut.deviceView(), tailOut.deviceView());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "bandBitsDeviceRoundTripKernel launch failed";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "bandBitsDeviceRoundTripKernel execution failed";

    hiOut.download();
    loOut.download();
    tailOut.download();
    auto hov = hiOut.hostView();
    auto lov = loOut.hostView();
    auto tov = tailOut.hostView();

    for (std::size_t idx = 0; idx < n; ++idx) {
        EXPECT_EQ(hov(idx), hv(idx)) << ops[idx].name << ": device fromBits/toBits hi diverged from host";
        EXPECT_EQ(lov(idx), lv(idx)) << ops[idx].name << ": device fromBits/toBits lo diverged from host";
        EXPECT_EQ(tov(idx), tv(idx)) << ops[idx].name << ": device fromBits/toBits tail diverged from host";
    }
}

#endif // !AETHER_CPP_MODE

} // namespace BandBitsTest
} // namespace aether_tests
