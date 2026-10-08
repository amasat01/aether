// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The CROSS-MODE anchor for the banded codec and its certified arithmetic
// core, DEVICE arm.
//
// Registers the SAME name as `test_BandGolden.cpp` — `BandGoldenTest.
// CrossModeBitExact` — and that is the whole point: the host arm evaluates the
// committed rows on the host, this one evaluates them INSIDE A KERNEL, and both
// compare against the same md5-fenced numbers in `tests/banded/golden_band.h`.
// Cross-mode bit-identity is therefore a RUN against a fixed statement, not two
// runs against each other (two arms that move together are invisible to a
// host-versus-device comparison and loud against a golden).
//
// ★ THE INGEST RUNS ON THE DEVICE TOO. `detail::cell8FromIEEE` takes the two
// 32-bit HALVES of a `double` rather than a `double`, precisely so the encode
// has one body for both arms — no `double` ever appears in device code (the
// 0-FP64 portability warrant). So the kernel below reproduces the whole journey:
// raw IEEE halves -> stored word -> decoded limbs, and then the six arithmetic
// results from the stored operand words.
//
// The comparison is EXACT — `EXPECT_EQ` on bit patterns — and decidable by
// construction: every error-free transform under `aether/banded/detail/Fp32.h`
// pins its own rounding with inline PTX on this arm, so nothing here depends on
// `-fmad`.
//
// GPU EXECUTION NOTE: requires a GPU to run (house convention: use the pinned
// device).

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "aether/backend/cuda/Launch.h"
#include "aether/banded/banded.h"

#include "tests/banded/golden_band.h"

namespace aether_tests {
namespace {

using aether::banded::Band;
using aether::banded::BandCell8;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;

constexpr int kBlock = 128;

/** @brief What one device thread reports for one codec row. */
struct CodecOut {
    std::uint64_t word;
    std::uint32_t hi, lo, tail;
};

/** @brief What one device thread reports for one arithmetic row: six carriers,
 *         three limbs each, in the golden's own field order. */
struct ChainOut {
    std::uint32_t r[6][3];
};

AETHER_KERNEL() void goldenCodecKernel(
    const std::uint64_t* __restrict__ inBits, CodecOut* __restrict__ out, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const std::uint64_t u = inBits[i];
    // The DEVICE-callable encode, from the raw IEEE halves — the same body the
    // host `BandedReal::fromDouble` reaches through.
    const BandCell8 w = bd::cell8FromIEEE(static_cast<std::uint32_t>(u & 0xFFFFFFFFu),
        static_cast<std::uint32_t>(u >> 32), 0, false);
    const Band b      = bd::bandFromCell8(w);
    CodecOut o;
    o.word = w.w;
    o.hi   = static_cast<std::uint32_t>(bd::floatAsInt(b.hi));
    o.lo   = static_cast<std::uint32_t>(bd::floatAsInt(b.lo));
    o.tail = static_cast<std::uint32_t>(bd::floatAsInt(b.tail));
    out[i] = o;
}

AETHER_KERNEL() void goldenChainKernel(const std::uint64_t* __restrict__ aWords,
    const std::uint64_t* __restrict__ bWords, ChainOut* __restrict__ out, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band a = static_cast<Band>(BandedReal::fromBits(aWords[i]));
    const Band b = static_cast<Band>(BandedReal::fromBits(bWords[i]));

    const Band res[6] = { bd::add(a, b), bd::sub(a, b), bd::mul(a, b), bd::div(a, b),
        bd::div(bd::mul(bd::sub(a, b), a), b),
        bd::copysign(bd::fmax(bd::abs(a), bd::abs(b)), b) };

    ChainOut o;
    for (int k = 0; k < 6; ++k) {
        o.r[k][0] = static_cast<std::uint32_t>(bd::floatAsInt(res[k].hi));
        o.r[k][1] = static_cast<std::uint32_t>(bd::floatAsInt(res[k].lo));
        o.r[k][2] = static_cast<std::uint32_t>(bd::floatAsInt(res[k].tail));
    }
    out[i] = o;
}

void expectLimbs(const char* what, std::size_t row, const std::uint32_t (&got)[3],
    const std::uint32_t (&want)[3])
{
    EXPECT_EQ(got[0], want[0]) << what << " hi, row " << row;
    EXPECT_EQ(got[1], want[1]) << what << " lo, row " << row;
    EXPECT_EQ(got[2], want[2]) << what << " tail, row " << row;
}

} // namespace

TEST(BandGoldenTest, CrossModeBitExact)
{
    ASSERT_EQ(std::size(golden::kBandCodecRows), golden::kBandCodecRowCount);
    ASSERT_EQ(std::size(golden::kBandChainRows), golden::kBandChainRowCount);
    ASSERT_GT(golden::kBandCodecRowCount, 100u)
        << "the golden codec corpus collapsed — a shrunken corpus certifies "
           "whatever is left";
    ASSERT_GT(golden::kBandChainRowCount, 100u);

    // ---------------- CODEC, in-kernel ----------------
    {
        const int n = static_cast<int>(golden::kBandCodecRowCount);
        std::vector<std::uint64_t> in(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            in[static_cast<std::size_t>(i)]
                = golden::kBandCodecRows[static_cast<std::size_t>(i)].inBits;

        std::uint64_t* dIn = nullptr;
        CodecOut* dOut     = nullptr;
        ASSERT_EQ(cudaMalloc(&dIn, sizeof(std::uint64_t) * n), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dOut, sizeof(CodecOut) * n), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dIn, in.data(), sizeof(std::uint64_t) * n,
                      cudaMemcpyHostToDevice),
            cudaSuccess);

        goldenCodecKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dIn, dOut, n);
        aether::cuda::checkLastLaunch("goldenCodecKernel");

        std::vector<CodecOut> got(static_cast<std::size_t>(n));
        ASSERT_EQ(cudaMemcpy(got.data(), dOut, sizeof(CodecOut) * n,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        cudaFree(dIn);
        cudaFree(dOut);

        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            const golden::BandCodecRow& r = golden::kBandCodecRows[i];
            EXPECT_EQ(got[i].word, r.word) << "device codec word, row " << i;
            EXPECT_EQ(got[i].hi, r.hi) << "device decoded hi, row " << i;
            EXPECT_EQ(got[i].lo, r.lo) << "device decoded lo, row " << i;
            EXPECT_EQ(got[i].tail, r.tail) << "device decoded tail, row " << i;
        }
    }

    // ---------------- ARITHMETIC, in-kernel ----------------
    {
        const int n = static_cast<int>(golden::kBandChainRowCount);
        std::vector<std::uint64_t> aw(static_cast<std::size_t>(n)),
            bw(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            aw[static_cast<std::size_t>(i)]
                = golden::kBandChainRows[static_cast<std::size_t>(i)].aWord;
            bw[static_cast<std::size_t>(i)]
                = golden::kBandChainRows[static_cast<std::size_t>(i)].bWord;
        }

        std::uint64_t *dA = nullptr, *dB = nullptr;
        ChainOut* dOut = nullptr;
        ASSERT_EQ(cudaMalloc(&dA, sizeof(std::uint64_t) * n), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dB, sizeof(std::uint64_t) * n), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dOut, sizeof(ChainOut) * n), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dA, aw.data(), sizeof(std::uint64_t) * n,
                      cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dB, bw.data(), sizeof(std::uint64_t) * n,
                      cudaMemcpyHostToDevice),
            cudaSuccess);

        goldenChainKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dA, dB, dOut, n);
        aether::cuda::checkLastLaunch("goldenChainKernel");

        std::vector<ChainOut> got(static_cast<std::size_t>(n));
        ASSERT_EQ(cudaMemcpy(got.data(), dOut, sizeof(ChainOut) * n,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dOut);

        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            const golden::BandChainRow& r = golden::kBandChainRows[i];
            expectLimbs("device add", i, got[i].r[0], r.add);
            expectLimbs("device sub", i, got[i].r[1], r.sub);
            expectLimbs("device mul", i, got[i].r[2], r.mul);
            expectLimbs("device div", i, got[i].r[3], r.div);
            expectLimbs("device chain", i, got[i].r[4], r.chain);
            expectLimbs("device facade", i, got[i].r[5], r.facade);
        }
    }

    // NON-VACUITY, identical to the host arm's: the corpus must actually carry
    // the reserved code points and the zeros it claims to.
    std::size_t specials = 0, zeros = 0;
    for (std::size_t i = 0; i < golden::kBandCodecRowCount; ++i) {
        const BandedReal r = BandedReal::fromBits(golden::kBandCodecRows[i].word);
        if (r.isNan() || r.isInf())
            ++specials;
        if (r.isZero())
            ++zeros;
    }
    EXPECT_GE(specials, 6u);
    EXPECT_GE(zeros, 4u);
}

} // namespace aether_tests
