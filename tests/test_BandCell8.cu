// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandCell8.cu
 * @brief `BandCell8` conformance (CUDA mode): the shared host battery
 *        (`test_BandCell8_common.h`) plus the DEVICE arms.
 *
 * The host battery runs identically here and in the CPP_MODE twin
 * `test_BandCell8.cpp`. This file adds the GPU path, which is where an 8-byte
 * cell earns its keep and where four things could differ from the host:
 *
 *   - The codec reads and writes float exponent fields through `floatAsInt` and
 *     `intAsFloat`. Those are a free reinterpretation on the GPU and a `memcpy`
 *     on the host — two different implementations that have to agree EXACTLY,
 *     because the claim is bit-identity and not closeness.
 *   - The one rounding in the codec is a float-to-int conversion, and the two
 *     targets reach it differently: `F2I.S32.F32.RN` on sm_61, the host's
 *     default-rounding-mode `__builtin_lrintf` off it. Both are
 *     round-to-nearest-even by contract, and this is where that contract is
 *     checked rather than assumed.
 *   - The lower limbs are rebuilt by SUBTRACTING two constructed floats. If the
 *     device ever compiled that pair into a contraction, or flushed a small
 *     result to zero, the two arms would part company here first.
 *   - `loadCell8`/`storeCell8` exist so that a cell crosses the memory boundary
 *     as ONE 64-bit transaction. On the host that is an ordinary copy; on the
 *     GPU it is the entire reason the type is 8 bytes wide, and the kernels
 *     below are the only place it is exercised as a real global-memory access.
 *
 * The arithmetic is integer work plus exact power-of-two scaling throughout, so
 * the two arms are required to agree bit for bit, and the gate says so.
 *
 * GPU EXECUTION NOTE: this file is built and its tests listed here; the
 * CUDA suite that actually launches these kernels runs separately.
 *
 * `BridgeAgreesWithTheHostOnDevice` (the 16-byte `BandCell` bridge) is not
 * carried here — `BandCell` arrives with the coefficient-table family — it
 * stays `deferred-with-addon` rather than being reproduced as a vacuous pass.
 */

#include "test_BandCell8_common.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace BandCell8Test {

// =========================================================================
//  Device arms. One thread per row.
// =========================================================================

/// Pack a `Band` into a word and unpack it again, on the GPU. Both ends are
/// written out so the host can score the round trip AND compare the
/// intermediate word against its own.
AETHER_KERNEL() void codecKernel(const Band* __restrict__ in,
    BandCell8* __restrict__ words, Band* __restrict__ back, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const BandCell8 w = bd::cell8FromBand(in[i]);
    words[i]          = w;
    back[i]           = bd::bandFromCell8(w);
}

/// Move a word through global memory using only the punned helpers. The load
/// and the store are the subject; nothing else happens in the body.
AETHER_KERNEL() void punKernel(
    const BandCell8* __restrict__ src, BandCell8* __restrict__ dst, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    bd::storeCell8(dst + i, bd::loadCell8(src + i));
}

/// The second tier and the rebias, on the GPU: the escape encode from a raw
/// double, the escape predicate, the escape decode, the FP32 tier, and a
/// rebiased store/load pair.
AETHER_KERNEL() void tierKernel(const std::uint64_t* __restrict__ doubles,
    const Band* __restrict__ bands, std::int32_t arrayBias,
    BandCell8* __restrict__ escWords, bd::EscapeParts* __restrict__ escBack,
    std::uint32_t* __restrict__ flags, float* __restrict__ heads,
    BandCell8* __restrict__ rebiased, Band* __restrict__ rebiasBack, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const std::uint64_t u = doubles[i];
    const BandCell8 e     = bd::cell8EscapedFromIEEE(
        static_cast<std::uint32_t>(u & 0xFFFFFFFFu),
        static_cast<std::uint32_t>(u >> 32));
    escWords[i] = e;
    escBack[i]  = bd::escapePartsFromCell8(e);
    flags[i]    = static_cast<std::uint32_t>(bd::cell8IsEscape(e));

    const BandCell8 r = bd::cell8FromBandRebias(bands[i], arrayBias);
    rebiased[i]       = r;
    rebiasBack[i]     = bd::bandFromCell8Rebias(r, arrayBias);
    heads[i]          = bd::floatFromCell8(bd::cell8FromBand(bands[i]));
}

// -------------------------------------------------------------------------
//  The codec on the GPU, against the host's own answer.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, CodecIsBitIdenticalOnDevice)
{
    std::vector<Band> in;
    for (const Row& r : sweepRows())
        in.push_back(r.b);
    const int n = static_cast<int>(in.size());
    ASSERT_GT(n, 3000) << "the sweep collapsed before it reached the device";

    Band* dIn               = nullptr;
    BandCell8* dWords       = nullptr;
    Band* dBack             = nullptr;
    const std::size_t bandB = static_cast<std::size_t>(n) * sizeof(Band);
    const std::size_t wordB = static_cast<std::size_t>(n) * sizeof(BandCell8);
    ASSERT_EQ(cudaMalloc(&dIn, bandB), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dWords, wordB), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dBack, bandB), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dIn, in.data(), bandB, cudaMemcpyHostToDevice), cudaSuccess);

    codecKernel<<<(n + 255) / 256, 256>>>(dIn, dWords, dBack, n);
    aether::cuda::checkLastLaunch("codecKernel");

    std::vector<BandCell8> words(static_cast<std::size_t>(n));
    std::vector<Band> back(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(words.data(), dWords, wordB, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(back.data(), dBack, bandB, cudaMemcpyDeviceToHost), cudaSuccess);

    std::uint64_t wordBad = 0, bandBad = 0, tagged = 0;
    int firstBad = -1;
    for (std::size_t i = 0; i < in.size(); i++) {
        const BandCell8 hostWord = bd::cell8FromBand(in[i]);
        if (words[i].w != hostWord.w) {
            wordBad++;
            if (firstBad < 0)
                firstBad = static_cast<int>(i);
        }
        if (!sameBandBits(back[i], bd::bandFromCell8(hostWord)))
            bandBad++;
        if (bd::cell8IsEscape(words[i]))
            tagged++;
    }

    std::printf("[BandCell8/device] %d rows: %llu word mismatches, %llu "
                "unpacked-Band mismatches, %llu words carrying an escape tag\n",
        n, static_cast<unsigned long long>(wordBad),
        static_cast<unsigned long long>(bandBad),
        static_cast<unsigned long long>(tagged));
    if (firstBad >= 0)
        std::printf("   FIRST OFFENDER row %d at 2^%d, pattern '%s'\n", firstBad,
            sweepRows()[static_cast<std::size_t>(firstBad)].e,
            sweepRows()[static_cast<std::size_t>(firstBad)].p->name);

    EXPECT_EQ(wordBad, 0u)
        << "the device and the host packed the same Band into different words "
           "-- the codec is exact bit reads, exact power-of-two scaling and one "
           "round-to-nearest conversion, so there is no seed or rounding "
           "difference that could permit this";
    EXPECT_EQ(bandBad, 0u)
        << "the device and the host unpacked the same word differently";
    EXPECT_EQ(tagged, 0u);

    cudaFree(dIn);
    cudaFree(dWords);
    cudaFree(dBack);
}

// -------------------------------------------------------------------------
//  The punned access across a real global-memory boundary.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, PunnedLoadStoreRoundTripsOnDevice)
{
    std::vector<BandCell8> src;
    for (const Row& r : sweepRows())
        src.push_back(bd::cell8FromBand(r.b));
    const int n = static_cast<int>(src.size());
    ASSERT_GT(n, 3000);

    BandCell8 *dSrc = nullptr, *dDst = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(BandCell8);
    ASSERT_EQ(cudaMalloc(&dSrc, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dDst, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(dDst, 0xA5, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dSrc, src.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    punKernel<<<(n + 255) / 256, 256>>>(dSrc, dDst, n);
    aether::cuda::checkLastLaunch("punKernel");

    std::vector<BandCell8> dst(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(dst.data(), dDst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    std::uint64_t bad = 0;
    for (std::size_t i = 0; i < src.size(); i++)
        if (src[i].w != dst[i].w)
            bad++;

    std::printf("[BandCell8/device] punned load+store: %d words, %llu altered\n", n,
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u)
        << "a word changed while crossing global memory through loadCell8 / "
           "storeCell8 -- the destination was pre-filled with 0xA5 bytes, so a "
           "partially-written word shows up here rather than passing as a "
           "coincidence";

    cudaFree(dSrc);
    cudaFree(dDst);
}

// -------------------------------------------------------------------------
//  The second tier and the rebias on the GPU.
//
//  The escape encode is the only part of this format that does integer
//  rounding on a 52-bit payload, and the rebias is the only part that does
//  modular arithmetic on an exponent field. Both are cheap to get subtly wrong
//  in a 64-bit shift on sm_61, where a 64-bit shift is a sequence rather than
//  an instruction, so both are compared against the host.
// -------------------------------------------------------------------------
TEST_F(BandCell8Cert, EscapeAndRebiasAgreeWithTheHostOnDevice)
{
    constexpr std::int32_t kArrayBias = 400;

    std::vector<std::uint64_t> doubles;
    std::vector<Band> bands;
    const std::uint64_t mants[] = { 0, (std::uint64_t{ 1 } << 52) - 1, 1,
        std::uint64_t{ 1 } << 51, 0x921FB54442D18ull, 0xFFull, 0x80ull, 0x180ull };
    for (int e = -1000; e <= 1000; e += 3) {
        if (!tier2Covers(e))
            continue;
        for (std::uint64_t m : mants)
            for (int s = 0; s < 2; s++) {
                // Rows the encoder is contracted to refuse never reach the
                // device either -- same fate computation as the host battery.
                if (!escapeRowFate(e, m, s).encodable)
                    continue;
                doubles.push_back((static_cast<std::uint64_t>(s) << 63)
                    | (static_cast<std::uint64_t>(e + 1023) << 52) | m);
            }
    }
    // One in-window Band per escape row, so the two arms of the kernel run over
    // the same index space.
    {
        std::size_t i = 0;
        while (bands.size() < doubles.size()) {
            const Pattern& p = patterns()[i % patterns().size()];
            const int e      = kFloorE + static_cast<int>(i % 220);
            bands.push_back(restBand(p, (i & 1) != 0, e, (i & 2) != 0));
            i++;
        }
    }
    const int n = static_cast<int>(doubles.size());
    ASSERT_GT(n, 3000);

    std::uint64_t* dDoubles     = nullptr;
    Band* dBands                = nullptr;
    BandCell8* dEsc             = nullptr;
    bd::EscapeParts* dEscBack   = nullptr;
    std::uint32_t* dFlags       = nullptr;
    float* dHeads               = nullptr;
    BandCell8* dRebias          = nullptr;
    Band* dRebiasBack           = nullptr;
    const std::size_t un        = static_cast<std::size_t>(n);
    ASSERT_EQ(cudaMalloc(&dDoubles, un * sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dBands, un * sizeof(Band)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dEsc, un * sizeof(BandCell8)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dEscBack, un * sizeof(bd::EscapeParts)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dFlags, un * sizeof(std::uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dHeads, un * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRebias, un * sizeof(BandCell8)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dRebiasBack, un * sizeof(Band)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dDoubles, doubles.data(), un * sizeof(std::uint64_t),
                  cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dBands, bands.data(), un * sizeof(Band), cudaMemcpyHostToDevice),
        cudaSuccess);

    tierKernel<<<(n + 255) / 256, 256>>>(dDoubles, dBands, kArrayBias, dEsc, dEscBack,
        dFlags, dHeads, dRebias, dRebiasBack, n);
    aether::cuda::checkLastLaunch("tierKernel");

    std::vector<BandCell8> esc(un), rebias(un);
    std::vector<bd::EscapeParts> escBack(un);
    std::vector<std::uint32_t> flags(un);
    std::vector<float> heads(un);
    std::vector<Band> rebiasBack(un);
    ASSERT_EQ(cudaMemcpy(esc.data(), dEsc, un * sizeof(BandCell8), cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(escBack.data(), dEscBack, un * sizeof(bd::EscapeParts),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(flags.data(), dFlags, un * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(heads.data(), dHeads, un * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(rebias.data(), dRebias, un * sizeof(BandCell8),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(rebiasBack.data(), dRebiasBack, un * sizeof(Band),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);

    std::uint64_t escBad = 0, escBackBad = 0, flagBad = 0, headBad = 0, rebiasBad = 0,
                  rebiasBackBad = 0;
    for (std::size_t i = 0; i < un; i++) {
        const BandCell8 he = bd::cell8EscapedFromIEEE(
            static_cast<std::uint32_t>(doubles[i] & 0xFFFFFFFFu),
            static_cast<std::uint32_t>(doubles[i] >> 32));
        if (esc[i].w != he.w)
            escBad++;
        if (!sameEscapeBits(escBack[i], bd::escapePartsFromCell8(he)))
            escBackBad++;
        if (flags[i] != static_cast<std::uint32_t>(bd::cell8IsEscape(he)))
            flagBad++;
        const BandCell8 hr = bd::cell8FromBandRebias(bands[i], kArrayBias);
        if (rebias[i].w != hr.w)
            rebiasBad++;
        if (!sameBandBits(rebiasBack[i], bd::bandFromCell8Rebias(hr, kArrayBias)))
            rebiasBackBad++;
        if (bits(heads[i]) != bits(bd::floatFromCell8(bd::cell8FromBand(bands[i]))))
            headBad++;
    }

    std::printf("[BandCell8/device] tiers: %d rows, escape %llu/%llu/%llu "
                "(word/decode/flag), rebias %llu/%llu (word/decode), float "
                "tier %llu\n",
        n, static_cast<unsigned long long>(escBad),
        static_cast<unsigned long long>(escBackBad),
        static_cast<unsigned long long>(flagBad),
        static_cast<unsigned long long>(rebiasBad),
        static_cast<unsigned long long>(rebiasBackBad),
        static_cast<unsigned long long>(headBad));

    EXPECT_EQ(escBad, 0u) << "the device rounded a 52-bit payload into 44 bits "
                             "differently from the host";
    EXPECT_EQ(escBackBad, 0u);
    EXPECT_EQ(flagBad, 0u);
    EXPECT_EQ(rebiasBad, 0u);
    EXPECT_EQ(rebiasBackBad, 0u);
    EXPECT_EQ(headBad, 0u);

    cudaFree(dDoubles);
    cudaFree(dBands);
    cudaFree(dEsc);
    cudaFree(dEscBack);
    cudaFree(dFlags);
    cudaFree(dHeads);
    cudaFree(dRebias);
    cudaFree(dRebiasBack);
}

} // namespace BandCell8Test
} // namespace aether_tests
