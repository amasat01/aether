// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandIngest.cu
 * @brief `BandIngest` (CUDA mode): the shared host battery
 *        (`test_BandIngest_common.h`) plus the DEVICE arms.
 *
 * The claim being made is bit-level equality with an exactly-computed
 * reference, so host and device have to agree exactly, not closely.
 *
 * `MaterializeFiftyThreeCompilesAndAgreesOnDevice` is not carried: it names
 * `SoftDouble` as its kernel's return type and `det::materialize<53>` over
 * `BandedDemandT` -- neither exists in aether's carried surface -- and
 * stays `deferred-with-addon`.
 *
 * GPU EXECUTION NOTE: this file is built and its tests listed here; the
 * CUDA suite that actually launches these kernels runs separately.
 */

#include "test_BandIngest_common.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace BandIngestTest {

struct IeeeWords {
    std::uint32_t lo, hi;
};
struct LimbBits {
    std::uint32_t hi, lo, tail;
};

AETHER_KERNEL() void ingestKernel(const IeeeWords* __restrict__ in,
    LimbBits* __restrict__ out, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band b = bd::bandFromIEEE(in[i].lo, in[i].hi);
    LimbBits r;
    std::memcpy(&r.hi, &b.hi, sizeof(float));
    std::memcpy(&r.lo, &b.lo, sizeof(float));
    std::memcpy(&r.tail, &b.tail, sizeof(float));
    out[i] = r;
}

TEST_F(BandIngest, LimbsMatchTheReferenceOnDevice)
{
    std::vector<IeeeWords> rows;
    for (int e = kWindowLow; e <= kWindowHigh; e++)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                rows.push_back(IeeeWords{ lo, hi });
            }
    const int n = static_cast<int>(rows.size());
    ASSERT_GT(n, 4000);

    IeeeWords* dIn = nullptr;
    LimbBits* dOut = nullptr;
    cudaMalloc(&dIn, static_cast<size_t>(n) * sizeof(IeeeWords));
    cudaMalloc(&dOut, static_cast<size_t>(n) * sizeof(LimbBits));
    cudaMemcpy(dIn, rows.data(), static_cast<size_t>(n) * sizeof(IeeeWords),
        cudaMemcpyHostToDevice);

    ingestKernel<<<(n + 255) / 256, 256>>>(dIn, dOut, n);
    cudaDeviceSynchronize();

    std::vector<LimbBits> out(static_cast<size_t>(n));
    cudaMemcpy(out.data(), dOut, static_cast<size_t>(n) * sizeof(LimbBits),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::uint64_t refBad = 0, hostBad = 0;
    for (std::size_t i = 0; i < rows.size(); i++) {
        const Band want = referenceBandFromIEEE(rows[i].lo, rows[i].hi);
        const Band host = bd::bandFromIEEE(rows[i].lo, rows[i].hi);
        if (out[i].hi != fbits(want.hi) || out[i].lo != fbits(want.lo)
            || out[i].tail != fbits(want.tail))
            refBad++;
        if (out[i].hi != fbits(host.hi) || out[i].lo != fbits(host.lo)
            || out[i].tail != fbits(host.tail))
            hostBad++;
    }

    std::printf("[BandIngest/device] %d rows: %llu differ from the reference, "
                "%llu differ from the host arm\n",
        n, static_cast<unsigned long long>(refBad),
        static_cast<unsigned long long>(hostBad));
    EXPECT_EQ(refBad, 0u)
        << "the device does not produce the correctly-rounded limb values";
    EXPECT_EQ(hostBad, 0u)
        << "the host and the device disagree on the same input";

    cudaFree(dIn);
    cudaFree(dOut);
}

TEST_F(BandIngest, LeadingLimbIsCorrectlyRoundedBelowTheFloorOnDevice)
{
    std::vector<IeeeWords> rows;
    for (int e = kTinyTop; e >= kTinyBottom; e--)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                rows.push_back(IeeeWords{ lo, hi });
            }
    for (int e = -153; e >= -1022; e -= 3)
        for (const Pattern& p : patterns())
            for (int s = 0; s < 2; s++) {
                std::uint32_t lo = 0, hi = 0;
                wordsFor(p, e, s == 1, lo, hi);
                rows.push_back(IeeeWords{ lo, hi });
            }
    const int n = static_cast<int>(rows.size());
    ASSERT_GT(n, 5000);

    IeeeWords* dIn = nullptr;
    LimbBits* dOut = nullptr;
    cudaMalloc(&dIn, static_cast<size_t>(n) * sizeof(IeeeWords));
    cudaMalloc(&dOut, static_cast<size_t>(n) * sizeof(LimbBits));
    cudaMemcpy(dIn, rows.data(), static_cast<size_t>(n) * sizeof(IeeeWords),
        cudaMemcpyHostToDevice);

    ingestKernel<<<(n + 255) / 256, 256>>>(dIn, dOut, n);
    cudaDeviceSynchronize();

    std::vector<LimbBits> out(static_cast<size_t>(n));
    cudaMemcpy(out.data(), dOut, static_cast<size_t>(n) * sizeof(LimbBits),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::uint64_t refBad = 0, hostBad = 0, subnormalHeads = 0;
    for (std::size_t i = 0; i < rows.size(); i++) {
        const Band want = referenceBandFromIEEE(rows[i].lo, rows[i].hi);
        const Band host = bd::bandFromIEEE(rows[i].lo, rows[i].hi);
        if (out[i].hi != fbits(want.hi) || out[i].lo != fbits(want.lo)
            || out[i].tail != fbits(want.tail))
            refBad++;
        if (out[i].hi != fbits(host.hi) || out[i].lo != fbits(host.lo)
            || out[i].tail != fbits(host.tail))
            hostBad++;
        if ((out[i].hi & 0x7F800000u) == 0u && (out[i].hi & 0x007FFFFFu) != 0u)
            subnormalHeads++;
    }

    std::printf("[BandIngest/device] sub-floor sweep, %d rows: %llu differ "
                "from the reference, %llu differ from the host arm; %llu rows "
                "produced a SUBNORMAL leading limb on the device\n",
        n, static_cast<unsigned long long>(refBad),
        static_cast<unsigned long long>(hostBad),
        static_cast<unsigned long long>(subnormalHeads));
    EXPECT_EQ(refBad, 0u);
    EXPECT_EQ(hostBad, 0u);
    EXPECT_GT(subnormalHeads, 100u)
        << "no device row produced a subnormal leading limb, so this arm did "
           "not exercise the regime it exists for";

    cudaFree(dIn);
    cudaFree(dOut);
}

// NOTE: `MaterializeFiftyThreeCompilesAndAgreesOnDevice` is not carried
// -- its kernel returns `SoftDouble`, a type aether does not
// carry. See `test_BandIngest_common.h`'s header comment.

} // namespace BandIngestTest
} // namespace aether_tests
