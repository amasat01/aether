// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandCell.cu
 * @brief `BandCell` conformance (CUDA mode): the shared host battery
 *        (`test_BandCell_common.h`) plus the DEVICE arms.
 *
 * The re-centring is exact power-of-two scaling and pure bit reads
 * throughout, so unlike the `rsqrt`-seeded primitives elsewhere there is no
 * host/device seed difference to allow for — the two arms are required to
 * agree bit for bit.
 *
 * ★ `roundTripKernel` is subject 5 of `tests/sass/check_band_fp64free.sh`
 * (fp64 audit): FIXED-NAME AUDIT SUBJECT, do not rename. It is
 * `BandCell`'s own codec — `cellFromBand`/`bandFromCell` — exercised over
 * device data, the same shape the audit's other subjects exercise for
 * `BandCell8`/`Ff2`/`RsqrtCore`. `roundTripFp64InjectedKernel` (below) is its
 * POSITIVE CONTROL. `punKernel` is a pure 128-bit load/store with no
 * arithmetic in its body at all — it cannot emit FP64 by construction, so it
 * carries no control and is not part of the audit: a kernel with zero
 * arithmetic instructions makes "0 FP64 hits" a vacuous claim, and a
 * positive control for it would have nothing real to poison.
 *
 * GPU EXECUTION NOTE: this file is built and its tests listed here; the
 * CUDA suite that actually launches these kernels runs separately.
 */

#include "test_BandCell_common.h"

#include "aether/backend/cuda/Launch.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace BandCellTest {

// =========================================================================
//  Device arms. One thread per row.
// =========================================================================

/**
 * @brief ★ THE FP64-FREE AUDIT SUBJECT (subject 5). Fixed name; do not
 * rename. Re-centre a band into a cell and fold it straight back, on the
 * GPU. Both ends are written out so the host can score the round trip AND
 * compare the intermediate cell against its own.
 *
 * Op inventory `tests/sass/check_band_fp64free.sh`'s subject-5 budget is
 * DERIVED from: TWO `bandScalePow2Split` calls (one inside `cellFromBand`,
 * one inside `bandFromCell`), each two `scalePow2f` calls of THREE FP32
 * multiplies (hi, lo, tail) = 12 FP32 multiplies, plus the bit-level
 * exponent extract/rebuild each of `cellFromBand`/`bandFromCell` does (int
 * shifts/masks, no float ops). Unlike `bandChainKernel`'s multi-op chain,
 * there is no algebra between separate certified ops here to amplify a
 * regression through — the whole body IS the codec — so the smallest
 * realistic regression is ONE extra `bandScalePow2Split` call (~14
 * instructions: 3 FP32 muls x 2 halves + int bookkeeping), the SAME
 * single-crossing class subjects 3/4 derive their budgets against.
 */
AETHER_KERNEL() void roundTripKernel(
    const DeviceRow* __restrict__ rows, BandCell* __restrict__ cells, DeviceRow* __restrict__ back, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band b     = Band{ rows[i].hi, rows[i].lo, rows[i].tail };
    const BandCell c = bd::cellFromBand(b);
    const Band r      = bd::bandFromCell(c);
    cells[i]          = c;
    back[i]            = DeviceRow{ r.hi, r.lo, r.tail, rows[i].E };
}

/**
 * @brief ★ THE POSITIVE CONTROL for `roundTripKernel`. Fixed name; do not
 * rename. The identical body with ONE deliberate FP64 operation. `poison`
 * is a KERNEL ARGUMENT, so the multiply cannot be constant-folded away —
 * @see `bandFp64InjectedKernel` (`tests/test_BandedReal.cu`) for the
 * identical pattern. The audit is REQUIRED to find FP64 here; without this
 * leg, "0 FP64 hits" on `roundTripKernel` is equally consistent with a
 * working scan and with a symbol lookup that quietly matched nothing.
 */
AETHER_KERNEL() void roundTripFp64InjectedKernel(const DeviceRow* __restrict__ rows,
    BandCell* __restrict__ cells, DeviceRow* __restrict__ back, int n, double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    const Band b      = Band{ rows[i].hi * spoilt, rows[i].lo, rows[i].tail };
    const BandCell c  = bd::cellFromBand(b);
    const Band r       = bd::bandFromCell(c);
    cells[i]           = c;
    back[i]             = DeviceRow{ r.hi, r.lo, r.tail, rows[i].E };
}

/// Move a cell through global memory using only the punned helpers. The load
/// and the store are the subject; nothing else happens in the body. Pure
/// 128-bit memory traffic, no arithmetic instructions at all -- NOT an
/// fp64-audit subject (see the file header).
AETHER_KERNEL() void punKernel(const BandCell* __restrict__ src, BandCell* __restrict__ dst, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    bd::storeCell(dst + i, bd::loadCell(src + i));
}

// -------------------------------------------------------------------------
//  14. The round trip, on the GPU, against the host's own answer.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, RoundTripIsBitExactOnDevice)
{
    const std::vector<DeviceRow> rs = deviceRows();
    const int n                     = static_cast<int>(rs.size());
    ASSERT_GT(n, 3000) << "the sweep collapsed before it reached the device";

    DeviceRow* dRows  = nullptr;
    BandCell* dCells = nullptr;
    DeviceRow* dBack = nullptr;
    const std::size_t rowB = static_cast<std::size_t>(n) * sizeof(DeviceRow);
    ASSERT_EQ(cudaMalloc(&dRows, rowB), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dCells, static_cast<std::size_t>(n) * sizeof(BandCell)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dBack, rowB), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dRows, rs.data(), rowB, cudaMemcpyHostToDevice), cudaSuccess);

    roundTripKernel<<<(n + 255) / 256, 256>>>(dRows, dCells, dBack, n);
    aether::cuda::checkLastLaunch("roundTripKernel");

    std::vector<BandCell> cells(static_cast<std::size_t>(n));
    std::vector<DeviceRow> back(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(cells.data(), dCells, static_cast<std::size_t>(n) * sizeof(BandCell),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(back.data(), dBack, rowB, cudaMemcpyDeviceToHost), cudaSuccess);

    std::uint64_t inverseBad = 0, hostDeviceBad = 0, restBad = 0;
    int firstBad = -1;
    for (std::size_t i = 0; i < rs.size(); i++) {
        const Band in{ rs[i].hi, rs[i].lo, rs[i].tail };
        const Band out{ back[i].hi, back[i].lo, back[i].tail };
        if (!sameBandBits(in, out)) {
            inverseBad++;
            if (firstBad < 0)
                firstBad = static_cast<int>(i);
        }
        if (!sameCellBits(cells[i], bd::cellFromBand(in)))
            hostDeviceBad++;
        const float a        = std::fabs(cells[i].hi);
        const bool zeroCell = bits(cells[i].hi) == 0u && cells[i].bias == 0;
        if (!zeroCell && !(a >= 1.0f && a < 2.0f))
            restBad++;
    }

    std::printf("[BandCell/device] %d rows: %llu round-trip failures, %llu host/device cell "
                "mismatches, %llu cells not at rest\n",
        n, static_cast<unsigned long long>(inverseBad), static_cast<unsigned long long>(hostDeviceBad),
        static_cast<unsigned long long>(restBad));
    if (firstBad >= 0)
        std::printf("   FIRST OFFENDER row %d at 2^%d\n", firstBad, rs[static_cast<std::size_t>(firstBad)].E);

    EXPECT_EQ(inverseBad, 0u) << "band -> cell -> band is not the identity on the device";
    EXPECT_EQ(hostDeviceBad, 0u)
        << "the device and the host disagree on the cell for the same band -- the re-centring is "
           "exact power-of-two arithmetic and pure bit reads, so there is no seed or rounding "
           "difference to permit this";
    EXPECT_EQ(restBad, 0u) << "a cell produced on the device is neither zero nor centred";

    cudaFree(dRows);
    cudaFree(dCells);
    cudaFree(dBack);
}

// -------------------------------------------------------------------------
//  15. The punned access across a real global-memory boundary.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, PunnedLoadStoreRoundTripsOnDevice)
{
    std::vector<BandCell> src;
    for (const Row& r : sweepRows())
        src.push_back(bd::cellFromBand(r.b));
    const int n = static_cast<int>(src.size());
    ASSERT_GT(n, 3000);

    BandCell *dSrc = nullptr, *dDst = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(BandCell);
    ASSERT_EQ(cudaMalloc(&dSrc, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dDst, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(dDst, 0xA5, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dSrc, src.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    punKernel<<<(n + 255) / 256, 256>>>(dSrc, dDst, n);
    aether::cuda::checkLastLaunch("punKernel");

    std::vector<BandCell> dst(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(dst.data(), dDst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    std::uint64_t bad = 0;
    for (std::size_t i = 0; i < src.size(); i++)
        if (!sameCellBits(src[i], dst[i]))
            bad++;

    std::printf("[BandCell/device] punned load+store: %d cells, %llu altered\n", n,
        static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u)
        << "a cell changed while crossing global memory through loadCell/storeCell -- the "
           "destination was pre-filled with 0xA5 bytes, so a partially-written cell shows up "
           "here rather than passing as a coincidence";

    cudaFree(dSrc);
    cudaFree(dDst);
}

// -------------------------------------------------------------------------
//  16. The fp64-free SASS audit's positive control, LAUNCHED (not merely
//      compiled): this row exists for two reasons at once, mirroring
//      `TheAuditedChainKernelAgreesWithTheHostAndItsControlDoesNot`
//      (`tests/test_BandedReal.cu`). It is the INSTANTIATION
//      `tests/sass/check_band_fp64free.sh` subject 5 needs, and it is the
//      known-answer check that `poison == 0.0` makes the injected kernel's
//      scale factor exactly `1.0f`, so it must compute the IDENTICAL round
//      trip `roundTripKernel` does -- the control's job is to carry an FP64
//      INSTRUCTION, not a different answer.
// -------------------------------------------------------------------------
TEST_F(BandCellCert, Fp64InjectedControlAgreesWithTheHostWhenUnpoisoned)
{
    const std::vector<DeviceRow> rs = deviceRows();
    const int n                     = static_cast<int>(rs.size());
    ASSERT_GT(n, 3000) << "the sweep collapsed before it reached the device";

    DeviceRow* dRows  = nullptr;
    BandCell* dCells = nullptr;
    DeviceRow* dBack = nullptr;
    const std::size_t rowB = static_cast<std::size_t>(n) * sizeof(DeviceRow);
    ASSERT_EQ(cudaMalloc(&dRows, rowB), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dCells, static_cast<std::size_t>(n) * sizeof(BandCell)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dBack, rowB), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dRows, rs.data(), rowB, cudaMemcpyHostToDevice), cudaSuccess);

    roundTripFp64InjectedKernel<<<(n + 255) / 256, 256>>>(dRows, dCells, dBack, n, 0.0);
    aether::cuda::checkLastLaunch("roundTripFp64InjectedKernel");

    std::vector<DeviceRow> back(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(back.data(), dBack, rowB, cudaMemcpyDeviceToHost), cudaSuccess);

    std::uint64_t bad = 0;
    for (std::size_t i = 0; i < rs.size(); i++) {
        const Band in{ rs[i].hi, rs[i].lo, rs[i].tail };
        const Band out{ back[i].hi, back[i].lo, back[i].tail };
        if (!sameBandBits(in, out))
            bad++;
    }
    std::printf(
        "[BandCell/device] fp64-injected control (poison=0): %d rows, %llu round-trip failures\n",
        n, static_cast<unsigned long long>(bad));
    EXPECT_EQ(bad, 0u) << "poison=0 makes the injected scale exactly 1.0f -- the control's job "
                           "is to carry an FP64 instruction, not a different answer";

    cudaFree(dRows);
    cudaFree(dCells);
    cudaFree(dBack);
}

} // namespace BandCellTest
} // namespace aether_tests
