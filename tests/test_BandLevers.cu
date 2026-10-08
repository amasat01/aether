// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandLevers.cu
 * @brief `BandAccum`/`BandAccumVec` (CUDA mode): the shared host battery
 *        (`test_BandLevers_common.h`) plus the DEVICE arm and the fp64-free
 *        SASS AUDIT SUBJECT for the accumulator.
 *
 * `BandAccum`'s mutators (`addTerm`/`subTerm`) are already exercised on the
 * device by every OTHER banded kernel that folds through `detail::add`/
 * `detail::sub` — those are the same certified bodies `Band`'s own operators
 * call, already proven FP64-free by `bandChainKernel` (`test_BandedReal.cu`).
 * What is NEW here, and needs its OWN audited kernel, is the ACCUMULATOR'S
 * OWN shape: `BandAccum` state carried across a LOOP of `addTerm`/`subTerm`
 * calls (rather than one expression evaluated once), then read back through
 * the exemplar's own new egress terminal `toBandedReal()` — a construction
 * `tests/sass/check_band_fp64free.sh`'s original audit never exercised.
 *
 * ★ `bandAccumFoldKernel` and `bandAccumFoldFp64InjectedKernel` are
 * FIXED-NAME AUDIT SUBJECTS. Do not rename, do not dead-strip. The second is
 * the POSITIVE CONTROL — the identical body plus one deliberate FP64
 * operation, which the scanner is REQUIRED to find.
 *
 * @section rsqrtatan A second audit subject
 * `rsqrtAtanPrimKernel`/`rsqrtAtanPrimFp64InjectedKernel` are a SECOND pair
 * of fixed-name audit subjects, added for `banded/RsqrtCore.h` (`rsqrt`/
 * `sqrt_`/`rsqrtCube`) and `banded/AtanSector.h` (`atanSector`) — neither was
 * exercised by ANY prior audited kernel (subjects 1/2 fold `add`/`sub`/`mul`/
 * `div`/`recip`; `atanSector` is a pure comparison+lookup with zero shared
 * arithmetic path). Same idiom throughout: fixed names, a deliberate
 * KERNEL-ARGUMENT FP64 poison in the injected twin, `tests/sass/
 * check_band_fp64free.sh` registers it as "Subject 4".
 *
 * GPU EXECUTION NOTE: this file is built and its tests listed here; the
 * CUDA suite that actually launches these kernels runs separately.
 */

#include "test_BandLevers_common.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace BandLeversTest {

using aether::SampleIndex;

constexpr int kBlock = 128;

/// @brief Term count per fold. Read off THIS number by the SASS budget
/// derivation in `tests/sass/check_band_fp64free.sh` — change one and
/// reconcile the other.
constexpr int kAccumFoldTerms = 8;

// =========================================================================
//  ★ THE FP64-FREE AUDIT SUBJECT. Fixed name; do not rename.
//
//  Folds `kAccumFoldTerms` packed terms through ONE `BandAccum`, alternating
//  addTerm/subTerm by the caller-supplied op byte, then packs the result via
//  `toBandedReal()` — the exemplar's DEVICEHOST egress terminal.
//
//  Op inventory (read off THIS body, not measured):
//    kAccumFoldTerms (=8) certified Band ops (each addTerm/subTerm is ONE
//    `detail::add`/`detail::sub` call -- algebraically the same cost, `sub`
//    being `add` of a free `neg`).
//  Codec boundary crossings:
//    kAccumFoldTerms (=8) unpacks (the per-term `BandedReal -> Band` loads)
//    + 1 pack (the `toBandedReal()` store) = 9.
// =========================================================================
AETHER_KERNEL() void bandAccumFoldKernel(BandedReal* __restrict__ out,
    const BandedReal* __restrict__ terms, const std::uint8_t* __restrict__ ops, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    BandAccum acc;
    for (int t = 0; t < kAccumFoldTerms; ++t) {
        const Band term = static_cast<Band>(terms[i * kAccumFoldTerms + t]);
        if (ops[i * kAccumFoldTerms + t] == 0u)
            acc.addTerm(term);
        else
            acc.subTerm(term);
    }
    out[i] = acc.toBandedReal();
}

/**
 * @brief ★ THE POSITIVE CONTROL. Fixed name; do not rename.
 *
 * The identical body with ONE deliberate FP64 operation scaling the FIRST
 * term's leading limb. `poison` is a KERNEL ARGUMENT, so the multiply cannot
 * be constant-folded away. The audit is REQUIRED to find FP64 here.
 */
AETHER_KERNEL() void bandAccumFoldFp64InjectedKernel(BandedReal* __restrict__ out,
    const BandedReal* __restrict__ terms, const std::uint8_t* __restrict__ ops, int n,
    double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    BandAccum acc;
    for (int t = 0; t < kAccumFoldTerms; ++t) {
        Band term = static_cast<Band>(terms[i * kAccumFoldTerms + t]);
        if (t == 0)
            term.hi = term.hi * spoilt;
        if (ops[i * kAccumFoldTerms + t] == 0u)
            acc.addTerm(term);
        else
            acc.subTerm(term);
    }
    out[i] = acc.toBandedReal();
}

// =========================================================================
//  Device row
// =========================================================================

TEST_F(BandLevers, TheAuditedAccumFoldKernelAgreesWithTheHostAndItsControlDoesNotOnDevice)
{
    /* This row is the INSTANTIATION the SASS audit needs (a header-only type
     * emits nothing to scan until something instantiates it) and the
     * KNOWN-ANSWER check that the audited body actually folds the
     * accumulator rather than something the optimiser collapsed. The
     * injected control is LAUNCHED too, so its presence in the binary is not
     * an accident of the linker. */
    const int n = 256;
    std::vector<BandedReal> terms(static_cast<std::size_t>(n) * kAccumFoldTerms);
    std::vector<std::uint8_t> ops(static_cast<std::size_t>(n) * kAccumFoldTerms);
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < kAccumFoldTerms; t++) {
            const std::size_t idx = static_cast<std::size_t>(i) * kAccumFoldTerms
                + static_cast<std::size_t>(t);
            terms[idx] = BandedReal::fromDouble(
                std::sin(0.037 * i + 0.61 * t) * std::ldexp(1.0, ((i + 3 * t) % 23) - 11));
            ops[idx] = static_cast<std::uint8_t>((i + t) % 2);
        }
    }

    BandedReal *dTerms = nullptr, *dOut = nullptr, *dInj = nullptr;
    std::uint8_t* dOps = nullptr;
    ASSERT_EQ(cudaMalloc(&dTerms, sizeof(BandedReal) * terms.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOps, sizeof(std::uint8_t) * ops.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dInj, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dTerms, terms.data(), sizeof(BandedReal) * terms.size(),
                  cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(dOps, ops.data(), sizeof(std::uint8_t) * ops.size(), cudaMemcpyHostToDevice),
        cudaSuccess);

    bandAccumFoldKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dOut, dTerms, dOps, n);
    aether::cuda::checkLastLaunch("bandAccumFoldKernel");
    bandAccumFoldFp64InjectedKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
        dInj, dTerms, dOps, n, 0.0);
    aether::cuda::checkLastLaunch("bandAccumFoldFp64InjectedKernel");

    std::vector<BandedReal> got(static_cast<std::size_t>(n)), inj(static_cast<std::size_t>(n));
    ASSERT_EQ(
        cudaMemcpy(got.data(), dOut, sizeof(BandedReal) * n, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(inj.data(), dInj, sizeof(BandedReal) * n, cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaFree(dTerms);
    cudaFree(dOps);
    cudaFree(dOut);
    cudaFree(dInj);

    std::size_t bad = 0, trivial = 0, injDiff = 0;
    for (int i = 0; i < n; i++) {
        BandAccum acc;
        for (int t = 0; t < kAccumFoldTerms; t++) {
            const std::size_t idx = static_cast<std::size_t>(i) * kAccumFoldTerms
                + static_cast<std::size_t>(t);
            const Band term = static_cast<Band>(terms[idx]);
            if (ops[idx] == 0u)
                acc.addTerm(term);
            else
                acc.subTerm(term);
        }
        const BandedReal want = acc.toBandedReal();
        if (got[static_cast<std::size_t>(i)].toBits() != want.toBits())
            bad++;
        if (got[static_cast<std::size_t>(i)].isZero())
            trivial++;
        /* poison = 0 makes the injected kernel's scale factor exactly 1.0f,
         * so the two agree numerically -- the control's job is to carry an
         * FP64 INSTRUCTION, not a different answer. */
        if (inj[static_cast<std::size_t>(i)].toBits() != want.toBits())
            injDiff++;
    }
    std::printf("[BandLevers/device] audited accum fold over %d samples: %zu "
                "mismatches, %zu trivially-zero results, %zu injected-arm "
                "differences\n",
        n, bad, trivial, injDiff);
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(injDiff, 0u);
    EXPECT_LT(trivial, static_cast<std::size_t>(n) / 4)
        << "most of the audited accum fold's outputs are zero, so the SASS "
           "audit is scanning a kernel the optimiser may have hollowed out";
}

// =========================================================================
//  ★ THE SECOND FP64-FREE AUDIT SUBJECT (Subject 4
//  in `tests/sass/check_band_fp64free.sh`). Fixed names; do not rename.
//
//  Exercises `rsqrt`/`sqrt_`/`rsqrtCube` (RsqrtCore.h) on ONE positive
//  operand and `atanSector` (AtanSector.h) on a leading-limb ratio derived
//  from a SECOND operand — a construction NEITHER prior audited kernel
//  reaches (subjects 1/2 never call `sqrt`/`rsqrt`; `atanSector` shares no
//  arithmetic path with any op subjects 1-3 exercise).
//
//  Op inventory (read off THIS body, not measured):
//    3 RsqrtCore-family calls (rsqrt, sqrt_, rsqrtCube — each pays its own
//    ONE rsqrtCore() Newton refinement, so 3 independent cores) + 1
//    atanSector lookup (pure FP32 compares + a literal table, no core).
//  Codec boundary crossings: 2 unpacks (a[i], b[i] -> Band) + 5 packs (the
//  three RsqrtCore results + the sector's t/phi, each toBandedReal()) = 7.
// =========================================================================
AETHER_KERNEL() void rsqrtAtanPrimKernel(const BandedReal* __restrict__ a,
    const BandedReal* __restrict__ b, int n, BandedReal* __restrict__ oRsqrt,
    BandedReal* __restrict__ oSqrt, BandedReal* __restrict__ oRsqrtCube,
    BandedReal* __restrict__ oSecT, BandedReal* __restrict__ oSecPhi)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    Band A = static_cast<Band>(a[i]);
    if (A.hi <= 0.0f)
        A = bd::neg(A);
    const Band B = static_cast<Band>(b[i]);

    oRsqrt[i]     = BandedReal(bd::rsqrt(A));
    oSqrt[i]      = BandedReal(bd::sqrt_(A));
    oRsqrtCube[i] = BandedReal(bd::rsqrtCube(A));

    const bd::AtanSector sec = bd::atanSector(std::fabs(B.hi), std::fabs(A.hi) + 1.0f);
    oSecT[i]                 = BandedReal(sec.t);
    oSecPhi[i]                = BandedReal(sec.phi);
}

/**
 * @brief ★ THE POSITIVE CONTROL. Fixed name; do not rename. Identical body,
 * ONE deliberate FP64 operation scaling operand `A`'s leading limb via a
 * KERNEL ARGUMENT (`poison`), so it cannot be constant-folded away.
 */
AETHER_KERNEL() void rsqrtAtanPrimFp64InjectedKernel(const BandedReal* __restrict__ a,
    const BandedReal* __restrict__ b, int n, BandedReal* __restrict__ oRsqrt,
    BandedReal* __restrict__ oSqrt, BandedReal* __restrict__ oRsqrtCube,
    BandedReal* __restrict__ oSecT, BandedReal* __restrict__ oSecPhi, double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    Band A              = static_cast<Band>(a[i]);
    if (A.hi <= 0.0f)
        A = bd::neg(A);
    A.hi         = A.hi * spoilt;
    const Band B = static_cast<Band>(b[i]);

    oRsqrt[i]     = BandedReal(bd::rsqrt(A));
    oSqrt[i]      = BandedReal(bd::sqrt_(A));
    oRsqrtCube[i] = BandedReal(bd::rsqrtCube(A));

    const bd::AtanSector sec = bd::atanSector(std::fabs(B.hi), std::fabs(A.hi) + 1.0f);
    oSecT[i]                 = BandedReal(sec.t);
    oSecPhi[i]                = BandedReal(sec.phi);
}

// -------------------------------------------------------------------------
//  The audit-only device row. Mirrors
//  `TheAuditedAccumFoldKernelAgreesWithTheHostAndItsControlDoesNotOnDevice`'s
//  own precedent (same file, same rationale): the SASS audit needs a header-only
//  unit to be INSTANTIATED before there is anything to scan, and the
//  known-answer check proves the audited body actually ran the family rather
//  than something the optimiser collapsed.
// -------------------------------------------------------------------------
TEST_F(BandLevers,
    TheAuditedRsqrtAtanPrimKernelAgreesWithTheHostAndItsControlDoesNotOnDevice)
{
    const int n = 512;
    std::vector<BandedReal> ha(static_cast<std::size_t>(n)), hb(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) {
        ha[static_cast<std::size_t>(i)] = BandedReal::fromDouble(
            std::exp(0.017 * i) * (1.0 + 0.5 * std::sin(0.29 * i)));
        hb[static_cast<std::size_t>(i)]
            = BandedReal::fromDouble(1.0 + 3.0 * std::fabs(std::sin(0.61 * i + 0.4)));
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(BandedReal);
    BandedReal *da, *db, *oRsqrt, *oSqrt, *oCube, *oSecT, *oSecPhi;
    BandedReal *iRsqrt, *iSqrt, *iCube, *iSecT, *iSecPhi;
    auto alloc = [&](BandedReal** p) { ASSERT_EQ(cudaMalloc(p, bytes), cudaSuccess); };
    alloc(&da);
    alloc(&db);
    alloc(&oRsqrt);
    alloc(&oSqrt);
    alloc(&oCube);
    alloc(&oSecT);
    alloc(&oSecPhi);
    alloc(&iRsqrt);
    alloc(&iSqrt);
    alloc(&iCube);
    alloc(&iSecT);
    alloc(&iSecPhi);
    ASSERT_EQ(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    rsqrtAtanPrimKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
        da, db, n, oRsqrt, oSqrt, oCube, oSecT, oSecPhi);
    aether::cuda::checkLastLaunch("rsqrtAtanPrimKernel");
    rsqrtAtanPrimFp64InjectedKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
        da, db, n, iRsqrt, iSqrt, iCube, iSecT, iSecPhi, 0.0);
    aether::cuda::checkLastLaunch("rsqrtAtanPrimFp64InjectedKernel");

    std::vector<BandedReal> gRsqrt(static_cast<std::size_t>(n)), gSqrt(static_cast<std::size_t>(n)),
        gCube(static_cast<std::size_t>(n)), gSecT(static_cast<std::size_t>(n)),
        gSecPhi(static_cast<std::size_t>(n));
    std::vector<BandedReal> jRsqrt(static_cast<std::size_t>(n));
    auto back = [&](std::vector<BandedReal>& h, BandedReal* d) {
        ASSERT_EQ(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    };
    back(gRsqrt, oRsqrt);
    back(gSqrt, oSqrt);
    back(gCube, oCube);
    back(gSecT, oSecT);
    back(gSecPhi, oSecPhi);
    back(jRsqrt, iRsqrt);
    cudaFree(da);
    cudaFree(db);
    cudaFree(oRsqrt);
    cudaFree(oSqrt);
    cudaFree(oCube);
    cudaFree(oSecT);
    cudaFree(oSecPhi);
    cudaFree(iRsqrt);
    cudaFree(iSqrt);
    cudaFree(iCube);
    cudaFree(iSecT);
    cudaFree(iSecPhi);

    std::size_t bad = 0, trivial = 0, injSame = 0;
    for (int i = 0; i < n; i++) {
        Band A = static_cast<Band>(ha[static_cast<std::size_t>(i)]);
        if (A.hi <= 0.0f)
            A = bd::neg(A);
        const Band B = static_cast<Band>(hb[static_cast<std::size_t>(i)]);

        const BandedReal wRsqrt = BandedReal(bd::rsqrt(A));
        const BandedReal wSqrt  = BandedReal(bd::sqrt_(A));
        const BandedReal wCube  = BandedReal(bd::rsqrtCube(A));
        const bd::AtanSector sec
            = bd::atanSector(std::fabs(B.hi), std::fabs(A.hi) + 1.0f);
        const BandedReal wSecT   = BandedReal(sec.t);
        const BandedReal wSecPhi = BandedReal(sec.phi);

        if (gRsqrt[static_cast<std::size_t>(i)].toBits() != wRsqrt.toBits())
            bad++;
        if (gSqrt[static_cast<std::size_t>(i)].toBits() != wSqrt.toBits())
            bad++;
        if (gCube[static_cast<std::size_t>(i)].toBits() != wCube.toBits())
            bad++;
        if (gSecT[static_cast<std::size_t>(i)].toBits() != wSecT.toBits())
            bad++;
        if (gSecPhi[static_cast<std::size_t>(i)].toBits() != wSecPhi.toBits())
            bad++;
        if (gRsqrt[static_cast<std::size_t>(i)].isZero())
            trivial++;
        // poison = 0 -> the injected kernel's scale factor is exactly 1.0f,
        // so the two agree numerically; the control's job is to carry an
        // FP64 INSTRUCTION, not a different answer.
        if (jRsqrt[static_cast<std::size_t>(i)].toBits()
            == gRsqrt[static_cast<std::size_t>(i)].toBits())
            injSame++;
    }
    std::printf("[BandLevers/device] audited rsqrt/atan-sector prim over %d "
                "samples: %zu mismatches, %zu trivially-zero rsqrt results, "
                "%zu injected-arm agreements\n",
        n, bad, trivial, injSame);
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(injSame, static_cast<std::size_t>(n));
    EXPECT_LT(trivial, static_cast<std::size_t>(n) / 4)
        << "most of the audited kernel's rsqrt outputs are zero, so the SASS "
           "audit is scanning a kernel the optimiser may have hollowed out";
}

} // namespace BandLeversTest
} // namespace aether_tests
