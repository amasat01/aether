// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathSass.cu
 * @brief Per-op SASS audit subjects: one isolated single-op kernel plus its
 *        FP64-injected positive control for each live Band math op (scanned
 *        by `tests/sass/check_band_fp64free.sh`).
 *
 * ★ Kernel SHAPE is raw `Band` in/out over flat `float[3*n]` arrays, NOT
 * `BandedReal`'s 8-byte codec — so the measured TOTAL/FP32 counts are
 * directly comparable to the frozen sentinels (`sub` 30, `mul` 33, `fma` 54
 * FP32): a codec crossing here would inflate every count by the pack/unpack
 * cost and the comparison would no longer be apples-to-apples. Fixed
 * names; do not rename, do not dead-strip (@see `bandChainKernel`'s
 * identical warning in `test_BandedReal.cu`).
 *
 * `asin`'s two branches (`asinReducedRaw`'s `|x|<=0.5` vs `|x|>0.5` legs)
 * CANNOT be separate SASS subjects here. A compiled kernel's SASS is a
 * STATIC property of the function — BOTH branches are always present in
 * `bandmathAsinKernel`'s disassembly regardless of what data
 * `TEST(BandMathSass, ...)` below happens to launch it with, so a
 * data-only split would produce BYTE-IDENTICAL disassembly for both
 * "subjects". This audits the aggregate kernel (both branches compiled
 * together) instead of fabricating a distinction the SASS scan cannot see.
 * `acos` shares the identical `asinReducedRaw` body and inherits the same
 * note.
 *
 * GPU EXECUTION NOTE: identical to `test_BandedReal.cu`'s own — this file
 * is built and its tests listed here; the CUDA suite that actually
 * launches these kernels runs separately. The device compile is still a
 * genuine check on its own.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"
#include "aether/backend/cuda/Launch.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace bandmath_sass {

using aether::banded::Band;
namespace bd = aether::banded::detail;

__device__ inline Band ld(const float* p, int i)
{
    return Band{ p[3 * i], p[3 * i + 1], p[3 * i + 2] };
}
__device__ inline void st(float* q, int i, Band b)
{
    q[3 * i]     = b.hi;
    q[3 * i + 1] = b.lo;
    q[3 * i + 2] = b.tail;
}

// ---- one isolated kernel per live op, plus its FP64-injected control ----
// `poison` (a KERNEL ARGUMENT, so it cannot be constant-folded) perturbs
// `a` by one deliberate FP64 multiply before the certified op runs —
// identical positive-control shape to `bandFp64InjectedKernel`.

#define BANDMATH_SASS_UNARY(NAME, EXPR)                                                   \
    AETHER_KERNEL() void NAME##Kernel(float* __restrict__ out, const float* __restrict__ a, int n) \
    {                                                                                      \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);             \
        if (i >= n)                                                                        \
            return;                                                                        \
        const Band x = ld(a, i);                                                           \
        st(out, i, (EXPR));                                                                \
    }                                                                                       \
    AETHER_KERNEL() void NAME##Fp64InjectedKernel(                                         \
        float* __restrict__ out, const float* __restrict__ a, int n, double poison)        \
    {                                                                                       \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);             \
        if (i >= n)                                                                        \
            return;                                                                        \
        const Band x0       = ld(a, i);                                                    \
        const float spoilt  = static_cast<float>(poison * 2.0 + 1.0);                      \
        const Band x        = Band{ x0.hi * spoilt, x0.lo, x0.tail };                      \
        st(out, i, (EXPR));                                                                \
    }

#define BANDMATH_SASS_BINARY(NAME, EXPR)                                                            \
    AETHER_KERNEL() void NAME##Kernel(                                                              \
        float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, int n)   \
    {                                                                                                \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);                       \
        if (i >= n)                                                                                  \
            return;                                                                                  \
        const Band x = ld(a, i);                                                                     \
        const Band y = ld(b, i);                                                                     \
        st(out, i, (EXPR));                                                                          \
    }                                                                                                \
    AETHER_KERNEL() void NAME##Fp64InjectedKernel(float* __restrict__ out,                           \
        const float* __restrict__ a, const float* __restrict__ b, int n, double poison)              \
    {                                                                                                 \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);                       \
        if (i >= n)                                                                                   \
            return;                                                                                   \
        const Band x0       = ld(a, i);                                                              \
        const float spoilt  = static_cast<float>(poison * 2.0 + 1.0);                                \
        const Band x        = Band{ x0.hi * spoilt, x0.lo, x0.tail };                                \
        const Band y        = ld(b, i);                                                              \
        st(out, i, (EXPR));                                                                           \
    }

BANDMATH_SASS_UNARY(bandmathAbs, bd::abs(x))
BANDMATH_SASS_BINARY(bandmathCopysign, bd::copysign(x, y))
BANDMATH_SASS_BINARY(bandmathFmax, bd::fmax(x, y))
BANDMATH_SASS_BINARY(bandmathFmin, bd::fmin(x, y))
BANDMATH_SASS_BINARY(bandmathAdd, bd::add(x, y))
BANDMATH_SASS_BINARY(bandmathSub, bd::sub(x, y))
BANDMATH_SASS_BINARY(bandmathMul, bd::mul(x, y))
// exp/log unary, pow binary.
BANDMATH_SASS_UNARY(bandmathExp, bd::exp(x))
BANDMATH_SASS_UNARY(bandmathLog, bd::log(x))
BANDMATH_SASS_BINARY(bandmathPow, bd::pow(x, y))
// `rsqrt` audits `BandRoot.h::rsqrtIeee`, not `RsqrtCore.h::rsqrt`
// directly -- the same body `BandedFacade::rsqrt` forwards to.
BANDMATH_SASS_UNARY(bandmathSqrt, bd::sqrt_(x))
BANDMATH_SASS_UNARY(bandmathRsqrt, bd::rsqrtIeee(x))
BANDMATH_SASS_UNARY(bandmathRsqrtCube, bd::rsqrtCube(x))
BANDMATH_SASS_UNARY(bandmathCbrt, bd::cbrt(x))
BANDMATH_SASS_BINARY(bandmathHypot, bd::hypot(x, y))
// sin/cos additions.
BANDMATH_SASS_UNARY(bandmathSin, bd::sin(x))
BANDMATH_SASS_UNARY(bandmathCos, bd::cos(x))

// floor/ceil/round/trunc (unary), fdim/fmod (binary), same macro shape.
BANDMATH_SASS_UNARY(bandmathFloor, bd::floor(x))
BANDMATH_SASS_UNARY(bandmathCeil, bd::ceil(x))
BANDMATH_SASS_UNARY(bandmathRound, bd::round(x))
BANDMATH_SASS_UNARY(bandmathTrunc, bd::trunc(x))
BANDMATH_SASS_BINARY(bandmathFdim, bd::fdim(x, y))
BANDMATH_SASS_BINARY(bandmathFmod, bd::fmod(x, y))
// `atan`/`atan2` are SEPARATE subjects -- atan2's general two-operand
// reduction vs atan's atan2(x,1) special case, which the compiler can
// partially constant-fold (predicted FP32 counts 480/490). `asin`/`acos`
// each audit the AGGREGATE kernel (both branches compiled together) --
// see the file header for why the harness cannot show the two branches
// as separate subjects.
BANDMATH_SASS_UNARY(bandmathAtan, bd::atan(x))
BANDMATH_SASS_BINARY(bandmathAtan2, bd::atan2(x, y))
BANDMATH_SASS_UNARY(bandmathAsin, bd::asin(x))
BANDMATH_SASS_UNARY(bandmathAcos, bd::acos(x))

#undef BANDMATH_SASS_UNARY
#undef BANDMATH_SASS_BINARY

// `sincos` returns TWO outputs (Band&, Band&), which does not fit either
// macro's shape (one Band in, one Band out) -- a bespoke pair, same
// poison-injection idiom as `BANDMATH_SASS_UNARY`'s own Fp64Injected arm.
AETHER_KERNEL() void bandmathSincosKernel(
    float* __restrict__ outS, float* __restrict__ outC, const float* __restrict__ a, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band x = ld(a, i);
    Band       s, c;
    bd::sincos(x, s, c);
    st(outS, i, s);
    st(outC, i, c);
}
AETHER_KERNEL() void bandmathSincosFp64InjectedKernel(float* __restrict__ outS,
    float* __restrict__ outC, const float* __restrict__ a, int n, double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band  x0     = ld(a, i);
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    const Band  x       = Band{ x0.hi * spoilt, x0.lo, x0.tail };
    Band        s, c;
    bd::sincos(x, s, c);
    st(outS, i, s);
    st(outC, i, c);
}
// The FIRST ternary SASS subject (`fma`), same shape as the unary/binary
// macros above, one extra operand array.
#define BANDMATH_SASS_TERNARY(NAME, EXPR)                                                                    \
    AETHER_KERNEL() void NAME##Kernel(float* __restrict__ out, const float* __restrict__ a,                  \
        const float* __restrict__ b, const float* __restrict__ c, int n)                                     \
    {                                                                                                         \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);                               \
        if (i >= n)                                                                                           \
            return;                                                                                          \
        const Band x = ld(a, i);                                                                             \
        const Band y = ld(b, i);                                                                             \
        const Band z = ld(c, i);                                                                             \
        st(out, i, (EXPR));                                                                                   \
    }                                                                                                         \
    AETHER_KERNEL() void NAME##Fp64InjectedKernel(float* __restrict__ out, const float* __restrict__ a,      \
        const float* __restrict__ b, const float* __restrict__ c, int n, double poison)                      \
    {                                                                                                         \
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);                               \
        if (i >= n)                                                                                           \
            return;                                                                                          \
        const Band x0       = ld(a, i);                                                                      \
        const float spoilt  = static_cast<float>(poison * 2.0 + 1.0);                                        \
        const Band x        = Band{ x0.hi * spoilt, x0.lo, x0.tail };                                        \
        const Band y        = ld(b, i);                                                                      \
        const Band z        = ld(c, i);                                                                      \
        st(out, i, (EXPR));                                                                                   \
    }

BANDMATH_SASS_TERNARY(bandmathFma, bd::fma(x, y, z))

#undef BANDMATH_SASS_TERNARY

// =========================================================================
//  Instantiation + KNOWN-ANSWER check (same two reasons bandChainKernel's
//  own launch exists for: the SASS audit needs an instantiation to scan,
//  and an audit of a kernel the optimiser folded to nothing is an audit of
//  nothing). NOT executed as part of this check -- only built and listed.
// =========================================================================
TEST(BandMathSass, KernelsAreLiveAndComputeTheClaimedOp)
{
    constexpr int n     = 64;
    constexpr int kBlock = 32;
    std::vector<float> av(3 * n), bv(3 * n);
    for (int i = 0; i < n; i++) {
        const float m       = 1.0f + 0.01f * static_cast<float>(i);
        av[3 * i]           = m;
        av[3 * i + 1]       = 0.0f;
        av[3 * i + 2]       = 0.0f;
        bv[3 * i]           = 2.0f - 0.003f * static_cast<float>(i);
        bv[3 * i + 1]       = 0.0f;
        bv[3 * i + 2]       = 0.0f;
    }
    float *dA = nullptr, *dB = nullptr, *dOut = nullptr, *dInj = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, sizeof(float) * 3 * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, sizeof(float) * 3 * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, sizeof(float) * 3 * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dInj, sizeof(float) * 3 * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, av.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, bv.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice), cudaSuccess);

    const int grid = (n + kBlock - 1) / kBlock;

    bandmathAbsKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathAbsKernel");
    bandmathAbsFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAbsFp64InjectedKernel");

    bandmathCopysignKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathCopysignKernel");
    bandmathCopysignFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathCopysignFp64InjectedKernel");

    bandmathFmaxKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathFmaxKernel");
    bandmathFmaxFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFmaxFp64InjectedKernel");

    bandmathFminKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathFminKernel");
    bandmathFminFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFminFp64InjectedKernel");

    bandmathAddKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathAddKernel");
    bandmathAddFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAddFp64InjectedKernel");

    bandmathSubKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathSubKernel");
    bandmathSubFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathSubFp64InjectedKernel");

    bandmathMulKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathMulKernel");
    bandmathMulFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathMulFp64InjectedKernel");

    bandmathExpKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathExpKernel");
    bandmathExpFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathExpFp64InjectedKernel");

    bandmathLogKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathLogKernel");
    bandmathLogFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathLogFp64InjectedKernel");

    bandmathPowKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathPowKernel");
    bandmathPowFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathPowFp64InjectedKernel");
    // sqrt/rsqrt/rsqrtCube/cbrt/hypot.
    bandmathSqrtKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathSqrtKernel");
    bandmathSqrtFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathSqrtFp64InjectedKernel");

    bandmathRsqrtKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathRsqrtKernel");
    bandmathRsqrtFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathRsqrtFp64InjectedKernel");

    bandmathRsqrtCubeKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathRsqrtCubeKernel");
    bandmathRsqrtCubeFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathRsqrtCubeFp64InjectedKernel");

    bandmathCbrtKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathCbrtKernel");
    bandmathCbrtFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathCbrtFp64InjectedKernel");

    bandmathHypotKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathHypotKernel");
    bandmathHypotFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathHypotFp64InjectedKernel");
    // atan/atan2/asin/acos.
    bandmathAtanKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathAtanKernel");
    bandmathAtanFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAtanFp64InjectedKernel");

    bandmathAtan2Kernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathAtan2Kernel");
    bandmathAtan2Fp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAtan2Fp64InjectedKernel");

    bandmathAsinKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathAsinKernel");
    bandmathAsinFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAsinFp64InjectedKernel");

    bandmathAcosKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathAcosKernel");
    bandmathAcosFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathAcosFp64InjectedKernel");

    // sin/cos.
    bandmathSinKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathSinKernel");
    bandmathSinFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathSinFp64InjectedKernel");

    bandmathCosKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathCosKernel");
    bandmathCosFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathCosFp64InjectedKernel");

    float *dOutC = nullptr, *dInjC = nullptr;
    ASSERT_EQ(cudaMalloc(&dOutC, sizeof(float) * 3 * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dInjC, sizeof(float) * 3 * n), cudaSuccess);
    bandmathSincosKernel<<<grid, kBlock>>>(dOut, dOutC, dA, n);
    aether::cuda::checkLastLaunch("bandmathSincosKernel");
    bandmathSincosFp64InjectedKernel<<<grid, kBlock>>>(dInj, dInjC, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathSincosFp64InjectedKernel");
    cudaFree(dOutC);
    cudaFree(dInjC);
    // floor/ceil/round/trunc/fdim/fmod.
    bandmathFloorKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathFloorKernel");
    bandmathFloorFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFloorFp64InjectedKernel");

    bandmathCeilKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathCeilKernel");
    bandmathCeilFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathCeilFp64InjectedKernel");

    bandmathRoundKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathRoundKernel");
    bandmathRoundFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathRoundFp64InjectedKernel");

    bandmathTruncKernel<<<grid, kBlock>>>(dOut, dA, n);
    aether::cuda::checkLastLaunch("bandmathTruncKernel");
    bandmathTruncFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathTruncFp64InjectedKernel");

    bandmathFdimKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathFdimKernel");
    bandmathFdimFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFdimFp64InjectedKernel");

    bandmathFmodKernel<<<grid, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandmathFmodKernel");
    bandmathFmodFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFmodFp64InjectedKernel");

    bandmathFmaKernel<<<grid, kBlock>>>(dOut, dA, dB, dA, n);
    aether::cuda::checkLastLaunch("bandmathFmaKernel");
    bandmathFmaFp64InjectedKernel<<<grid, kBlock>>>(dInj, dA, dB, dA, n, 0.0);
    aether::cuda::checkLastLaunch("bandmathFmaFp64InjectedKernel");

    std::vector<float> got(3 * n);
    ASSERT_EQ(cudaMemcpy(got.data(), dOut, sizeof(float) * 3 * n, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    cudaFree(dInj);

    // Known-answer: bandmathMulKernel's last launch left `dOut` holding
    // mul(a,b); a coarse sanity check that SOMETHING was computed (not the
    // certification — that is BandMathCert's job).
    bool anyNonzero = false;
    for (float v : got)
        if (v != 0.0f)
            anyNonzero = true;
    EXPECT_TRUE(anyNonzero) << "SASS subject kernels computed an all-zero output — "
                                "the optimiser may have folded the chain away";
}

} // namespace bandmath_sass
} // namespace aether_tests
