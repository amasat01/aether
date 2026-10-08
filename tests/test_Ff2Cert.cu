// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Ff2Cert.cu
 * @brief `Ff2` (w=45 rung, df64) certification (CUDA mode): the host
 *        DD-oracle certs (shared `test_Ff2Cert_common.h`) plus the 0-FP64
 *        device-parity cert.
 *
 * `DevicePrimitivesMatchHostExactly` covers this; a device-vs-host parity
 * check for the recurrence step (`det::recStepP`) is not covered here.
 */

#include "test_Ff2Cert_common.h"

#include "aether/backend/cuda/Launch.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace aether_tests {
namespace Ff2CertTest {

using aether::banded::Band;

static std::int64_t ulpDist(double a, double b)
{
    if (std::isnan(a) || std::isnan(b))
        return 0x7FFFFFFFFFFFFFFFLL;
    if (a == b)
        return 0;
    std::uint64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if (ia >> 63)
        ia = 0x8000000000000000ULL - ia;
    if (ib >> 63)
        ib = 0x8000000000000000ULL - ib;
    return static_cast<std::int64_t>(ia > ib ? ia - ib : ib - ia);
}

// =========================================================================
//  Device parity: primitives (add/sub/mul/fma/div/sqrt/rsqrt) on device.
// =========================================================================
AETHER_KERNEL() void ff2PrimKernel(const BandedReal* __restrict__ a,
    const BandedReal* __restrict__ b, const BandedReal* __restrict__ c, int n,
    BandedReal* __restrict__ oAdd, BandedReal* __restrict__ oSub, BandedReal* __restrict__ oMul,
    BandedReal* __restrict__ oFma, BandedReal* __restrict__ oDiv, BandedReal* __restrict__ oSqrt,
    BandedReal* __restrict__ oRsqrt)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    Ff2 A = static_cast<Band>(a[i]);
    Ff2 B = static_cast<Band>(b[i]);
    Ff2 C = static_cast<Band>(c[i]);
    Ff2 P = A;
    if (P.hi < 0.0f)
        P = -P;
    oAdd[i]   = (A + B).toBandedReal();
    oSub[i]   = (A - B).toBandedReal();
    oMul[i]   = (A * B).toBandedReal();
    oFma[i]   = aether::banded::fma(A, B, C).toBandedReal();
    oDiv[i]   = (A / B).toBandedReal();
    oSqrt[i]  = aether::banded::sqrt(P).toBandedReal();
    oRsqrt[i] = aether::banded::rsqrt(P).toBandedReal();
}

/**
 * @brief The positive control for the Ff-cores 0-FP64 SASS audit
 * (`tests/sass/check_band_fp64free.sh`'s "Subject 3"). Fixed name; do not
 * rename, do not dead-strip. The identical body as `ff2PrimKernel` with one
 * deliberate FP64 operation on a kernel argument (`poison`, so it cannot be
 * constant-folded away) — @see `test_BandedReal.cu`'s `bandFp64InjectedKernel`
 * for the identical idiom this mirrors.
 */
AETHER_KERNEL() void ff2PrimFp64InjectedKernel(const BandedReal* __restrict__ a,
    const BandedReal* __restrict__ b, const BandedReal* __restrict__ c, int n,
    BandedReal* __restrict__ oAdd, BandedReal* __restrict__ oSub, BandedReal* __restrict__ oMul,
    BandedReal* __restrict__ oFma, BandedReal* __restrict__ oDiv, BandedReal* __restrict__ oSqrt,
    BandedReal* __restrict__ oRsqrt, double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band a0      = static_cast<Band>(a[i]);
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    Ff2 A               = Band{ a0.hi * spoilt, a0.lo, a0.tail };
    Ff2 B = static_cast<Band>(b[i]);
    Ff2 C = static_cast<Band>(c[i]);
    Ff2 P = A;
    if (P.hi < 0.0f)
        P = -P;
    oAdd[i]   = (A + B).toBandedReal();
    oSub[i]   = (A - B).toBandedReal();
    oMul[i]   = (A * B).toBandedReal();
    oFma[i]   = aether::banded::fma(A, B, C).toBandedReal();
    oDiv[i]   = (A / B).toBandedReal();
    oSqrt[i]  = aether::banded::sqrt(P).toBandedReal();
    oRsqrt[i] = aether::banded::rsqrt(P).toBandedReal();
}

TEST_F(Ff2Cert, DevicePrimitivesMatchHostExactly)
{
    const int n = isMinimalMode() ? 2000 : 20000;
    std::mt19937_64 rng(0x0FF2C0DEu ^ 4u);
    std::vector<BandedReal> ha(static_cast<std::size_t>(n)), hb(static_cast<std::size_t>(n)),
        hc(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) {
        ha[static_cast<std::size_t>(i)] = BandedReal::fromDouble(mag(rng, -4, 4));
        hb[static_cast<std::size_t>(i)] = BandedReal::fromDouble(mag(rng, -4, 4));
        hc[static_cast<std::size_t>(i)] = BandedReal::fromDouble(mag(rng, -4, 4));
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(BandedReal);
    BandedReal *da, *db, *dc, *oAdd, *oSub, *oMul, *oFma, *oDiv, *oSqrt, *oRsqrt;
    auto alloc = [&](BandedReal** p) { ASSERT_EQ(cudaMalloc(p, bytes), cudaSuccess); };
    alloc(&da);
    alloc(&db);
    alloc(&dc);
    alloc(&oAdd);
    alloc(&oSub);
    alloc(&oMul);
    alloc(&oFma);
    alloc(&oDiv);
    alloc(&oSqrt);
    alloc(&oRsqrt);
    ASSERT_EQ(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dc, hc.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    ff2PrimKernel<<<(n + 127) / 128, 128>>>(
        da, db, dc, n, oAdd, oSub, oMul, oFma, oDiv, oSqrt, oRsqrt);
    aether::cuda::checkLastLaunch("ff2PrimKernel");

    std::vector<BandedReal> rAdd(static_cast<std::size_t>(n)), rSub(static_cast<std::size_t>(n)),
        rMul(static_cast<std::size_t>(n)), rFma(static_cast<std::size_t>(n)),
        rDiv(static_cast<std::size_t>(n)), rSqrt(static_cast<std::size_t>(n)),
        rRsqrt(static_cast<std::size_t>(n));
    auto back = [&](std::vector<BandedReal>& h, BandedReal* d) {
        ASSERT_EQ(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    };
    back(rAdd, oAdd);
    back(rSub, oSub);
    back(rMul, oMul);
    back(rFma, oFma);
    back(rDiv, oDiv);
    back(rSqrt, oSqrt);
    back(rRsqrt, oRsqrt);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::int64_t worstUlp = 0;
    double worstBitsDev   = 100;
    for (int i = 0; i < n; i++) {
        Ff2 A = static_cast<Band>(ha[static_cast<std::size_t>(i)]);
        Ff2 B = static_cast<Band>(hb[static_cast<std::size_t>(i)]);
        Ff2 C = static_cast<Band>(hc[static_cast<std::size_t>(i)]);
        Ff2 P = A;
        if (P.hi < 0.0f)
            P = -P;
        auto cmp = [&](const BandedReal& dev, Ff2 host) {
            worstUlp = std::max(worstUlp, ulpDist(dev.toDouble(), host.toDouble()));
        };
        cmp(rAdd[static_cast<std::size_t>(i)], A + B);
        cmp(rSub[static_cast<std::size_t>(i)], A - B);
        cmp(rMul[static_cast<std::size_t>(i)], A * B);
        cmp(rFma[static_cast<std::size_t>(i)], aether::banded::fma(A, B, C));
        cmp(rDiv[static_cast<std::size_t>(i)], A / B);
        cmp(rSqrt[static_cast<std::size_t>(i)], aether::banded::sqrt(P));
        cmp(rRsqrt[static_cast<std::size_t>(i)], aether::banded::rsqrt(P));
        ddref::DD dA = ff2ExactDf(A), dB = ff2ExactDf(B);
        worstBitsDev = std::min(worstBitsDev,
            effBits(rMul[static_cast<std::size_t>(i)].toDouble(), ddref::toDouble(ddref::mul(dA, dB))));
    }
    std::printf("[DevicePrimitivesMatchHostExactly] n=%d device-vs-host worst=%ld ulp; "
                "device mul vs oracle worst=%.2f bits\n",
        n, static_cast<long>(worstUlp), worstBitsDev);
    EXPECT_LE(worstUlp, 2) << "device 0-FP64 Ff2 ops must match host IEEE";
    EXPECT_GE(worstBitsDev, 45.0) << "device mul must certify >= 45 bits";

    for (auto p : { da, db, dc, oAdd, oSub, oMul, oFma, oDiv, oSqrt, oRsqrt })
        cudaFree(p);
}

} // namespace Ff2CertTest
} // namespace aether_tests
