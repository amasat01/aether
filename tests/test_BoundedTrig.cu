// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BoundedTrig.cu
 * @brief ULP-parity test for the zero-stack FP64 sin/cos/sincos
 *        device path in `aether/math/detail/BoundedTrig.h`. The sweep
 *        generator (`makeSweep()`, including the 1e10/1e12/1e15 slow-path
 *        coverage) and the ULP-distance metric (`ulpDistD`) gate at <= 1
 *        ULP (`kUlpTol`), matching libdevice's documented worst case.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace BoundedTrigTest {

/* ULP distance for FP64. NaN sentinel = max int64. Equal values return 0. */
static int64_t ulpDistD(double a, double b)
{
    if (std::isnan(a) || std::isnan(b))
        return 0x7FFFFFFFFFFFFFFFLL;
    if (a == b)
        return 0;
    uint64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if (ia >> 63)
        ia = 0x8000000000000000ULL - ia;
    if (ib >> 63)
        ib = 0x8000000000000000ULL - ib;
    return static_cast<int64_t>(ia > ib ? ia - ib : ib - ia);
}

/* Argument sweep covering boundary cases, full period, and larger in-range
 * values. Length kept moderate so each test runs in < 1 s. */
static std::vector<double> makeSweep()
{
    std::vector<double> v;
    v.reserve(2000);

    /* Boundaries + named angles. */
    for (double a :
        { 0.0, 1.0e-300, 1.0e-15, 1.0e-8, 0.1, 0.5, 1.0, M_PI_4, M_PI_2, 3 * M_PI_4, M_PI, 5 * M_PI_4, 3 * M_PI_2, 7 * M_PI_4,
            2 * M_PI }) {
        v.push_back(a);
        v.push_back(-a);
    }

    /* Dense sweep across [-2pi, +2pi] -- exercises every quadrant + sign. */
    constexpr int N = 1024;
    for (int i = -N; i <= N; ++i) {
        v.push_back(2 * M_PI * static_cast<double>(i) / N);
    }

    /* Larger in-range values up to ~1e7 (a representative worst-case PM
     * angle for long-duration propagations is ~2e7 rad). Fast-path
     * Cody-Waite handles everything below 2^31 ~= 2.15e9. */
    for (double scale : { 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 }) {
        for (int k = 0; k < 17; ++k) {
            const double offset = k * 0.37 - 3.0; /* arbitrary phase */
            v.push_back(scale + offset);
            v.push_back(-scale - offset);
        }
    }

    /* Slow-path coverage (|x| > 2^31, statically-unrolled Payne-Hanek).
     * The reduction is full-range (test_TrigTotal covers up to DBL_MAX,
     * Inf and NaN). These points gate
     * the slow path at 1 ULP, matching libdevice's slow path. */
    for (double scale : { 1e10, 1e12, 1e15 }) {
        for (int k = 0; k < 17; ++k) {
            const double offset = k * 0.37 - 3.0;
            v.push_back(scale + offset);
            v.push_back(-scale - offset);
        }
    }

    return v;
}

/* The cwSin/cwCos dispatch fast Cody-Waite for
 * |x| < 2^31 (the common case) and statically-unrolled Payne-Hanek for
 * the rare large-argument tail. Both feed a unified Horner polynomial
 * with per-coefficient FSEL. Matches libdevice's 1-ULP worst case
 * across the full sweep. */
constexpr int64_t kUlpTol = 1;

__global__ void kernelSin(const double* __restrict__ in, double* __restrict__ out, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    out[i] = aether::math::sin(in[i]);
}

__global__ void kernelCos(const double* __restrict__ in, double* __restrict__ out, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    out[i] = aether::math::cos(in[i]);
}

__global__ void kernelSinCos(const double* __restrict__ in, double* __restrict__ outSin, double* __restrict__ outCos, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    double s, c;
    aether::math::sincos(in[i], &s, &c);
    outSin[i] = s;
    outCos[i] = c;
}

static void deviceParityTest(const std::string& label, bool doSin, bool doCos, bool doSinCos)
{
    const auto sweep = makeSweep();
    const int N       = static_cast<int>(sweep.size());

    double *dIn, *dSin, *dCos;
    cudaMalloc(&dIn, N * sizeof(double));
    cudaMalloc(&dSin, N * sizeof(double));
    cudaMalloc(&dCos, N * sizeof(double));
    cudaMemcpy(dIn, sweep.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    constexpr int BS = 128;
    const int grid    = (N + BS - 1) / BS;

    std::vector<double> hSin(N, 0.0), hCos(N, 0.0);

    if (doSin) {
        kernelSin<<<grid, BS>>>(dIn, dSin, N);
        cudaMemcpy(hSin.data(), dSin, N * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (doCos) {
        kernelCos<<<grid, BS>>>(dIn, dCos, N);
        cudaMemcpy(hCos.data(), dCos, N * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (doSinCos) {
        kernelSinCos<<<grid, BS>>>(dIn, dSin, dCos, N);
        cudaMemcpy(hSin.data(), dSin, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hCos.data(), dCos, N * sizeof(double), cudaMemcpyDeviceToHost);
    }

    int64_t maxSinUlp = 0, maxCosUlp = 0;
    double worstSinX = 0, worstCosX = 0;
    for (int i = 0; i < N; ++i) {
        if (doSin || doSinCos) {
            const int64_t u = ulpDistD(hSin[i], std::sin(sweep[i]));
            if (u > maxSinUlp) {
                maxSinUlp = u;
                worstSinX = sweep[i];
            }
        }
        if (doCos || doSinCos) {
            const int64_t u = ulpDistD(hCos[i], std::cos(sweep[i]));
            if (u > maxCosUlp) {
                maxCosUlp = u;
                worstCosX = sweep[i];
            }
        }
    }

    cudaFree(dIn);
    cudaFree(dSin);
    cudaFree(dCos);

    if (doSin || doSinCos)
        EXPECT_LE(maxSinUlp, kUlpTol) << label << " sin max ULP=" << maxSinUlp << " at x=" << worstSinX;
    if (doCos || doSinCos)
        EXPECT_LE(maxCosUlp, kUlpTol) << label << " cos max ULP=" << maxCosUlp << " at x=" << worstCosX;
}

TEST(BoundedTrigTest, DeviceAetherSin_MatchesStdSin)
{
    deviceParityTest("device aether::math::sin", true, false, false);
}

TEST(BoundedTrigTest, DeviceAetherCos_MatchesStdCos)
{
    deviceParityTest("device aether::math::cos", false, true, false);
}

TEST(BoundedTrigTest, DeviceAetherSinCos_MatchesStd)
{
    deviceParityTest("device aether::math::sincos", false, false, true);
}

} // namespace BoundedTrigTest
} // namespace aether_tests
