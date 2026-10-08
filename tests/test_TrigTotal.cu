// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Totality of FP64 `aether::math::sin`/`cos`/`sincos` on the device route
// (`aether/math/detail/BoundedTrig.h`): every double — zeros, subnormals,
// Inf, NaN, the fast/slow reduction cut at 2^31, and huge arguments up to
// DBL_MAX — within `kUlpTol` of the host libm, NaN exactly where libm is
// NaN. test_TrigTotal.cpp is the host-route twin; the corpus and scoring
// live in test_TrigTotal_common.h.

#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "test_TrigTotal_common.h"

namespace aether_tests {
namespace TrigTotalTest {

__global__ void kernelTrig(const double* __restrict__ in, double* __restrict__ outSin, double* __restrict__ outCos,
    double* __restrict__ outSc, double* __restrict__ outCc, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double x = in[i];
    outSin[i]      = aether::math::sin(x);
    outCos[i]      = aether::math::cos(x);
    double s, c;
    aether::math::sincos(x, &s, &c);
    outSc[i] = s;
    outCc[i] = c;
}

constexpr bool kUsePinned = true;

static Columns evaluate()
{
    Columns r;
    r.xs         = makeCorpus();
    const int n  = static_cast<int>(r.xs.size());
    const auto b = n * sizeof(double);
    double* d    = nullptr;
    EXPECT_EQ(cudaMalloc(&d, 5 * b), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, r.xs.data(), b, cudaMemcpyHostToDevice), cudaSuccess);
    kernelTrig<<<(n + 127) / 128, 128>>>(d, d + n, d + 2 * n, d + 3 * n, d + 4 * n, n);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (auto* col : { &r.sin, &r.cos, &r.sc, &r.cc })
        col->resize(n);
    EXPECT_EQ(cudaMemcpy(r.sin.data(), d + n, b, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.cos.data(), d + 2 * n, b, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.sc.data(), d + 3 * n, b, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(r.cc.data(), d + 4 * n, b, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d);
    return r;
}

TEST(TrigTotalTest, SpecialsHaveCStandardResults)
{
    expectCStandardSpecials(evaluate());
}

TEST(TrigTotalTest, SinIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.sin, kSin, kUsePinned, "sin");
}

TEST(TrigTotalTest, CosIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.cos, kCos, kUsePinned, "cos");
}

TEST(TrigTotalTest, SinCosIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.sc, kSin, kUsePinned, "sincos.sin");
    score(r.xs, r.cc, kCos, kUsePinned, "sincos.cos");
}

} // namespace TrigTotalTest
} // namespace aether_tests
