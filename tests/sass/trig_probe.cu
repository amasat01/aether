// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// STACK=0 ptxas probe for the zero-stack FP64 sin/cos/sincos device
// path (aether/math/detail/BoundedTrig.h). Standalone TU, compiled
// directly via bare `nvcc` (mirrors
// `tools/volatile_probe.cu`'s placement/rationale) — NEVER wired into
// tests/CMakeLists.txt's glob, NEVER executed. The whole point of
// BoundedTrig.h's statically-unrolled Payne-Hanek reduction is to
// eliminate libdevice's hidden alloca (40 B/thread) on the slow path; this
// probe exists to keep that invariant ptxas-checked, not just asserted in
// a docstring.

#include <aether/aether.h>

AETHER_KERNEL()
void trigProbeSinKernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = aether::math::sin(in[i]);
}

AETHER_KERNEL()
void trigProbeCosKernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = aether::math::cos(in[i]);
}

AETHER_KERNEL()
void trigProbeSinCosKernel(const double* __restrict__ in, double* __restrict__ sinOut, double* __restrict__ cosOut, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double s, c;
    aether::math::sincos(in[i], &s, &c);
    sinOut[i] = s;
    cosOut[i] = c;
}
