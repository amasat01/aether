// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Reproducer for the masked-call miscompile that AETHER_HVM_MASKED guards
// against (see aether/math/detail/HostVectorMath.h): GCC 13.1-13.3 and
// 14.1-14.2 if-convert a conditional call of a function with masked vector
// variants using the inverted mask. Three loops over 64 doubles, each
// compared with a scalar reference:
//   a  conditional aether math call (the miscompiled pattern),
//   b  unconditional aether math call,
//   c  if-conversion without a math call.
// Prints one line per loop ("a: OK" / "a: MISMATCH"); exit status is 0 when
// every loop is correct and 1 otherwise. Build with -O3 -march=native
// -DAETHER_HOST_VECTOR_MATH. The loop statements sit on the lines marked
// LOOP_A / LOOP_B / LOOP_C, which check_hvm_masked.sh looks up in the
// compiler's vectoriser report.

#if !defined(AETHER_HOST_VECTOR_MATH)
#error "hvm_masked_repro.cpp must be compiled with -DAETHER_HOST_VECTOR_MATH"
#endif

#include "aether/math/math.h"

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {
constexpr std::size_t N = 64;

__attribute__((noinline)) void loopA(const double* a, const double* b, const double* c, double lim, double* r)
{
    for (std::size_t i = 0; i < N; ++i) // LOOP_A
        r[i] = c[i] > lim ? a[i] : aether::math::tanh(b[i]) + 0.1;
}

__attribute__((noinline)) void loopB(const double* b, double* r)
{
    for (std::size_t i = 0; i < N; ++i) // LOOP_B
        r[i] = aether::math::tanh(b[i]) * 2.0;
}

__attribute__((noinline)) void loopC(const double* a, const double* b, const double* c, double lim, double* r)
{
    for (std::size_t i = 0; i < N; ++i) // LOOP_C
        r[i] = c[i] > lim ? a[i] * 3.0 : b[i] + 1.0;
}

bool close(const double* x, const double* ref)
{
    for (std::size_t i = 0; i < N; ++i)
        if (!(std::fabs(x[i] - ref[i]) <= 1e-13 * (1.0 + std::fabs(ref[i]))))
            return false;
    return true;
}
} // namespace

int main()
{
    double a[N], b[N], c[N], r[N], ref[N];
    const double lim = 0.5;
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = 100.0 + static_cast<double>(i);
        b[i] = 0.05 * static_cast<double>(i) - 1.5;
        c[i] = (i % 3 == 0) ? 1.0 : 0.0; // mixed active / inactive lanes
    }
    int bad = 0;

    loopA(a, b, c, lim, r);
    for (std::size_t i = 0; i < N; ++i)
        ref[i] = c[i] > lim ? a[i] : std::tanh(b[i]) + 0.1;
    bool ok = close(r, ref);
    std::printf("a: %s\n", ok ? "OK" : "MISMATCH");
    bad += !ok;

    loopB(b, r);
    for (std::size_t i = 0; i < N; ++i)
        ref[i] = std::tanh(b[i]) * 2.0;
    ok = close(r, ref);
    std::printf("b: %s\n", ok ? "OK" : "MISMATCH");
    bad += !ok;

    loopC(a, b, c, lim, r);
    for (std::size_t i = 0; i < N; ++i)
        ref[i] = c[i] > lim ? a[i] * 3.0 : b[i] + 1.0;
    ok = close(r, ref);
    std::printf("c: %s\n", ok ? "OK" : "MISMATCH");
    bad += !ok;

    return bad ? 1 : 0;
}
