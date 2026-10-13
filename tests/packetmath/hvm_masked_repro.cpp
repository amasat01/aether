// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Reproducer for the masked-call miscompile that AETHER_HVM_MASKED guards
// against (see aether/math/detail/HostVectorMath.h): GCC 13.1-13.3 and
// 14.1-14.2 if-convert a conditional call of a function with masked vector
// variants using the inverted mask. Three loops over 64 doubles, each
// compared with a scalar reference:
//   a  conditional aether math call (the miscompiled pattern),
//   b  unconditional aether math call,
//   c  if-conversion without a math call,
//   d  an early-exit iteration over a tile of lanes (per-lane active flags, a
//      bounded trip loop, the math result feeding a select): the shape of a
//      lane-interchanged kernel, which the affected compilers miscompile on
//      AVX2 even when they leave loop a scalar.
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

constexpr std::size_t TILE = 8;

// One trip: s' = 1.1 s + g t; a lane whose s' exceeds lim stops with (s', t),
// any other lane continues with t' = tanh(t) + 0.1. Twelve trips at most.
__attribute__((noinline)) void loopD(const double* g, const double* x0, double lim, double* y)
{
    for (std::size_t lo = 0; lo < N; lo += TILE) {
        alignas(64) double s[TILE], t[TILE], gg[TILE], on[TILE];
        for (std::size_t l = 0; l < TILE; ++l) {
            s[l] = x0[lo + l];
            t[l] = x0[lo + l];
            gg[l] = g[lo + l];
            on[l] = 1.0;
        }
        for (int trip = 0; trip < 12; ++trip) {
            long live = 0;
#pragma GCC ivdep
            for (std::size_t l = 0; l < TILE; ++l) { // LOOP_D
                const bool lane = on[l] != 0.0;
                const double tb2 = s[l] * 1.1 + gg[l] * t[l];
                const bool stop = tb2 > lim;
                const double th = aether::math::tanh(t[l]);
                const double n1 = stop ? t[l] : th + 0.1;
                s[l] = lane ? tb2 : s[l];
                t[l] = lane ? n1 : t[l];
                const bool more = lane && !stop;
                on[l] = more ? 1.0 : 0.0;
                live |= more ? 1L : 0L;
            }
            if (live == 0) break;
        }
        for (std::size_t l = 0; l < TILE; ++l)
            y[lo + l] = s[l] + t[l];
    }
}

double refD(double g, double x0, double lim)
{
    double s = x0, t = x0;
    for (int trip = 0; trip < 12; ++trip) {
        const double tb2 = s * 1.1 + g * t;
        s = tb2;
        if (tb2 > lim) break;
        t = std::tanh(t) + 0.1;
    }
    return s + t;
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

    double g[N], x0[N];
    for (std::size_t i = 0; i < N; ++i) {
        g[i] = 0.2 + 0.03 * static_cast<double>(i % 11);
        x0[i] = 0.05 * static_cast<double>(i % 7) - 0.1;
    }
    loopD(g, x0, 2.0, r);
    for (std::size_t i = 0; i < N; ++i)
        ref[i] = refD(g[i], x0[i], 2.0);
    ok = close(r, ref);
    std::printf("d: %s\n", ok ? "OK" : "MISMATCH");
    bad += !ok;

    return bad ? 1 : 0;
}
