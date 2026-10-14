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
// every loop is correct and 1 otherwise. Build with the flags in
// check_hvm_masked.sh (-O3 -fno-trapping-math -march=native ...; the
// -fno-trapping-math is what lets GCC if-convert the floating-point select). The loop statements sit on the lines marked
// LOOP_A / LOOP_B / LOOP_C / LOOP_D, which check_hvm_masked.sh looks up in the
// compiler's vectoriser report.

#if !defined(AETHER_HOST_VECTOR_MATH)
#error "hvm_masked_repro.cpp must be compiled with -DAETHER_HOST_VECTOR_MATH"
#endif

#include "aether/math/math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// The lane-interchanged iteration, as a code generator emits it: a tile of
// lanes (eight vector registers of doubles) advances one trip at a time, each
// lane keeping its carried values in per-lane arrays. A finished lane keeps
// its old value (`on ? v : old`, one select per array), a lane stays active
// while its stop test fails, and the trip loop ends when no lane is live.
#if defined(__AVX512F__)
constexpr std::ptrdiff_t VECTOR_BYTES = 64;
#elif defined(__AVX__)
constexpr std::ptrdiff_t VECTOR_BYTES = 32;
#else
constexpr std::ptrdiff_t VECTOR_BYTES = 16;
#endif
constexpr std::ptrdiff_t TILE = 8 * VECTOR_BYTES / static_cast<std::ptrdiff_t>(sizeof(double));

template<class T>
inline void lanePut(T* lanes, std::ptrdiff_t t, const T& v)
{
    lanes[t] = v;
}

template<class T>
inline void laneKeep(T* lanes, std::ptrdiff_t t, bool on, const T& v)
{
    lanes[t] = on ? v : lanes[t];
}

template<class T>
inline T laneGet(const T* lanes, std::ptrdiff_t t)
{
    return lanes[t];
}

// One trip: s' = 1.1 s + g t; a lane whose s' exceeds lim stops with (s', t),
// any other lane continues with t' = tanh(t) + 0.1. Twelve trips at most.
__attribute__((noinline)) void loopD(const double* g, const double* x0, double lim, double* y)
{
    for (std::ptrdiff_t lo = 0; lo < static_cast<std::ptrdiff_t>(N); lo += TILE) {
        const std::ptrdiff_t tn = std::min<std::ptrdiff_t>(TILE, static_cast<std::ptrdiff_t>(N) - lo);
        alignas(64) double on[TILE] = {};
        alignas(64) double ron[TILE] = {};
        alignas(64) double sL[TILE] = {};
        alignas(64) double tL[TILE] = {};
        alignas(64) double gL[TILE] = {};
        alignas(64) double limL[TILE] = {};
        for (std::ptrdiff_t t = 0; t < tn; ++t) {
            lanePut(sL, t, x0[lo + t]);
            lanePut(tL, t, x0[lo + t]);
            lanePut(gL, t, g[lo + t]);
            lanePut(limL, t, lim);
            on[t] = 1.0;
        }
        std::int64_t any = 0;
#pragma GCC ivdep
        for (std::ptrdiff_t l = 0; l < TILE; ++l) {
            const bool in = l < tn && on[l] != 0.0;
            ron[l] = in ? 1.0 : 0.0;
            any |= in ? std::int64_t{1} : std::int64_t{0};
        }
        for (int trip = 0; any != 0 && trip < 12; ++trip) {
            std::int64_t live = 0;
#pragma GCC ivdep
            for (std::ptrdiff_t l = 0; l < TILE; ++l) { // LOOP_D
                const bool laneOn = ron[l] != 0.0;
                const double s = laneGet(sL, l);
                const double t = laneGet(tL, l);
                const double gi = laneGet(gL, l);
                const double limi = laneGet(limL, l);
                const double tb2 = s * 1.1 + gi * t;
                const bool stop = tb2 > limi;
                const double th = aether::math::tanh(t);
                const double n0 = stop ? tb2 : tb2;
                const double n1 = stop ? t : th + 0.1;
                laneKeep(sL, l, laneOn, n0);
                laneKeep(tL, l, laneOn, n1);
                const bool more = laneOn && !stop;
                ron[l] = more ? 1.0 : 0.0;
                live |= more ? std::int64_t{1} : std::int64_t{0};
            }
            if (live == 0) break;
        }
#pragma GCC ivdep
        for (std::ptrdiff_t t = 0; t < tn; ++t)
            y[lo + t] = laneGet(sL, t) + laneGet(tL, t);
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
