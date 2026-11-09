// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The aarch64 GCC route of aether::math::pow (aether/math/detail/HostVectorMath.h):
// an unconditional scalar loop that GCC vectorises through _ZGVnN2vv_aether_hvm_pow.
// Exit status 0 when the loop matches the scalar symbol bit for bit and the exact
// and special cases hold, 1 otherwise. check_hvm_arm.sh builds this file and looks
// the loops up in the compiler's vectoriser report (the LOOP and TRIP markers). TRIP has
// the shape of a lane-interchanged trip body: one flag tested around an unconditional pow,
// which GCC's jump threading copies onto conditional paths it cannot vectorise.

#if !defined(AETHER_HOST_VECTOR_MATH)
#error "hvm_arm_repro.cpp must be compiled with -DAETHER_HOST_VECTOR_MATH"
#endif

#include "aether/math/math.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
constexpr int kN = 64;
double base[kN], expo[kN], out[kN], step[kN], acc[kN], rej[kN], kept[kN];

__attribute__((noinline)) void loop()
{
    for (int i = 0; i < kN; ++i) out[i] = aether::math::pow(base[i], expo[i]); // LOOP
}

__attribute__((noinline)) void trip()
{
#pragma GCC ivdep
    for (int i = 0; i < kN; ++i) { // TRIP
        const bool ok = expo[i] < 0.5;
        const double p = aether::math::pow(base[i], ok ? -0.1111111111111111 : -0.125);
        step[i] = 0.9 * p;
        acc[i] = ok ? 1.0 : 0.0;
        rej[i] = ok ? 0.0 : 1.0;
        kept[i] = ok ? base[i] * 2.0 : kept[i];
    }
}

bool same(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
} // namespace

int main()
{
    for (int i = 0; i < kN; ++i) {
        base[i] = 0.05 + 0.31 * i;
        expo[i] = -2.0 + 0.0625 * i;
    }
    loop();
    int bad = 0;
    for (int i = 0; i < kN; ++i)
        if (!same(out[i], ::aether_hvm_pow(base[i], expo[i]))) {
            std::printf("lane %d differs from the scalar symbol\n", i);
            ++bad;
        }
    trip();
    for (int i = 0; i < kN; ++i) {
        const bool ok = expo[i] < 0.5;
        const double want = 0.9 * ::aether_hvm_pow(base[i], ok ? -0.1111111111111111 : -0.125);
        if (!same(step[i], want) || acc[i] != (ok ? 1.0 : 0.0) || rej[i] != (ok ? 0.0 : 1.0) || kept[i] != (ok ? base[i] * 2.0 : 0.0)) {
            std::printf("trip lane %d is wrong\n", i);
            ++bad;
        }
    }
    const struct { double x, y, want; } exact[] = {
        {2.0, 10.0, 1024.0}, {9.0, 0.5, 3.0}, {4.0, -0.5, 0.5}, {10.0, 2.0, 100.0}, {0.5, -3.0, 8.0}, {-8.0, 3.0, -512.0},
        {0.0, -1.0, HUGE_VAL}, {3.0, 0.0, 1.0}, {1.0, HUGE_VAL, 1.0}, {HUGE_VAL, 2.0, HUGE_VAL}};
    for (const auto& row : exact)
        if (!same(aether::math::pow(row.x, row.y), row.want)) {
            std::printf("pow(%g, %g) wrong\n", row.x, row.y);
            ++bad;
        }
    if (!std::isnan(aether::math::pow(-1.0, 0.5))) {
        std::printf("pow(-1, 0.5) is not NaN\n");
        ++bad;
    }
    return bad ? 1 : 0;
}
