// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// Nested `select` over value leaves read a stale stack value under GCC points-to
// analysis when the leaves were held by value. aether-only, 60 lines; driven by
// check_select_pta.sh. Exact result is 0 for every sample (every term carries a
// zero plane); the 1e200 stack poison shows up verbatim when the bug is present.
#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using Real = double;
using Vec = aether::View<Real, aether::extents<3, aether::dyn>, aether::layout_right>;
using CVec = const Vec;
static Vec vec(Real* p, std::size_t n) { using Ext = aether::extents<3, aether::dyn>; return Vec(p, aether::layout_right::mapping<Ext>(Ext(n)), aether::Device(kDLCPU)); }
static CVec cvec(const Real* p, std::size_t n) { using Ext = aether::extents<3, aether::dyn>; return Vec(const_cast<Real*>(p), aether::layout_right::mapping<Ext>(Ext(n)), aether::Device(kDLCPU)); }

__attribute__((noinline)) void kernel(Real* out_p, const Real* x_p, const Real* a_p, const Real* b_p,
                                      const unsigned char* term, Real s1, Real s2, long n)
{
    Vec out = vec(out_p, n); const CVec x = cvec(x_p, n), a = cvec(a_p, n), b = cvec(b_p, n);
    for (long j = 0; j < n; ++j) {
        const aether::SampleIndex i = aether::SampleIndex::make(static_cast<std::size_t>(j));
        if (!term[j]) {
            const bool t = term[j] != 0;
            const auto zero = aether::constant<aether::extents<3>>(static_cast<Real>(0.0));
            const auto x_i = x[i].get();
            const auto f1 = (s1 * aether::math::sqrt(x_i.dot(x_i)));   // a scalar derived from x
            const auto a_i = a[i].get();
            const auto b_i = b[i].get();
            const auto hz0 = aether::constant<aether::extents<>>(t);
            const auto hz1 = aether::constant<aether::extents<>>(t);
#if defined(SHAPE_SINGLE)
            out[i] = aether::select(hz0, zero, ((f1 * a_i) + (s2 * b_i)));
#elif defined(SHAPE_LARGE)
            const auto f0 = (s2 * a_i.dot(x_i));                        // exact 0: a is a zero plane
            const auto f3 = (f1 * b_i.dot(x_i));                        // exact 0: b is a zero plane
            out[i] = aether::select(hz1, zero, aether::select(hz0, zero,
                (((f0 * x_i) + (f1 * a_i)) + ((s2 * b_i) + (f3 * x_i)))));
#else
            out[i] = aether::select(hz1, zero, aether::select(hz0, zero, ((f1 * a_i) + (s2 * b_i))));
#endif
        }
    }
}
__attribute__((noinline)) void dirty() { volatile double junk[4096]; for (int i = 0; i < 4096; ++i) junk[i] = 1e200; }
int main(int argc, char** argv)
{
    const long n = argc > 1 ? atol(argv[1]) : 6;
    Real* x = new Real[3 * n]; Real* a = new Real[3 * n](); Real* b = new Real[3 * n](); Real* out = new Real[3 * n];
    unsigned char* term = new unsigned char[n]();
    for (long i = 0; i < 3 * n; ++i) { x[i] = std::sin(1.3 * i + 0.4) * 1.7; out[i] = 777.0; }
    double mx = 0;
    for (int call = 0; call < 2; ++call) {
        dirty();
        kernel(out, x, a, b, term, 0.3, 1.0, n);
        for (long i = 0; i < 3 * n; ++i) mx = std::fmax(mx, std::fabs(out[i]));
    }
    std::printf("max|out|=%.3g (exact 0)\n", mx);
    return mx != 0;
}
