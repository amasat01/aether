// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::rt (aether/expr/Rt.h) against its static twin, bit for bit. One body
// per sample computes BOTH forms in the same TU (production flags, no
// contraction override); the host compares the bits. test_ExprRt.cpp runs the
// bodies in a host loop, test_ExprRt.cu in one kernel.
#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace expr_rt {
namespace {

using aether::Array;
using aether::SampleIndex;

constexpr std::size_t kSamples = 37;

#if defined(__CUDACC__)
template<class Body>
AETHER_KERNEL()
void bodyKernel(Body body, std::size_t n)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() < n)
        body(i);
}
template<class Body>
void runSamples(Body body)
{
    bodyKernel<<<(kSamples + 63) / 64, 64>>>(body, kSamples);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
#else
template<class Body>
void runSamples(Body body)
{
    for (std::size_t g = 0; g < kSamples; ++g)
        body(SampleIndex::make(g));
}
#endif

// Samples 0..5 carry special values (NaN, -0.0, +-inf), the rest are random
// with a wide exponent spread so that a different summation order shows.
inline double value(std::mt19937& rng, std::size_t sample, std::size_t k, std::size_t n)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<int> e(-30, 30);
    const double r = std::ldexp(u(rng), e(rng));
    switch (sample) {
    case 0: return k == n / 2 ? std::numeric_limits<double>::quiet_NaN() : r;
    case 1: return -0.0;
    case 2: return k == n - 1 ? inf : r;
    case 3: return k == 0 ? -inf : r;
    case 4: return k % 2 == 0 ? -0.0 : r;
    case 5: return k == 0 ? -inf : (k == n / 2 ? inf : r);
    default: return r;
    }
}

// Element k of sample g, flattened row-major for rank 2.
template<class V>
AETHER_DEVICEHOST() constexpr double& elemRef(V& v, std::size_t k, std::size_t g)
{
    if constexpr (V::element_extents::Rank == 1) {
        return v(k, g);
    } else {
        constexpr std::size_t C = V::element_extents::static_extent(1);
        return v(k / C, k % C, g);
    }
}

template<class T, std::size_t... Es>
void fill(Array<T, Es...>& a, unsigned seed)
{
    std::mt19937 rng(seed);
    constexpr std::size_t n = (Es * ... * std::size_t{ 1 });
    auto v                  = a.hostView();
    for (std::size_t g = 0; g < kSamples; ++g) {
        for (std::size_t k = 0; k < n; ++k) {
            elemRef(v, k, g) = value(rng, g, k, n);
        }
    }
    a.upload();
}

template<class T, std::size_t... Es>
void expectSameBits(Array<T, Es...>& s, Array<T, Es...>& r, const char* what)
{
    s.download();
    r.download();
    constexpr std::size_t n = (Es * ... * std::size_t{ 1 });
    auto sv                 = s.hostView();
    auto rv                 = r.hostView();
    std::size_t bad         = 0;
    for (std::size_t g = 0; g < kSamples; ++g) {
        for (std::size_t k = 0; k < n; ++k) {
            const double a = elemRef(sv, k, g);
            const double b = elemRef(rv, k, g);
            if (std::bit_cast<std::uint64_t>(a) != std::bit_cast<std::uint64_t>(b) && bad++ < 3)
                ADD_FAILURE() << what << ": sample " << g << " element " << k << " static " << a << " rt " << b;
        }
    }
    EXPECT_EQ(bad, 0u) << what;
}

// ---------------------------------------------------------------------------
// Rank-1 reductions: static `a[i].get().f(...)` vs `rt::f(a[i], ...)` (proxy)
// and `rt::f(item, ...)` (Item operand).
// ---------------------------------------------------------------------------

enum class Red { Dot, Sum, SquaredNorm, Norm, RNorm, RSquaredNorm, CubedNorm, RCubedNorm, MaxNorm, IsFinite };

template<Red Op, class VA, class VO>
struct ReduceBody {
    VA a, b;
    VO s, r, t;

    template<class X, class Y>
    AETHER_DEVICEHOST() static double rtForm(const X& x, [[maybe_unused]] const Y& y)
    {
        if constexpr (Op == Red::Dot) return aether::rt::dot(x, y);
        else if constexpr (Op == Red::Sum) return aether::rt::sum(x);
        else if constexpr (Op == Red::SquaredNorm) return aether::rt::squaredNorm(x);
        else if constexpr (Op == Red::Norm) return aether::rt::norm(x);
        else if constexpr (Op == Red::RNorm) return aether::rt::rNorm(x);
        else if constexpr (Op == Red::RSquaredNorm) return aether::rt::rSquaredNorm(x);
        else if constexpr (Op == Red::CubedNorm) return aether::rt::cubedNorm(x);
        else if constexpr (Op == Red::RCubedNorm) return aether::rt::rCubedNorm(x);
        else if constexpr (Op == Red::MaxNorm) return aether::rt::maxNorm(x);
        else return aether::rt::isFinite(x) ? 1.0 : 0.0;
    }

    AETHER_DEVICEHOST() void operator()(const SampleIndex& i)
    {
        const std::size_t g = i.global();
        const VA& cb        = b; // ConstSampleRef operand
        const auto ia       = a[i].get();
        const auto ib       = b[i].get();
        if constexpr (Op == Red::Dot) s(0, g) = ia.dot(ib);
        if constexpr (Op == Red::Sum) s(0, g) = ia.sum();
        if constexpr (Op == Red::SquaredNorm) s(0, g) = ia.squaredNorm();
        if constexpr (Op == Red::Norm) s(0, g) = ia.norm();
        if constexpr (Op == Red::RNorm) s(0, g) = ia.rNorm();
        if constexpr (Op == Red::RSquaredNorm) s(0, g) = ia.rSquaredNorm();
        if constexpr (Op == Red::CubedNorm) s(0, g) = ia.cubedNorm();
        if constexpr (Op == Red::RCubedNorm) s(0, g) = ia.rCubedNorm();
        if constexpr (Op == Red::MaxNorm) s(0, g) = ia.maxNorm();
        if constexpr (Op == Red::IsFinite) s(0, g) = aether::isFinite(ia) ? 1.0 : 0.0;
        r(0, g) = rtForm(a[i], cb[i]);
        t(0, g) = rtForm(ia, ib);
    }
};

template<std::size_t N, Red Op>
void checkReduce(const char* what)
{
    Array<double, N> a(kSamples), b(kSamples);
    Array<double, 1> s(kSamples), r(kSamples), t(kSamples);
    fill(a, 11 + N);
    fill(b, 23 + N);
    runSamples(ReduceBody<Op, typename Array<double, N>::ViewT, typename Array<double, 1>::ViewT>{
        a.deviceView(), b.deviceView(), s.deviceView(), r.deviceView(), t.deviceView() });
    expectSameBits(s, r, what);
    expectSameBits(s, t, what);
}

template<std::size_t N>
void checkAllReductions()
{
    checkReduce<N, Red::Dot>("dot");
    checkReduce<N, Red::Sum>("sum");
    checkReduce<N, Red::SquaredNorm>("squaredNorm");
    checkReduce<N, Red::Norm>("norm");
    checkReduce<N, Red::RNorm>("rNorm");
    checkReduce<N, Red::RSquaredNorm>("rSquaredNorm");
    checkReduce<N, Red::CubedNorm>("cubedNorm");
    checkReduce<N, Red::RCubedNorm>("rCubedNorm");
    checkReduce<N, Red::MaxNorm>("maxNorm");
    checkReduce<N, Red::IsFinite>("isFinite");
}

// ---------------------------------------------------------------------------
// Rank-2: matDot vs `get().matDot(get())`; matVec/matMat (Set/Add/Sub) vs
// `view[i] = / += / -= M * v` (the fused AssignDispatch path for Set).
// ---------------------------------------------------------------------------

template<class VM, class VO>
struct MatDotBody {
    VM l, m;
    VO s, r;
    AETHER_DEVICEHOST() void operator()(const SampleIndex& i)
    {
        s(0, i.global()) = l[i].get().matDot(m[i].get());
        r(0, i.global()) = aether::rt::matDot(l[i], m[i]);
    }
};

template<std::size_t R, std::size_t C>
void checkMatDot()
{
    Array<double, R, C> l(kSamples), m(kSamples);
    Array<double, 1> s(kSamples), r(kSamples);
    fill(l, 31 + R * C);
    fill(m, 37 + R * C);
    runSamples(MatDotBody<typename Array<double, R, C>::ViewT, typename Array<double, 1>::ViewT>{
        l.deviceView(), m.deviceView(), s.deviceView(), r.deviceView() });
    expectSameBits(s, r, "matDot");
}

template<int Mode, class VL, class VR, class VO>
struct ProductBody {
    VL l;
    VR x;
    VO s, r;
    AETHER_DEVICEHOST() void operator()(const SampleIndex& i)
    {
        constexpr bool vec = VR::element_extents::Rank == 1;
        if constexpr (Mode == 0) {
            s[i] = l * x;
            if constexpr (vec) aether::rt::matVec(r[i], l[i], x[i]);
            else aether::rt::matMat(r[i], l[i], x[i]);
        } else if constexpr (Mode == 1) {
            s[i] += l * x;
            if constexpr (vec) aether::rt::matVecAdd(r[i], l[i], x[i]);
            else aether::rt::matMatAdd(r[i], l[i], x[i]);
        } else {
            s[i] -= l * x;
            if constexpr (vec) aether::rt::matVecSub(r[i], l[i], x[i]);
            else aether::rt::matMatSub(r[i], l[i], x[i]);
        }
    }
};

template<int Mode, class AL, class AX, class AO>
void checkProduct(const char* what)
{
    AL l(kSamples);
    AX x(kSamples);
    AO s(kSamples), r(kSamples);
    fill(l, 41);
    fill(x, 43);
    fill(s, 47); // the same start for += / -=
    fill(r, 47);
    runSamples(ProductBody<Mode, typename AL::ViewT, typename AX::ViewT, typename AO::ViewT>{
        l.deviceView(), x.deviceView(), s.deviceView(), r.deviceView() });
    expectSameBits(s, r, what);
}

template<class AL, class AX, class AO>
void checkAllModes(const char* set, const char* add, const char* sub)
{
    checkProduct<0, AL, AX, AO>(set);
    checkProduct<1, AL, AX, AO>(add);
    checkProduct<2, AL, AX, AO>(sub);
}

template<std::size_t R, std::size_t C>
void checkMatVec()
{
    checkAllModes<Array<double, R, C>, Array<double, C>, Array<double, R>>("matVec", "matVecAdd", "matVecSub");
}

template<std::size_t R, std::size_t K, std::size_t C>
void checkMatMat()
{
    checkAllModes<Array<double, R, K>, Array<double, K, C>, Array<double, R, C>>("matMat", "matMatAdd", "matMatSub");
}

} // namespace

TEST(ExprRt, ReductionsN1) { checkAllReductions<1>(); }
TEST(ExprRt, ReductionsN2) { checkAllReductions<2>(); }
TEST(ExprRt, ReductionsN3) { checkAllReductions<3>(); }
TEST(ExprRt, ReductionsN7) { checkAllReductions<7>(); }
TEST(ExprRt, ReductionsN16) { checkAllReductions<16>(); }
TEST(ExprRt, ReductionsN144) { checkAllReductions<144>(); }

TEST(ExprRt, MatDot)
{
    checkMatDot<1, 1>();
    checkMatDot<1, 2>();
    checkMatDot<3, 1>();
    checkMatDot<1, 7>();
    checkMatDot<4, 4>();
    checkMatDot<12, 12>();
}

TEST(ExprRt, MatVec)
{
    checkMatVec<1, 1>();
    checkMatVec<2, 2>();
    checkMatVec<3, 3>();
    checkMatVec<3, 7>();
    checkMatVec<16, 16>();
    checkMatVec<16, 144>();
}

TEST(ExprRt, MatMat)
{
    checkMatMat<1, 1, 1>();
    checkMatMat<2, 2, 2>();
    checkMatMat<3, 3, 3>();
    checkMatMat<3, 7, 2>();
    checkMatMat<16, 16, 3>();
    checkMatMat<2, 144, 2>(); // long K; a 16x144 left operand costs minutes of nvcc per mode
}

} // namespace expr_rt
} // namespace aether_tests
