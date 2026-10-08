// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Emulation-scalar seam probe (host / AETHER_CPP_MODE build;
// test_ProbeEmulationScalar.cu is the identical CUDA-build twin).
//
// Question: is aether's rank-generic core (Item/View/expr/assign/reductions)
// already scalar-generic — usable with a future non-native (e.g.
// SoftDouble/emulation) scalar type — or does it silently assume
// `double`/`float` somewhere an emulation-scalar rollout would trip over?
// `ProbeSd` is a stub non-native scalar: a plain aggregate wrapping one
// `double`, with the arithmetic/compare operators the core machinery below
// needs, and deliberately no `operator double() const` (an implicit scalar
// conversion would silently promote through `double` and mask exactly the
// kind of dtype-genericity bug this probe exists to catch).
//
// Scope/finding: this TU exercises every path that compiles with `ProbeSd`
// today — Item construction, the `Sum`/`CWiseScale` expression nodes (`+`,
// `-`, `s*e`, `e*s`, `e/s`), `View`'s `operator[]` `SampleRef` assignment
// proxy, and the reductions that need only `+`/`*`/`<`/`>`/unary `-` (`dot`,
// `squaredNorm`, `rSquaredNorm`, `sum`, `maxNorm`) — proving the
// rank-generic core is scalar-generic today for all of that surface. It
// does not exercise `.norm()`/`.rNorm()`/`.cubedNorm()`/`isFinite()` (route
// through `aether::math::{sqrt,isfinite}`, aether/math/math.h) or DLPack
// export (`aether::dtype_of<T>`, aether/dtype/DType.h) — both are gated
// by a `static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
// …)` that fires for `ProbeSd` exactly as documented in those headers.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

// The stub non-native scalar (D-a). Plain aggregate — no user-declared
// constructors — so `T{0}`/`T{1}` (aggregate-init, used throughout
// aether/expr/Reduce.h and aether/expr/Expression.h, e.g. `element_type{1}
// / squaredNorm()`) work exactly as they do for `double`.
struct ProbeSd {
    double v;
};

// AETHER_DEVICEHOST() on every operator (folded in here rather than
// reproduced as a separate RED — see the header comment above):
// an UNANNOTATED `constexpr` free function defaults to `__host__`-only under
// nvcc; aether's expression nodes (`Sum::eval`, `aether/expr/nodes/
// Arithmetic.h`) are `AETHER_DEVICEHOST()`-qualified, so a `__device__`-path
// instantiation calling a host-only `ProbeSd::operator+` is a hard nvcc
// error ("calling a constexpr __host__ function from a __device__
// function") — confirmed for real by this file's own .cu build. A
// non-native scalar type must annotate its own operators device-callable to
// participate in aether's rank-generic core at all; this is NOT optional
// scaffolding, it is part of the D-a finding.
AETHER_DEVICEHOST() constexpr ProbeSd operator+(ProbeSd a, ProbeSd b) { return ProbeSd{ a.v + b.v }; }
AETHER_DEVICEHOST() constexpr ProbeSd operator-(ProbeSd a, ProbeSd b) { return ProbeSd{ a.v - b.v }; }
AETHER_DEVICEHOST() constexpr ProbeSd operator-(ProbeSd a) { return ProbeSd{ -a.v }; }
AETHER_DEVICEHOST() constexpr ProbeSd operator*(ProbeSd a, ProbeSd b) { return ProbeSd{ a.v * b.v }; }
AETHER_DEVICEHOST() constexpr ProbeSd operator/(ProbeSd a, ProbeSd b) { return ProbeSd{ a.v / b.v }; }
AETHER_DEVICEHOST() constexpr bool operator<(ProbeSd a, ProbeSd b) { return a.v < b.v; }
AETHER_DEVICEHOST() constexpr bool operator>(ProbeSd a, ProbeSd b) { return a.v > b.v; }
AETHER_DEVICEHOST() constexpr bool operator==(ProbeSd a, ProbeSd b) { return a.v == b.v; }

class ProbeEmulationScalarTest : public ::testing::Test { };

// ---------------------------------------------------------------------
// Item + the Sum/CWiseScale expression nodes (Operations.h: `+`, `-`,
// `s*e`, `e*s`, `e/s`) — needs only ProbeSd's `+`, `-`, `*`, `/`.
// ---------------------------------------------------------------------

TEST_F(ProbeEmulationScalarTest, ItemArithmeticExpressionsEvaluateComponentwise)
{
    aether::Item<ProbeSd, 3> a{ ProbeSd{ 1.0 }, ProbeSd{ 2.0 }, ProbeSd{ 3.0 } };
    aether::Item<ProbeSd, 3> b{ ProbeSd{ 10.0 }, ProbeSd{ 20.0 }, ProbeSd{ 30.0 } };

    aether::Item<ProbeSd, 3> sum = a + b;
    EXPECT_DOUBLE_EQ(sum(0).v, 11.0);
    EXPECT_DOUBLE_EQ(sum(1).v, 22.0);
    EXPECT_DOUBLE_EQ(sum(2).v, 33.0);

    aether::Item<ProbeSd, 3> diff = b - a;
    EXPECT_DOUBLE_EQ(diff(0).v, 9.0);
    EXPECT_DOUBLE_EQ(diff(1).v, 18.0);
    EXPECT_DOUBLE_EQ(diff(2).v, 27.0);

    const ProbeSd s{ 2.5 };
    aether::Item<ProbeSd, 3> scaledL = s * a;
    aether::Item<ProbeSd, 3> scaledR = a * s;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(scaledL(i).v, s.v * a(i).v);
        EXPECT_DOUBLE_EQ(scaledR(i).v, a(i).v * s.v);
    }

    aether::Item<ProbeSd, 3> divided = b / s;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(divided(i).v, b(i).v * (ProbeSd{ 1.0 } / s).v);
}

// ---------------------------------------------------------------------
// View::operator[] SampleRef assignment proxy (Assign.h) over a batched
// Array<ProbeSd,3> — needs only `+` (mirrors test_ExprAssign.cpp's
// ViewSubscriptAssignOverwrites, ProbeSd standing in for double).
// ---------------------------------------------------------------------

TEST_F(ProbeEmulationScalarTest, ViewSubscriptAssignBuildsAndEvaluatesAnExpressionTree)
{
    constexpr std::size_t N = 4;
    aether::Array<ProbeSd, 3> a(N), b(N), out(N);
    auto av = a.hostView();
    auto bv = b.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            av(c, i) = ProbeSd{ static_cast<double>(c) + static_cast<double>(i) };
            bv(c, i) = ProbeSd{ static_cast<double>(c) * 2.0 };
        }
    }

    auto ov = out.hostView();
    for (std::size_t i = 0; i < N; ++i)
        ov[aether::SampleIndex::make(i)] = av[aether::SampleIndex::make(i)].get() + bv[aether::SampleIndex::make(i)].get();

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(ov(c, i).v, av(c, i).v + bv(c, i).v);
}

// ---------------------------------------------------------------------
// Reductions that need only `+`/`*`/`<`/`>`/unary `-` (Reduce.h's
// SumOp/TimesOp/MaxOp/AbsOp, Expression.h's dot/squaredNorm/rSquaredNorm/
// sum/maxNorm) — deliberately EXCLUDES norm()/rNorm()/cubedNorm()/
// isFinite() (need aether::math::sqrt/isfinite — see this file's own
// header comment).
// ---------------------------------------------------------------------

TEST_F(ProbeEmulationScalarTest, ReductionsThatNeedOnlyPlusTimesLessGreaterCompileAndComputeCorrectly)
{
    aether::Item<ProbeSd, 3> a{ ProbeSd{ 2.0 }, ProbeSd{ -3.0 }, ProbeSd{ 4.0 } };
    aether::Item<ProbeSd, 3> b{ ProbeSd{ 5.0 }, ProbeSd{ 1.0 }, ProbeSd{ -2.0 } };

    EXPECT_DOUBLE_EQ(a.dot(b).v, 2.0 * 5.0 + (-3.0) * 1.0 + 4.0 * (-2.0));
    EXPECT_DOUBLE_EQ(a.squaredNorm().v, 2.0 * 2.0 + (-3.0) * (-3.0) + 4.0 * 4.0);
    EXPECT_DOUBLE_EQ(a.rSquaredNorm().v, 1.0 / (2.0 * 2.0 + (-3.0) * (-3.0) + 4.0 * 4.0));
    EXPECT_DOUBLE_EQ(a.sum().v, 2.0 + (-3.0) + 4.0);
    EXPECT_DOUBLE_EQ(a.maxNorm().v, 4.0); // max(|2|, |-3|, |4|) = 4
}

} // namespace
} // namespace aether_tests
