// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Reduction expression tests (CUDA build; test_ExprReduce.cpp is the
// identical host-build twin): `dot`/`norm`/`squaredNorm`/`cubedNorm`/
// `rNorm`/`rSquaredNorm`/`rCubedNorm`/`maxNorm`/`sum` (aether/expr/Reduce.h,
// member functions declared on `Expression` in aether/expr/Expression.h)
// and the free `aether::isFinite` are DEVICEHOST-safe with no CUDA-specific
// behaviour a plain host build cannot already exercise.
//
// The reduction's exact fold order (`RecursiveReduce` walks the component
// index descending, N-1 down to 0; see aether/expr/Reduce.h's docstring)
// matches every hand-computed reference below, so EXPECT_DOUBLE_EQ is
// meaningful (a reference computed via a different sequence can
// legitimately disagree in the last ULP).

#include <cmath>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Item;
using aether::Vec3d;
using aether::Vec4d;

class ExprReduceTest : public ::testing::Test { };

TEST_F(ExprReduceTest, DotMatchesHandComputedVec3d)
{
    // Dot product.
    Vec3d a;
    for (std::size_t i = 0; i < 3; ++i)
        a(i) = static_cast<double>(i) + 1.5;
    Vec3d b = a * 3.0;

    // Fold order matches RecursiveReduce: (a2*b2) + ((a1*b1) + (a0*b0)).
    const double expected = (a(2) * b(2)) + ((a(1) * b(1)) + (a(0) * b(0)));
    EXPECT_DOUBLE_EQ(a.dot(b), expected);
}

TEST_F(ExprReduceTest, SumMatchesHandComputedVec3d)
{
    // Component sum.
    Vec3d a;
    for (std::size_t i = 0; i < 3; ++i)
        a(i) = static_cast<double>(i) * 2.5 - 1.0;

    const double expected = a(2) + (a(1) + a(0));
    EXPECT_DOUBLE_EQ(a.sum(), expected);
}

TEST_F(ExprReduceTest, NormFamilyMatchesHandComputedVec3d)
{
    // aether's reductions are sample-free by design (no dual, index-taking
    // overload), so this checks every norm flavor against a genuine
    // hand-computed reference rather than a self-consistency check.
    Vec3d a;
    a(0) = 3.0;
    a(1) = -4.0;
    a(2) = 12.0;

    const double sqNorm = (a(2) * a(2)) + ((a(1) * a(1)) + (a(0) * a(0)));
    EXPECT_DOUBLE_EQ(a.squaredNorm(), sqNorm);
    EXPECT_DOUBLE_EQ(a.norm(), std::sqrt(sqNorm));
    EXPECT_DOUBLE_EQ(a.rNorm(), 1.0 / std::sqrt(sqNorm));
    EXPECT_DOUBLE_EQ(a.rSquaredNorm(), 1.0 / sqNorm);
    EXPECT_DOUBLE_EQ(a.cubedNorm(), sqNorm * std::sqrt(sqNorm));
    EXPECT_DOUBLE_EQ(a.rCubedNorm(), 1.0 / (sqNorm * std::sqrt(sqNorm)));
}

TEST_F(ExprReduceTest, MaxNormMatchesHandComputedVec3d)
{
    Vec3d a;
    a(0) = -2.0;
    a(1) = 5.0;
    a(2) = -7.0;
    // fold: max(|a2|, max(|a1|, |a0|))
    const double expected = std::max(std::abs(a(2)), std::max(std::abs(a(1)), std::abs(a(0))));
    EXPECT_DOUBLE_EQ(a.maxNorm(), expected);
}

TEST_F(ExprReduceTest, DotAndNormWorkOnAQuaternionToo)
{
    // Rank-generic reductions: the same dot()/norm() used above on a
    // Vec3d must also work, unmodified, on a Vec4d (quaternion shape).
    Vec4d q;
    for (std::size_t i = 0; i < 4; ++i)
        q(i) = static_cast<double>(i) - 1.25;
    Vec4d r = q * 2.0;

    const double dotExpected = (q(3) * r(3)) + ((q(2) * r(2)) + ((q(1) * r(1)) + (q(0) * r(0))));
    EXPECT_DOUBLE_EQ(q.dot(r), dotExpected);

    const double sqNorm
        = (q(3) * q(3)) + ((q(2) * q(2)) + ((q(1) * q(1)) + (q(0) * q(0))));
    EXPECT_DOUBLE_EQ(q.squaredNorm(), sqNorm);
    EXPECT_DOUBLE_EQ(q.norm(), std::sqrt(sqNorm));
}

TEST_F(ExprReduceTest, IsFiniteDetectsNonFiniteComponent)
{
    Vec3d finiteVec = Vec3d::filled(1.0);
    EXPECT_TRUE(aether::isFinite(finiteVec));

    Vec3d withNan = Vec3d::filled(1.0);
    withNan(1)    = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(aether::isFinite(withNan));

    Vec3d withInf = Vec3d::filled(1.0);
    withInf(2)    = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(aether::isFinite(withInf));
}

TEST_F(ExprReduceTest, ViewMaterializedDotAndSumMatchRawAccessorArithmetic)
{
    // The reduction surface reached via the batched `view[i].get()`
    // materialization path — see also test_ExprArithmetic.cpp's
    // ViewAsExpressionLeafMatchesRawAccessorArithmetic.
    constexpr std::size_t N = 4;
    aether::Array<double, 3> arrA(N), arrB(N);
    auto av = arrA.hostView();
    auto bv = arrB.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            av(c, i) = static_cast<double>(c) + static_cast<double>(i) * 1.5 - 2.0;
            bv(c, i) = static_cast<double>(c) * 0.5 - static_cast<double>(i);
        }
    }

    const aether::SampleIndex i1 = aether::SampleIndex::make(1);
    Vec3d a                      = av[i1].get();
    Vec3d b                      = bv[i1].get();

    const double dotExpected = (a(2) * b(2)) + ((a(1) * b(1)) + (a(0) * b(0)));
    EXPECT_DOUBLE_EQ(a.dot(b), dotExpected);

    const double sumExpected = a(2) + (a(1) + a(0));
    EXPECT_DOUBLE_EQ(a.sum(), sumExpected);
}

} // namespace
} // namespace aether_tests
