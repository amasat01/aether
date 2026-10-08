// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Geometric expression tests (host / AETHER_CPP_MODE build;
// test_ExprGeometric.cu is the identical CUDA-build twin). `cross`/
// `unitVector` (aether/expr/nodes/Geometric.h, member functions declared on
// `Expression` in aether/expr/Expression.h) are DEVICEHOST-safe with no
// CUDA-specific behaviour a plain host build cannot already exercise.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Vec3d;

class ExprGeometricTest : public ::testing::Test { };

TEST_F(ExprGeometricTest, CrossKnownAxisValues)
{
    // Known-values half: x cross y = z, etc.
    Vec3d x = Vec3d::Zeros();
    Vec3d y = Vec3d::Zeros();
    Vec3d z = Vec3d::Zeros();
    x(0)    = 1.0;
    y(1)    = 1.0;
    z(2)    = 1.0;

    Vec3d yCrossZ = y.cross(z);
    Vec3d zCrossX = z.cross(x);
    Vec3d xCrossY = x.cross(y);
    EXPECT_DOUBLE_EQ(x(0), yCrossZ(0));
    EXPECT_DOUBLE_EQ(y(1), zCrossX(1));
    EXPECT_DOUBLE_EQ(z(2), xCrossY(2));

    Vec3d zCrossY = z.cross(y);
    Vec3d xCrossZ = x.cross(z);
    Vec3d yCrossX = y.cross(x);
    EXPECT_DOUBLE_EQ(-x(0), zCrossY(0));
    EXPECT_DOUBLE_EQ(-y(1), xCrossZ(1));
    EXPECT_DOUBLE_EQ(-z(2), yCrossX(2));
}

TEST_F(ExprGeometricTest, CrossWholeOperationMatchesHandComputed)
{
    // The whole-operation half of the cross-product check.
    Vec3d a;
    for (std::size_t i = 0; i < 3; ++i)
        a(i) = static_cast<double>(i) - 0.5;
    Vec3d b = a * 3.0;

    Vec3d expected;
    expected(0) = a(1) * b(2) - a(2) * b(1);
    expected(1) = a(2) * b(0) - a(0) * b(2);
    expected(2) = a(0) * b(1) - a(1) * b(0);

    Vec3d actual = a.cross(b);
    EXPECT_DOUBLE_EQ(expected(0), actual(0));
    EXPECT_DOUBLE_EQ(expected(1), actual(1));
    EXPECT_DOUBLE_EQ(expected(2), actual(2));
}

TEST_F(ExprGeometricTest, UnitVectorMatchesHandComputed)
{
    // The `.unitVector()` check — aether has no `.normalize()` alias.
    Vec3d a;
    a(0) = 3.0;
    a(1) = -4.0;
    a(2) = 12.0;

    const double n = a.norm();
    Vec3d unit      = a.unitVector();
    EXPECT_DOUBLE_EQ(unit(0), a(0) * a.rNorm());
    EXPECT_DOUBLE_EQ(unit(1), a(1) * a.rNorm());
    EXPECT_DOUBLE_EQ(unit(2), a(2) * a.rNorm());
    // Sanity: the unit vector's norm is 1 (within FP tolerance — a different
    // FP sequence than the exact bit-match above, so a tolerance here).
    EXPECT_NEAR(unit.norm(), 1.0, 1e-12);
    (void)n;
}

TEST_F(ExprGeometricTest, ViewMaterializedCrossAndUnitVectorMatchRawAccessorArithmetic)
{
    // The geometric surface reached via the batched `view[i].get()`
    // materialization path (mirrors test_ExprReduce.cpp's analogous batched
    // test).
    constexpr std::size_t N = 4;
    aether::Array<double, 3> arrA(N), arrB(N);
    auto av = arrA.hostView();
    auto bv = arrB.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            av(c, i) = static_cast<double>(c) * 2.0 - static_cast<double>(i);
            bv(c, i) = static_cast<double>(c) - static_cast<double>(i) * 0.5 + 1.0;
        }
    }

    const aether::SampleIndex i2 = aether::SampleIndex::make(2);
    Vec3d a                      = av[i2].get();
    Vec3d b                      = bv[i2].get();

    Vec3d crossExpected;
    crossExpected(0) = a(1) * b(2) - a(2) * b(1);
    crossExpected(1) = a(2) * b(0) - a(0) * b(2);
    crossExpected(2) = a(0) * b(1) - a(1) * b(0);
    Vec3d crossActual = a.cross(b);
    EXPECT_DOUBLE_EQ(crossExpected(0), crossActual(0));
    EXPECT_DOUBLE_EQ(crossExpected(1), crossActual(1));
    EXPECT_DOUBLE_EQ(crossExpected(2), crossActual(2));

    Vec3d unitActual = a.unitVector();
    EXPECT_DOUBLE_EQ(unitActual(0), a(0) * a.rNorm());
}

} // namespace
} // namespace aether_tests
