// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Small-matrix closed-form expression tests (host / AETHER_CPP_MODE build;
// test_ExprInverse.cu is the identical CUDA-build twin). `trace()`/`det()`
// (eager scalars) and `inverse()` (lazy, backed by
// `aether/expr/nodes/Inverse.h`) — all three declared on `Expression` in
// `aether/expr/Expression.h` — are DEVICEHOST-safe with no CUDA-specific
// behaviour a plain host build cannot already exercise. Scoped to 2x2/3x3
// closed forms only (no LU/solve).
//
// `trace()`/`det()` on integer-valued matrices are bit-exact (every
// intermediate is an exactly-representable double integer); `A *
// A.inverse() ~= I` uses a combined abs+rel 1e-12 tolerance via the
// `absRelTol` helper below.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Item;
using aether::Mat33d;

/** @brief Combined abs+rel tolerance around `expected` (abs+rel 1e-12). */
double absRelTol(double expected)
{
    return 1e-12 * (1.0 + std::abs(expected));
}

class ExprInverseTest : public ::testing::Test { };

TEST_F(ExprInverseTest, Trace2x2And3x3MatchHandComputed)
{
    Item<double, 2, 2> m2{ 4.0, 3.0, 6.0, 3.0 };
    EXPECT_DOUBLE_EQ(m2.trace(), 4.0 + 3.0);

    Mat33d m3{ 2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0 };
    EXPECT_DOUBLE_EQ(m3.trace(), 2.0 + 3.0 + 2.0);
}

TEST_F(ExprInverseTest, Det2x2And3x3ExactOnIntegerValuedCases)
{
    // a=4,b=3,c=6,d=3 -> det = a*d - b*c = 12 - 18 = -6.
    Item<double, 2, 2> m2{ 4.0, 3.0, 6.0, 3.0 };
    EXPECT_DOUBLE_EQ(m2.det(), -6.0);

    // Cofactor expansion by hand:
    // det = 2*(3*2-2*0) - 0*(1*2-2*1) + 1*(1*0-3*1) = 2*6 - 0 + 1*(-3) = 9.
    Mat33d m3{ 2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0 };
    EXPECT_DOUBLE_EQ(m3.det(), 9.0);

    // Identity: det == 1, trace == N.
    EXPECT_DOUBLE_EQ(Mat33d::Identity().det(), 1.0);
    EXPECT_DOUBLE_EQ(Mat33d::Identity().trace(), 3.0);
}

TEST_F(ExprInverseTest, Inverse2x2TimesOriginalApproxIdentity)
{
    Item<double, 2, 2> m{ 4.0, 3.0, 6.0, 3.0 };
    Item<double, 2, 2> inv = m.inverse();
    Item<double, 2, 2> prod = m * inv;
    Item<double, 2, 2> ident = Item<double, 2, 2>::Identity();
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            EXPECT_NEAR(prod(r, c), ident(r, c), absRelTol(ident(r, c)));
}

TEST_F(ExprInverseTest, Inverse3x3TimesOriginalApproxIdentity)
{
    Mat33d m{ 2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0 };
    Mat33d inv  = m.inverse();
    Mat33d prod = m * inv;
    Mat33d ident = Mat33d::Identity();
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_NEAR(prod(r, c), ident(r, c), absRelTol(ident(r, c)));

    // Order matters for a non-symmetric matrix in general, but A*A^-1 ==
    // A^-1*A always holds for a true inverse — check both orders.
    Mat33d prodOther = inv * m;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_NEAR(prodOther(r, c), ident(r, c), absRelTol(ident(r, c)));
}

TEST_F(ExprInverseTest, InverseComponentsMatchHandComputedCofactorFormula)
{
    // Direct component check against the closed-form adjugate/det formula
    // (independent derivation from Inverse.h's own, cross-checking the
    // implementation rather than just the round-trip product above).
    Mat33d m{ 2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0 };
    const double a = m(0, 0), b = m(0, 1), c = m(0, 2);
    const double d = m(1, 0), e = m(1, 1), f = m(1, 2);
    const double g = m(2, 0), h = m(2, 1), k = m(2, 2);
    const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);

    Mat33d inv = m.inverse();
    EXPECT_DOUBLE_EQ(inv(0, 0), (e * k - f * h) / det);
    EXPECT_DOUBLE_EQ(inv(0, 1), (c * h - b * k) / det);
    EXPECT_DOUBLE_EQ(inv(0, 2), (b * f - c * e) / det);
    EXPECT_DOUBLE_EQ(inv(1, 0), (f * g - d * k) / det);
    EXPECT_DOUBLE_EQ(inv(1, 1), (a * k - c * g) / det);
    EXPECT_DOUBLE_EQ(inv(1, 2), (c * d - a * f) / det);
    EXPECT_DOUBLE_EQ(inv(2, 0), (d * h - e * g) / det);
    EXPECT_DOUBLE_EQ(inv(2, 1), (b * g - a * h) / det);
    EXPECT_DOUBLE_EQ(inv(2, 2), (a * e - b * d) / det);
}

} // namespace
} // namespace aether_tests
