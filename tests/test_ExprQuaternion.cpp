// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Quaternion expression tests (host / AETHER_CPP_MODE build;
// test_ExprQuaternion.cu is the identical CUDA-build twin).
// `quatMul`/`quatConj`/`quatReciprocal`/`quatRotate`/`asPureQuaternion`/
// `asBack3DVector` (aether/expr/nodes/Quaternion.h, member functions
// declared on `Expression` in aether/expr/Expression.h) are DEVICEHOST-safe
// with no CUDA-specific behaviour a plain host build cannot already
// exercise.
//
// Tolerance: abs+rel 1e-12 — fused/reassociated quaternion arithmetic
// legitimately diverges from a naive scalar reference by a few ULP, more
// where terms cancel toward zero, so a plain EXPECT_DOUBLE_EQ flakes on
// these.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Item;
using aether::Vec3d;
using aether::Vec4d;

class ExprQuaternionTest : public ::testing::Test {
protected:
    static void expectNear(double test, double truth) { EXPECT_NEAR(test, truth, 1e-12 + 1e-12 * std::abs(truth)); }
};

TEST_F(ExprQuaternionTest, ConjugateNegatesVectorPart)
{
    // Quaternion conjugate.
    Vec4d q;
    for (std::size_t i = 0; i < 4; ++i)
        q(i) = static_cast<double>(i) - 1.25;

    Vec4d conj = q.quatConj();
    expectNear(conj(0), q(0));
    expectNear(conj(1), -q(1));
    expectNear(conj(2), -q(2));
    expectNear(conj(3), -q(3));
}

TEST_F(ExprQuaternionTest, ReciprocalMatchesHandComputed)
{
    // Quaternion reciprocal.
    Vec4d q;
    q(0) = 0.5;
    q(1) = -1.5;
    q(2) = 2.25;
    q(3) = -0.75;

    Vec4d rec = q.quatReciprocal();
    const double sq = q.squaredNorm();
    expectNear(rec(0), q(0) / sq);
    expectNear(rec(1), -q(1) / sq);
    expectNear(rec(2), -q(2) / sq);
    expectNear(rec(3), -q(3) / sq);
}

TEST_F(ExprQuaternionTest, MultiplyByConjugateGivesSquaredNorm)
{
    // Quaternion multiplication, first half.
    Vec4d q;
    q(0) = 1.5;
    q(1) = -2.0;
    q(2) = 0.5;
    q(3) = 3.0;

    Vec4d conj = q.quatConj();
    Vec4d res  = q.quatMul(conj);
    expectNear(res(0), q.squaredNorm());
    expectNear(res(1), 0.0);
    expectNear(res(2), 0.0);
    expectNear(res(3), 0.0);
}

TEST_F(ExprQuaternionTest, AsPureQuaternionAndAsBack3DVector)
{
    // asPureQuaternion/asBack3DVector expansion (aether does not carry
    // asFrontQuaternion/asFront3DVector).
    Vec3d v;
    v(0) = 1.5;
    v(1) = -2.5;
    v(2) = 3.5;

    Vec4d pure = v.asPureQuaternion();
    expectNear(pure(0), 0.0);
    expectNear(pure(1), v(0));
    expectNear(pure(2), v(1));
    expectNear(pure(3), v(2));

    Vec4d q;
    q(0) = 0.25;
    q(1) = 1.0;
    q(2) = -1.0;
    q(3) = 2.0;
    Vec4d conj = q.quatConj();
    Vec3d back = conj.asBack3DVector();
    expectNear(back(0), -q(1));
    expectNear(back(1), -q(2));
    expectNear(back(2), -q(3));
}

TEST_F(ExprQuaternionTest, RotateIdentityIsNoop)
{
    // Rotating by the identity quaternion is a no-op.
    Vec4d ident;
    ident(0) = 1.0;
    ident(1) = 0.0;
    ident(2) = 0.0;
    ident(3) = 0.0;

    Vec3d v;
    v(0) = 1.2;
    v(1) = -3.4;
    v(2) = 5.6;

    Vec3d res = ident.quatRotate(v);
    expectNear(res(0), v(0));
    expectNear(res(1), v(1));
    expectNear(res(2), v(2));
}

TEST_F(ExprQuaternionTest, Rotate90DegAboutZ)
{
    // (cos45,0,0,sin45) rotates (1,0,0) -> (0,1,0) (90 deg about Z).
    const double s = std::sqrt(0.5);
    Vec4d q;
    q(0) = s;
    q(1) = 0.0;
    q(2) = 0.0;
    q(3) = s;

    Vec3d v;
    v(0) = 1.0;
    v(1) = 0.0;
    v(2) = 0.0;

    Vec3d res = q.quatRotate(v);
    EXPECT_NEAR(res(0), 0.0, 1e-14);
    EXPECT_NEAR(res(1), 1.0, 1e-14);
    EXPECT_NEAR(res(2), 0.0, 1e-14);
}

TEST_F(ExprQuaternionTest, Rotate180DegAboutX)
{
    // (0,1,0,0) rotates (0,1,0) -> (0,-1,0) (180 deg about X).
    Vec4d q;
    q(0) = 0.0;
    q(1) = 1.0;
    q(2) = 0.0;
    q(3) = 0.0;

    Vec3d v;
    v(0) = 0.0;
    v(1) = 1.0;
    v(2) = 0.0;

    Vec3d res = q.quatRotate(v);
    EXPECT_NEAR(res(0), 0.0, 1e-14);
    EXPECT_NEAR(res(1), -1.0, 1e-14);
    EXPECT_NEAR(res(2), 0.0, 1e-14);
}

TEST_F(ExprQuaternionTest, RotateMatchesSandwichProduct)
{
    // quatRotate() must match the quaternion sandwich formula q*v*q^-1
    // computed directly via quatMul/quatConj.
    const double w = 0.4, x = -0.3, y = 0.7, z = 0.2;
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    Vec4d q;
    q(0) = w / n;
    q(1) = x / n;
    q(2) = y / n;
    q(3) = z / n;

    Vec3d v;
    v(0) = 0.6;
    v(1) = -1.1;
    v(2) = 2.3;

    Vec3d rodrigues = q.quatRotate(v);
    Vec4d sandwich   = q.quatMul(v.asPureQuaternion()).quatMul(q.quatConj());

    constexpr double tol = 1e-12;
    EXPECT_NEAR(rodrigues(0), sandwich(1), tol);
    EXPECT_NEAR(rodrigues(1), sandwich(2), tol);
    EXPECT_NEAR(rodrigues(2), sandwich(3), tol);
}

TEST_F(ExprQuaternionTest, RotatePreservesNorm)
{
    // Rotation preserves the vector's norm.
    const double w = -0.2, x = 0.9, y = -0.4, z = 0.1;
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    Vec4d q;
    q(0) = w / n;
    q(1) = x / n;
    q(2) = y / n;
    q(3) = z / n;

    Vec3d v;
    v(0) = -0.7;
    v(1) = 1.4;
    v(2) = 0.3;

    Vec3d res = q.quatRotate(v);
    EXPECT_NEAR(res.squaredNorm(), v.squaredNorm(), 1e-12);
}

} // namespace
} // namespace aether_tests
