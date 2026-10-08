// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Elementwise expression tests (host / AETHER_CPP_MODE build;
// test_ExprElementwise.cu is the identical CUDA-build twin).
// `cwiseMin`/`cwiseMax`/`cwiseAbs`/`clamp`/`cwiseSign`
// (aether/expr/nodes/Elementwise.h, free functions — not `Expression`
// members, mirroring `CWiseScale`'s own call-site shape per that file's
// docstring) are DEVICEHOST-safe with no CUDA-specific behaviour a plain
// host build cannot already exercise.
//
// Every check compares against the same `std::`/comparison expression the
// implementation itself evaluates (same operation order), so
// EXPECT_DOUBLE_EQ/EXPECT_FLOAT_EQ (bit-for-bit), never a tolerance, except
// where noted.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Mat33d;
using aether::Vec3d;

class ExprElementwiseTest : public ::testing::Test { };

TEST_F(ExprElementwiseTest, CwiseMinBroadcastsScalarBoundComponentwise)
{
    Vec3d a{ -2.0, 0.5, 5.0 };
    Vec3d out = aether::cwiseMin(a, 1.0);
    EXPECT_DOUBLE_EQ(out(0), std::fmin(a(0), 1.0));
    EXPECT_DOUBLE_EQ(out(1), std::fmin(a(1), 1.0));
    EXPECT_DOUBLE_EQ(out(2), std::fmin(a(2), 1.0));
    EXPECT_DOUBLE_EQ(out(0), -2.0);
    EXPECT_DOUBLE_EQ(out(1), 0.5);
    EXPECT_DOUBLE_EQ(out(2), 1.0);
}

TEST_F(ExprElementwiseTest, CwiseMaxBroadcastsScalarBoundComponentwise)
{
    Vec3d a{ -2.0, 0.5, 5.0 };
    Vec3d out = aether::cwiseMax(a, 1.0);
    EXPECT_DOUBLE_EQ(out(0), std::fmax(a(0), 1.0));
    EXPECT_DOUBLE_EQ(out(1), std::fmax(a(1), 1.0));
    EXPECT_DOUBLE_EQ(out(2), std::fmax(a(2), 1.0));
    EXPECT_DOUBLE_EQ(out(0), 1.0);
    EXPECT_DOUBLE_EQ(out(1), 1.0);
    EXPECT_DOUBLE_EQ(out(2), 5.0);
}

TEST_F(ExprElementwiseTest, CwiseAbsComponentwise)
{
    Vec3d a{ -3.5, 0.0, 4.25 };
    Vec3d out = aether::cwiseAbs(a);
    EXPECT_DOUBLE_EQ(out(0), std::abs(a(0)));
    EXPECT_DOUBLE_EQ(out(1), std::abs(a(1)));
    EXPECT_DOUBLE_EQ(out(2), std::abs(a(2)));
    EXPECT_DOUBLE_EQ(out(0), 3.5);
    EXPECT_DOUBLE_EQ(out(2), 4.25);
}

TEST_F(ExprElementwiseTest, ClampBoundsComponentwiseToLoHi)
{
    // Below lo, inside [lo,hi], above hi — the whole range in one vector.
    Vec3d a{ -10.0, 0.5, 10.0 };
    Vec3d out = aether::clamp(a, -1.0, 1.0);
    EXPECT_DOUBLE_EQ(out(0), std::fmin(std::fmax(a(0), -1.0), 1.0));
    EXPECT_DOUBLE_EQ(out(1), std::fmin(std::fmax(a(1), -1.0), 1.0));
    EXPECT_DOUBLE_EQ(out(2), std::fmin(std::fmax(a(2), -1.0), 1.0));
    EXPECT_DOUBLE_EQ(out(0), -1.0);
    EXPECT_DOUBLE_EQ(out(1), 0.5);
    EXPECT_DOUBLE_EQ(out(2), 1.0);
}

TEST_F(ExprElementwiseTest, CwiseSignComponentwiseIncludingZerosAndNegatives)
{
    Vec3d a{ -7.0, 0.0, 3.0 };
    Vec3d out = aether::cwiseSign(a);
    EXPECT_DOUBLE_EQ(out(0), -1.0);
    EXPECT_DOUBLE_EQ(out(1), 0.0);
    EXPECT_DOUBLE_EQ(out(2), 1.0);

    Vec3d negZero{ -0.0, -0.0, -0.0 };
    Vec3d signOfNegZero = aether::cwiseSign(negZero);
    EXPECT_DOUBLE_EQ(signOfNegZero(0), 0.0);
    EXPECT_DOUBLE_EQ(signOfNegZero(1), 0.0);
    EXPECT_DOUBLE_EQ(signOfNegZero(2), 0.0);
}

TEST_F(ExprElementwiseTest, RankGenericWorksOnAMatrixItemToo)
{
    // One rank-generic operator set: the same cwiseMin/cwiseAbs used above
    // on a rank-1 Vec3d must also work, unmodified, on a rank-2 Mat33d.
    Mat33d a = Mat33d::Zeros();
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            a(r, c) = static_cast<double>(r) - static_cast<double>(c) * 2.0;

    Mat33d minOut   = aether::cwiseMin(a, 0.0);
    Mat33d absOut   = aether::cwiseAbs(a);
    Mat33d clampOut = aether::clamp(a, -1.0, 1.0);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(minOut(r, c), std::fmin(a(r, c), 0.0));
            EXPECT_DOUBLE_EQ(absOut(r, c), std::abs(a(r, c)));
            EXPECT_DOUBLE_EQ(clampOut(r, c), std::fmin(std::fmax(a(r, c), -1.0), 1.0));
        }
    }
}

TEST_F(ExprElementwiseTest, ViewMaterializedCwiseOpsMatchRawAccessorArithmetic)
{
    // The batched `view[i].get()` materialization path (mirrors
    // test_ExprGeometric.cpp's analogous batched test).
    constexpr std::size_t N = 4;
    aether::Array<double, 3> arr(N);
    auto v = arr.hostView();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            v(c, i) = static_cast<double>(c) * 3.0 - static_cast<double>(i);

    const aether::SampleIndex i2 = aether::SampleIndex::make(2);
    Vec3d a                      = v[i2].get();

    Vec3d clampActual = aether::clamp(a, -1.0, 1.0);
    for (std::size_t c = 0; c < 3; ++c)
        EXPECT_DOUBLE_EQ(clampActual(c), std::fmin(std::fmax(a(c), -1.0), 1.0));
}

} // namespace
} // namespace aether_tests
