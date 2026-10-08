// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Math dispatch tests (CUDA build; test_Math.cpp is the identical
// host-build twin). Exercises
// `aether::math::{abs,floor,fma,fmax,fmin,min,pow,sqrt,isfinite,sign}`
// (aether/math/math.h) on the host route (`std::`) for both `float` and
// `double` — the device route (the CUDA math intrinsics, and for
// `sin`/`cos`/`sincos` the zero-stack `detail::cwSin`/`cwCos`/
// `cwSinCos` path in `aether/math/detail/BoundedTrig.h`) is exercised by
// `test_Math.cu`'s CUDA-mode kernel build and by
// `test_BoundedTrig.{cpp,cu}`'s dedicated ULP-parity suite. `sign` has no
// device/host split, so this file's host-route coverage is the whole story
// for it.
//
// `exp`/`exp2`/`exp10`/`log`/`log2`/`log10` (`aether/math/explog.h`) get
// their own paired test file, `test_MathExpLog.{cpp,cu}` — mirroring the
// `test_Expr<NodeFile>` convention `expr/nodes/` already uses (one
// node-header, one test pair).
//
// These are original value checks against `std::` known answers, one case
// per function, `float` and `double`.

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class MathDispatchTest : public ::testing::Test { };

TEST_F(MathDispatchTest, AbsMatchesStdForFloatAndDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::abs(-3.5), 3.5);
    EXPECT_DOUBLE_EQ(aether::math::abs(3.5), 3.5);
    EXPECT_FLOAT_EQ(aether::math::abs(-2.5f), 2.5f);
}

TEST_F(MathDispatchTest, FloorMatchesStdForFloatAndDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::floor(2.9), std::floor(2.9));
    EXPECT_DOUBLE_EQ(aether::math::floor(-2.1), std::floor(-2.1));
    EXPECT_FLOAT_EQ(aether::math::floor(2.9f), std::floor(2.9f));
}

TEST_F(MathDispatchTest, FmaMatchesStdForFloatAndDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::fma(2.0, 3.0, 4.0), std::fma(2.0, 3.0, 4.0));
    EXPECT_DOUBLE_EQ(aether::math::fma(2.0, 3.0, 4.0), 10.0);
    EXPECT_FLOAT_EQ(aether::math::fma(2.0f, 3.0f, 4.0f), std::fma(2.0f, 3.0f, 4.0f));
}

TEST_F(MathDispatchTest, FmaxFminMinMatchKnownAnswers)
{
    EXPECT_DOUBLE_EQ(aether::math::fmax(3.0, 7.0), 7.0);
    EXPECT_DOUBLE_EQ(aether::math::fmax(7.0, 3.0), 7.0);
    EXPECT_DOUBLE_EQ(aether::math::fmin(3.0, 7.0), 3.0);
    EXPECT_DOUBLE_EQ(aether::math::min(3.0, 7.0), 3.0);
    EXPECT_DOUBLE_EQ(aether::math::min(-1.0, -5.0), -5.0);
    EXPECT_FLOAT_EQ(aether::math::fmax(3.0f, 7.0f), 7.0f);
}

TEST_F(MathDispatchTest, PowMatchesStdForFloatAndDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::pow(2.0, 10.0), std::pow(2.0, 10.0));
    EXPECT_DOUBLE_EQ(aether::math::pow(2.0, 10.0), 1024.0);
    EXPECT_FLOAT_EQ(aether::math::pow(2.0f, 10.0f), std::pow(2.0f, 10.0f));
}

TEST_F(MathDispatchTest, SqrtMatchesStdForFloatAndDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::sqrt(2.0), std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(aether::math::sqrt(16.0), 4.0);
    EXPECT_FLOAT_EQ(aether::math::sqrt(16.0f), 4.0f);
}

TEST_F(MathDispatchTest, IsFiniteDetectsInfAndNan)
{
    EXPECT_TRUE(aether::math::isfinite(1.0));
    EXPECT_FALSE(aether::math::isfinite(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(aether::math::isfinite(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(aether::math::isfinite(1.0f));
    EXPECT_FALSE(aether::math::isfinite(std::numeric_limits<float>::infinity()));
}

TEST_F(MathDispatchTest, SignMatchesConventionForFloatAndDouble)
{
    // +1/-1/0, with x==0 (either IEEE sign) and NaN all landing on 0 — the
    // convention documented on aether::math::sign itself (math.h).
    EXPECT_DOUBLE_EQ(aether::math::sign(3.5), 1.0);
    EXPECT_DOUBLE_EQ(aether::math::sign(-3.5), -1.0);
    EXPECT_DOUBLE_EQ(aether::math::sign(0.0), 0.0);
    EXPECT_DOUBLE_EQ(aether::math::sign(-0.0), 0.0);
    EXPECT_DOUBLE_EQ(aether::math::sign(std::numeric_limits<double>::quiet_NaN()), 0.0);

    EXPECT_FLOAT_EQ(aether::math::sign(2.5f), 1.0f);
    EXPECT_FLOAT_EQ(aether::math::sign(-2.5f), -1.0f);
    EXPECT_FLOAT_EQ(aether::math::sign(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(aether::math::sign(-0.0f), 0.0f);
}

TEST_F(MathDispatchTest, SinCosSincosMatchStdOnTheHostRoute)
{
    // Host route (this TU's own build mode, and — for a CUDA-mode build —
    // the HOST PASS of test_Math.cu) is a plain std:: call for BOTH float
    // and double, regardless of AETHER_DEVICE_COMPILE: the zero-stack
    // detail::cwSin/cwCos/cwSinCos path is DEVICE-ONLY (see
    // aether/math/math.h's docstring). The device path's accuracy is
    // test_BoundedTrig.{cpp,cu}'s job.
    const double x = 1.23456789;
    EXPECT_DOUBLE_EQ(aether::math::sin(x), std::sin(x));
    EXPECT_DOUBLE_EQ(aether::math::cos(x), std::cos(x));

    double s = 0.0, c = 0.0;
    aether::math::sincos(x, &s, &c);
    EXPECT_DOUBLE_EQ(s, std::sin(x));
    EXPECT_DOUBLE_EQ(c, std::cos(x));

    const float xf = 0.5f;
    EXPECT_FLOAT_EQ(aether::math::sin(xf), std::sin(xf));
    EXPECT_FLOAT_EQ(aether::math::cos(xf), std::cos(xf));
}

} // namespace
} // namespace aether_tests
