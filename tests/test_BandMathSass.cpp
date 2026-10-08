// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathSass.cpp
 * @brief CPP_MODE twin of `test_BandMathSass.cu`: no SASS to disassemble in
 *        a pure-C++ build, so this TU drops the `__global__` kernels and
 *        instead smoke-checks that the SAME `aether::banded::detail::`
 *        bodies the CUDA subjects audit are live and compute a non-trivial
 *        answer on the host arm too.
 */

#include <gtest/gtest.h>

#include "aether/banded/banded.h"

namespace aether_tests {
namespace bandmath_sass {

using aether::banded::Band;
namespace bd = aether::banded::detail;

TEST(BandMathSass, HostOpsAreLiveAndComputeTheClaimedOp)
{
    const Band a{ 1.25f, 0.0f, 0.0f };
    const Band b{ -2.0f, 0.0f, 0.0f };

    EXPECT_EQ(bd::abs(b).hi, 2.0f);
    EXPECT_EQ(bd::copysign(a, b).hi, -1.25f);
    EXPECT_EQ(bd::fmax(a, b).hi, a.hi);
    EXPECT_EQ(bd::fmin(a, b).hi, b.hi);
    EXPECT_FLOAT_EQ(bd::add(a, b).hi + bd::add(a, b).lo, -0.75f);
    EXPECT_FLOAT_EQ(bd::sub(a, b).hi + bd::sub(a, b).lo, 3.25f);
    EXPECT_FLOAT_EQ(bd::mul(a, b).hi + bd::mul(a, b).lo, -2.5f);

    // exp/log/pow live on the host arm too.
    const Band one{ 1.0f, 0.0f, 0.0f };
    const Band two{ 2.0f, 0.0f, 0.0f };
    EXPECT_NEAR(bd::exp(Band{ 0.0f, 0.0f, 0.0f }).hi, 1.0f, 1e-6f);
    EXPECT_NEAR(bd::log(one).hi, 0.0f, 1e-6f);
    EXPECT_NEAR(bd::pow(two, Band{ 10.0f, 0.0f, 0.0f }).hi, 1024.0f, 1e-2f);
    // sqrt/rsqrt/rsqrtCube/cbrt/hypot.
    const Band four{ 4.0f, 0.0f, 0.0f };
    const Band nine{ 9.0f, 0.0f, 0.0f };
    EXPECT_NEAR(bd::sqrt_(four).hi, 2.0f, 1e-5f);
    EXPECT_NEAR(bd::rsqrtIeee(four).hi, 0.5f, 1e-5f);
    EXPECT_NEAR(bd::rsqrtCube(four).hi, 0.125f, 1e-5f);
    EXPECT_NEAR(bd::cbrt(nine).hi, 2.08008f, 1e-4f);
    EXPECT_NEAR(bd::hypot(Band{ 3.0f, 0.0f, 0.0f }, four).hi, 5.0f, 1e-5f);

    // sin/cos/sincos live on the host arm too.
    EXPECT_NEAR(bd::sin(Band{ 0.0f, 0.0f, 0.0f }).hi, 0.0f, 1e-6f);
    EXPECT_NEAR(bd::cos(Band{ 0.0f, 0.0f, 0.0f }).hi, 1.0f, 1e-6f);
    Band s, c;
    bd::sincos(one, s, c);
    EXPECT_NEAR(s.hi, 0.841470985f, 1e-5f);
    EXPECT_NEAR(c.hi, 0.540302306f, 1e-5f);
    // floor/ceil/round/trunc/fdim/fmod/fma.
    const Band twoAndHalf{ 2.5f, 0.0f, 0.0f };
    const Band negTwoAndHalf{ -2.5f, 0.0f, 0.0f };
    EXPECT_EQ(bd::floor(twoAndHalf).hi, 2.0f);
    EXPECT_EQ(bd::ceil(twoAndHalf).hi, 3.0f);
    EXPECT_EQ(bd::round(twoAndHalf).hi, 3.0f);        // half away from zero
    EXPECT_EQ(bd::round(negTwoAndHalf).hi, -3.0f);
    EXPECT_EQ(bd::trunc(negTwoAndHalf).hi, -2.0f);    // toward zero
    EXPECT_EQ(bd::fdim(a, b).hi, a.hi - b.hi);         // a=1.25 > b=-2.0
    EXPECT_EQ(bd::fdim(b, a).hi, 0.0f);                // b < a
    EXPECT_NEAR(bd::fmod(Band{ 7.5f, 0.0f, 0.0f }, Band{ 2.0f, 0.0f, 0.0f }).hi, 1.5f, 1e-6f);
    EXPECT_NEAR(bd::fma(a, b, four).hi, a.hi * b.hi + four.hi, 1e-5f);
}

} // namespace bandmath_sass
} // namespace aether_tests
