// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// math/explog.h dispatch tests (CUDA build; test_MathExpLog.cpp is the
// identical host-build twin). Exercises
// `aether::math::{exp,exp2,exp10,log,log2,log10}` (aether/math/explog.h) on
// the host route (`std::`) for both `float` and `double` — the device route
// (the CUDA math intrinsics `expf`/`::exp`/...) is exercised by this file's
// `.cu` pairing's CUDA-mode kernel build; there is no separate
// device-accuracy ULP suite for this facade (unlike sin/cos/sincos's
// `test_BoundedTrig.{cpp,cu}`) since the device leg is a bare CUDA math
// intrinsic call, not a bounded-argument algorithm this repo owns.
//
// Tolerance: on aether's actual host route, `aether::math::exp` etc. call
// `std::exp` etc. verbatim (no alternate formula) — so, mirroring
// `test_Math.cpp`'s own established convention for this exact situation
// (`pow`/`sqrt`/`abs`/...), every check below is bit-exact
// (`EXPECT_DOUBLE_EQ`/`EXPECT_FLOAT_EQ`), not a tolerance — a stronger
// check than any epsilon would be.

#include <cmath>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class MathExpLogTest : public ::testing::Test { };

TEST_F(MathExpLogTest, ExpSweepMatchesStdForDouble)
{
    const double xs[] = { -10.0, -5.0, -1.0, -0.001, 0.0, 0.001, 1.0, 2.5, 10.0 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::exp(x), std::exp(x));
}

TEST_F(MathExpLogTest, Exp2SweepMatchesStdForDouble)
{
    const double xs[] = { -8.0, -1.0, -0.001, 0.0, 0.001, 1.0, 3.0, 16.0 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::exp2(x), std::exp2(x));
}

TEST_F(MathExpLogTest, Exp10SweepMatchesStdForDouble)
{
    // Host route: std::pow(T(10), x) — the STDEXPR for exp10 (no ::exp10
    // on host, a non-standard GNU extension; see explog.h's docstring).
    const double xs[] = { -4.0, -1.0, -0.001, 0.0, 0.001, 1.0, 2.0, 6.0 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::exp10(x), std::pow(10.0, x));
}

TEST_F(MathExpLogTest, LogSweepMatchesStdForDouble)
{
    const double xs[] = { 0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1.0e6 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::log(x), std::log(x));
}

TEST_F(MathExpLogTest, Log2SweepMatchesStdForDouble)
{
    const double xs[] = { 0.001, 0.1, 0.5, 1.0, 2.0, 8.0, 1024.0 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::log2(x), std::log2(x));
}

TEST_F(MathExpLogTest, Log10SweepMatchesStdForDouble)
{
    const double xs[] = { 0.001, 0.1, 0.5, 1.0, 10.0, 1000.0, 1.0e6 };
    for (const double x : xs)
        EXPECT_DOUBLE_EQ(aether::math::log10(x), std::log10(x));
}

TEST_F(MathExpLogTest, KnownIdentitiesHoldForDouble)
{
    EXPECT_DOUBLE_EQ(aether::math::exp(0.0), 1.0);
    EXPECT_DOUBLE_EQ(aether::math::log(1.0), 0.0);
    EXPECT_DOUBLE_EQ(aether::math::exp2(0.0), 1.0);
    EXPECT_DOUBLE_EQ(aether::math::log2(1.0), 0.0);
    EXPECT_DOUBLE_EQ(aether::math::exp10(0.0), 1.0);
    EXPECT_DOUBLE_EQ(aether::math::log10(1.0), 0.0);
    // log2(8.0)/log10(1000.0) against std:: rather than a bare literal `3.0`
    // — a correctly-rounded log implementation lands exactly on an exact
    // power, but this test does not need to ASSUME that; std:: is what the
    // host route actually forwards to (bit-exact by construction either way).
    EXPECT_DOUBLE_EQ(aether::math::log2(8.0), std::log2(8.0));
    EXPECT_DOUBLE_EQ(aether::math::log10(1000.0), std::log10(1000.0));
}

TEST_F(MathExpLogTest, SweepMatchesStdForFloat)
{
    const float xs[] = { -5.0f, -0.5f, 0.0f, 0.5f, 2.0f, 5.0f };
    for (const float x : xs) {
        EXPECT_FLOAT_EQ(aether::math::exp(x), std::exp(x));
        EXPECT_FLOAT_EQ(aether::math::exp2(x), std::exp2(x));
        EXPECT_FLOAT_EQ(aether::math::exp10(x), std::pow(10.0f, x));
    }
    const float pxs[] = { 0.01f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f };
    for (const float x : pxs) {
        EXPECT_FLOAT_EQ(aether::math::log(x), std::log(x));
        EXPECT_FLOAT_EQ(aether::math::log2(x), std::log2(x));
        EXPECT_FLOAT_EQ(aether::math::log10(x), std::log10(x));
    }
}

} // namespace
} // namespace aether_tests
