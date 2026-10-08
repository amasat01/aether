// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Expression-depth canary (CUDA build; test_ExprDepth.cpp is the identical
// host-build twin).
//
// A chained expression of ~48 nested `Sum` nodes must compile and evaluate
// correctly in both modes — deep expression-template chains can silently
// blow past a compiler's template-instantiation-depth default. 48 terms of
// `Vec3d::filled(1.0)` summed gives a closed-form answer (48.0 per
// component), checked exactly (integer-valued sums have no rounding path
// to diverge on).

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Vec3d;

class ExprDepthTest : public ::testing::Test { };

TEST_F(ExprDepthTest, FortyEightTermChainedSumCompilesAndEvaluatesExactly)
{
    Vec3d a = Vec3d::filled(1.0);
    // clang-format off
    Vec3d sum48 = a + a + a + a + a + a + a + a + a + a
                + a + a + a + a + a + a + a + a + a + a
                + a + a + a + a + a + a + a + a + a + a
                + a + a + a + a + a + a + a + a + a + a
                + a + a + a + a + a + a + a + a; // 48 operands total
    // clang-format on
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(sum48(i), 48.0);
}

TEST_F(ExprDepthTest, DeepChainMixedWithScaleStillMatchesHandComputed)
{
    // A second, differently-shaped deep chain — alternating +/scale so the
    // canary also exercises Sum<CWiseScale<...>, ...> nesting, not just
    // Sum<Sum<...>, Item> (the pure-add case above).
    Vec3d a = Vec3d::filled(1.0);
    Vec3d b = Vec3d::filled(2.0);
    // clang-format off
    Vec3d chain = a + 1.0 * b + a + 1.0 * b + a + 1.0 * b + a + 1.0 * b
                + a + 1.0 * b + a + 1.0 * b + a + 1.0 * b + a + 1.0 * b
                + a + 1.0 * b + a + 1.0 * b + a + 1.0 * b + a + 1.0 * b;
    // clang-format on
    // 12 copies of `a` (1.0) + 12 copies of `1.0*b` (2.0) = 12 + 24 = 36.
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(chain(i), 36.0);
}

} // namespace
} // namespace aether_tests
