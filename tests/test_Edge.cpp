// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Edge rules (host build): the oracle is the rule stated in words, applied index by index to a small extent.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/view/Edge.h>

namespace aether_tests {
namespace {

class EdgeTest : public ::testing::Test { };

TEST_F(EdgeTest, ClampHoldsAnIndexAtTheNearestValidOne)
{
    const int in[] = { -5, -1, 0, 2, 3, 4, 9 };
    const std::size_t out[] = { 0, 0, 0, 2, 3, 3, 3 };
    for (std::size_t k = 0; k < 7; ++k)
        EXPECT_EQ(aether::edge::clamp(in[k], 4), out[k]);
}

TEST_F(EdgeTest, WrapCountsAnOutOfRangeIndexAroundTheExtent)
{
    const int in[] = { -5, -4, -1, 0, 3, 4, 9 };
    const std::size_t out[] = { 3, 0, 3, 0, 3, 0, 1 };
    for (std::size_t k = 0; k < 7; ++k)
        EXPECT_EQ(aether::edge::wrap(in[k], 4), out[k]);
}

TEST_F(EdgeTest, InsideIsTrueExactlyForIndicesBelowTheExtent)
{
    EXPECT_FALSE(aether::edge::inside(-1, 4));
    EXPECT_TRUE(aether::edge::inside(0, 4));
    EXPECT_TRUE(aether::edge::inside(3, 4));
    EXPECT_FALSE(aether::edge::inside(4, 4));
}

TEST_F(EdgeTest, UncheckedPassesAValidIndexThrough)
{
    EXPECT_EQ(aether::edge::unchecked(2, 4), 2u);
}

} // namespace
} // namespace aether_tests
