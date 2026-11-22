// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Edge rules (host build): the oracle is the rule stated in words, applied index by index to a small extent.

#include <cstddef>
#include <cstdint>

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

TEST_F(EdgeTest, WrapEqualsTheMathematicalRemainderForEveryExtentUpToSeven)
{
    for (std::int64_t extent = 1; extent <= 7; ++extent)
        for (std::int64_t index = -3 * extent - 2; index <= 3 * extent + 2; ++index) {
            std::int64_t expected = index;    // the reference: step by the extent into [0, extent)
            while (expected < 0) expected += extent;
            while (expected >= extent) expected -= extent;
            EXPECT_EQ(aether::edge::wrap(index, static_cast<std::size_t>(extent)), static_cast<std::size_t>(expected));
        }
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
