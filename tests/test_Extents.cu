// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Extents tests (CUDA build; test_Extents.cpp is the identical host-build
// twin): Extents is DEVICEHOST-safe with no CUDA-specific behaviour that a
// plain host build cannot already exercise.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/layout/Extents.h>

namespace aether_tests {
namespace {

class ExtentsTest : public ::testing::Test { };

TEST_F(ExtentsTest, AllStaticIsDefaultConstructible)
{
    aether::extents<3, 4> ex;
    EXPECT_EQ(ex.rank(), 2u);
    EXPECT_EQ(ex.rank_dynamic(), 0u);
    EXPECT_EQ(ex.extent(0), 3u);
    EXPECT_EQ(ex.extent(1), 4u);
}

TEST_F(ExtentsTest, StaticExtentReportsDynSentinelForDynamicModes)
{
    EXPECT_EQ((aether::extents<3, aether::dyn>::static_extent(0)), 3u);
    EXPECT_EQ((aether::extents<3, aether::dyn>::static_extent(1)), aether::dyn);
}

TEST_F(ExtentsTest, OneDynamicModeResolvesFromConstructorArgument)
{
    aether::extents<3, aether::dyn> ex(7);
    EXPECT_EQ(ex.rank(), 2u);
    EXPECT_EQ(ex.rank_dynamic(), 1u);
    EXPECT_EQ(ex.extent(0), 3u);
    EXPECT_EQ(ex.extent(1), 7u);
}

TEST_F(ExtentsTest, RankOneDynamicScalarExtents)
{
    aether::extents<aether::dyn> ex(11);
    EXPECT_EQ(ex.rank(), 1u);
    EXPECT_EQ(ex.rank_dynamic(), 1u);
    EXPECT_EQ(ex.extent(0), 11u);
}

TEST_F(ExtentsTest, ThreeModeMixedStaticAndDynamic)
{
    aether::extents<3, 3, aether::dyn> ex(9);
    EXPECT_EQ(ex.rank(), 3u);
    EXPECT_EQ(ex.rank_dynamic(), 1u);
    EXPECT_EQ(ex.extent(0), 3u);
    EXPECT_EQ(ex.extent(1), 3u);
    EXPECT_EQ(ex.extent(2), 9u);
}

TEST_F(ExtentsTest, MultipleDynamicModesInOrder)
{
    // extents<dyn, 3, dyn>: two dynamic modes; constructor arguments apply
    // to the dyn slots IN ORDER (first dyn slot gets the first argument).
    aether::extents<aether::dyn, 3, aether::dyn> ex(5, 8);
    EXPECT_EQ(ex.rank_dynamic(), 2u);
    EXPECT_EQ(ex.extent(0), 5u);
    EXPECT_EQ(ex.extent(1), 3u);
    EXPECT_EQ(ex.extent(2), 8u);
}

TEST_F(ExtentsTest, ConstexprEvaluatesAtCompileTime)
{
    constexpr aether::extents<3, 3> ex;
    static_assert(ex.rank() == 2, "extents::rank() must be constexpr");
    static_assert(ex.extent(0) == 3, "extents::extent() must be constexpr");
    EXPECT_EQ(ex.extent(1), 3u);
}

} // namespace
} // namespace aether_tests
