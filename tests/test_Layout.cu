// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Layout tests (CUDA build; test_Layout.cpp is the identical host-build
// twin): the mapping formulas are pure host-computable arithmetic, with no
// CUDA-specific behaviour a plain host build cannot already exercise.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/layout/Extents.h>
#include <aether/layout/Layout.h>
#include <aether/layout/detail/Carray.h>

namespace aether_tests {
namespace {

class LayoutTest : public ::testing::Test { };

TEST_F(LayoutTest, RowMajorRank1ScalarIsIdentity)
{
    aether::extents<aether::dyn> ex(10);
    aether::layout_right::mapping<aether::extents<aether::dyn>> map(ex);
    for (std::size_t i = 0; i < 10; ++i)
        EXPECT_EQ(map(i), i);
    EXPECT_EQ(map.required_span_size(), 10u);
}

TEST_F(LayoutTest, RowMajorVectorExtentsSoaFormula)
{
    // extents<3, dyn>(N) -> operator()(c, i) == c*N + i — THE SoA anchor (L3).
    constexpr std::size_t N = 5;
    aether::extents<3, aether::dyn> ex(N);
    aether::layout_right::mapping<aether::extents<3, aether::dyn>> map(ex);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(map(c, i), c * N + i);
    EXPECT_EQ(map.required_span_size(), 3u * N);
}

TEST_F(LayoutTest, RowMajorMatrixExtentsFormula)
{
    // extents<R, C, dyn>(N) -> operator()(r, c, n) == (r*C + c)*N + n (L3).
    constexpr std::size_t R = 3, C = 3, N = 7;
    aether::extents<R, C, aether::dyn> ex(N);
    aether::layout_right::mapping<aether::extents<R, C, aether::dyn>> map(ex);
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t n = 0; n < N; ++n)
                EXPECT_EQ(map(r, c, n), (r * C + c) * N + n);
    EXPECT_EQ(map.required_span_size(), R * C * N);
}

TEST_F(LayoutTest, LayoutStrideDotProductMatchesHandComputedOffsets)
{
    aether::extents<3, aether::dyn> ex(4);
    // Deliberately non-default strides (a gapped layout): stride(0)=1
    // (component-minor), stride(1)=8 (a 4-element gap per sample beyond the
    // tight 3-component width).
    aether::detail::Carray<std::size_t, 2> strides{ 1, 8 };
    aether::layout_stride::mapping<aether::extents<3, aether::dyn>> map(ex, strides);

    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_EQ(map(c, i), c * 1 + i * 8);

    // required_span_size() = 1 + sum((extent(r)-1)*stride(r))
    //                       = 1 + (3-1)*1 + (4-1)*8 = 1 + 2 + 24 = 27
    EXPECT_EQ(map.required_span_size(), 27u);
}

TEST_F(LayoutTest, LayoutStrideRequiredSpanSizeIsZeroWhenAnyExtentIsZero)
{
    aether::extents<3, aether::dyn> ex(0);
    aether::detail::Carray<std::size_t, 2> strides{ 1, 8 };
    aether::layout_stride::mapping<aether::extents<3, aether::dyn>> map(ex, strides);
    EXPECT_EQ(map.required_span_size(), 0u);
}

TEST_F(LayoutTest, ConstexprEvaluatesAtCompileTime)
{
    constexpr aether::extents<3, 3> ex;
    constexpr aether::layout_right::mapping<aether::extents<3, 3>> map(ex);
    static_assert(map(1, 2) == 5, "layout_right::mapping::operator() must be constexpr");
    static_assert(map.required_span_size() == 9, "required_span_size() must be constexpr");
    EXPECT_EQ(map(1, 2), 5u);
}

} // namespace
} // namespace aether_tests
