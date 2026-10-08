// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Item tests (CUDA build; test_Item.cpp is the identical host-build twin):
// Item is DEVICEHOST-safe with no CUDA-specific behaviour a plain host
// build cannot already exercise.

#include <cstddef>
#include <type_traits>

#include <gtest/gtest.h>

#include <aether/view/Item.h>

namespace aether_tests {
namespace {

class ItemTest : public ::testing::Test { };

TEST_F(ItemTest, DefaultConstructedIsZeroInitialized)
{
    aether::Item<double, 3> v;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(v(i), 0.0);
}

TEST_F(ItemTest, ZerosOnesFilledFactories)
{
    auto z = aether::Item<double, 4>::Zeros();
    auto o = aether::Item<double, 4>::Ones();
    auto f = aether::Item<double, 4>::filled(7.5);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(z(i), 0.0);
        EXPECT_EQ(o(i), 1.0);
        EXPECT_EQ(f(i), 7.5);
    }
}

TEST_F(ItemTest, OperatorParenReadWriteRoundTrip)
{
    aether::Item<double, 3> v;
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    EXPECT_EQ(v(0), 1.0);
    EXPECT_EQ(v(1), 2.0);
    EXPECT_EQ(v(2), 3.0);
}

TEST_F(ItemTest, CompileTimeGetMatchesRuntimeOperatorParen)
{
    aether::Item<double, 3> v;
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    EXPECT_EQ((v.get<0>()), v(0));
    EXPECT_EQ((v.get<1>()), v(1));
    EXPECT_EQ((v.get<2>()), v(2));
}

TEST_F(ItemTest, MatrixRowMajorAccessMatchesHandComputedOffset)
{
    // Item<T,R,C>: (r,c) -> offset r*C+c (matches layout_right's formula).
    // Deliberately non-square (3x4, not 3x3): a second TEST_F in this same
    // TU below instantiates the canonical Mat33-shaped Item<double,3,3> —
    // re-instantiating that EXACT specialization from a second gtest
    // TEST_F body in one translation unit trips an nvcc host-frontend
    // parser defect ("template argument N is invalid") that is otherwise
    // unrelated to Item's own correctness (isolated and confirmed via a
    // minimal repro against the fetched GoogleTest headers; not a locked
    // Item.h shape issue).
    constexpr std::size_t R = 3, C = 4;
    auto m           = aether::Item<double, R, C>::Zeros();
    std::size_t next = 0;
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t c = 0; c < C; ++c)
            m(r, c) = static_cast<double>(next++);
    EXPECT_EQ(m.data()[0 * C + 0], m(0, 0));
    EXPECT_EQ(m.data()[1 * C + 2], m(1, 2));
    EXPECT_EQ(m.data()[2 * C + 1], m(2, 1));
}

TEST_F(ItemTest, SizeReportsElementCount)
{
    using Vec3 = aether::Item<double, 3>;
    using Mat33 = aether::Item<double, 3, 3>;
    EXPECT_EQ(Vec3::size(), 3u);
    EXPECT_EQ(Mat33::size(), 9u);
}

TEST_F(ItemTest, IsTriviallyCopyable)
{
    // Local aliases (not the bare multi-argument template) inside the trait
    // — avoids embedding a nested `Item<T, A, B>` comma list directly inside
    // a macro-adjacent expression.
    using Vec3 = aether::Item<double, 3>;
    using Mat33 = aether::Item<double, 3, 3>;
    static_assert(std::is_trivially_copyable_v<Vec3>, "Item must be trivially copyable (L5)");
    static_assert(std::is_trivially_copyable_v<Mat33>, "Item must be trivially copyable (L5)");
    SUCCEED();
}

TEST_F(ItemTest, ConstexprConstructibleAndFillable)
{
    constexpr auto z = aether::Item<double, 3>::Zeros();
    static_assert(z.get<0>() == 0.0, "Item::Zeros() must be constexpr");
    constexpr auto o = aether::Item<double, 3>::Ones();
    static_assert(o.get<1>() == 1.0, "Item::Ones() must be constexpr");
    EXPECT_EQ(z.get<0>(), 0.0);
    EXPECT_EQ(o.get<1>(), 1.0);
}

// The N-scalar constructor.

TEST_F(ItemTest, NScalarConstructorVectorMatchesOperatorParen)
{
    aether::Item<double, 3> v{ 1.0, 2.0, 3.0 };
    EXPECT_EQ(v(0), 1.0);
    EXPECT_EQ(v(1), 2.0);
    EXPECT_EQ(v(2), 3.0);
}

TEST_F(ItemTest, NScalarConstructorMatrixIsRowMajor)
{
    // Item<T,2,2>: (r,c) -> offset r*2+c — the N-scalar ctor lists values in
    // that SAME row-major order (matches operator()/get<>()'s convention).
    aether::Item<double, 2, 2> m{ 1.0, 2.0, 3.0, 4.0 };
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 1), 2.0);
    EXPECT_EQ(m(1, 0), 3.0);
    EXPECT_EQ(m(1, 1), 4.0);
}

TEST_F(ItemTest, NScalarConstructorIsConstexpr)
{
    constexpr aether::Item<double, 3> v{ 1.0, 2.0, 3.0 };
    static_assert(v.get<0>() == 1.0, "Item's N-scalar ctor must be constexpr");
    static_assert(v.get<2>() == 3.0, "Item's N-scalar ctor must be constexpr");
    EXPECT_EQ(v.get<1>(), 2.0);
}

} // namespace
} // namespace aether_tests
