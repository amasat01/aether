// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Expression assignment tests (CUDA build; test_ExprAssign.cpp is the
// identical host-build twin): `RecursiveAssign`, `View::operator[]`'s
// `SampleRef` proxy (`=`, `+=`, `-=`, `.get()`), and `Item`'s
// converting-constructor/assignment-from-expression are all DEVICEHOST-safe
// with no CUDA-specific behaviour a plain host build cannot already
// exercise.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;
using aether::Vec3d;
using aether::Vec6d;

class ExprAssignTest : public ::testing::Test { };

TEST_F(ExprAssignTest, ViewSubscriptAssignOverwrites)
{
    constexpr std::size_t N = 5;
    Array<double, 3> a(N), b(N), out(N);
    auto av = a.hostView();
    auto bv = b.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            av(c, i) = static_cast<double>(c) + static_cast<double>(i);
            bv(c, i) = static_cast<double>(c) * 2.0;
        }
    }
    auto ov = out.hostView();
    for (std::size_t i = 0; i < N; ++i)
        ov[SampleIndex::make(i)] = av[SampleIndex::make(i)].get() + bv[SampleIndex::make(i)].get();

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) + bv(c, i));
}

TEST_F(ExprAssignTest, ViewSubscriptPlusEqualsAccumulates)
{
    constexpr std::size_t N = 3;
    Array<double, 3> out(N), delta(N);
    auto ov = out.hostView();
    auto dv = delta.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            ov(c, i) = static_cast<double>(i) - 1.0;
            dv(c, i) = static_cast<double>(c) + 0.5;
        }
    }
    // Snapshot the pre-op values via hand computation (ov(c,i) itself
    // changes below).
    double before[3][N];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            before[c][i] = ov(c, i);

    for (std::size_t i = 0; i < N; ++i)
        ov[SampleIndex::make(i)] += dv[SampleIndex::make(i)].get();

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(ov(c, i), before[c][i] + dv(c, i));
}

TEST_F(ExprAssignTest, ViewSubscriptMinusEqualsSubtracts)
{
    constexpr std::size_t N = 3;
    Array<double, 3> out(N), delta(N);
    auto ov = out.hostView();
    auto dv = delta.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            ov(c, i) = static_cast<double>(i) * 3.0;
            dv(c, i) = static_cast<double>(c) + 1.0;
        }
    }
    double before[3][N];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            before[c][i] = ov(c, i);

    for (std::size_t i = 0; i < N; ++i)
        ov[SampleIndex::make(i)] -= dv[SampleIndex::make(i)].get();

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(ov(c, i), before[c][i] - dv(c, i));
}

TEST_F(ExprAssignTest, SampleRefGetMaterializesAnItemSnapshot)
{
    constexpr std::size_t N = 4;
    Array<double, 3> a(N);
    auto av = a.hostView();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            av(c, i) = static_cast<double>(c) * 10.0 + static_cast<double>(i);

    const SampleIndex i2 = SampleIndex::make(2);
    Vec3d snap            = av[i2].get();
    for (std::size_t c = 0; c < 3; ++c)
        EXPECT_DOUBLE_EQ(snap(c), av(c, 2));

    // The snapshot is a genuine copy — mutating the source view afterwards
    // must not change it.
    av(0, 2) = 999.0;
    EXPECT_NE(snap(0), av(0, 2));
}

TEST_F(ExprAssignTest, ItemConstructedFromExpression)
{
    Vec3d a, b;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) + 1.0;
        b(i) = static_cast<double>(i) * 2.0;
    }
    Vec3d c(a + b); // converting ctor from expression (L5)
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) + b(i));
}

TEST_F(ExprAssignTest, ItemAssignedFromExpression)
{
    Vec3d a, b, c;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) - 3.0;
        b(i) = static_cast<double>(i) * 0.25;
        c(i) = -1.0; // pre-existing value, must be fully overwritten
    }
    c = a - b; // operator= from expression (L5)
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) - b(i));
}

TEST_F(ExprAssignTest, RankGenericAssignWorksForSixComponentVectors)
{
    // The same RecursiveAssign engine, unmodified, over a rank-1
    // element_extents<6> shape (a second fixed reference kernel, axpy6).
    constexpr std::size_t N = 3;
    Array<double, 6> out(N), a(N), b(N);
    auto ov = out.hostView();
    auto av = a.hostView();
    auto bv = b.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 6; ++c) {
            av(c, i) = static_cast<double>(c) - static_cast<double>(i);
            bv(c, i) = static_cast<double>(c) * 0.5;
        }
    }
    const double s = -2.0;
    for (std::size_t i = 0; i < N; ++i)
        ov[SampleIndex::make(i)] = av[SampleIndex::make(i)].get() + s * bv[SampleIndex::make(i)].get();

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) + s * bv(c, i));
}

TEST_F(ExprAssignTest, Vec6ItemFromExpressionMatchesHandComputed)
{
    Vec6d a, b;
    for (std::size_t i = 0; i < 6; ++i) {
        a(i) = static_cast<double>(i) + 0.5;
        b(i) = static_cast<double>(i) - 2.5;
    }
    Vec6d c = a + b;
    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) + b(i));
}

} // namespace
} // namespace aether_tests
