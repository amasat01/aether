// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Slicing expression tests (CUDA build; test_ExprStructural.cpp is the
// identical host-build twin). `segment`/`head`/`tail`
// (aether/expr/nodes/Structural.h, member functions declared on
// `Expression` in aether/expr/Expression.h) are DEVICEHOST-safe with no
// CUDA-specific behaviour a plain host build cannot already exercise.
//
// aether's slicing is read-only (a `Segment` lazy expression node, not an
// assignable view), so this exercises segment/head/tail used as expression
// operands, not as write-side targets.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Item;
using aether::Vec3d;
using aether::Vec4d;

class ExprStructuralTest : public ::testing::Test { };

TEST_F(ExprStructuralTest, SegmentReadsContiguousComponents)
{
    Vec4d q;
    q(0) = 1.0;
    q(1) = 2.0;
    q(2) = 3.0;
    q(3) = 4.0;

    Item<double, 2> mid = q.segment<1, 2>();
    EXPECT_DOUBLE_EQ(mid(0), q(1));
    EXPECT_DOUBLE_EQ(mid(1), q(2));
}

TEST_F(ExprStructuralTest, HeadReadsLeadingComponents)
{
    Vec3d v;
    v(0) = 5.0;
    v(1) = 6.0;
    v(2) = 7.0;

    Item<double, 2> h = v.head<2>();
    EXPECT_DOUBLE_EQ(h(0), v(0));
    EXPECT_DOUBLE_EQ(h(1), v(1));
}

TEST_F(ExprStructuralTest, TailReadsTrailingComponents)
{
    Vec3d v;
    v(0) = 5.0;
    v(1) = 6.0;
    v(2) = 7.0;

    Item<double, 2> t = v.tail<2>();
    EXPECT_DOUBLE_EQ(t(0), v(1));
    EXPECT_DOUBLE_EQ(t(1), v(2));
}

TEST_F(ExprStructuralTest, SegmentComposesWithArithmetic)
{
    // segment()/head()/tail() are lazy nodes (isLeaf = false) — they must
    // compose with the arithmetic operators like any other expression.
    Vec4d a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    a(3) = 4.0;
    Vec4d b;
    b(0) = 10.0;
    b(1) = 20.0;
    b(2) = 30.0;
    b(3) = 40.0;

    Item<double, 2> sum = a.head<2>() + b.tail<2>();
    EXPECT_DOUBLE_EQ(sum(0), a(0) + b(2));
    EXPECT_DOUBLE_EQ(sum(1), a(1) + b(3));
}

TEST_F(ExprStructuralTest, ViewMaterializedSegmentMatchesRawAccessorArithmetic)
{
    // Batched materialization path (mirrors test_ExprReduce.cpp's analogous test).
    constexpr std::size_t N = 4;
    aether::Array<double, 4> arr(N);
    auto view = arr.hostView();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 4; ++c)
            view(c, i) = static_cast<double>(c) * 10.0 + static_cast<double>(i);

    const aether::SampleIndex i3 = aether::SampleIndex::make(3);
    Vec4d q                      = view[i3].get();

    Item<double, 3> back = q.tail<3>();
    EXPECT_DOUBLE_EQ(back(0), view(1, 3));
    EXPECT_DOUBLE_EQ(back(1), view(2, 3));
    EXPECT_DOUBLE_EQ(back(2), view(3, 3));
}

} // namespace
} // namespace aether_tests
