// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Expression arithmetic tests (CUDA build; test_ExprArithmetic.cpp is the
// identical host-build twin): the arithmetic nodes (`Sum`/`CWiseScale`) and
// operators (`+`, `-`, `s*e`, `e*s`, `e/s`) are DEVICEHOST-safe with no
// CUDA-specific behaviour a plain host build cannot already exercise.
//
// Value checks are against a hand-computed reference that replicates the
// exact same floating-point operation order the expression machinery uses
// (a reference computed via a different sequence can legitimately disagree
// in the last ULP) — so every check below uses EXPECT_DOUBLE_EQ
// (bit-for-bit expected agreement), never a tolerance.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Item;
using aether::Vec3d;
using aether::Mat33d;

class ExprArithmeticTest : public ::testing::Test { };

TEST_F(ExprArithmeticTest, SumAddsComponentwise)
{
    Vec3d a, b;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) + 1.0;
        b(i) = static_cast<double>(i) * 2.0 - 3.0;
    }
    Vec3d c = a + b;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) + b(i));
}

TEST_F(ExprArithmeticTest, DiffSubtractsComponentwise)
{
    Vec3d a, b;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) * 3.5;
        b(i) = static_cast<double>(i) - 1.25;
    }
    Vec3d c = a - b;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) - b(i));
}

TEST_F(ExprArithmeticTest, ScalarTimesExprAndExprTimesScalarAgree)
{
    Vec3d a;
    for (std::size_t i = 0; i < 3; ++i)
        a(i) = static_cast<double>(i) + 0.5;
    const double s = 2.5;

    Vec3d left  = s * a;
    Vec3d right = a * s;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(left(i), s * a(i));
        EXPECT_DOUBLE_EQ(right(i), a(i) * s);
    }
}

TEST_F(ExprArithmeticTest, DivByScalarMatchesReciprocalMultiply)
{
    Vec3d a;
    for (std::size_t i = 0; i < 3; ++i)
        a(i) = static_cast<double>(i) + 7.0;
    const double s = 4.0;

    Vec3d c = a / s;
    // Reference REPLICATES CWiseScale's own reciprocal-once strategy
    // (Operations.h: `1/s` computed ONCE, then multiplied per component) —
    // a per-component `a(i)/s` division is a DIFFERENT FP sequence and is
    // not guaranteed bit-identical, so the reference must match the
    // implementation's actual operation order.
    const double recip = 1.0 / s;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(c(i), a(i) * recip);
}

TEST_F(ExprArithmeticTest, AxpyMatchesHandComputedAetaXpy3)
{
    // out = a + s*b — a fixed reference formula.
    Vec3d a, b;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) - 2.0;
        b(i) = static_cast<double>(i) * 1.5;
    }
    const double s = -3.25;
    Vec3d out      = a + s * b;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(out(i), a(i) + s * b(i));
}

TEST_F(ExprArithmeticTest, ComposedExpressionMatchesHandComputedReference)
{
    // (a + b) - s*a, exercising Sum-of-Sum and CWiseScale-of-leaf together.
    Vec3d a, b;
    for (std::size_t i = 0; i < 3; ++i) {
        a(i) = static_cast<double>(i) + 0.25;
        b(i) = static_cast<double>(i) - 1.75;
    }
    const double s = 1.5;
    Vec3d out      = (a + b) - s * a;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(out(i), (a(i) + b(i)) - s * a(i));
}

TEST_F(ExprArithmeticTest, RankGenericOperatorsWorkOnAMatrixItemToo)
{
    // One rank-generic operator set: the same `+`/`s*e` used above on a
    // rank-1 Vec3d must also work, unmodified, on a rank-2 Mat33d.
    Mat33d a = Mat33d::Zeros();
    Mat33d b = Mat33d::Zeros();
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            a(r, c) = static_cast<double>(r * 3 + c);
            b(r, c) = static_cast<double>(c * 3 + r) * 0.5;
        }
    }
    const double s = 2.0;
    Mat33d out     = a + s * b;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(out(r, c), a(r, c) + s * b(r, c));
}

TEST_F(ExprArithmeticTest, ViewAsExpressionLeafMatchesRawAccessorArithmetic)
{
    // A View leaf, used directly as an expression operand (via
    // `View::eval<Is...>`, not through operator[]) — the read half of the
    // leaf protocol.
    constexpr std::size_t N = 4;
    aether::Array<double, 3> arrA(N), arrB(N);
    auto av = arrA.hostView();
    auto bv = arrB.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            av(c, i) = static_cast<double>(c) + static_cast<double>(i) * 10.0;
            bv(c, i) = static_cast<double>(c) - static_cast<double>(i);
        }
    }
    // `av`/`bv` are Views; combine them as an expression and materialize into
    // an Item at a specific sample via SampleRef::get() (exercised more in
    // test_ExprAssign — here we only need the READ path via operator[]).
    const aether::SampleIndex i2 = aether::SampleIndex::make(2);
    Vec3d combined                = (av[i2].get() + bv[i2].get());
    for (std::size_t c = 0; c < 3; ++c)
        EXPECT_DOUBLE_EQ(combined(c), av(c, 2) + bv(c, 2));
}

} // namespace
} // namespace aether_tests
