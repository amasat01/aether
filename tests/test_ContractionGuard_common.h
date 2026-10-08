// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The host error-free transformations stay exact under FMA contraction.
// Shared by test_ContractionGuard.cpp (C++ build) and test_ContractionGuard.cu
// (CUDA build); both TUs are compiled with -ffp-contract=fast on purpose
// (tests/CMakeLists.txt), the gcc/clang default and what a consumer's fast
// host profile asks for.
//
// Known answer, by construction: a = b = 1 + 2^-30, so a*b = 1 + 2^-29 +
// 2^-60 exactly and its rounded product is p = 1 + 2^-29. A two-sum residual
// is defined on the ROUNDED operands, so
//   * twoSumErr_(p, -1, p - 1) is 0 (p - 1 = 2^-29 exactly), and
//   * compensatedSum_ of the term a*b into (1, 0) leaves value 2 + 2^-29,
//     comp 0 (1 + p is exact).
// If the product were fused into the sums (AETHER_FP_BARRIER removed from
// aether/accum/atomic.h), the residual would see the unrounded 2^-60 tail
// and both come out 2^-60 when the barriers are removed, which this test
// confirms fails without them.

#include <cmath>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

// Runtime operands the optimiser cannot fold.
double opaque(double x)
{
    volatile double v = x;
    return v;
}

TEST(ContractionGuard, TwoSumResidualOfARoundedProductIsExact)
{
    const double a = opaque(1.0 + 0x1p-30), b = opaque(1.0 + 0x1p-30), c = opaque(-1.0);
    const double p = a * b;
    const double s = p + c;
    const double e = aether::accum::atomic::detail::twoSumErr_(p, c, s);
    // (p itself is not inspected here: a non-arithmetic use of the product
    // would by itself stop gcc from contracting it, and the row would pass
    // without the guard.)
    EXPECT_EQ(s, 0x1p-29);
    EXPECT_EQ(e, 0.0) << "residual " << e << ": a product was fused into the two-sum";
}

TEST(ContractionGuard, CompensatedSumOfAProductTermIsExact)
{
    const double a = opaque(1.0 + 0x1p-30), b = opaque(1.0 + 0x1p-30);
    double value = opaque(1.0), comp = opaque(0.0);
    aether::accum::atomic::detail::compensatedSum_(&value, &comp, a * b);
    EXPECT_EQ(value, 2.0 + 0x1p-29);
    EXPECT_EQ(comp, 0.0) << "compensation " << comp << ": the term was fused into the sum";
}

// Non-vacuity of the subject: the same expression without the guard really
// is contracted in this TU (otherwise the two rows above prove nothing).
TEST(ContractionGuard, ThisTuContracts)
{
    const double a = opaque(1.0 + 0x1p-30), b = opaque(1.0 + 0x1p-30), c = opaque(-1.0);
#if defined(__FMA__) || defined(__AVX512F__) || defined(__aarch64__)
    const double fused = a * b + c; // contracted: the 2^-60 tail survives
    EXPECT_EQ(fused, 0x1p-29 + 0x1p-60);
#else
    GTEST_SKIP() << "no hardware FMA on this target: nothing to contract";
    (void)a, (void)b, (void)c;
#endif
}

} // namespace
} // namespace aether_tests
