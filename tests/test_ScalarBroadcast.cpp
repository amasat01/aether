// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Scalar-broadcast tests (host / AETHER_CPP_MODE build;
// test_ScalarBroadcast.cu is the identical CUDA-build twin). Covers
// `aether::detail::Constant<T,Extents>` and `SampleRef`'s scalar `= += -=`
// overloads: a scalar RHS routes through the same
// assign/assignAdd/assignSub terminal an expression RHS does, so a scalar
// store and an equivalent expression store must be bit-identical, and a
// `Constant<BandedReal,...>` broadcast must compute in `Band` with no
// `Constant`-specific code.
//
// The compile-fail complement — `v[i] = 2` (int into a `double` view) must
// not compile — cannot live here by construction: see
// tests/compile_fail/check_constview_no_assign.sh.

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "aether/banded/banded.h"

namespace aether_tests {
namespace {

using aether::banded::Band;
using aether::banded::BandedReal;

using aether::Device;
using aether::extents;
using aether::Item;
using aether::Mat33d;
using aether::SampleIndex;
using aether::Vec3d;

class ScalarBroadcastTest : public ::testing::Test { };

TEST_F(ScalarBroadcastTest, AssignScalarSetsEveryComponentOfAVectorRankView)
{
    constexpr std::size_t N = 4;
    std::vector<double> buf(3 * N, -1.0);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), Device(kDLCPU), N);

    for (std::size_t i = 0; i < N; ++i)
        v[SampleIndex::make(i)] = 2.0;

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(v(c, i), 2.0);
}

TEST_F(ScalarBroadcastTest, AssignScalarSetsEveryComponentOfAMatrixRankView)
{
    constexpr std::size_t N = 2;
    std::vector<double> buf(3 * 3 * N, -1.0);
    auto v = aether::make_view<double, 3, 3, aether::dyn>(buf.data(), Device(kDLCPU), N);

    v[SampleIndex::make(0)] = 7.5;

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(v(r, c, 0), 7.5);
}

TEST_F(ScalarBroadcastTest, PlusEqualsScalarAddsToEveryComponent)
{
    Vec3d a{ 1.0, 2.0, 3.0 };
    std::vector<double> buf(3, 0.0);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), Device(kDLCPU), 1);
    v[SampleIndex::make(0)] = a;

    v[SampleIndex::make(0)] += 0.5;

    EXPECT_DOUBLE_EQ(v(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(v(1, 0), 2.5);
    EXPECT_DOUBLE_EQ(v(2, 0), 3.5);
}

TEST_F(ScalarBroadcastTest, MinusEqualsScalarSubtractsFromEveryComponent)
{
    Vec3d a{ 1.0, 2.0, 3.0 };
    std::vector<double> buf(3, 0.0);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), Device(kDLCPU), 1);
    v[SampleIndex::make(0)] = a;

    v[SampleIndex::make(0)] -= 0.25;

    EXPECT_DOUBLE_EQ(v(0, 0), 0.75);
    EXPECT_DOUBLE_EQ(v(1, 0), 1.75);
    EXPECT_DOUBLE_EQ(v(2, 0), 2.75);
}

TEST_F(ScalarBroadcastTest, ScalarStoreIsBitIdenticalToTheSameStoreWrittenAsAnItemExpression)
{
    // `v[i] = 2.0` (the scalar overload, `Constant` leaf) vs `w[i] =
    // Item<double,3>::filled(2.0)` (an ordinary expression leaf) both route
    // through the same assign terminal, so the stores must be bit-for-bit
    // identical, never merely numerically close.
    std::vector<double> vbuf(3, -9.0), wbuf(3, -9.0);
    auto v = aether::make_view<double, 3, aether::dyn>(vbuf.data(), Device(kDLCPU), 1);
    auto w = aether::make_view<double, 3, aether::dyn>(wbuf.data(), Device(kDLCPU), 1);
    const SampleIndex i0 = SampleIndex::make(0);

    v[i0] = 2.0;
    w[i0] = Item<double, 3>::filled(2.0);

    for (std::size_t c = 0; c < 3; ++c) {
        EXPECT_EQ(v(c, 0), w(c, 0));
        // Bit-for-bit, not merely value-equal (guards a hypothetical future
        // change that computes the scalar store via a different rounding
        // path than the expression store).
        EXPECT_EQ(0, std::memcmp(&v(c, 0), &w(c, 0), sizeof(double)));
    }
}

TEST_F(ScalarBroadcastTest, ConstantFactoryBroadcastsOverAnItemShapedExpression)
{
    // `aether::constant<Extents>(v)` used directly in an expression tree,
    // not just via `SampleRef`'s internal wrap.
    Vec3d a{ 1.0, 2.0, 3.0 };
    Vec3d out = a + aether::constant<extents<3>>(10.0);
    EXPECT_DOUBLE_EQ(out(0), 11.0);
    EXPECT_DOUBLE_EQ(out(1), 12.0);
    EXPECT_DOUBLE_EQ(out(2), 13.0);
}

TEST_F(ScalarBroadcastTest, BandedRealViewBroadcastComputesInBandAndPacksOnce)
{
    // Reuses the witness shape from test_BandedReal_common.h's
    // ExpressionAlgebraOverBandedRealLeavesComputesInBandAndPacksOnce: two
    // independently hand-computed references over the same chain
    // `((a + b) - a) * s` — one packing once (the working-carrier policy
    // this row measures), one packing at every node — chosen so the two
    // disagree (0.75 vs 1.0), making the comparison decidable on codec
    // words, never a tolerance. Here `b` is bound through
    // `aether::constant<Extents>(v)` — a `Constant<BandedReal,...>`
    // broadcast leaf composed into the same expression tree via the
    // ordinary `+`/`-`/`*` operators — instead of a second View leaf, so
    // the comparison shows `Constant` costs the same zero extra packs a
    // View leaf does.
    constexpr std::size_t D = 3;

    const BandedReal wA = BandedReal::fromDouble(1.0);
    const BandedReal wB = BandedReal::fromDouble(3.0 * std::ldexp(1.0, -57));
    const BandedReal wS = BandedReal::fromDouble(std::ldexp(1.0, 55));

    // Once-packed hand reference: the whole chain in Band, a single encode.
    const BandedReal wantOnce = BandedReal::fromBand(((wA + wB) - wA) * wS);
    // Per-node-packed hand reference: an encode at every node.
    const BandedReal node1       = wA + wB;
    const BandedReal node2       = node1 - wA;
    const BandedReal wantPerNode = node2 * wS;

    EXPECT_DOUBLE_EQ(wantOnce.toDouble(), 0.75);
    EXPECT_DOUBLE_EQ(wantPerNode.toDouble(), 1.0);
    // Non-vacuity control for the witness itself (mirrors
    // test_BandedReal_common.h's identical control): if the two references
    // agreed, every assertion below would pass under EITHER policy.
    ASSERT_NE(wantOnce.toBits(), wantPerNode.toBits())
        << "the witness does not discriminate: the once-packed and per-node "
           "references are the same codec word, so the row is inert";

    // `Constant<BandedReal,...>` must itself compute in `Band` with no
    // `Constant`-specific code — the type-level half of the claim, checked
    // before the value-level one below.
    static_assert(std::is_same_v<aether::working_type_t<BandedReal>, Band>,
        "BandedReal must map to the Band working carrier");

    std::vector<BandedReal> waBuf(D, wA), woBuf(D);
    const aether::Device dev(kDLCPU);
    auto wav = aether::make_view<BandedReal, D, aether::dyn>(waBuf.data(), dev, 1);
    auto wov = aether::make_view<BandedReal, D, aether::dyn>(woBuf.data(), dev, 1);
    const SampleIndex i0 = SampleIndex::make(0);

    const aether::Item<BandedReal, D> A = wav[i0].get();
    // `b` bound through the `constant<Extents>` factory instead of a second View leaf.
    const auto cB = aether::constant<extents<D>>(wB);
    static_assert(std::is_same_v<decltype(cB)::working_type, Band>,
        "Constant<BandedReal,...> must compute in Band (working-type trait)");

    // ONE expression, ONE store — the exact shape a chain with a genuine
    // View leaf uses in test_BandedReal_common.h; only the operand kind
    // (Constant vs. View) differs.
    wov[i0] = ((A + cB) - A) * wS;

    for (std::size_t d = 0; d < D; d++) {
        EXPECT_EQ(woBuf[d].toBits(), wantOnce.toBits())
            << "component " << d << ": the scalar-broadcast chain did not deliver the ONCE-packed value";
        EXPECT_NE(woBuf[d].toBits(), wantPerNode.toBits())
            << "component " << d << ": the scalar-broadcast chain still packs at every node";
    }
}

} // namespace
} // namespace aether_tests
