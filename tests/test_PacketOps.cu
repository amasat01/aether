// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// SIMD unit tests for backend/cpu/simd/ (CUDA build; test_PacketOps.cpp is
// the identical host-build twin): the SIMD packet layer (Packet/PacketMask/
// PreferredWidth, PacketItem) is host-only with no CUDA-specific behaviour
// that a plain host build cannot already exercise, but the headers must
// still compile in CUDA-mode builds (host side).
//
// Packet/Mask unit tests including tail masks. Packet-vs-scalar
// bit-identity (the expr-layer packetGet/packetAssign/packetEval
// comparison) lives in test_PacketExpr.{cpp,cu} instead — this file stays
// scoped to the primitives themselves.

#include <cmath>
#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/backend/cpu/PacketItem.h>
#include <aether/backend/cpu/simd/simd.h>

namespace aether_tests {
namespace {

using aether::simd::Packet;
using aether::simd::PacketMask;
using aether::simd::PreferredWidth;

class PreferredWidthTest : public ::testing::Test { };
class PacketTest : public ::testing::Test { };
class PacketMaskTest : public ::testing::Test { };
class PacketItemTest : public ::testing::Test { };

// ─── PreferredWidth / PacketTraits ─────────────────────────────────────────

TEST_F(PreferredWidthTest, NativeWidthIsAtLeastOneAndMatchesPacketWidth)
{
    EXPECT_GE(PreferredWidth<double>, 1u);
    EXPECT_GE(PreferredWidth<float>, 1u);
    EXPECT_EQ((Packet<double, PreferredWidth<double>>::width), PreferredWidth<double>);
    EXPECT_EQ((Packet<float, PreferredWidth<float>>::width), PreferredWidth<float>);
}

TEST_F(PreferredWidthTest, ScalarFallbackWidthIsOne)
{
    EXPECT_EQ((Packet<double, 1>::width), 1u);
}

// ─── Packet<T,W>: load/store/broadcast/zero ────────────────────────────────

TEST_F(PacketTest, BroadcastFillsEveryLane)
{
    constexpr std::size_t W = PreferredWidth<double>;
    auto p = Packet<double, W>::broadcast(3.5);
    double out[W];
    Packet<double, W>::store(out, p);
    for (std::size_t k = 0; k < W; ++k)
        EXPECT_EQ(out[k], 3.5);
}

TEST_F(PacketTest, ZeroIsAllZero)
{
    constexpr std::size_t W = PreferredWidth<double>;
    auto p = Packet<double, W>::zero();
    double out[W];
    Packet<double, W>::store(out, p);
    for (std::size_t k = 0; k < W; ++k)
        EXPECT_EQ(out[k], 0.0);
}

TEST_F(PacketTest, LoadStoreRoundTripIsExact)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double in[W];
    for (std::size_t k = 0; k < W; ++k)
        in[k] = static_cast<double>(k) * 1.25 - 3.0;
    auto p = Packet<double, W>::load(in);
    double out[W] = {};
    Packet<double, W>::store(out, p);
    EXPECT_EQ(0, std::memcmp(in, out, sizeof(in)));
}

TEST_F(PacketTest, ArithmeticMatchesPerLaneScalarArithmetic)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) * 2.0 + 1.0;
        bIn[k] = static_cast<double>(k) * -0.5 + 3.0;
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);

    double sumOut[W], diffOut[W], prodOut[W], quotOut[W], negOut[W];
    Packet<double, W>::store(sumOut, a + b);
    Packet<double, W>::store(diffOut, a - b);
    Packet<double, W>::store(prodOut, a * b);
    Packet<double, W>::store(quotOut, a / b);
    Packet<double, W>::store(negOut, -a);

    for (std::size_t k = 0; k < W; ++k) {
        EXPECT_EQ(sumOut[k], aIn[k] + bIn[k]) << "lane " << k;
        EXPECT_EQ(diffOut[k], aIn[k] - bIn[k]) << "lane " << k;
        EXPECT_EQ(prodOut[k], aIn[k] * bIn[k]) << "lane " << k;
        EXPECT_EQ(quotOut[k], aIn[k] / bIn[k]) << "lane " << k;
        EXPECT_EQ(negOut[k], -aIn[k]) << "lane " << k;
    }
}

TEST_F(PacketTest, FmaddMatchesPerLaneStdFma)
{
    // fma is a SINGLE-rounding fused op by definition — the correct
    // per-lane reference is std::fma, not a separate mul-then-add (which
    // would legitimately disagree in the last bit).
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W], cIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) + 1.0000000000000002; // 1 + 2^-52
        bIn[k] = static_cast<double>(k) + 1.0000000000000002;
        cIn[k] = -static_cast<double>(k) - 1.0;
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);
    auto c = Packet<double, W>::load(cIn);
    double out[W];
    Packet<double, W>::store(out, fmadd(a, b, c));
    for (std::size_t k = 0; k < W; ++k)
        EXPECT_EQ(out[k], std::fma(aIn[k], bIn[k], cIn[k])) << "lane " << k;
}

TEST_F(PacketTest, ComparisonsMatchPerLaneScalarComparisons)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) - 1.0;
        bIn[k] = static_cast<double>(W - k) - 2.0; // crosses aIn[k] at some lane
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);

    auto ge = cmpGe(a, b);
    auto lt = cmpLt(a, b);
    auto gt = cmpGt(a, b);
    for (std::size_t k = 0; k < W; ++k) {
        EXPECT_EQ(ge.lane(k), aIn[k] >= bIn[k]) << "lane " << k;
        EXPECT_EQ(lt.lane(k), aIn[k] < bIn[k]) << "lane " << k;
        EXPECT_EQ(gt.lane(k), aIn[k] > bIn[k]) << "lane " << k;
    }
}

// ─── Packet<T,W>: maskLoad / maskStore — TAIL MASKS ─────────────────

TEST_F(PacketTest, MaskLoadOnlyReadsActiveLanesRestAreZero)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double src[W];
    for (std::size_t k = 0; k < W; ++k)
        src[k] = static_cast<double>(k) + 10.0;

    for (std::size_t active = 0; active <= W; ++active) {
        auto mask = PacketMask<double, W>::firstN(active);
        auto p = Packet<double, W>::maskLoad(src, mask);
        double out[W];
        Packet<double, W>::store(out, p);
        for (std::size_t k = 0; k < W; ++k) {
            if (k < active)
                EXPECT_EQ(out[k], src[k]) << "active=" << active << " lane=" << k;
            else
                EXPECT_EQ(out[k], 0.0) << "active=" << active << " lane=" << k << " (inactive must read as zero)";
        }
    }
}

TEST_F(PacketTest, MaskStoreOnlyWritesActiveLanesRestUntouched)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double valIn[W];
    for (std::size_t k = 0; k < W; ++k)
        valIn[k] = static_cast<double>(k) * -3.0 + 1.0;
    auto val = Packet<double, W>::load(valIn);

    for (std::size_t active = 0; active <= W; ++active) {
        double dst[W];
        for (std::size_t k = 0; k < W; ++k)
            dst[k] = -999.0; // sentinel: untouched lanes must keep this
        auto mask = PacketMask<double, W>::firstN(active);
        Packet<double, W>::maskStore(dst, mask, val);
        for (std::size_t k = 0; k < W; ++k) {
            if (k < active)
                EXPECT_EQ(dst[k], valIn[k]) << "active=" << active << " lane=" << k;
            else
                EXPECT_EQ(dst[k], -999.0) << "active=" << active << " lane=" << k << " (inactive must be untouched)";
        }
    }
}

// ─── PacketMask<T,W> ────────────────────────────────────────────────────

TEST_F(PacketMaskTest, AllTrueAllFalse)
{
    constexpr std::size_t W = PreferredWidth<double>;
    auto allTrue = PacketMask<double, W>::allTrue();
    auto allFalse = PacketMask<double, W>::allFalse();
    EXPECT_TRUE(allTrue.isAllTrue());
    EXPECT_TRUE(allTrue.anyTrue());
    EXPECT_FALSE(allFalse.anyTrue());
    EXPECT_FALSE(allFalse.isAllTrue());
    for (std::size_t k = 0; k < W; ++k) {
        EXPECT_TRUE(allTrue.lane(k)) << k;
        EXPECT_FALSE(allFalse.lane(k)) << k;
    }
}

TEST_F(PacketMaskTest, FirstNSetsExactlyTheFirstNLanes)
{
    constexpr std::size_t W = PreferredWidth<double>;
    for (std::size_t n = 0; n <= W; ++n) {
        auto m = PacketMask<double, W>::firstN(n);
        EXPECT_EQ(m.anyTrue(), n > 0) << "n=" << n;
        EXPECT_EQ(m.isAllTrue(), n == W) << "n=" << n;
        for (std::size_t k = 0; k < W; ++k)
            EXPECT_EQ(m.lane(k), k < n) << "n=" << n << " lane=" << k;
    }
}

TEST_F(PacketMaskTest, BooleanAlgebraAndOrNot)
{
    constexpr std::size_t W = PreferredWidth<double>;
    if (W < 2)
        GTEST_SKIP() << "boolean-algebra combinations need at least 2 lanes";
    auto first = PacketMask<double, W>::firstN(1);
    auto all = PacketMask<double, W>::allTrue();
    auto notFirst = ~first;
    auto andRes = first & all;
    auto orRes = first | notFirst;

    EXPECT_TRUE(orRes.isAllTrue());
    for (std::size_t k = 0; k < W; ++k) {
        EXPECT_EQ(andRes.lane(k), first.lane(k)) << k;
        EXPECT_EQ(notFirst.lane(k), !first.lane(k)) << k;
    }
}

TEST_F(PacketMaskTest, ScalarFallbackMaskBehavesLikeABool)
{
    using M1 = PacketMask<double, 1>;
    EXPECT_TRUE(M1::allTrue().anyTrue());
    EXPECT_FALSE(M1::allFalse().anyTrue());
    EXPECT_TRUE(M1::firstN(1).anyTrue());
    EXPECT_FALSE(M1::firstN(0).anyTrue());
    EXPECT_TRUE(M1::allTrue().lane(0));
    EXPECT_FALSE(M1::allFalse().lane(0));
}

// ─── PacketItem<T,W,Es...> ──────────────────────────────────────────────
//
// `PacketItem` is a PRIMARY class template (no explicit specializations,
// unlike `Packet<T,W>`/`PacketMask<T,W>` above, which are fully
// specialized per ISA). Every `PacketItem<double, ..., 3>` instantiation
// below deliberately names the SAME shared, file-scope `kPacketItemW`
// constant (rather than each TEST_F re-declaring its own per-function
// `constexpr std::size_t W = PreferredWidth<double>;` local of the SAME
// name) — a per-function local of the same name, repeated across multiple
// gtest-generated TestBody() member functions in ONE .cu translation
// unit, was observed to make nvcc's host-compiler pass reject a LATER
// occurrence with a spurious "template argument 2 is invalid" / "does not
// name a type" (confirmed via a minimal, non-SIMD repro — a 2-parameter
// class template with no Packet/vector type involved at all reproduces it
// identically). A single shared constant,
// referenced directly (not re-aliased into a per-function local of the
// same name), avoids it.
inline constexpr std::size_t kPacketItemW = PreferredWidth<double>;

TEST_F(PacketItemTest, ReadWriteRoundTripPerComponent)
{
    aether::PacketItem<double, kPacketItemW, 3> item;
    item.template packet<0>() = Packet<double, kPacketItemW>::broadcast(1.0);
    item.template packet<1>() = Packet<double, kPacketItemW>::broadcast(2.0);
    item.template packet<2>() = Packet<double, kPacketItemW>::broadcast(3.0);

    double out[kPacketItemW];
    Packet<double, kPacketItemW>::store(out, item.template packet<0>());
    for (std::size_t k = 0; k < kPacketItemW; ++k)
        EXPECT_EQ(out[k], 1.0);
    Packet<double, kPacketItemW>::store(out, item.template packet<1>());
    for (std::size_t k = 0; k < kPacketItemW; ++k)
        EXPECT_EQ(out[k], 2.0);
    Packet<double, kPacketItemW>::store(out, item.template packet<2>());
    for (std::size_t k = 0; k < kPacketItemW; ++k)
        EXPECT_EQ(out[k], 3.0);
}

TEST_F(PacketItemTest, SizeAndRankMatchElementExtents)
{
    using Item3 = aether::PacketItem<double, kPacketItemW, 3>;
    static_assert(Item3::Rank == 1, "PacketItem<T,W,3>::Rank must be 1");
    static_assert(Item3::Size == 3, "PacketItem<T,W,3>::Size must be 3");
    static_assert(Item3::isLeaf, "PacketItem must be a leaf");
    EXPECT_EQ(Item3::size(), 3u);
}

TEST_F(PacketItemTest, DataPointerAddressesEveryComponentInOrder)
{
    aether::PacketItem<double, kPacketItemW, 3> item;
    item.template packet<0>() = Packet<double, kPacketItemW>::broadcast(10.0);
    item.template packet<1>() = Packet<double, kPacketItemW>::broadcast(20.0);
    item.template packet<2>() = Packet<double, kPacketItemW>::broadcast(30.0);

    auto* p = item.data();
    double out[kPacketItemW];
    Packet<double, kPacketItemW>::store(out, p[0]);
    EXPECT_EQ(out[0], 10.0);
    Packet<double, kPacketItemW>::store(out, p[1]);
    EXPECT_EQ(out[0], 20.0);
    Packet<double, kPacketItemW>::store(out, p[2]);
    EXPECT_EQ(out[0], 30.0);
}

} // namespace
} // namespace aether_tests
