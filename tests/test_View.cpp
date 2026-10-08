// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// View tests (host / AETHER_CPP_MODE build; test_View.cu covers the same
// fixture), but the covered device-kind set differs by construction: this
// build has only kDLCPU available, so test_View.cu additionally covers
// kDLCUDA/kDLCUDAHost chunks.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/layout/Extents.h>
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class ViewTest : public ::testing::Test { };

TEST_F(ViewTest, SoaProofPointerArithmeticMatchesClosedFormOverAFullSweep)
{
    // THE SoA PROOF: &v(c,i) - v.data() == c*N + i.
    constexpr std::size_t C = 3, N = 11;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto v     = aether::make_view<double, C, aether::dyn>(chunk, N);
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(&v(c, i) - v.data(), static_cast<std::ptrdiff_t>(c * N + i));
        }
    }
}

TEST_F(ViewTest, SoaProofHoldsForAMatrixView)
{
    constexpr std::size_t R = 3, C = 2, N = 6;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), R * C * N * sizeof(double));
    auto v     = aether::make_view<double, R, C, aether::dyn>(chunk, N);
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t n = 0; n < N; ++n) {
                EXPECT_EQ(&v(r, c, n) - v.data(), static_cast<std::ptrdiff_t>((r * C + c) * N + n));
            }
        }
    }
}

TEST_F(ViewTest, SamplesSizeRankExtentAccessors)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 11 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 11);
    EXPECT_EQ(v.rank(), 2u);
    EXPECT_EQ(v.samples(), 11u);
    EXPECT_EQ(v.size(), 33u);
    EXPECT_EQ(v.extent(0), 3u);
    EXPECT_EQ(v.extent(1), 11u);
    EXPECT_EQ(v.device(), aether::Device(kDLCPU));
}

TEST_F(ViewTest, ReadWriteRoundTrip)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < 5; ++i)
            v(c, i) = static_cast<double>(c * 10 + i);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < 5; ++i)
            EXPECT_EQ(v(c, i), static_cast<double>(c * 10 + i));
}

TEST_F(ViewTest, MakeViewThrowsWhenChunkTooSmall)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double) - 1);
    EXPECT_THROW((aether::make_view<double, 3, aether::dyn>(chunk, 5)), aether::Error);
}

TEST_F(ViewTest, MakeViewAcceptsAnExactlySizedChunk)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    EXPECT_NO_THROW((aether::make_view<double, 3, aether::dyn>(chunk, 5)));
}

TEST_F(ViewTest, RawPointerWrapOverForeignMemory)
{
    double raw[6] = { 0, 1, 2, 3, 4, 5 };
    auto v        = aether::make_view<double, 3, aether::dyn>(raw, aether::Device(kDLCPU), 2);
    EXPECT_EQ(v.data(), raw);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < 2; ++i)
            EXPECT_EQ(v(c, i), raw[c * 2 + i]);
}

TEST_F(ViewTest, RawPointerWrapThrowsOnNullWithNonZeroSpan)
{
    EXPECT_THROW(
        (aether::make_view<double, 3, aether::dyn>(nullptr, aether::Device(kDLCPU), 2)), aether::Error);
}

TEST_F(ViewTest, RawPointerWrapAllowsNullWithZeroSpan)
{
    auto v = aether::make_view<double, 3, aether::dyn>(nullptr, aether::Device(kDLCPU), 0);
    EXPECT_EQ(v.data(), nullptr);
    EXPECT_EQ(v.size(), 0u);
}

TEST_F(ViewTest, RawPointerWrapThrowsWhenSpanExceedsOffsetRange)
{
    // aether addresses elements with a 32-bit `offset_t`
    // (aether/index/Offset.h), so a view whose required span does not fit is
    // refused at construction — the alternative is a silently wrapped address
    // computed inside device code, where nothing can throw.
    //
    // The raw-pointer wrap is the right instrument: make_view never touches
    // the memory, so an over-range shape needs no allocation to express. The
    // pointer below is never dereferenced.
    double raw[1] = { 0.0 };

    // extents<3,3,dyn>: span == 9 * N. 9 * 477'218'589 == 4'294'967'301 > UINT32_MAX.
    EXPECT_THROW((aether::make_view<double, 3, 3, aether::dyn>(raw, aether::Device(kDLCPU), 477218589u)),
        aether::Error);
}

TEST_F(ViewTest, RawPointerWrapAcceptsTheLargestAddressableSpan)
{
    // The other half of the boundary — without it the throw above would also
    // pass a guard that rejected EVERYTHING. 9 * 477'218'588 == 4'294'967'292,
    // the largest multiple of 9 that an offset_t can address.
    double raw[1] = { 0.0 };

    auto v = aether::make_view<double, 3, 3, aether::dyn>(raw, aether::Device(kDLCPU), 477218588u);
    EXPECT_EQ(v.samples(), 477218588u);
    EXPECT_EQ(v.data(), raw);
    static_assert(std::is_same_v<decltype(v.samples()), aether::offset_t>,
        "View::samples() must report the narrow offset type");
}

TEST_F(ViewTest, AsConstProducesAReadOnlyViewOverTheSameData)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5);
    v(0, 0)    = 42.0;
    auto cv    = v.as_const();
    EXPECT_EQ(cv(0, 0), 42.0);
    EXPECT_EQ(cv.data(), v.data());
}

TEST_F(ViewTest, AsVolatileProducesAVolatileQualifiedViewOverTheSameData)
{
    // View::as_volatile() -> same data, volatile-qualified element access.
    // Functional round-trip only here (host build) -- the CSE-blocking
    // SASS-level property is tools/volatile_audit.sh's job (ptxas/cuobjdump
    // only).
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * sizeof(double));
    auto v     = aether::make_view<double, 3>(chunk);
    v(0)       = 1.5;
    v(1)       = -2.25;
    v(2)       = 4.0;

    auto vv = v.as_volatile();
    static_assert(decltype(vv)::isVolatile, "as_volatile() result must report isVolatile == true");
    static_assert(!decltype(v)::isVolatile, "the source view must remain non-volatile");
    EXPECT_EQ(vv(0), 1.5);
    EXPECT_EQ(vv(1), -2.25);
    EXPECT_EQ(vv(2), 4.0);
    EXPECT_EQ(vv.data(), v.data());

    // Writes through the volatile view are visible through the original
    // (same underlying memory -- as_volatile() is a re-qualification, not a copy).
    vv(1) = 7.0;
    EXPECT_EQ(v(1), 7.0);
}

} // namespace
} // namespace aether_tests
