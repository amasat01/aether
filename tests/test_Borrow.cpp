// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk::borrow() tests (host / AETHER_CPP_MODE build; test_Borrow.cu is the
// CUDA-build twin over the same fixture). Fills a vector, borrows it as a
// Chunk, views it, computes in place, verifies the result landed in the
// vector's own storage (never copied), and checks the borrowed Chunk never
// double-frees that storage when it goes out of scope before the vector
// does.

#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/layout/Extents.h>
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class BorrowTest : public ::testing::Test { };

TEST_F(BorrowTest, BorrowedChunkDoesNotOwnAndAliasesTheSourcePointer)
{
    std::vector<double> buf(8, 0.0);
    auto chunk = aether::Chunk::borrow(buf.data(), buf.size() * sizeof(double), aether::Device(kDLCPU));
    EXPECT_FALSE(chunk.owns());
    EXPECT_EQ(chunk.data(), reinterpret_cast<std::byte*>(buf.data()));
    EXPECT_EQ(chunk.size(), buf.size() * sizeof(double));
    EXPECT_EQ(chunk.device(), aether::Device(kDLCPU));
}

TEST_F(BorrowTest, AllocatedChunkStillOwns)
{
    // The default (`allocate()`) path is unaffected — sanity check that
    // `owns()` correctly distinguishes the two origins.
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    EXPECT_TRUE(chunk.owns());
}

TEST_F(BorrowTest, NullWithZeroBytesIsValidEmptyBorrow)
{
    auto chunk = aether::Chunk::borrow(nullptr, 0, aether::Device(kDLCPU));
    EXPECT_EQ(chunk.data(), nullptr);
    EXPECT_EQ(chunk.size(), 0u);
    EXPECT_FALSE(chunk.owns());
}

TEST_F(BorrowTest, NullWithNonzeroBytesThrows)
{
    EXPECT_THROW(aether::Chunk::borrow(nullptr, 64, aether::Device(kDLCPU)), aether::Error);
}

TEST_F(BorrowTest, VectorBackedViewComputesInPlaceAndSurvivesBorrowedChunkTeardown)
{
    constexpr std::size_t C = 3, N = 8;
    std::vector<double> buf(C * N, 0.0);
    {
        auto chunk = aether::Chunk::borrow(buf.data(), buf.size() * sizeof(double), aether::Device(kDLCPU));
        auto view  = aether::make_view<double, C, aether::dyn>(chunk, N);
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t i = 0; i < N; ++i)
                view(c, i) = static_cast<double>(c * 100 + i);
        // `chunk` (and `view`, which borrows from it) go out of scope here.
        // A genuine free would leave `buf` dangling for the checks below; a
        // correct no-op leaves it fully intact.
    }
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(buf[c * N + i], static_cast<double>(c * 100 + i));
    // `buf`'s OWN destructor runs at this test's scope exit — if
    // `Chunk::borrow` had actually freed this memory above, THIS would be
    // the double-free.
}

TEST_F(BorrowTest, MoveTransfersNonOwningStateAndSourceStaysEmpty)
{
    std::vector<double> buf(4, 0.0);
    auto a = aether::Chunk::borrow(buf.data(), buf.size() * sizeof(double), aether::Device(kDLCPU));
    aether::Chunk b(std::move(a));
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_FALSE(b.owns());
    EXPECT_EQ(b.data(), reinterpret_cast<std::byte*>(buf.data()));
    // Both `a` and `b` go out of scope here — a double-free would touch
    // `buf`'s memory twice; a no-op (owns()==false throughout) touches it
    // zero times.
}

TEST_F(BorrowTest, MoveAssignmentOntoAnOwningChunkFreesTheOldAllocationNotTheBorrowedOne)
{
    std::vector<double> buf(4, 0.0);
    auto owning = aether::Chunk::allocate(aether::Device(kDLCPU), 32);
    auto borrowed = aether::Chunk::borrow(buf.data(), buf.size() * sizeof(double), aether::Device(kDLCPU));
    owning = std::move(borrowed); // frees the ORIGINAL 32-byte allocation, then adopts the borrow
    EXPECT_FALSE(owning.owns());
    EXPECT_EQ(owning.data(), reinterpret_cast<std::byte*>(buf.data()));
    // `owning`'s destructor at scope exit must be a no-op on `buf` too.
}

} // namespace
} // namespace aether_tests
