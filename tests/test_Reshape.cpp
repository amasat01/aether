// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Reshape tests (host / AETHER_CPP_MODE build; test_Reshape.cu is the
// identical CUDA-build twin): index folding is pure host-computable
// arithmetic over a `layout_right` view, with no CUDA-specific behaviour a
// plain host build cannot already exercise.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>
#include <aether/layout/Extents.h>
#include <aether/view/MakeReshape.h> // reshaped() lives here, split out of view/Reshape.h
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/Reshape.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class ReshapeTest : public ::testing::Test { };

TEST_F(ReshapeTest, FlatVsFoldedIndexEquivalenceSweep)
{
    constexpr std::size_t C = 3, Major = 4, Minor = 5;
    constexpr std::size_t N = Major * Minor;
    auto chunk               = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto flat                = aether::make_view<double, C, aether::dyn>(chunk, N);
    auto folded               = aether::reshaped<Major, aether::dyn>(flat);

    EXPECT_EQ(folded.rank(), 3u);
    EXPECT_EQ(folded.extent(0), C);
    EXPECT_EQ(folded.extent(1), Major);
    EXPECT_EQ(folded.extent(2), Minor);

    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t maj = 0; maj < Major; ++maj) {
            for (std::size_t min = 0; min < Minor; ++min) {
                const std::size_t flatI = maj * Minor + min;
                EXPECT_EQ(&folded(c, maj, min), &flat(c, flatI));
            }
        }
    }
}

TEST_F(ReshapeTest, FoldedWritesAreVisibleThroughTheFlatView)
{
    constexpr std::size_t C = 2, Major = 3, Minor = 4;
    constexpr std::size_t N = Major * Minor;
    auto chunk               = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto flat                = aether::make_view<double, C, aether::dyn>(chunk, N);
    auto folded               = aether::reshaped<Major, aether::dyn>(flat);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t maj = 0; maj < Major; ++maj)
            for (std::size_t min = 0; min < Minor; ++min)
                folded(c, maj, min) = static_cast<double>(c * 100 + maj * 10 + min);

    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t maj = i / Minor;
            const std::size_t min = i % Minor;
            EXPECT_EQ(flat(c, i), static_cast<double>(c * 100 + maj * 10 + min));
        }
    }
}

TEST_F(ReshapeTest, ScalarViewReshapesDirectlyIntoModes)
{
    constexpr std::size_t Major = 6, Minor = 5;
    constexpr std::size_t N = Major * Minor;
    auto chunk                = aether::Chunk::allocate(aether::Device(kDLCPU), N * sizeof(double));
    auto flat                 = aether::make_view<double, aether::dyn>(chunk, N);
    auto folded                = aether::reshaped<Major, aether::dyn>(flat);
    EXPECT_EQ(folded.rank(), 2u);
    for (std::size_t maj = 0; maj < Major; ++maj)
        for (std::size_t min = 0; min < Minor; ++min)
            EXPECT_EQ(&folded(maj, min), &flat(maj * Minor + min));
}

TEST_F(ReshapeTest, DynModeCanComeFirstInsteadOfLast)
{
    // reshaped<dyn, Minor>: the dynamic mode need not be Ms...'s LAST slot.
    constexpr std::size_t C = 2, Minor = 4;
    constexpr std::size_t N = 12; // 12 / 4 = 3 -> the resolved dyn extent
    auto chunk                = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto flat                 = aether::make_view<double, C, aether::dyn>(chunk, N);
    auto folded                = aether::reshaped<aether::dyn, Minor>(flat);
    EXPECT_EQ(folded.extent(1), 3u);
    EXPECT_EQ(folded.extent(2), Minor);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t maj = 0; maj < 3; ++maj)
            for (std::size_t min = 0; min < Minor; ++min)
                EXPECT_EQ(&folded(c, maj, min), &flat(c, maj * Minor + min));
}

TEST_F(ReshapeTest, BadProductThrows)
{
    constexpr std::size_t C = 3, N = 17; // 17 is prime — no Major=4 factor divides it.
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto flat  = aether::make_view<double, C, aether::dyn>(chunk, N);
    EXPECT_THROW((aether::reshaped<4, aether::dyn>(flat)), aether::Error);
}

} // namespace
} // namespace aether_tests
