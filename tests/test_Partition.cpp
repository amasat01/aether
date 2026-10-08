// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Partition.h tests (host / AETHER_CPP_MODE build; test_Partition.cu covers
// device-side compute over a partitioned view). Closed-form
// `padded_block_samples` arithmetic, closed-form `partition_view` offset +
// stride derivation (including the short last block), and a DLPack export
// case (padded/non-compact strides, zero-copy pointer identity into the
// block).

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/dtype/dlpack.h>
#include <aether/err/Error.h>
#include <aether/interop/DLPack.h>
#include <aether/layout/Extents.h>
#include <aether/layout/Layout.h>
#include <aether/residency/Partition.h>
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class PartitionTest : public ::testing::Test { };

// ---------------------------------------------------------------------
// padded_block_samples — closed-form pitch arithmetic.
// ---------------------------------------------------------------------

TEST_F(PartitionTest, PaddedBlockSamplesCeilDivThenRoundsUpToPadTo)
{
    // ceil(100/4) = 25 -> round up to 32 -> 32.
    EXPECT_EQ(aether::padded_block_samples(100, 4, 32), 32u);
    // ceil(100/3) = 34 -> round up to 32 -> 64.
    EXPECT_EQ(aether::padded_block_samples(100, 3, 32), 64u);
    // ceil(96/3) = 32 -> already a multiple of 32 -> 32 (no rounding needed).
    EXPECT_EQ(aether::padded_block_samples(96, 3, 32), 32u);
}

TEST_F(PartitionTest, PaddedBlockSamplesDefaultPadToIs32)
{
    // ceil(10/4) = 3 -> round up to the DEFAULT pad_to=32 -> 32.
    EXPECT_EQ(aether::padded_block_samples(10, 4), 32u);
}

TEST_F(PartitionTest, PaddedBlockSamplesHonorsAnExplicitFinerPadTo)
{
    // ceil(10/4) = 3 -> round up to 4 -> 4.
    EXPECT_EQ(aether::padded_block_samples(10, 4, 4), 4u);
    // pad_to=1 -> no rounding at all.
    EXPECT_EQ(aether::padded_block_samples(10, 4, 1), 3u);
}

TEST_F(PartitionTest, PaddedBlockSamplesRejectsZeroPartsOrPadTo)
{
    EXPECT_THROW(aether::padded_block_samples(100, 0, 32), aether::Error);
    EXPECT_THROW(aether::padded_block_samples(100, 4, 0), aether::Error);
}

// ---------------------------------------------------------------------
// partition_view — closed-form offset/extent/stride derivation.
// ---------------------------------------------------------------------

class PartitionViewTest : public ::testing::Test {
protected:
    static constexpr std::size_t C = 3, N = 100;

    PartitionViewTest()
        : chunk(aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double)))
        , full(aether::make_view<double, C, aether::dyn>(chunk, N))
    {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t i = 0; i < N; ++i)
                full(c, i) = static_cast<double>(c * 1000 + i);
    }

    aether::Chunk chunk;
    aether::View<double, aether::extents<C, aether::dyn>> full;
};

TEST_F(PartitionViewTest, FourWayPartitionOfHundredSamplesHasOneShortLastBlock)
{
    // n=100, parts=4, pad_to=32 -> padded=32: 3 full blocks + one short
    // last block.
    const aether::PartitionSpec spec{ 4, 32 };
    const std::size_t expectedOffsets[4] = { 0, 32, 64, 96 };
    const std::size_t expectedReal[4]    = { 32, 32, 32, 4 }; // last block is SHORT: 100 - 96 = 4

    for (std::size_t r = 0; r < 4; ++r) {
        auto block = aether::partition_view(full, spec, r);
        EXPECT_EQ(block.samples(), expectedReal[r]) << "r=" << r;
        EXPECT_EQ(block.data(), full.data() + expectedOffsets[r]) << "r=" << r;
        EXPECT_EQ(block.device(), full.device()) << "r=" << r;

        // Component stride is copied VERBATIM from `full`'s own mapping
        // (N=100) — a uniform block pitch (padded), never a per-block
        // re-derived component stride (the "padded stride" case).
        EXPECT_EQ(block.mapping().stride(0), N) << "r=" << r;
        EXPECT_EQ(block.mapping().stride(1), 1u) << "r=" << r;

        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t i = 0; i < expectedReal[r]; ++i)
                EXPECT_EQ(block(c, i), full(c, expectedOffsets[r] + i)) << "r=" << r << " c=" << c << " i=" << i;
    }
}

TEST_F(PartitionViewTest, ExactlyDivisiblePartitionHasNoShortBlock)
{
    // n=96, parts=3, pad_to=32 -> padded=32 exactly -> every block real=32.
    aether::Chunk exactChunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * 96 * sizeof(double));
    auto exact = aether::make_view<double, C, aether::dyn>(exactChunk, 96);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < 96; ++i)
            exact(c, i) = static_cast<double>(c * 1000 + i);

    const aether::PartitionSpec spec{ 3, 32 };
    for (std::size_t r = 0; r < 3; ++r) {
        auto block = aether::partition_view(exact, spec, r);
        EXPECT_EQ(block.samples(), 32u) << "r=" << r;
        EXPECT_EQ(block.data(), exact.data() + r * 32) << "r=" << r;
    }
}

TEST_F(PartitionViewTest, OutOfRangeBlockIndexThrows)
{
    const aether::PartitionSpec spec{ 4, 32 };
    EXPECT_THROW(aether::partition_view(full, spec, std::size_t{ 4 }), aether::Error);
}

// ---------------------------------------------------------------------
// DLPack export of a partition-block view: correct (non-compact,
// "padded") strides + zero-copy pointer identity into the block.
// ---------------------------------------------------------------------

TEST_F(PartitionViewTest, DlpackExportOfAFullBlockEmitsPaddedStridesAndZeroCopyPointer)
{
    const aether::PartitionSpec spec{ 4, 32 };
    auto block = aether::partition_view(full, spec, std::size_t{ 1 }); // offset 32, real 32 (a full, non-edge block)
    ASSERT_EQ(block.samples(), 32u);
    ASSERT_EQ(block.data(), full.data() + 32);

    DLManagedTensorVersioned* exported = aether::interop::toDLPack(block);
    ASSERT_NE(exported, nullptr);

    // Hand-verified against the closed form above: shape = (C, real),
    // strides = (N, 1) — not (real, 1), which is what a compact tensor of
    // this shape would carry. This is the "padded stride" case.
    EXPECT_EQ(exported->dl_tensor.data, static_cast<void*>(full.data() + 32)); // zero-copy INTO the block
    EXPECT_EQ(exported->dl_tensor.ndim, 2);
    EXPECT_EQ(exported->dl_tensor.shape[0], static_cast<std::int64_t>(C));
    EXPECT_EQ(exported->dl_tensor.shape[1], static_cast<std::int64_t>(32));
    EXPECT_EQ(exported->dl_tensor.strides[0], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);

    aether::interop::DLPackImport reimported = aether::interop::fromDLPack(exported);
    EXPECT_EQ(reimported.view.data, static_cast<void*>(full.data() + 32));
    EXPECT_EQ(reimported.view.rank, 2u);
    EXPECT_EQ(reimported.view.extents[0], C);
    EXPECT_EQ(reimported.view.extents[1], 32u);
    EXPECT_EQ(reimported.view.strides[0], N);
    EXPECT_EQ(reimported.view.strides[1], 1u);
    // `reimported.owner`'s destructor frees `exported` at scope exit.
}

TEST_F(PartitionViewTest, DlpackExportOfTheShortLastBlockEmitsRealExtentButStillPaddedStride)
{
    const aether::PartitionSpec spec{ 4, 32 };
    auto block = aether::partition_view(full, spec, std::size_t{ 3 }); // offset 96, real 4 (SHORT last block)
    ASSERT_EQ(block.samples(), 4u);
    ASSERT_EQ(block.data(), full.data() + 96);

    DLManagedTensorVersioned* exported = aether::interop::toDLPack(block);
    ASSERT_NE(exported, nullptr);

    // The SHORT block's shape reflects its real (4) extent, but the
    // component stride is STILL N=100 — the same "padded" stride every
    // other block carries, never re-derived from the short real extent.
    EXPECT_EQ(exported->dl_tensor.data, static_cast<void*>(full.data() + 96));
    EXPECT_EQ(exported->dl_tensor.shape[0], static_cast<std::int64_t>(C));
    EXPECT_EQ(exported->dl_tensor.shape[1], static_cast<std::int64_t>(4));
    EXPECT_EQ(exported->dl_tensor.strides[0], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);

    aether::interop::DLPackImport reimported = aether::interop::fromDLPack(exported);
    EXPECT_EQ(reimported.view.extents[1], 4u);
    EXPECT_EQ(reimported.view.strides[0], N);
}

} // namespace
} // namespace aether_tests
