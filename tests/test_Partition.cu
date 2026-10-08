// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// partition_view device-side compute test (CUDA build; test_Partition.cpp
// covers the closed-form host tests plus a DLPack export case). Runs a
// kernel over one partitioned block's device View and verifies (after
// download) that only that block's real samples in the full device array
// changed — `partition_view` slices the original device memory, so a kernel
// launched over block r must be indistinguishable from the same kernel run
// over the full view restricted to `[r*padded, r*padded + real)`.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/residency/Partition.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;

class PartitionDeviceTest : public ::testing::Test { };

// The block view's Layout is `layout_stride` (`partition_view`'s return
// type) — distinct from the `Vec3dView` (`layout_right`) alias, so the
// kernel parameter is spelled out explicitly.
using Block3dView = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;

AETHER_KERNEL()
void doubleInPlaceKernel(Block3dView block)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= block.samples())
        return;
    block(0, i.global()) *= 2.0;
    block(1, i.global()) *= 2.0;
    block(2, i.global()) *= 2.0;
}

TEST_F(PartitionDeviceTest, KernelOverOneBlockOnlyTouchesThatBlocksRealSamples)
{
    constexpr std::size_t C = 3, N = 100;
    Array<double, C> a(N);

    std::vector<double> original(C * N);
    auto hv = a.hostView();
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            original[c * N + i] = static_cast<double>(c * 1000 + i);
            hv(c, i)             = original[c * N + i];
        }
    }
    a.upload();

    // n=100, parts=4, pad_to=32 -> padded=32; block r=1 is a full,
    // non-edge block: offset 32, real 32.
    const aether::PartitionSpec spec{ 4, 32 };
    const std::size_t r = 1;
    auto block           = aether::partition_view(a.deviceView(), spec, r);
    ASSERT_EQ(block.samples(), 32u);

    const auto cfg = aether::cuda::launchConfig(block.samples());
    doubleInPlaceKernel<<<cfg.blocks, cfg.threads>>>(block);
    aether::cuda::checkLastLaunch("doubleInPlaceKernel");

    a.download();
    auto after = a.hostView();
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < N; ++i) {
            const double orig     = original[c * N + i];
            const bool inBlock    = (i >= 32 && i < 64); // r*padded=32 .. r*padded+real=64
            const double expected = inBlock ? orig * 2.0 : orig;
            EXPECT_EQ(after(c, i), expected) << "c=" << c << " i=" << i;
        }
    }
}

} // namespace
} // namespace aether_tests
