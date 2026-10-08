// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Array capacity device-side compute test (CUDA build). Paired with
// test_ArrayCapacity.cpp (closed-form host tests + the DLPack case; house
// convention — see test_Array.*/test_Partition.*). A device kernel writes
// into an Array's RESERVED TAIL (`deviceSpare()`), the host `commit()`s the
// written count, and a download shows the exact values — the same round
// trip `AppendCounter` drives via a device-side atomic slot counter instead
// of this test's fixed count.
//
// GPU EXECUTION NOTE: this file is built and its tests LISTED
// (--gtest_list_tests); the suite is never RUN here — that happens
// elsewhere, on the GPU.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;

class ArrayCapacityDeviceTest : public ::testing::Test { };

// `deviceSpare()`'s return type — `layout_stride`, same convention as
// `test_Partition.cu`'s `Block3dView`.
using Spare3dView = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;

AETHER_KERNEL()
void fillSpareKernel(Spare3dView spare)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= spare.samples())
        return;
    spare(0, i.global()) = static_cast<double>(i.global());
    spare(1, i.global()) = static_cast<double>(i.global()) * 2.0;
    spare(2, i.global()) = static_cast<double>(i.global()) * 3.0;
}

TEST_F(ArrayCapacityDeviceTest, KernelWritesIntoSpareHostCommitsValuesExact)
{
    constexpr std::size_t C = 3;
    Array<double, C> a(0);
    a.reserve(10); // -> capacity 32 (kCapacityAlignment)
    ASSERT_EQ(a.capacity(), 32u);
    ASSERT_EQ(a.samples(), 0u);

    auto spare = a.deviceSpare(); // [0, 32) -- the WHOLE reserved capacity (samples()==0)
    ASSERT_EQ(spare.samples(), 32u);

    const auto cfg = aether::cuda::launchConfig(spare.samples());
    fillSpareKernel<<<cfg.blocks, cfg.threads>>>(spare);
    aether::cuda::checkLastLaunch("fillSpareKernel");

    a.commit(spare.samples()); // adopt every written slot
    EXPECT_EQ(a.samples(), 32u);

    a.download();
    auto after = a.hostView();
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(after(0, i), static_cast<double>(i));
        EXPECT_EQ(after(1, i), static_cast<double>(i) * 2.0);
        EXPECT_EQ(after(2, i), static_cast<double>(i) * 3.0);
    }
}

} // namespace
} // namespace aether_tests
