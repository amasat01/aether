// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// residency/PartitionedArray.h tests (host / AETHER_CPP_MODE build;
// test_PartitionedArray.cu covers a different scope — see that file's own
// docstring): this file covers the partition math (closed-form parity with
// Partition.h, via PartitionedArray's own accessors) and the
// AETHER_CPP_MODE single-CPU-device degenerate path (one "device" that is
// itself `kDLCPU` — the only device kind this build has, per Device.h's
// `backend_enabled` gate) end-to-end: fill the host canonical copy,
// scatter(), mutate the device-side block, gather(), verify the host
// canonical copy sees the mutation — the same `aether::copy` CPU<->CPU
// legal path (`aether/chunk/Copy.h`) a real CUDAHost<->CUDA multi-GPU
// scatter/gather would use, just at parts=1.
//
// `aether::PartitionedArray` itself lives in
// aether/residency/PartitionedArray.h.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Device;
using aether::PartitionedArray;

class PartitionedArrayTest : public ::testing::Test { };

// ---------------------------------------------------------------------
// Partition math — closed-form parity with Partition.h's own
// padded_block_samples worked example (test_Partition.cpp's
// FourWayPartitionOfHundredSamplesHasOneShortLastBlock), reached through
// PartitionedArray's own accessors instead of the free functions directly.
// ---------------------------------------------------------------------

TEST_F(PartitionedArrayTest, PartitionMathMatchesPartitionHClosedForm)
{
    using PA = PartitionedArray<double, aether::dyn>;
    std::vector<Device> devices{ Device(kDLCPU), Device(kDLCPU), Device(kDLCPU), Device(kDLCPU) };
    // n=100, parts=4, pad_to=32 -> padded=32: 3 full blocks + one short
    // last block.
    PA pa(devices, 100, 32);

    EXPECT_EQ(pa.parts(), 4u);
    EXPECT_EQ(pa.samples(), 100u);
    EXPECT_EQ(pa.padded(), 32u);
    const std::size_t expectedReal[4] = { 32, 32, 32, 4 }; // last block SHORT: 100 - 96 = 4
    for (std::size_t r = 0; r < 4; ++r)
        EXPECT_EQ(pa.realCount(r), expectedReal[r]) << "r=" << r;
}

TEST_F(PartitionedArrayTest, ExactlyDivisiblePartitionHasNoShortBlock)
{
    using PA = PartitionedArray<double, aether::dyn>;
    std::vector<Device> devices{ Device(kDLCPU), Device(kDLCPU), Device(kDLCPU) };
    PA pa(devices, 96, 32); // n=96, parts=3, pad_to=32 -> padded=32 exactly.

    for (std::size_t r = 0; r < 3; ++r)
        EXPECT_EQ(pa.realCount(r), 32u) << "r=" << r;
}

TEST_F(PartitionedArrayTest, RejectsEmptyDeviceList)
{
    // padded_block_samples's own "parts must be > 0" check (Partition.h),
    // reached before any Chunk::allocate — the SAME error path
    // test_Partition.cpp's PaddedBlockSamplesRejectsZeroPartsOrPadTo covers
    // directly.
    using PA = PartitionedArray<double, aether::dyn>;
    EXPECT_THROW(PA(std::vector<Device>{}, 100), aether::Error);
}

// ---------------------------------------------------------------------
// AETHER_CPP_MODE single-CPU-device degenerate path — RUN, not just
// built: rank-2 shape (3 leading components x N samples) exercises the
// pitched (innerSize > 1, multiple row-strips per rank) copy path, not
// just the innerSize == 1 collapse a plain rank-1 array would give.
// ---------------------------------------------------------------------

TEST_F(PartitionedArrayTest, SingleCpuDeviceDegeneratePathScatterMutateGatherRoundTripsExactly)
{
    constexpr std::size_t N = 57;
    std::vector<Device> devices{ Device(kDLCPU) }; // parts=1 — the degenerate case.
    PartitionedArray<double, 3, aether::dyn> pa(devices, N);

    EXPECT_EQ(pa.parts(), 1u);
    EXPECT_EQ(pa.device(0), Device(kDLCPU));

    auto host = pa.hostView();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            host(c, i) = static_cast<double>(c * 1000 + i);

    pa.scatter();

    auto dev0 = pa.deviceView(0);
    ASSERT_EQ(dev0.samples(), N); // parts=1 -> no short block possible: real(0) == N always.
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(dev0(c, i), host(c, i)) << "c=" << c << " i=" << i;

    // Mutate the device-side ("device") block only, then gather() back —
    // proves scatter()/gather() move REAL data through independent storage,
    // not aliases of the same chunk.
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            dev0(c, i) += 0.5;

    pa.gather();

    auto hostAfter = pa.hostView();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(hostAfter(c, i), static_cast<double>(c * 1000 + i) + 0.5) << "c=" << c << " i=" << i;
}

} // namespace
} // namespace aether_tests
