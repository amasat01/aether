// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Array tests (host / AETHER_CPP_MODE build). Paired with test_Array.cu:
// AETHER_CPP_MODE has a single kDLCPU chunk, so hostView() and
// deviceView() alias the SAME memory and upload()/download() are no-ops;
// the "kernel" that the .cu twin runs on-device is here a plain host loop
// over View + SampleIndex.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/array/Array.h>
#include <aether/index/SampleIndex.h>

namespace aether_tests {
namespace {

class ArrayTest : public ::testing::Test { };

TEST_F(ArrayTest, SamplesAndSize)
{
    aether::Array<double, 3> arr(10);
    EXPECT_EQ(arr.samples(), 10u);
    EXPECT_EQ(arr.size(), 30u);
}

TEST_F(ArrayTest, ScalarArraySamplesAndSize)
{
    aether::Array<double> arr(7);
    EXPECT_EQ(arr.samples(), 7u);
    EXPECT_EQ(arr.size(), 7u);
}

TEST_F(ArrayTest, HostAndDeviceViewAliasTheSameChunkInCppMode)
{
    aether::Array<double, 3> arr(4);
    auto hv = arr.hostView();
    auto dv = arr.deviceView();
    EXPECT_EQ(hv.data(), dv.data());
}

TEST_F(ArrayTest, LifecycleFillUploadHostLoopDownloadBitExact)
{
    constexpr std::size_t C = 3, N = 20;
    aether::Array<double, C> arr(N);

    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 1000 + i);

    arr.upload(); // no-op in AETHER_CPP_MODE, exercised for API parity with the .cu twin

    // THE HOST-LOOP TWIN of the .cu file's real kernel: pos(c,i) += 1 via
    // View + SampleIndex, over the array's own "device" (== host, in this
    // mode) view.
    auto dv = arr.deviceView();
    for (std::size_t i = 0; i < N; ++i) {
        auto idx = aether::SampleIndex::make(i);
        if (idx.global() >= dv.samples())
            continue;
        for (std::size_t c = 0; c < C; ++c)
            dv(c, idx.global()) += 1.0;
    }

    arr.download(); // no-op in AETHER_CPP_MODE

    auto verify = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(verify(c, i), static_cast<double>(c * 1000 + i + 1));
}

} // namespace
} // namespace aether_tests
