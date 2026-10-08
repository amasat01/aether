// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::DeviceFlag` tests on the CPU host path (AETHER_CPP_MODE);
// test_DeviceFlag.cu mirrors this over the CUDA build.

#include <omp.h>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

class DeviceFlagTest : public ::testing::Test { };

TEST_F(DeviceFlagTest, InitiallyClear)
{
    aether::DeviceFlag flag;
    EXPECT_FALSE(flag.isSet());
}

TEST_F(DeviceFlagTest, SetMakesIsSetTrue)
{
    aether::DeviceFlag flag;
    flag.deviceView().set();
    EXPECT_TRUE(flag.isSet());
}

TEST_F(DeviceFlagTest, CheckAndClearThrowsWhenSetThenClears)
{
    aether::DeviceFlag flag;
    flag.deviceView().set();
    ASSERT_TRUE(flag.isSet());
    EXPECT_THROW(flag.checkAndClear("DeviceFlagTest"), aether::Error);
    // checkAndClear() clears the flag as part of the host-throw pattern.
    EXPECT_FALSE(flag.isSet());
    EXPECT_NO_THROW(flag.checkAndClear("DeviceFlagTest"));
}

TEST_F(DeviceFlagTest, CheckAndClearDoesNotThrowWhenClear)
{
    aether::DeviceFlag flag;
    EXPECT_NO_THROW(flag.checkAndClear("DeviceFlagTest"));
}

TEST_F(DeviceFlagTest, ClearResetsWithoutThrowing)
{
    aether::DeviceFlag flag;
    flag.deviceView().set();
    ASSERT_TRUE(flag.isSet());
    flag.clear();
    EXPECT_FALSE(flag.isSet());
}

TEST_F(DeviceFlagTest, ConcurrentSetFromManyThreadsIsIdempotent)
{
    aether::DeviceFlag flag;
    auto view = flag.deviceView();
#pragma omp parallel for
    for (int k = 0; k < 4096; ++k) {
        view.set();
    }
    EXPECT_TRUE(flag.isSet());
}

} // namespace
} // namespace aether_tests
