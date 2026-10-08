// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::DeviceFlag` tests (CUDA mode); test_DeviceFlag.cpp mirrors this
// over the host build.

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

AETHER_KERNEL()
void setFlagKernel(aether::DeviceFlagView flag)
{
    flag.set();
}

AETHER_KERNEL()
void raceSetFlagKernel(aether::DeviceFlagView flag)
{
    // Every thread races to set the same sticky flag — must never crash
    // and must always end up set (idempotent OR).
    flag.set();
}

class DeviceFlagTest : public ::testing::Test { };

TEST_F(DeviceFlagTest, InitiallyClear)
{
    aether::DeviceFlag flag;
    EXPECT_FALSE(flag.isSet());
}

TEST_F(DeviceFlagTest, SetMakesIsSetTrue)
{
    aether::DeviceFlag flag;
    setFlagKernel<<<1, 1>>>(flag.deviceView());
    aether::cuda::checkLastLaunch("setFlagKernel");
    EXPECT_TRUE(flag.isSet());
}

TEST_F(DeviceFlagTest, CheckAndClearThrowsWhenSetThenClears)
{
    aether::DeviceFlag flag;
    setFlagKernel<<<1, 1>>>(flag.deviceView());
    aether::cuda::checkLastLaunch("setFlagKernel");
    ASSERT_TRUE(flag.isSet());
    EXPECT_THROW(flag.checkAndClear("DeviceFlagTest"), aether::Error);
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
    setFlagKernel<<<1, 1>>>(flag.deviceView());
    aether::cuda::checkLastLaunch("setFlagKernel");
    ASSERT_TRUE(flag.isSet());
    flag.clear();
    EXPECT_FALSE(flag.isSet());
}

TEST_F(DeviceFlagTest, ConcurrentSetFromManyThreadsIsIdempotent)
{
    aether::DeviceFlag flag;
    raceSetFlagKernel<<<64, 256>>>(flag.deviceView());
    aether::cuda::checkLastLaunch("raceSetFlagKernel");
    EXPECT_TRUE(flag.isSet());
}

} // namespace
} // namespace aether_tests
