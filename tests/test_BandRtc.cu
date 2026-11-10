// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandRtc.cu
 * @brief Runs `tests/rtc/check_banded_rtc.sh`: the banded family compiles under NVRTC
 *        with only the allowed std headers, and a host-only symbol is rejected.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

TEST(BandRtc, CompilesUnderNvrtcWithIncludeBudgetAndRejectsHostSymbol)
{
    const std::string scratch = std::string(::testing::TempDir()) + "band_rtc";
    const std::string cmd = std::string("bash ") + AETHER_RTC_ROW + "/check_banded_rtc.sh " + scratch;
    EXPECT_EQ(std::system(cmd.c_str()), 0) << cmd;
}
