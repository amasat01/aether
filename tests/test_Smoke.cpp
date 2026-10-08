// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Bootstrap smoke test (host / AETHER_CPP_MODE build).
//
// Paired with test_Smoke.cu (same fixture, same case names) — only one of
// the two compiles into any given aether_tests binary: tests/CMakeLists.txt
// globs test_*.cpp in AETHER_CPP_MODE and test_*.cu otherwise.

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/macros.h>

namespace aether_tests {
namespace {

AETHER_DEVICEHOST() int answer()
{
    return 42;
}

class SmokeTest : public ::testing::Test { };

TEST_F(SmokeTest, MacrosCompile)
{
    EXPECT_EQ(answer(), 42);
}

TEST_F(SmokeTest, VersionVisible)
{
    EXPECT_EQ(aether::version_major, 0);
}

} // namespace
} // namespace aether_tests
