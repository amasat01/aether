// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::version()`/`gitRevision()`/`abi()` (`aether/version.h`, a
// committed header — no configure_file, no generated file in the source
// tree, so a header-only checkout needs no CMake run to be includable).
// Paired with test_Version.cpp (identical content): host-only, no device
// code.

#include <cctype>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/aether.h>

#ifndef AETHER_TEST_EXPECTED_VERSION
#error "AETHER_TEST_EXPECTED_VERSION must be defined by tests/CMakeLists.txt (from PROJECT_VERSION)"
#endif

namespace aether_tests {

class VersionTest : public ::testing::Test { };

TEST_F(VersionTest, StringsAreNonEmpty)
{
    EXPECT_GT(std::strlen(aether::version()), 0u);
    EXPECT_GT(std::strlen(aether::gitRevision()), 0u);
    EXPECT_GT(std::strlen(aether::abi()), 0u);
}

TEST_F(VersionTest, VersionMatchesCMakeProjectVersion)
{
    EXPECT_STREQ(aether::version(), AETHER_TEST_EXPECTED_VERSION);
}

TEST_F(VersionTest, AbiMatchesTheHandMaintainedLiteral)
{
    EXPECT_STREQ(aether::abi(), "aether-abi-1");
}

// Under a normal CMake build (this test binary), the `aether` INTERFACE
// target's `AETHER_GIT_REVISION` compile definition is always populated
// from a real `git rev-parse HEAD` (this worktree is a git checkout) — so
// gitRevision() must be a full, real 40-character lowercase hex SHA-1
// here, not the "unknown" fallback. (The fallback path itself —
// gitRevision() == "unknown" under a bare `-I<repo_root>` compile with no
// CMake in play, e.g. tests/headers/check_header_diet.sh — is verified
// out-of-band: adding it here would need a second, CMake-def-free compile
// target/probe, which is not "cheap" inside this suite.)
TEST_F(VersionTest, GitRevisionIsFortyHexCharsUnderThisBuild)
{
    const char* rev = aether::gitRevision();
    const std::size_t len = std::strlen(rev);
    ASSERT_EQ(len, 40u) << "gitRevision() = \"" << rev << "\" — expected a full SHA-1 (40 hex chars) "
                            "under a normal CMake build of a git checkout";
    for (std::size_t i = 0; i < len; ++i) {
        EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(rev[i])))
            << "gitRevision()[" << i << "] = '" << rev[i] << "' is not a hex digit";
    }
}

} // namespace aether_tests
