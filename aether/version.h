// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file version.h
 * @brief `aether::version()`/`gitRevision()`/`abi()`: a committed header —
 *        a header-only library must be includable from a fresh checkout
 *        without running CMake, so there is no configure_file/generated
 *        header in the source tree at all.
 *
 * `AETHER_VERSION`/`AETHER_ABI` are literals here, kept in sync with
 * `CMakeLists.txt`'s `project(aether VERSION ...)` by
 * `VersionTest.VersionMatchesCMakeProjectVersion`
 * (`tests/test_Version.{cpp,cu}`), which reads the CMake value through a
 * compile definition rather than re-deriving it — so a drift between this
 * literal and `project(VERSION)` fails that test, not silently.
 *
 * `gitRevision()` returns the `AETHER_GIT_REVISION` compile definition
 * when the build defines one — root `CMakeLists.txt` computes it via `git
 * rev-parse HEAD` at CONFIGURE time and threads it as an INTERFACE compile
 * definition on the `aether` target (so every consumer, including a
 * downstream `find_package(aether)` via the exported target's
 * `INTERFACE_COMPILE_DEFINITIONS`, sees the same value) — and falls back
 * to the literal "unknown" below otherwise: a bare `-I<repo_root>` compile
 * with no CMake in play at all (`tests/headers/check_header_diet.sh`)
 * never defines the macro, so this header still compiles standalone and
 * answers "unknown" rather than failing.
 */

namespace aether {

/** @brief This checkout's `project(... VERSION)` string. */
constexpr const char* version() { return "0.2.0"; }

#ifndef AETHER_GIT_REVISION
#define AETHER_GIT_REVISION "unknown"
#endif

/** @brief `git rev-parse HEAD` at CMake configure time (via the
 *  `AETHER_GIT_REVISION` compile definition on the `aether` target), or
 *  "unknown" — outside a git checkout, or when this header is compiled
 *  without CMake at all (the macro is simply never defined). */
constexpr const char* gitRevision() { return AETHER_GIT_REVISION; }

/** @brief Hand-maintained ABI tag — bump ONLY on a genuine binary-layout
 *  break (never on a source-compatible addition). */
constexpr const char* abi() { return "aether-abi-1"; }

} // namespace aether
