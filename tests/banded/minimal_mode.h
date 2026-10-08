// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file minimal_mode.h
 * @brief `isMinimalMode()` — the shrink switch the banded batteries consult.
 *
 * The banded conformance batteries carry BULK arms on top of their enumerated
 * corpora (hundreds of thousands of random rows against the exact oracle).
 * Those are cheap in a release build and expensive under a sanitizer, so the
 * sanitize driver (`tests/sanitize/run_sanitize.py`, which sets
 * `AETHER_TEST_MINIMAL=1`) asks tests to shrink.
 *
 * ★ ONLY the BULK arms may consult it. The ENUMERATED corpora — the pattern
 * list, the exponent sweep, the specials, the tier edges — are never sampled
 * and never shrunk: a corpus that thins under a flag is a corpus whose
 * coverage claim depends on the flag.
 */

#include <cstdlib>

namespace aether_tests {

/** @brief True when `AETHER_TEST_MINIMAL=1` is set in the environment. */
inline bool isMinimalMode()
{
    const char* e = std::getenv("AETHER_TEST_MINIMAL");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
}

} // namespace aether_tests
