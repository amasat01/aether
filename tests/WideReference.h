// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A test whose reference is `long double` (or falls back to it where there is no
// `__float128`) cannot resolve the error it bounds where `long double` is `double`
// itself (Apple arm64): there it reports SKIPPED. x86-64 (64-bit `long double`,
// `__float128`) and Linux aarch64 (113-bit `long double`) always run it.

#include <limits>

#include <gtest/gtest.h>

#define AETHER_SKIP_WITHOUT_WIDE_REFERENCE()                                                     \
    do {                                                                                         \
        if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) { \
            GTEST_SKIP() << "the reference needs a long double wider than double";              \
        }                                                                                        \
    } while (0)
