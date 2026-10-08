// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Language-split canary: host code MUST compile at C++23 — it uses
// `if consteval` (P1938), a C++23-only construct. Compiled with `-std=c++23`.
//
// Compile-only (`-c`); no `main()` needed. See check_langsplit.sh.

#include "aether/macros.h"

namespace aether_langsplit {

constexpr int classify(int x)
{
    if consteval {
        return x + 1;
    } else {
        return x - 1;
    }
}

AETHER_HOST() int useClassify()
{
    constexpr int compileTimeValue = classify(1);
    return compileTimeValue + classify(2);
}

} // namespace aether_langsplit
