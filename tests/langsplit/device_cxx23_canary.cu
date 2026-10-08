// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Language-split canary: this TU uses an explicit object parameter
// (`this Self&&`, P0847 "deducing this"), a C++23-only construct. Compiled
// with `-std=c++20` it MUST FAIL — that failure is the positive control
// proving the std=20 cap on device code is actually enforced, not merely
// documented. See check_langsplit.sh.
//
// Compile-only (`-c`); no `main()` needed.

#include "aether/macros.h"

namespace aether_langsplit {

struct Widget {
    int value = 0;

    // Explicit object parameter — ill-formed before C++23.
    AETHER_DEVICEHOST() int get(this Widget&& self)
    {
        return self.value;
    }
};

AETHER_DEVICEHOST() int useWidget()
{
    Widget w;
    return w.get();
}

} // namespace aether_langsplit
