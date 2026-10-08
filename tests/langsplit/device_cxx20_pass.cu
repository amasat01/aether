// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Language-split canary: device code MUST compile at CUDA std 20
// (nvcc's device-code standard ceiling). Compiled with `-std=c++20`.
//
// Compile-only (`-c`); no `main()` needed. See check_langsplit.sh.

#include "aether/macros.h"

namespace aether_langsplit {

AETHER_KERNEL()
void trivialKernel(int* out)
{
    *out = 1;
}

AETHER_DEVICEHOST() int identity(int x)
{
    return x;
}

} // namespace aether_langsplit
