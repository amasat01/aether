// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/nvrtc/shim_include/__aether_nvrtc_prelude.h — passed to NVRTC via
// --pre-include=__aether_nvrtc_prelude.h (see tools/nvrtc/README.md).
//
// aether/math/detail/BoundedTrig.h calls the GCC compiler builtin
// `__builtin_memcpy` directly (for FP64 bit-reinterpretation, e.g.
// doubleAsU64/applySign). Real nvcc's own device frontend (cicc) resolves
// this fine, but NVRTC's standalone frontend does not — confirmed by
// experiment ("identifier __builtin_memcpy is undefined"), and confirmed by
// reading that libcu++ itself deliberately avoids this exact trap: its own
// bit_cast (cuda/std/__bit/bit_cast.h) uses the portable
// `__builtin_bit_cast` intrinsic instead, never __builtin_memcpy. This is
// an aether-source-level NVRTC gap (the real fix is for aether to switch to
// __builtin_bit_cast or a union-based reinterpret), out of this tool's edit
// scope — worked around here with a plain byte-copy definition under the
// SAME identifier, so aether's existing call sites need no change.
#pragma once

__host__ __device__ inline void* __builtin_memcpy(void* dst, const void* src, unsigned long n)
{
    unsigned char* d       = static_cast<unsigned char*>(dst);
    const unsigned char* s = static_cast<const unsigned char*>(src);
    for (unsigned long i = 0; i < n; ++i)
        d[i] = s[i];
    return dst;
}
