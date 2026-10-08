// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/volatile_probe.cu — SASS-level CSE-audit probe for
// `View::as_volatile()` (aether/view/View.h). Standalone TU, compiled
// directly by `tools/volatile_audit.sh` via bare `nvcc` — never wired into
// tests/CMakeLists.txt's glob: a probe compiled only by its own shell
// script, never executed, never linked into a gtest binary.
//
// Two twin kernels, deliberate triple-read of the same shared-memory
// location:
//   - volatileProbePlainKernel   — plain (non-volatile) View over __shared__
//     memory. The compiler is free to CSE the three reads into one LDS.
//   - volatileProbeVolatileKernel — the SAME shared buffer, but read through
//     `as_volatile()`. Each read must be a genuine load — CSE is blocked —
//     and neither kernel may show a stack-backed mirror (STL/LDL).
//
// `tools/volatile_audit.sh` compiles this file (ptxas/cuobjdump only, no GPU
// execution), disassembles both kernels, and asserts (a) volatile emits
// strictly more `LDS` than plain (CSE blocked) and (b) both emit zero
// `STL`/`LDL` (no stack-mirror traffic).

#include <aether/aether.h>

// `aether::make_view()` is host-only (it allocates/validates against a
// `Chunk`); this probe builds the View directly over `__shared__` memory via
// the AETHER_DEVICEHOST()-qualified constructor + mapping, the same pattern
// device-side kernel code (not test-harness code) uses.
using ProbeExtents = aether::extents<3>;
using ProbeMapping = aether::layout_right::mapping<ProbeExtents>;

AETHER_KERNEL()
void volatileProbePlainKernel(double* out)
{
    AETHER_SHARED() double buf[3];
    const unsigned t = threadIdx.x % 3;
    buf[t] = static_cast<double>(threadIdx.x);
    __syncthreads();

    const ProbeMapping map{ ProbeExtents{} };
    aether::View<double, ProbeExtents> v(buf, map, aether::Device(kDLCUDA));
    // Deliberate triple-read of the SAME location: a plain view lets the
    // compiler CSE this to one load.
    const double acc = v(0) + v(0) + v(0);
    out[threadIdx.x] = acc;
}

AETHER_KERNEL()
void volatileProbeVolatileKernel(double* out)
{
    AETHER_SHARED() double buf[3];
    const unsigned t = threadIdx.x % 3;
    buf[t] = static_cast<double>(threadIdx.x);
    __syncthreads();

    const ProbeMapping map{ ProbeExtents{} };
    aether::View<double, ProbeExtents> v(buf, map, aether::Device(kDLCUDA));
    auto vv = v.as_volatile();
    // Same triple-read; volatile-qualified access must NOT be CSE'd.
    const double acc = vv(0) + vv(0) + vv(0);
    out[threadIdx.x] = acc;
}
