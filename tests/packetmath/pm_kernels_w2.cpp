// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Packet-math kernels at width 2, built for plain x86-64 (SSE2, no AVX, no
// FMA): the Packet<double, 2> specialization only exists there, and this is
// also the build that exercises the non-FMA (Dekker-split) code paths.

#include "tests/packetmath/PacketMathKernelImpl.h"

#if defined(__AVX__) || defined(__FMA__)
#error "pm_kernels_w2.cpp must be built for SSE2 without AVX/FMA"
#endif

namespace aether_pm_test {
namespace {
bool always() { return true; }
} // namespace

const Kernels& kernelsW2()
{
    static const Kernels k{ 2, &always, &unaryImpl<2>, &binaryImpl<2> };
    return k;
}

} // namespace aether_pm_test
