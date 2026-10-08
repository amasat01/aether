// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Packet-math kernels at widths 1 and 4, built with the target's default
// flags (x86-64-v3: AVX2 + FMA). See PacketMathKernels.h.

#include "tests/packetmath/PacketMathKernelImpl.h"

#if !defined(__AVX__)
#error "pm_kernels_w1_w4.cpp expects an AVX target (the suite's default -march)"
#endif

namespace aether_pm_test {
namespace {
bool always() { return true; }
} // namespace

const Kernels& kernelsW1()
{
    static const Kernels k{ 1, &always, &unaryImpl<1>, &binaryImpl<1> };
    return k;
}

const Kernels& kernelsW4()
{
    static const Kernels k{ 4, &always, &unaryImpl<4>, &binaryImpl<4> };
    return k;
}

} // namespace aether_pm_test
