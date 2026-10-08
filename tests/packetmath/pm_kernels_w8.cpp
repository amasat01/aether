// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Packet-math kernels at width 8, built with -mavx512f added to the target
// flags; the tests run them only when the CPU reports AVX-512F. Nothing here
// is shared inline code with the other TUs (width-8 instantiations only).

#if defined(__AVX512F__)
// Under AVX-512, PacketMask<T, W<=8> stores a __mmask8, and Packet.h's AVX2
// (width 4/8) cmp* friends brace-initialize it from an `unsigned`
// movemask -> -Wnarrowing in any TU that sees both specializations with
// -mavx512f. Pre-existing in Packet.h and harmless (movemask of 4/8 lanes
// fits 8 bits); silenced here rather than changing that shared header.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#include "tests/packetmath/PacketMathKernelImpl.h"
#pragma GCC diagnostic pop
#else
#include "tests/packetmath/PacketMathKernels.h"
#endif

namespace aether_pm_test {
namespace {
bool cpuHasAvx512() { return __builtin_cpu_supports("avx512f"); }
#if !defined(__AVX512F__)
void noUnary(int, const double*, double*, std::size_t) {}
void noBinary(int, const double*, const double*, double*, std::size_t) {}
bool never() { return false; }
#endif
} // namespace

const Kernels& kernelsW8()
{
#if defined(__AVX512F__)
    static const Kernels k{ 8, &cpuHasAvx512, &unaryImpl<8>, &binaryImpl<8> };
#else
    (void)&cpuHasAvx512;
    static const Kernels k{ 8, &never, &noUnary, &noBinary };
#endif
    return k;
}

} // namespace aether_pm_test
