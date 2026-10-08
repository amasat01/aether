// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Width-generic kernel bodies, included by each pm_kernels_w*.cpp TU (never
// by two TUs for the same width). Each loop applies the PUBLIC packet entry
// points (aether::simd::packetExp, ...) over whole packets; inputs are padded
// by the caller to a multiple of the width.

#include <cstddef>

#include "aether/backend/cpu/simd/math/PacketMath.h"
#include "tests/packetmath/PacketMathKernels.h"

namespace aether_pm_test {
namespace {

template<std::size_t W>
void unaryImpl(int fn, const double* x, double* out, std::size_t n)
{
    using namespace aether::simd;
    using P = Packet<double, W>;
    for (std::size_t i = 0; i < n; i += W) {
        const P v = P::load(x + i);
        P r;
        switch (fn) {
        case kExp: r = packetExp(v); break;
        case kExp2: r = packetExp2(v); break;
        case kExp10: r = packetExp10(v); break;
        case kExpm1: r = packetExpm1(v); break;
        case kLog: r = packetLog(v); break;
        case kLog2: r = packetLog2(v); break;
        case kLog10: r = packetLog10(v); break;
        case kLog1p: r = packetLog1p(v); break;
        case kSin: r = packetSin(v); break;
        case kCos: r = packetCos(v); break;
        case kSinCosSin: {
            P c;
            packetSinCos(v, &r, &c);
            break;
        }
        case kSinCosCos: {
            P s;
            packetSinCos(v, &s, &r);
            break;
        }
        case kTan: r = packetTan(v); break;
        case kAtan: r = packetAtan(v); break;
        case kAsin: r = packetAsin(v); break;
        case kAcos: r = packetAcos(v); break;
        case kSinh: r = packetSinh(v); break;
        case kCosh: r = packetCosh(v); break;
        case kTanh: r = packetTanh(v); break;
        case kAsinh: r = packetAsinh(v); break;
        case kAcosh: r = packetAcosh(v); break;
        case kAtanh: r = packetAtanh(v); break;
        case kFloor: r = packetFloor(v); break;
        case kCeil: r = packetCeil(v); break;
        case kTrunc: r = packetTrunc(v); break;
        case kRound: r = packetRound(v); break;
        case kRint: r = packetRint(v); break;
        case kAbs: r = packetAbs(v); break;
        case kSqrt: r = packetSqrt(v); break;
        case kRsqrt: r = packetRsqrt(v); break;
        case kCbrt: r = packetCbrt(v); break;
        default: r = v; break;
        }
        P::store(out + i, r);
    }
}

template<std::size_t W>
void binaryImpl(int fn, const double* x, const double* y, double* out, std::size_t n)
{
    using namespace aether::simd;
    using P = Packet<double, W>;
    for (std::size_t i = 0; i < n; i += W) {
        const P a = P::load(x + i);
        const P b = P::load(y + i);
        P r;
        switch (fn) {
        case kPow: r = packetPow(a, b); break;
        case kAtan2: r = packetAtan2(a, b); break;
        case kHypot: r = packetHypot(a, b); break;
        case kFmax: r = packetFmax(a, b); break;
        case kFmin: r = packetFmin(a, b); break;
        case kFdim: r = packetFdim(a, b); break;
        case kCopysign: r = packetCopysign(a, b); break;
        default: r = a; break;
        }
        P::store(out + i, r);
    }
}

} // namespace
} // namespace aether_pm_test
