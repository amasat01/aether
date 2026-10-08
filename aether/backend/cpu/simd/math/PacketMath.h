// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketMath.h
 * @brief Vector transcendental and rounding math on
 *        `aether::simd::Packet<double, W>`: `packetExp(p)`, `packetPow(a, b)`,
 *        `packetSin(p)`, ... — one call evaluates every lane.
 *
 * HOST-ONLY and opt-in: nothing in aether includes this header by default
 * (not `aether/aether.h`, not `aether::math`). Widths: whatever
 * `Packet<double, W>` exists for the TU's ISA flags — 8 (AVX-512F), 4 (AVX),
 * 2 (SSE2 without AVX) — plus the scalar width 1, which runs the SAME
 * algorithm (so a lane's result does not depend on the width it was
 * computed at, for one set of compile flags with `-ffp-contract=off`).
 * `double` only; `float` packets are not covered.
 *
 * `AETHER_HOST_VECTOR_MATH` routes the `aether::math` facade to these
 * functions (see `aether/math/detail/HostVectorMath.h`); `packetSinCos`
 * and `packetFma` are packet-level only (the facade's `sincos` reaches the
 * hook through its `sin`/`cos` legs, and `fma` stays on `std::fma`).
 *
 * ACCURACY. Every function is FAITHFULLY ROUNDED: the error against the
 * exact result is below 1 ULP for every input in its domain, at every width
 * (each result is one of the two doubles that bracket the exact value, and
 * the exact value itself whenever it is representable). This is verified
 * against a 128-bit mpmath reference over more than 10^6 inputs per function
 * (`make faithful-gate`; corpus by `tools/packet_math_ref/gen_golden.py`,
 * pinned by md5) and, in the unit suite, over a committed hard-case set.
 * Largest error observed on that corpus:
 *
 * | function | max error vs exact (ULP) | notes |
 * |---|---|---|
 * | exp, exp2, exp10, pow | 0.75 | in the subnormal range (double rounding into it); 0.6 elsewhere |
 * | expm1 | 0.58 | |
 * | log, log2, log10, log1p | 0.50 | log carried in double-double to ~2^-68 |
 * | sin, cos, sincos, tan | 0.57, 0.57, 0.57, 0.58 | Payne-Hanek reduction beyond 2^19 (per lane, scalar) |
 * | atan, atan2 | 0.66, 0.65 | atan2: full signed-zero/inf table |
 * | asin, acos | 0.59, 0.64 | |
 * | sinh, cosh, tanh | 0.57, 0.52, 0.54 | |
 * | asinh, acosh, atanh | 0.50 | |
 * | cbrt, rsqrt, hypot | 0.50, 0.50, 0.75 | hypot: 0.75 for subnormal results |
 * | floor, ceil, trunc, round, rint, fabs, copysign, fmax, fmin, fdim, sqrt | exact / correctly rounded | |
 *
 * Against glibc the distance is at most 1 ULP wherever glibc is itself
 * within 1 ULP of the exact result (log10, sinh, tanh, acosh, atanh: 2;
 * cbrt: 3, where glibc's own error is larger). `tests/test_PacketMath*`
 * asserts both per width (AVX-512 8, AVX2 4, SSE2 2, scalar 1), over
 * corpora that include zeros, signed zeros, subnormals, infinities, NaN,
 * overflow/underflow thresholds and exact-case inputs; NaN/inf classes must
 * match exactly. Results are not always correctly rounded and floating-point
 * exception flags are not raised the way libm raises them. `fma` is one
 * rounding where the ISA has FMA and a multiply then an add otherwise.
 *
 * Not ported: `fmod` (the quotient-truncation formula is not exact for
 * large |x/y|; `std::fmod` is exact and stays the reference).
 */

#include <cstddef>

#include "aether/backend/cpu/simd/Packet.h"
#include "aether/backend/cpu/simd/math/PacketArcTrig.h"
#include "aether/backend/cpu/simd/math/PacketExpLog.h"
#include "aether/backend/cpu/simd/math/PacketHyperbolic.h"
#include "aether/backend/cpu/simd/math/PacketMathCore.h"
#include "aether/backend/cpu/simd/math/PacketMisc.h"
#include "aether/backend/cpu/simd/math/PacketRounding.h"
#include "aether/backend/cpu/simd/math/PacketSinCos.h"

namespace aether {
namespace simd {

namespace pm {
template<std::size_t W>
AETHER_PM_INLINE Carrier<W> raw(Packet<double, W> p)
{
    if constexpr (W == 1) {
        return p.val_;
    } else {
        return p.reg_;
    }
}
} // namespace pm

#define AETHER_PM_PACKET_UNARY(NAME, IMPL)                                                         \
    template<std::size_t W>                                                                        \
    AETHER_PM_INLINE Packet<double, W> NAME(Packet<double, W> x)                                   \
    {                                                                                              \
        return Packet<double, W>{ pm::IMPL(pm::raw(x)) };                                          \
    }
#define AETHER_PM_PACKET_BINARY(NAME, IMPL)                                                        \
    template<std::size_t W>                                                                        \
    AETHER_PM_INLINE Packet<double, W> NAME(Packet<double, W> a, Packet<double, W> b)              \
    {                                                                                              \
        return Packet<double, W>{ pm::IMPL(pm::raw(a), pm::raw(b)) };                              \
    }

AETHER_PM_PACKET_UNARY(packetExp, vexp)
AETHER_PM_PACKET_UNARY(packetExp2, vexp2)
AETHER_PM_PACKET_UNARY(packetExp10, vexp10)
AETHER_PM_PACKET_UNARY(packetExpm1, vexpm1)
AETHER_PM_PACKET_UNARY(packetLog, vlog)
AETHER_PM_PACKET_UNARY(packetLog2, vlog2)
AETHER_PM_PACKET_UNARY(packetLog10, vlog10)
AETHER_PM_PACKET_UNARY(packetLog1p, vlog1p)
AETHER_PM_PACKET_BINARY(packetPow, vpow)
AETHER_PM_PACKET_UNARY(packetSin, vsin)
AETHER_PM_PACKET_UNARY(packetCos, vcos)
AETHER_PM_PACKET_UNARY(packetTan, vtan)
AETHER_PM_PACKET_UNARY(packetAtan, vatan)
AETHER_PM_PACKET_BINARY(packetAtan2, vatan2)
AETHER_PM_PACKET_UNARY(packetAsin, vasin)
AETHER_PM_PACKET_UNARY(packetAcos, vacos)
AETHER_PM_PACKET_UNARY(packetSinh, vsinh)
AETHER_PM_PACKET_UNARY(packetCosh, vcosh)
AETHER_PM_PACKET_UNARY(packetTanh, vtanh)
AETHER_PM_PACKET_UNARY(packetAsinh, vasinh)
AETHER_PM_PACKET_UNARY(packetAcosh, vacosh)
AETHER_PM_PACKET_UNARY(packetAtanh, vatanh)
AETHER_PM_PACKET_UNARY(packetFloor, vfloor)
AETHER_PM_PACKET_UNARY(packetCeil, vceil)
AETHER_PM_PACKET_UNARY(packetTrunc, vtrunc)
AETHER_PM_PACKET_UNARY(packetRound, vround)
AETHER_PM_PACKET_UNARY(packetRint, vrint)
AETHER_PM_PACKET_UNARY(packetAbs, abs)
AETHER_PM_PACKET_BINARY(packetCopysign, copysign)
AETHER_PM_PACKET_BINARY(packetFmax, vfmax)
AETHER_PM_PACKET_BINARY(packetFmin, vfmin)
AETHER_PM_PACKET_BINARY(packetFdim, vfdim)
AETHER_PM_PACKET_UNARY(packetSqrt, sqrt)
AETHER_PM_PACKET_UNARY(packetRsqrt, vrsqrt)
AETHER_PM_PACKET_UNARY(packetCbrt, vcbrt)
AETHER_PM_PACKET_BINARY(packetHypot, vhypot)

#undef AETHER_PM_PACKET_UNARY
#undef AETHER_PM_PACKET_BINARY

/** @brief `a*b + c` per lane, one rounding where the ISA has FMA. */
template<std::size_t W>
AETHER_PM_INLINE Packet<double, W> packetFma(Packet<double, W> a, Packet<double, W> b, Packet<double, W> c)
{
    return Packet<double, W>{ pm::fma(pm::raw(a), pm::raw(b), pm::raw(c)) };
}

/** @brief sin and cos of every lane, sharing one argument reduction. */
template<std::size_t W>
AETHER_PM_INLINE void packetSinCos(Packet<double, W> x, Packet<double, W>* sinOut, Packet<double, W>* cosOut)
{
    pm::Carrier<W> s, c;
    pm::vsincos(pm::raw(x), s, c);
    *sinOut = Packet<double, W>{ s };
    *cosOut = Packet<double, W>{ c };
}

} // namespace simd
} // namespace aether
