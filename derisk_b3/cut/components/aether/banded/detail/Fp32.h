// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Fp32.h
 * @brief The FP32 error-free-transform (EFT) primitives every certified banded
 *        op is built from, plus the bit-reinterpret and rounding helpers the
 *        codec needs.
 *
 * @section contraction Every FMA is explicit
 * An EFT recovers the rounding error of the step before it, and that recovery
 * is exact only if the step before it actually rounded. A compiler that fuses
 * a multiply into a neighbouring add deletes precisely the rounding these
 * functions exist to pin: the value stays close, but the lower limbs stop
 * meaning what the accuracy derivation says they mean.
 *
 * So the rounding is pinned at both ends and on both arms:
 *  - **device**: inline PTX (`mul.rn.f32`, `add.rn.f32`, `sub.rn.f32`,
 *    `fma.rn.f32`). An asm operand can never be fused into a neighbouring FFMA,
 *    and inline asm can never be out-of-lined into a libdevice call the way
 *    `__fmaf_rn` can be.
 *  - **host**: `AETHER_BAND_FP_BARRIER`, an empty asm with the value as an
 *    in/out register operand. gcc and clang both default to
 *    `-ffp-contract=fast` and fuse across statement and inlined-call
 *    boundaries; `#pragma STDC FP_CONTRACT OFF` is accepted and silently
 *    ignored by gcc. `mulRN` guards its result (nothing may pull the product
 *    it just rounded into somebody else's add); `addRN`/`subRN` guard their
 *    operands (nobody else's product may be pulled into the sum they are about
 *    to round). A fusion needs a multiply on one side and a sum on the other,
 *    so blocking either end suffices — these block both.
 *
 * The consequence: the banded results are independent of `-fmad`/
 * `-ffp-contract`.
 */

#include <cstdint>

#include "aether/macros.h"
#include "aether/banded/detail/Fp32Arm.h" // variant-supplied: AETHER_BAND_FP_BARRIER and the AETHER_BAND_IMPL_* bodies

namespace aether {
namespace banded {
namespace detail {

// =====================================================================
//  Bit reinterpretation
// =====================================================================

/** @brief Reinterpret float bits as `int32` (zero-cost on the device). */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() int floatAsInt(float f)
{
    AETHER_BAND_IMPL_FLOAT_AS_INT(f)
}

/** @brief Reinterpret `int32` bits as float (zero-cost on the device). */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float intAsFloat(int i)
{
    AETHER_BAND_IMPL_INT_AS_FLOAT(i)
}

/** @brief `|x|` by clearing the sign bit — never `std::fabs`, which would put
 *         a `std::` call on the device arm for one integer AND. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float absf(float x)
{
    return intAsFloat(floatAsInt(x) & 0x7FFFFFFF);
}

// =====================================================================
//  Round-to-nearest guarded FP32 arithmetic
// =====================================================================

/**
 * @brief Single-rounding FP32 fused multiply-add, `a*b + c`.
 *
 * Device: inline PTX `fma.rn.f32`, which is bit-identical to IEEE `fmaf` and —
 * unlike `__fmaf_rn`, measured — can never degrade into an out-of-line
 * libdevice call. Host: `__builtin_fmaf`, the compiler's own single-rounding
 * FMA, so the host arm is not a `std::` call either.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float fmaRN(
    float a, float b, float c)
{
    AETHER_BAND_IMPL_FMA(a, b, c)
}

/** @brief Round-to-nearest FP32 multiply, guarded against FMA contraction. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float mulRN(
    float a, float b)
{
    AETHER_BAND_IMPL_MUL(a, b)
}

/** @brief Round-to-nearest FP32 add, guarded against algebraic elimination. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float addRN(
    float a, float b)
{
    AETHER_BAND_IMPL_ADD(a, b)
}

/** @brief Round-to-nearest FP32 subtract, guarded the same way. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float subRN(
    float a, float b)
{
    AETHER_BAND_IMPL_SUB(a, b)
}

// =====================================================================
//  The three error-free transforms
// =====================================================================

/**
 * @brief Knuth Two-Sum: `a + b` exactly, as `(s, e)` with `s = RN(a+b)`.
 *        Makes no assumption about the operands' order. 6 FP32 ops.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void twoSum(
    float a, float b, float& s, float& e)
{
    s        = addRN(a, b);
    float bp = subRN(s, a);
    float ap = subRN(s, bp);
    float eb = subRN(b, bp);
    float ea = subRN(a, ap);
    e        = addRN(ea, eb);
}

/**
 * @brief Fast2Sum: `s = RN(a+b)`, `e` the exact error, assuming `|a| >= |b|`.
 *        3 FP32 ops. The assumption is the caller's; a cancelling sum must use
 *        `twoSum` (and, at the carrier level, `normalizeSafe`).
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void fast2Sum(
    float a, float b, float& s, float& e)
{
    s        = addRN(a, b);
    float bp = subRN(s, a);
    e        = subRN(b, bp);
}

/** @brief Two-Product via FMA: `p = RN(a*b)`, `e` the exact error. 2 FP32 ops. */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void twoProd(
    float a, float b, float& p, float& e)
{
    p = mulRN(a, b);
    e = fmaRN(a, b, -p);
}

// =====================================================================
//  Round-to-nearest-even float -> integer (codec side)
// =====================================================================

/** @brief Round-to-nearest-even `float` -> `int32`. One hardware instruction
 *         on the device; `__builtin_lrintf` (which honours the current, i.e.
 *         default RNE, rounding mode) on the host. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::int32_t roundToInt32(
    float x)
{
    AETHER_BAND_IMPL_ROUND32(x)
}

/** @brief Round-to-nearest-even `float` -> `int64`. @see roundToInt32. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::int64_t roundToInt64(
    float x)
{
    AETHER_BAND_IMPL_ROUND64(x)
}

} // namespace detail
} // namespace banded
} // namespace aether
