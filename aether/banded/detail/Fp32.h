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
 *
 * @section std No `std::` on the device arm
 * Every `std::memcpy`/`__builtin_*` call below sits inside the `#ifndef
 * __CUDA_ARCH__` half of its own function; the device compilation pass never
 * sees one.
 */

#include <cstdint>
#include <cstring>

#include "aether/macros.h"

// ---------------------------------------------------------------------------
// AETHER_BAND_FP_BARRIER — the host half of the rounding guard.
//
// `x` must be a modifiable lvalue of float type; it is named twice, so do not
// pass an expression with side effects. Costs zero instructions: the value
// must exist in a register before the asm and is unknown to the optimiser
// after it, so no fusion can cross it.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
// The PTX arms below already pin the rounding; nothing to add.
#define AETHER_BAND_FP_BARRIER(x) ((void)0)
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif defined(__aarch64__) || defined(__arm__)
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+r"(x))
#endif
#else
// Unknown compiler: fall back to a `volatile` round-trip rather than to
// nothing. It costs a stack store/load, but a guard that quietly evaporates on
// an unrecognised toolchain is how this class of defect gets in.
#define AETHER_BAND_FP_BARRIER(x)                                              \
    do {                                                                       \
        volatile float aetherBandBarrier_ = (x);                               \
        (x)                               = aetherBandBarrier_;                \
    } while (0)
#endif

namespace aether {
namespace banded {
namespace detail {

// =====================================================================
//  Bit reinterpretation
// =====================================================================

/** @brief Reinterpret float bits as `int32` (zero-cost on the device). */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() int floatAsInt(float f)
{
#ifdef __CUDA_ARCH__
    return __float_as_int(f);
#else
    int r;
    std::memcpy(&r, &f, sizeof(int));
    return r;
#endif
}

/** @brief Reinterpret `int32` bits as float (zero-cost on the device). */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float intAsFloat(int i)
{
#ifdef __CUDA_ARCH__
    return __int_as_float(i);
#else
    float r;
    std::memcpy(&r, &i, sizeof(float));
    return r;
#endif
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
#ifdef __CUDA_ARCH__
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
#else
    return __builtin_fmaf(a, b, c);
#endif
}

/** @brief Round-to-nearest FP32 multiply, guarded against FMA contraction. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float mulRN(
    float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    float r = a * b;
    AETHER_BAND_FP_BARRIER(r); // this product may not fuse into anyone's sum
    return r;
#endif
}

/** @brief Round-to-nearest FP32 add, guarded against algebraic elimination. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float addRN(
    float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    float x = a, y = b;
    AETHER_BAND_FP_BARRIER(x); // no caller's product may fuse into this sum
    AETHER_BAND_FP_BARRIER(y);
    return x + y;
#endif
}

/** @brief Round-to-nearest FP32 subtract, guarded the same way. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float subRN(
    float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    float x = a, y = b;
    AETHER_BAND_FP_BARRIER(x); // nor into this difference
    AETHER_BAND_FP_BARRIER(y);
    return x - y;
#endif
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
#ifdef __CUDA_ARCH__
    return __float2int_rn(x);
#else
    return static_cast<std::int32_t>(__builtin_lrintf(x));
#endif
}

/** @brief Round-to-nearest-even `float` -> `int64`. @see roundToInt32. */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::int64_t roundToInt64(
    float x)
{
#ifdef __CUDA_ARCH__
    return __float2ll_rn(x);
#else
    return static_cast<std::int64_t>(__builtin_llrintf(x));
#endif
}

} // namespace detail
} // namespace banded
} // namespace aether
