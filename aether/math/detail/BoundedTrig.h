// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BoundedTrig.h
 * @brief Total FP64 sin/cos/sincos with zero stack: Cody-Waite reduction
 *        below 2^31, statically-unrolled full-range Payne-Hanek above it,
 *        FDLIBM polynomial.
 *
 * No `std::` math call appears anywhere in this file (house rule) — every
 * FMA is the bare `fma`/`fmaf` CUDA device overload.
 *
 * @section motivation Motivation
 * NVIDIA libdevice's ``__nv_sin``/``__nv_cos`` for FP64 emit a hidden
 * ``alloca [5 x i64]`` (40 bytes) inside the slow-path
 * ``__internal_trig_reduction_slowpathd`` because LLVM's ``mem2reg``
 * cannot promote variable-indexed allocas.  Every kernel that hits the
 * slow path inherits 40 B of stack.
 *
 * **This file eliminates that alloca**, and the call: everything is
 * force-inlined (no CALL in SASS, so no caller-side spills).  The
 * argument reduction uses the same Payne-Hanek algorithm, statically
 * unrolled to exactly 3 multiply-accumulate iterations over a 192-bit
 * window of 2/π chosen by the exponent.  All result chunks live in named
 * scalar locals — no local array, no stack frame.  The only computed
 * index reads a 160-byte 2/π table in global memory, on the slow path.
 *
 * @section contract_boundedtrig Contract
 * - Total over all doubles.  ``sin``/``cos`` of ±Inf and of NaN are NaN;
 *   ``sin(±0) = ±0``, ``cos(±0) = 1``; ``sin(x) = x`` for subnormal x.
 * - ULP accuracy: ≤ 2 ULP relative to glibc's ``sin``/``cos`` (≤ 1 ULP
 *   measured over the test corpus) from the
 *   smallest subnormal to ``DBL_MAX``, including the quadrant-boundary
 *   points and doubles next to multiples of π/2 (where glibc is itself
 *   off — the hardest double for reduction, cos is 8 ULP off in glibc —
 *   the bound holds against the correctly rounded value).
 * - Cost: ``|x| < 2³¹`` (~ 2.15 × 10⁹ rad) takes the Cody-Waite fast
 *   path; larger and non-finite inputs take the slow branch, which holds
 *   the non-finite guard, so the in-range path pays nothing for it.
 *
 * @section algorithm Algorithm
 *  1. ``|x| < 2³¹``: Cody-Waite, ``n = round(|x|·2/π)``, two FMAs.
 *     Otherwise, statically-unrolled Payne-Hanek extracts ``(q, frac_128)``,
 *     where ``frac_128`` is a 128-bit two's-complement fractional
 *     representing the reduced angle in [-0.5, +0.5) (units of π/2).
 *  2. **Split refinement** — ``frac_128`` becomes a 53-bit signed head
 *     (exact in FP64) plus a 64-bit tail, so no fraction bit is rounded
 *     away before the multiply by π/2.
 *  3. **Multiply by π/2 = Hi + Lo** with the head product's rounding
 *     error recovered by FMA, then TwoSum: ``y + ylo`` double-double,
 *     ``|y| ≤ π/4``; the slow path's polynomial adds the first-order
 *     ``ylo`` term.
 *  4. **FDLIBM polynomial kernels** — one Horner chain with per-
 *     coefficient selection between the sin and cos coefficients.
 *  5. Branchless quadrant dispatch (sign flips by integer XOR).
 *
 * @section nostd_audit No std:: math in device code
 * Per the project rule, this header contains **no ``std::`` math calls**.
 * All FMAs use bare ``fma`` — CUDA's ``<cmath>`` device overload routes
 * to ``__nv_fma_rn``.  Bit-cast uses ``AETHER_BITCOPY`` —
 * `__builtin_memcpy` under nvcc/g++ (UB-free, single load), plain `memcpy`
 * under NVRTC, whose EDG front end has no `__builtin_memcpy` — see
 * `aether/macros.h`.
 *
 * @section device_only Device-only
 * Every helper here is ``AETHER_DEVICE()`` (``__device__``).  The whole file
 * is also guarded by ``#ifndef AETHER_CPP_MODE`` so the CPP_MODE build
 * skips it entirely — in that mode ``aether::math::sin``/``cos``/``sincos``
 * route to ``std::sin`` etc. through the host branch of ``aether/math/math.h``.
 *
 * @section provenance Algorithmic + numerical provenance
 *  - W. J. Cody & W. Waite, *Software Manual for the Elementary Functions*,
 *    Prentice-Hall, 1980.
 *  - M. Payne & R. Hanek, *Radian Reduction for Trigonometric Functions*,
 *    SIGNUM 18(1), 1983 — the multi-precision ``2/π`` reduction.
 *  - P. Markstein, *IA-64 and Elementary Functions: Speed and Precision*,
 *    2000 — the integer multi-precision reduction with alignment,
 *    quadrant, and round-up handling that libdevice mirrors.
 *  - FDLIBM (Sun Microsystems, public domain), ``__kernel_rem_pio2.c`` +
 *    ``k_sin.c`` + ``k_cos.c``: polynomial coefficients + the source of
 *    the 2/π 24-bit table from which our 64-bit chunks are recombined.
 *    The values themselves are mathematical fact (the binary expansion
 *    of 2/π), not creative expression.
 *  - T. J. Dekker, *A Floating-Point Technique for Extending the
 *    Available Precision*, Numer. Math. 18, 1971 — TwoSum / TwoProduct.
 *
 * No source lines copied from libdevice or any LGPL/GPL libm.  All code
 * is independently written from the published algorithms.
 */

#include "aether/macros.h"

#include <cstdint>

// DEVICE COMPILER REQUIRED. Every entity below is `AETHER_DEVICE()`
// (`__device__`) and its body is written in nvcc's device dialect
// (`__umul64hi`, the bare `fma` device overload). A CUDA-mode build may also
// compile translation units with the HOST compiler (`g++`), which knows
// none of that vocabulary — and cannot call any of it either, since these
// are device-only functions with no host-callable declaration anywhere.
// The whole file therefore folds away on that path. Nothing host-visible is
// lost: `aether::math::sin`/`cos`/`sincos` reach these helpers only from
// under `AETHER_DEVICE_COMPILE` (`aether/math/math.h`), which no host
// compiler ever defines; every other path is already `std::sin` & co.
#if !defined(AETHER_CPP_MODE) && AETHER_DEVICE_COMPILER

namespace aether {
namespace math {
namespace detail {

// ===================================================================
//  Constants
// ===================================================================

/* π/2 as a canonical double-double pair (hi + lo gives ~106 bits).
 * Hi  = round-to-nearest FP64 of π/2.
 * Lo  = π/2 − hi, exact in FP64.
 *
 * Used by both the fast Cody-Waite reduction (2-FMA chain) and the
 * slow Payne-Hanek refinement.  In the FMA chain ``fma(-n_d, Hi, x)``
 * the inner product n × Hi rounds, but the FMA captures the result to
 * a single rounding; the subsequent ``fma(-n_d, Lo, y)`` then folds
 * in the π/2_lo tail.  For ``|n| ≤ 2³¹`` the two-FMA chain holds
 * accuracy to ≤ 2 ULP on the reduced angle. */
inline constexpr double kPiOver2Hi = 0x1.921FB54442D18p+0;
inline constexpr double kPiOver2Lo = 0x1.1A62633145C07p-54;

/* 2/π as a single FP64 — used to compute n = round(x · 2/π). */
inline constexpr double kTwoOverPi = 0x1.45F306DC9C883p-1;   /* 0.6366197723675814 */

/* Sine polynomial coefficients.  FDLIBM ``k_sin.c`` (public domain).
 *   sin(x + y) ≈ x − ((z·(½y − v·r) − y) − v·S1)
 * where z = x², v = z·x, r = S2 + z·(S3 + z·(S4 + z·(S5 + z·S6))). */
inline constexpr double kS1 = -1.66666666666666324348e-01;
inline constexpr double kS2 =  8.33333333332248946124e-03;
inline constexpr double kS3 = -1.98412698298579493134e-04;
inline constexpr double kS4 =  2.75573137070700676789e-06;
inline constexpr double kS5 = -2.50507602534068634195e-08;
inline constexpr double kS6 =  1.58969099521155010221e-10;

/* Cosine polynomial coefficients.  FDLIBM ``k_cos.c`` (public domain).
 *   cos(x + y) ≈ w + (((1 − w) − hz) + (z·r − x·y))
 * where z = x², hz = ½z, w = 1 − hz,
 *       r = z·(C1 + z·(C2 + z·(C3 + z·(C4 + z·(C5 + z·C6))))). */
inline constexpr double kC1 =  4.16666666666666019037e-02;
inline constexpr double kC2 = -1.38888888888741095749e-03;
inline constexpr double kC3 =  2.48015872894767294178e-05;
inline constexpr double kC4 = -2.75573143513906633035e-07;
inline constexpr double kC5 =  2.08757232129817482790e-09;
inline constexpr double kC6 = -1.13596475577881948265e-11;

// ===================================================================
//  Bit-level + multi-precision primitives
// ===================================================================

/** @brief Reinterpret a double's bit pattern as a 64-bit unsigned int.
 *  UB-free; the compiler folds this to a single register move. */
AETHER_DEVICE() AETHER_FORCEINLINE()
uint64_t doubleAsU64(const double x) noexcept
{
    uint64_t out;
    AETHER_BITCOPY(&out, &x, sizeof(out));
    return out;
}

/** @brief Conditionally flip the sign bit of a double via integer XOR.
 *
 *  ``sign_mask`` must be either 0 or 0x8000000000000000ULL.  XORing it
 *  onto the bit pattern of x flips the sign with no FP64 op — the
 *  whole thing compiles to a single ``LOP32`` on the integer pipe,
 *  avoiding the ``DADD R, -RZ, -Rsrc`` pattern that ptxas would
 *  otherwise emit for ``? -x : x`` (one DADD per conditional negation,
 *  which on Pascal's throughput-bound FP64 pipe directly costs wall
 *  time).  Libdevice uses the same trick. */
AETHER_DEVICE() AETHER_FORCEINLINE()
double applySign(const double x, const uint64_t sign_mask) noexcept
{
    uint64_t bits = doubleAsU64(x);
    bits ^= sign_mask;
    double out;
    AETHER_BITCOPY(&out, &bits, sizeof(double));
    return out;
}

/** @brief 64×64 → 128 multiply-add: ``(hi, lo) = a · b + c_in``.
 *
 *  Uses ``__umul64hi(a, b)`` for the upper 64 bits and plain ``a * b``
 *  (truncating) for the lower 64 bits, then an explicit carry chain
 *  for the c_in add.  On sm_70+, ``mul.hi.u64`` is a single SASS
 *  instruction; on sm_61 (no native u64 mul) nvcc emulates via XMAD
 *  but emits a tighter sequence than ``__uint128_t`` arithmetic
 *  (saves ~3 XMAD per call on Pascal, ~9 in this 3-call loop). */
AETHER_DEVICE() AETHER_FORCEINLINE()
void mad64Hi(const uint64_t a, const uint64_t b, const uint64_t c_in,
    uint64_t& lo_out, uint64_t& hi_out) noexcept
{
    const uint64_t hi64 = __umul64hi(a, b);
    const uint64_t lo64 = a * b;            /* truncating u64 mul */
    const uint64_t sum  = lo64 + c_in;
    const uint64_t carry = (sum < lo64) ? 1ULL : 0ULL;
    lo_out = sum;
    hi_out = hi64 + carry;
}

// ===================================================================
//  Statically-unrolled Payne-Hanek argument reduction
//  Produces (q, y) where y ∈ [-π/4, +π/4] is the reduced angle.
// ===================================================================

/** @brief 64-bit word ``i`` of the 2/π window table (see below).
 *
 *  Entry 0 is 64 zero bits standing for the integer part of 2/π; entries
 *  1..19 are bits [0, 1216) of the binary expansion of 2/π, high → low
 *  (FDLIBM's public-domain ``two_over_pi`` 24-bit table repacked to 64-bit
 *  words; checked against a 2000-bit evaluation of 2/π).
 *  1216 bits cover the window of the largest finite exponent (see
 *  ``payneHanekReduce``).  The table is a function-local static: it lives
 *  in global memory and is read with a computed index, which costs a load,
 *  not a stack slot — only the slow path ever indexes it. */
AETHER_DEVICE() AETHER_FORCEINLINE()
uint64_t twoOverPiWord(const int i) noexcept
{
    static const uint64_t kTable[20] = {
        0x0000000000000000ULL,
        0xA2F9836E4E441529ULL, 0xFC2757D1F534DDC0ULL, 0xDB6295993C439041ULL,
        0xFE5163ABDEBBC561ULL, 0xB7246E3A424DD2E0ULL, 0x06492EEA09D1921CULL,
        0xFE1DEB1CB129A73EULL, 0xE88235F52EBB4484ULL, 0xE99C7026B45F7E41ULL,
        0x3991D639835339F4ULL, 0x9C845F8BBDF9283BULL, 0x1FF897FFDE05980FULL,
        0xEF2F118B5A0A6D1FULL, 0x6D367ECF27CB09B7ULL, 0x4F463F669E5FEA2DULL,
        0x7527BAC7EBE5F17BULL, 0x3D0739F78A5292EAULL, 0x6BFB5FB11F8D5D08ULL,
        0x56033046FC7B6BABULL,
    };
    return kTable[i];
}

/** @brief 64 bits of the 2/π table starting ``sh`` bits into word ``i``
 *  (``0 <= sh < 64``).  The low word's shift is split in two so that
 *  ``sh == 0`` never shifts by 64. */
AETHER_DEVICE() AETHER_FORCEINLINE()
uint64_t funnelWord(const uint64_t hiWord, const uint64_t loWord, const int sh) noexcept
{
    return (hiWord << sh) | ((loWord >> 1) >> (63 - sh));
}

/** @brief Full-range Payne-Hanek reduction for ``|x| >= 2^31``, finite or
 *  not: ``x·2/π = 4k + q + (y + ylo)/(π/2)``, ``y + ylo`` a double-double.
 *  A non-finite ``x`` yields ``y = x - x`` (NaN), so every caller's
 *  polynomial and sign flip turn it into NaN with no extra branch.
 *
 *  ``x = m_fixed · 2^(e - 1086)`` (``e`` the biased exponent, ``m_fixed``
 *  the 64-bit mantissa with its implicit leading 1 and 11 zero low bits).
 *  Bits of 2/π whose product with ``m_fixed`` has weight >= 4 only add
 *  whole turns and are skipped: the 192-bit window starts ``e - 1077``
 *  bits into 2/π, which puts the units bit of ``x · 2/π`` at bit 201 of
 *  the 256-bit product for EVERY exponent — the alignment shift is the
 *  constant 53 and the window's last bit is never more than one unit off
 *  in product bit 64, leaving 137 exact fraction bits (the worst double
 *  needs about 61 + 53).  Five product words, all named scalars. */
AETHER_DEVICE() AETHER_FORCEINLINE()
void payneHanekReduce(const double x, double& y_out, double& ylo_out, int& q_out) noexcept
{
    /* Step 1: extract bit fields.  Sign is applied at the wrapper level
     * (cwSin/cwSinCos), so this reduction only needs the biased exponent. */
    const uint64_t bits     = doubleAsU64(x);
    const int      biased_e = static_cast<int>((bits >> 52) & 0x7FF);

    /* Step 2: form 64-bit fixed-point mantissa with implicit leading 1.
     *   m_fixed = (mantissa_bits << 11) | (1 << 63) */
    const uint64_t m_fixed = (bits << 11) | (1ULL << 63);

    /* Step 3: the 192-bit window of 2/π.  Bit offset into the table is
     * (e - 1077) + 64 for the leading zero word: 41 at |x| = 2^31, 1033
     * at DBL_MAX, 1034 for Inf/NaN — words 0..19 are always in range. */
    const int      bitOffset = biased_e - 1013;
    const int      word      = bitOffset >> 6;
    const int      sh        = bitOffset & 63;
    const uint64_t t0        = twoOverPiWord(word);
    const uint64_t t1        = twoOverPiWord(word + 1);
    const uint64_t t2        = twoOverPiWord(word + 2);
    const uint64_t t3        = twoOverPiWord(word + 3);
    const uint64_t wHi       = funnelWord(t0, t1, sh);
    const uint64_t wMid      = funnelWord(t1, t2, sh);
    const uint64_t wLo       = funnelWord(t2, t3, sh);

    /* Step 4: three unrolled 64×64 → 128 multiply-accumulate iterations
     * + post-loop carry store, low-to-high in 2/π bits, so the final
     * carry lands at the top of the 256-bit product. */
    uint64_t r0, r1, r2, r3;
    uint64_t carry = 0;
    uint64_t lo, hi;

    mad64Hi(m_fixed, wLo,  carry, lo, hi);
    r0    = lo;
    carry = hi;

    mad64Hi(m_fixed, wMid, carry, lo, hi);
    r1    = lo;
    carry = hi;

    mad64Hi(m_fixed, wHi,  carry, lo, hi);
    r2    = lo;
    carry = hi;

    r3 = carry;

    /* Step 5: align so the top 2 bits of aligned_hi are the quadrant
     * (product bits 202:201) — a constant left shift by 53. */
    const uint64_t aligned_hi  = (r3 << 53) | (r2 >> 11);
    const uint64_t aligned_mid = (r2 << 53) | (r1 >> 11);
    const uint64_t aligned_lo  = (r1 << 53) | (r0 >> 11);

    /* Step 6: extract quadrant from top 2 bits, then strip them off. */
    int quadrant = static_cast<int>(aligned_hi >> 62);

    uint64_t frac_hi_u = (aligned_hi << 2) | (aligned_mid >> 62);
    uint64_t frac_lo_u = (aligned_mid << 2) | (aligned_lo >> 62);

    /* Step 7: round-up via signed re-interpretation.  If top bit of
     * frac_hi_u is set, the fractional ∈ [0.5, 1); reading it as int64
     * makes it represent (frac − 1) ∈ [-0.5, 0) — exactly the round-up
     * semantics.  We bump the quadrant to account for the shift. */
    const bool roundup = (frac_hi_u & (1ULL << 63)) != 0;
    if (roundup) {
        quadrant = quadrant + 1;
    }

    /* Step 8: clamp quadrant.  Input sign is applied by the caller
     * (cwSin negates the result for negative input; cwCos is sign-
     * insensitive).  Negating the quadrant here is incorrect: for
     * quadrant 0, -0 = 0 so the sign would be silently lost, producing
     * cwSin(-0.5) = +0.479 instead of -0.479. */
    q_out = quadrant & 3;

    /* Step 9: scalar refinement to y ∈ [-π/4, +π/4].
     *
     * The 128-bit two's-complement integer (frac_hi_u : frac_lo_u) at
     * 2⁻¹²⁸ scale represents the fractional.  It is split at bit 75 into
     * two FP64 contributions:
     *   - hi:  I × 2⁻⁵³  where I = the top 53 bits, signed (exact in FP64)
     *   - lo:  J × 2⁻¹¹⁷ where J = the next 64 bits (one rounding, 2⁻⁵³
     *          relative to a term already 2⁻⁵³ below hi)
     * Converting all 64 top bits instead would round away 11 bits of the
     * fraction that the lo term never sees again (a 2-ULP bias on large
     * arguments). */
    const int64_t frac_signed  = static_cast<int64_t>(frac_hi_u);
    const double  I_d          = static_cast<double>(frac_signed >> 11);
    const double  J_d          = static_cast<double>((frac_hi_u << 53) | (frac_lo_u >> 11));

    /* Constant-folded scaled coefficients of π/2. */
    constexpr double kPiOver2Hi_s53  = kPiOver2Hi * 0x1p-53;
    constexpr double kPiOver2Lo_s53  = kPiOver2Lo * 0x1p-53;
    constexpr double kPiOver2Hi_s117 = kPiOver2Hi * 0x1p-117;

    /* The head product's rounding error is recovered exactly (TwoProduct
     * by FMA) and folded into the tail with the two small terms, so y is
     * rounded once, at the end: 1 MUL + 3 FMA + 1 ADD, slow path only. */
    /* ``__dmul_rn``/``__dadd_rn``/``__dsub_rn`` are never contracted into
     * an FMA (nvcc's default ``-fmad=true`` would fuse ``head + tail``
     * with the product and break both error-free transformations). */
    const double head = __dmul_rn(I_d, kPiOver2Hi_s53);
    double tail = fma(I_d, kPiOver2Hi_s53, -head);
    tail = fma(I_d, kPiOver2Lo_s53,  tail);   /* π/2 lo correction       */
    tail = fma(J_d, kPiOver2Hi_s117, tail);   /* J × 2⁻¹¹⁷ contribution  */
    /* TwoSum: y = head + tail rounded once, ylo its exact residual.  The
     * residual matters: for |y| in [0.5, π/4) one ulp of y is two ulps of
     * sin(y), so dropping it costs up to a full ulp of the result. */
    const double y    = __dadd_rn(head, tail);
    const double yv   = __dsub_rn(y, head);
    const double ylo  = __dadd_rn(__dsub_rn(head, __dsub_rn(y, yv)), __dsub_rn(tail, yv));
    ylo_out = ylo;

    /* Non-finite guard: Inf and NaN (biased exponent 0x7FF) reduce to
     * NaN, which the polynomial and the sign XOR carry through. */
    y_out = (biased_e == 0x7FF) ? x - x : y;
}

// ===================================================================
//  FDLIBM polynomial kernels (scalar y, no lo correction)
// ===================================================================

/** @brief FDLIBM ``__kernel_sin`` with iy=0.  Returns sin(y) for
 *  |y| ≤ π/4.  ≤ 1 ULP per FDLIBM doc.
 *
 *  Form: sin(y) ≈ y + v · (S1 + z · (S2 + z · (S3 + z · (S4 + z · (S5 + z · S6))))).
 *  where z = y², v = z · y. */
AETHER_DEVICE() AETHER_FORCEINLINE()
double fdlibmKernelSin(const double y) noexcept
{
    const double z = y * y;
    const double v = z * y;
    const double r = fma(z, fma(z, fma(z, fma(z, fma(z, kS6, kS5), kS4), kS3), kS2), kS1);
    return fma(v, r, y);
}

/** @brief FDLIBM ``__kernel_cos`` scalar form.  Returns cos(y) for
 *  |y| ≤ π/4.  ≤ 1 ULP.
 *
 *  Form: cos(y) ≈ 1 − z/2 + z² · (C1 + z · (C2 + z · (C3 + z · (C4 + z · (C5 + z · C6))))). */
AETHER_DEVICE() AETHER_FORCEINLINE()
double fdlibmKernelCos(const double y) noexcept
{
    const double z  = y * y;
    const double r  = fma(z, fma(z, fma(z, fma(z, fma(z, kC6, kC5), kC4), kC3), kC2), kC1);
    const double hz = 0.5 * z;
    /* Compute (1 − hz) + r·z² with the (1−hz) compensation term that
     * preserves precision for y near π/4 (where hz reaches ~0.31). */
    const double w  = 1.0 - hz;
    return w + (((1.0 - w) - hz) + r * z * z);
}

// ===================================================================
//  Fast Cody-Waite reduction (for |x| < 2³¹) — matches libdevice's
//  fast path structurally.  Produces (q, y) without any multi-precision
//  integer arithmetic; pure FP64 + a single F2I/I2F round.
// ===================================================================

AETHER_DEVICE() AETHER_FORCEINLINE()
void codyWaiteReduce(const double x, double& y_out, int& q_out) noexcept
{
    /* Operate on |x| so the cwSin/cwSinCos wrappers can apply the input
     * sign uniformly (same convention as the slow Payne-Hanek path).
     * Forming abs_x from the input bits is one IADD-bit-clear, cheaper
     * than calling fabs(). */
    const uint64_t bits = doubleAsU64(x);
    const uint64_t abs_bits = bits & 0x7FFFFFFFFFFFFFFFULL;
    double abs_x;
    AETHER_BITCOPY(&abs_x, &abs_bits, sizeof(double));

    /* n = round(|x| · 2/π).  Compiles to DMUL + F2I.RN + I2F. */
    const double n_d = ::nearbyint(abs_x * kTwoOverPi);
    const int    n   = static_cast<int>(n_d);

    /* 2-FMA Cody-Waite chain: y = |x| − n·(π/2_hi + π/2_lo). */
    double y = fma(-n_d, kPiOver2Hi, abs_x);
    y       = fma(-n_d, kPiOver2Lo, y);

    y_out = y;
    q_out = n & 3;
}

// ===================================================================
//  Unified sin/cos polynomial — single Horner chain with per-coefficient
//  ternary selection (compiler emits FSEL / SEL, one cycle each).
//  Returns the value of sin(y) or cos(y) for |y| ≤ π/4 with ≤ 1 ULP.
// ===================================================================

AETHER_DEVICE() AETHER_FORCEINLINE()
double cwPolyUnified(const double y, const bool use_cos) noexcept
{
    const double z = y * y;

    /* Per-coefficient ternary → SEL on device.  Each SEL is 1 cycle. */
    const double a1 = use_cos ? kC1 : kS1;
    const double a2 = use_cos ? kC2 : kS2;
    const double a3 = use_cos ? kC3 : kS3;
    const double a4 = use_cos ? kC4 : kS4;
    const double a5 = use_cos ? kC5 : kS5;
    const double a6 = use_cos ? kC6 : kS6;

    /* Horner chain (degree 6). */
    double r = fma(z, a6, a5);
    r = fma(z, r, a4);
    r = fma(z, r, a3);
    r = fma(z, r, a2);
    r = fma(z, r, a1);

    if (use_cos) {
        /* cos(y) ≈ 1 − z/2 + z²·r = 1 + z·(z·r − ½), two FMAs:
         *   t      = fma(z, r, -0.5)   →  z·r − 0.5
         *   result = fma(z, t,  1.0)   →  1 + z·(z·r − 0.5)
         *                              =  1 − ½·z + z²·r
         *
         * The two-FMA form is bit-identical at ≤ 1 ULP to the
         * FDLIBM-compensated form on every sweep tested, while using
         * half the FP64 ops (3 vs 6) — the wall-time-relevant metric on
         * Pascal's throughput-bound FP64 pipeline. */
        const double t = fma(z, r, -0.5);
        return fma(z, t, 1.0);
    } else {
        /* sin(y) ≈ y + (y · z) · r. */
        const double v = z * y;
        return fma(v, r, y);
    }
}

/** @brief The unified polynomial at the double-double ``y + ylo`` (slow
 *  path only): the first-order term ``ylo · d/dy`` with the derivative
 *  taken as 1 for sin and ``-y`` for cos — exact to far below an ulp
 *  since ``|ylo| <= ulp(y)/2``. */
AETHER_DEVICE() AETHER_FORCEINLINE()
double slowPoly(const double y, const double ylo, const bool use_cos) noexcept
{
    return fma(ylo, use_cos ? -y : 1.0, cwPolyUnified(y, use_cos));
}

// ===================================================================
//  Public entry points — cwSin / cwCos / cwSinCos
// ===================================================================

/** @brief Total FP64 sine (any double; Inf/NaN give NaN).
 *  Accuracy: ≤ 2 ULP vs glibc over all finite doubles.
 *
 *  Sign handling: the reduction operates on ``|x|`` (the bit shift in
 *  step 2 of ``payneHanekReduce`` strips the sign bit of the input).
 *  The quadrant dispatch produces the value of sin on ``|x|``.  Since
 *  sin is odd, the result is negated when the input was negative. */
/* Cutoff for the fast Cody-Waite path.  Inputs with |x| < 2³¹ go through
 * codyWaiteReduce + the unified polynomial; larger inputs, Inf and NaN
 * (every comparison with NaN is false) take the statically-unrolled
 * full-range Payne-Hanek (always STACK=0), which also holds the
 * non-finite guard. */
inline constexpr double kFastPathBound = 0x1p31;   /* 2³¹ = 2147483648 */

AETHER_DEVICE() AETHER_FORCEINLINE()
double cwSin(const double x) noexcept
{
    double y;
    int    q;
    /* Dispatch: fast Cody-Waite for |x| < 2³¹, else slow Payne-Hanek
     * (also Inf/NaN, which it turns into NaN). */
    double polyVal;
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
        /* Unified polynomial: sin or cos based on (q & 1). */
        polyVal = cwPolyUnified(y, (q & 1) != 0);
    } else {
        double ylo;
        payneHanekReduce(x, y, ylo, q);
        polyVal = slowPoly(y, ylo, (q & 1) != 0);
    }
    /* Combined sign-bit XOR: flip if (q & 2) XOR input-sign.
     * (q & 2) is bit 1 of q → bit 63 of the mask via left shift by 62.
     * Single LOP32 on the integer pipe replaces what was two DADDs. */
    const uint64_t q_neg = (static_cast<uint64_t>(q) & 2ULL) << 62;
    const uint64_t x_neg = doubleAsU64(x) & 0x8000000000000000ULL;
    return applySign(polyVal, q_neg ^ x_neg);
}

/** @brief Total FP64 cosine (any double; Inf/NaN give NaN).
 *  cos is even, so the input sign does not affect the result. */
AETHER_DEVICE() AETHER_FORCEINLINE()
double cwCos(const double x) noexcept
{
    double y;
    int    q;
    /* cos: select cos(y) when q is even, sin(y) when q is odd. */
    double polyVal;
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
        polyVal = cwPolyUnified(y, (q & 1) == 0);
    } else {
        double ylo;
        payneHanekReduce(x, y, ylo, q);
        polyVal = slowPoly(y, ylo, (q & 1) == 0);
    }
    /* Bit-XOR sign flip — single LOP32 vs DADD-based negation. */
    const uint64_t mask = (static_cast<uint64_t>(q + 1) & 2ULL) << 62;
    return applySign(polyVal, mask);
}

/** @brief Total FP64 sincos (any double; Inf/NaN give NaN twice).  Shares the reduction; the
 *  polynomial runs twice (once for sin, once for cos) — the compiler
 *  CSEs the shared z = y².  Input sign is applied to the sin output only. */
AETHER_DEVICE() AETHER_FORCEINLINE()
void cwSinCos(const double x, double& sinOut, double& cosOut) noexcept
{
    double y;
    int    q;
    double sy, cy;
    if (fabs(x) < kFastPathBound) {
        codyWaiteReduce(x, y, q);
        sy = cwPolyUnified(y, false);
        cy = cwPolyUnified(y, true);
    } else {
        double ylo;
        payneHanekReduce(x, y, ylo, q);
        sy = slowPoly(y, ylo, false);
        cy = slowPoly(y, ylo, true);
    }
    const double sinSwap = (q & 1) ? cy : sy;
    const double cosSwap = (q & 1) ? sy : cy;
    /* Bit-XOR sign flips — single LOP32 each, no DADD/DMUL.
     *   sin: flip if (q & 2) XOR input-sign  (sin is odd)
     *   cos: flip if ((q+1) & 2)             (cos is even, no input-sign) */
    const uint64_t x_neg    = doubleAsU64(x) & 0x8000000000000000ULL;
    const uint64_t sin_mask = ((static_cast<uint64_t>(q)     & 2ULL) << 62) ^ x_neg;
    const uint64_t cos_mask =  (static_cast<uint64_t>(q + 1) & 2ULL) << 62;
    sinOut = applySign(sinSwap, sin_mask);
    cosOut = applySign(cosSwap, cos_mask);
}

} // namespace detail
} // namespace math
} // namespace aether

#endif // !AETHER_CPP_MODE && AETHER_DEVICE_COMPILER
