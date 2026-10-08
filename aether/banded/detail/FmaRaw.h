// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file FmaRaw.h
 * @brief `BandRaw`-level unnormalized FMA primitives (`fmaRaw`/
 *        `fmaRawNoCancel`/`fmaRawS`/`fmaRawSNoCancel`), shared by
 *        `BandExpLog.h` (`exp`/`log`'s internal use) and `BandRound.h`
 *        (the public checked `Band fma(Band,Band,Band)`).
 *
 * Both consumers land in the same tree via `aether/banded/banded.h`'s
 * umbrella, so this family is defined exactly once, here, and included
 * from both rather than duplicated (which would be a duplicate-definition
 * compile error).
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

namespace aether {
namespace banded {
namespace detail {

/// @brief `a*b + c` on `BandRaw` operands, unnormalized. Mirrors `addRaw`'s
/// own `BandRawCancel` return: a general FMA knows the sign of neither the
/// product nor the addend, so when they oppose the leading limb cancels and
/// the result needs the SAME "cancellation-capable, normalize-only" typing
/// `addRaw` carries.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawCancel fmaRaw(
    BandRaw a, BandRaw b, BandRaw c)
{
    float p_hi, p_lo;
    twoProd(a.hi, b.hi, p_hi, p_lo);
    float q_hi, q_lo;
    twoProd(a.hi, b.lo, q_hi, q_lo);
    float t_hi, t_lo;
    twoProd(a.lo, b.hi, t_hi, t_lo);

    float sA, eA;
    twoSum(p_lo, q_hi, sA, eA);
    float sB, eB;
    twoSum(sA, t_hi, sB, eB);

    float main_sum, e_main;
    twoSum(p_hi, c.hi, main_sum, e_main);

    float mA, emA;
    twoSum(e_main, sB, mA, emA);
    float mB, emB;
    twoSum(mA, c.lo, mB, emB);

    float resid = addRN(eA, eB);
    resid       = resid + emA;
    resid       = resid + emB;
    resid       = resid + q_lo;
    resid       = resid + t_lo;
    resid       = fmaRN(a.lo, b.lo, resid);
    resid       = fmaRN(a.tail, b.hi, resid);
    resid       = fmaRN(b.tail, a.hi, resid);
    resid       = resid + c.tail;

    return BandRawCancel{ main_sum, mB, resid };
}

/// @brief `fmaRaw` for a caller who can PROVE, at compile time, that the
/// product and the addend cannot cancel.
/// @warning the assertion is the CALLER's — see each call site for its own
/// one-line argument.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw fmaRawNoCancel(
    BandRaw a, BandRaw b, BandRaw c)
{
    const BandRawCancel r = fmaRaw(a, b, c);
    return BandRaw{ r.hi, r.lo, r.tail };
}

/// @brief `a*b + c` where the addend `c` is a scalar (one float, not a
/// carrier) — the specialization the compiler cannot perform on its own,
/// because every FP32 primitive here is explicit inline-asm/barrier-guarded
/// precisely so a structurally-zero limb survives to the codec.
/// @warning the caller owes the structural claim that `c` is exact in one
/// float (`1`, `1/2`, any power of two — `1/6` is not and must go through
/// the general `fmaRaw` as a full three-limb `Band`, exactly as `exp`'s
/// `c3` does).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawCancel fmaRawS(
    BandRaw a, BandRaw b, float c)
{
    float p_hi, p_lo;
    twoProd(a.hi, b.hi, p_hi, p_lo);
    float q_hi, q_lo;
    twoProd(a.hi, b.lo, q_hi, q_lo);
    float t_hi, t_lo;
    twoProd(a.lo, b.hi, t_hi, t_lo);

    float sA, eA;
    twoSum(p_lo, q_hi, sA, eA);
    float sB, eB;
    twoSum(sA, t_hi, sB, eB);

    float main_sum, e_main;
    twoSum(p_hi, c, main_sum, e_main);

    float mA, emA;
    twoSum(e_main, sB, mA, emA);
    // `twoSum(mA, c.lo)` with `c.lo == 0` returns `(mA, 0)` exactly — deleted,
    // together with the `resid + emB` / `resid + c.tail` terms it fed. 8 FP32.

    float resid = addRN(eA, eB);
    resid       = resid + emA;
    resid       = resid + q_lo;
    resid       = resid + t_lo;
    resid       = fmaRN(a.lo, b.lo, resid);
    resid       = fmaRN(a.tail, b.hi, resid);
    resid       = fmaRN(b.tail, a.hi, resid);

    return BandRawCancel{ main_sum, mA, resid };
}

/// @brief `fmaRawS` for a caller who can prove, at compile time, that the
/// product and the scalar addend cannot cancel.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw fmaRawSNoCancel(
    BandRaw a, BandRaw b, float c)
{
    const BandRawCancel r = fmaRawS(a, b, c);
    return BandRaw{ r.hi, r.lo, r.tail };
}

} // namespace detail
} // namespace banded
} // namespace aether
