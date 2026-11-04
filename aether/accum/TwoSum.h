// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file TwoSum.h
 * @brief The exact residual of a rounded sum, defined once for every
 *        compensated-sum consumer (`atomic.h`, `run.h`, hawk's emitted text).
 *
 * `twoSumErr(a, b, s)` with `s == fl(a + b)` returns `(a + b) - s` exactly,
 * for any order and signs of `a` and `b`:
 *   - device: Fast2Sum with the larger operand picked by an integer compare of
 *     the magnitude bits (integer pipe, not the FP64 pipe): 3 floating-point
 *     ops where Knuth's two-sum takes 6;
 *   - host: Knuth's two-sum, unless `AETHER_TWOSUM_FAST_HOST` selects the
 *     ordered form (the integer compare is a blend on a CPU, so it is kept
 *     only where measured faster).
 * Rules both forms pin (the bits agree on every input):
 *   - a non-finite `s` gives NaN;
 *   - a zero residual is `+0`, never `-0`.
 *
 * Includes only what an NVRTC-compiled kernel already has.
 */

#include <cstdint>

#include "aether/macros.h"

#ifndef __CUDA_ARCH__
#include <bit>
#endif

namespace aether {
namespace accum {

/** @brief The bits of `x` as a signed integer of its width. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto floatBits(E x)
{
#if defined(__CUDA_ARCH__)
    if constexpr (sizeof(E) == 8)
        return __double_as_longlong(x);
    else
        return __float_as_int(x);
#else
    if constexpr (sizeof(E) == 8)
        return std::bit_cast<std::int64_t>(x);
    else
        return std::bit_cast<std::int32_t>(x);
#endif
}

/** @brief True when `x` is neither infinite nor NaN: read from the exponent
 *         field, no floating-point op. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool isFiniteBits(E x)
{
    if constexpr (sizeof(E) == 8)
        return ((floatBits(x) >> 52) & 0x7ff) != 0x7ff;
    else
        return ((floatBits(x) >> 23) & 0xff) != 0xff;
}

/** @brief The exact residual `(a + b) - s` of `s == fl(a + b)` (see the file
 *         comment for the form and the two pinned rules). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErr(E a, E b, E s)
{
    // The residual is exact only for the ROUNDED a, b and s: a product feeding
    // any of them must not be contracted into the subtractions below.
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
#if defined(__CUDA_ARCH__) || defined(AETHER_TWOSUM_FAST_HOST)
    using I          = decltype(floatBits(a));
    constexpr I mask = sizeof(E) == 8 ? I(0x7fffffffffffffffLL) : I(0x7fffffff);
    const bool aBig  = (floatBits(a) & mask) >= (floatBits(b) & mask);
    const E big      = aBig ? a : b;
    const E small    = aBig ? b : a;
    E z              = s - big;
    AETHER_FP_BARRIER(z);
    const E e = small - z;
    return isFiniteBits(s) ? e + E(0) : s - s;
#else
    const E bb = s - a;
    return ((a - (s - bb)) + (b - bb)) + E(0);
#endif
}

} // namespace accum
} // namespace aether
