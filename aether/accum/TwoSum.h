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
#include <type_traits>

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

/** @brief `x` with a zero of either sign read as `+0`, by integer ops (no
 *         floating-point op: the FP64 pipe is the scarce one). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E positiveZero(E x)
{
    using U = std::conditional_t<sizeof(E) == 8, std::uint64_t, std::uint32_t>;
    U b     = static_cast<U>(floatBits(x));
    b       = (b << 1) == U(0) ? U(0) : b;
#if defined(__CUDA_ARCH__)
    if constexpr (sizeof(E) == 8)
        return __longlong_as_double(static_cast<long long>(b));
    else
        return __int_as_float(static_cast<int>(b));
#else
    return std::bit_cast<E>(b);
#endif
}

/** @brief `twoSumErr` for a caller that gates on `isFiniteBits(s)` itself: the
 *         residual is garbage when `s` is not finite (the `+0` rule holds). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErrFinite(E a, E b, E s)
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
    return positiveZero(small - z);
#else
    const E bb = s - a;
    return positiveZero((a - (s - bb)) + (b - bb));
#endif
}

/** @brief The exact residual `(a + b) - s` of `s == fl(a + b)` (see the file
 *         comment for the form and the two pinned rules). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErr(E a, E b, E s)
{
    return isFiniteBits(s) ? twoSumErrFinite(a, b, s) : s - s;
}

/** @brief `twoSumErr` for a commit that waits on an atomic's returned value
 *         (a latency path, not a throughput path): Knuth's branch-free two-sum
 *         with the non-finite and `+0` rules as selects. The same bits as
 *         `twoSumErr` on every input; no integer magnitude compare. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErrAtomic(E a, E b, E s)
{
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const E bb = s - a;
    const E e  = (a - (s - bb)) + (b - bb);
    const E z = positiveZero(e);
    const E n = s - s;
    return isFiniteBits(s) ? z : n;
}

/** @brief `twoSumErr` for a caller that SUMS the residual into a compensation
 *         that starts at `+0`: the `+0` rule and the non-finite test are
 *         redundant there, so the host spelling is Knuth's two-sum alone (no
 *         branch, no integer round trip in a dependent chain). A `+0`-started
 *         IEEE sum never becomes `-0`, and Knuth's form is already NaN when
 *         `s` is not finite, so the accumulated value is bit-identical to
 *         `twoSumErr`'s. The device keeps `twoSumErr` (Fast2Sum, measured).
 *         NOT for a caller that returns or stores the residual on its own. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErrAccum(E a, E b, E s)
{
#if defined(__CUDA_ARCH__)
    return twoSumErr(a, b, s);
#else
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const E bb = s - a;
    return (a - (s - bb)) + (b - bb);
#endif
}

} // namespace accum
} // namespace aether
