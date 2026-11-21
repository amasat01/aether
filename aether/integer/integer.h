// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file integer.h
 * @brief `aether::integer` — integer arithmetic with defined results for every operand.
 *
 * Addition, subtraction, multiplication and negation wrap around (computed in the unsigned type of
 * the same width, so no operation is undefined), division and remainder round toward negative
 * infinity and give 0 for a zero divisor, a shift by the width or more gives 0 (or -1 for a
 * negative value shifted right), and a real converts with saturation and NaN to 0. The same code
 * runs on host and device. Defined for `std::int32_t`, `std::uint32_t` and `long long`.
 */

#include <cstdint>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace integer {

/** @brief The unsigned twin, the limits and the width of an integer type `T`. */
template<class T>
struct Traits;
template<>
struct Traits<std::int32_t> {
    using U                          = std::uint32_t;
    static constexpr bool is_signed  = true;
    static constexpr std::int32_t lo = -2147483647 - 1, hi = 2147483647;
    static constexpr unsigned bits = 32u;
};
template<>
struct Traits<std::uint32_t> {
    using U                           = std::uint32_t;
    static constexpr bool is_signed   = false;
    static constexpr std::uint32_t lo = 0u, hi = 4294967295u;
    static constexpr unsigned bits = 32u;
};
template<>
struct Traits<long long> {
    using U                         = unsigned long long;
    static constexpr bool is_signed = true;
    static constexpr long long lo = -9223372036854775807LL - 1, hi = 9223372036854775807LL;
    static constexpr unsigned bits = 64u;
};

/** @brief `a + b`, wrapping on overflow. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T add(T a, T b)
{
    using U = typename Traits<T>::U;
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}
/** @brief `a - b`, wrapping on overflow. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T sub(T a, T b)
{
    using U = typename Traits<T>::U;
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}
/** @brief `a * b`, wrapping on overflow. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T mul(T a, T b)
{
    using U = typename Traits<T>::U;
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}
/** @brief `-a`, wrapping (the most negative value stays itself). */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T neg(T a)
{
    using U = typename Traits<T>::U;
    return static_cast<T>(U(0) - static_cast<U>(a));
}
/** @brief `|a|`, wrapping (the most negative value stays itself). */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T abs(T a)
{
    if constexpr (Traits<T>::is_signed)
        return a < T(0) ? neg(a) : a;
    else
        return a;
}
/** @brief The smaller of `a` and `b`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T min(T a, T b)
{
    return b < a ? b : a;
}
/** @brief The larger of `a` and `b`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T max(T a, T b)
{
    return a < b ? b : a;
}
/** @brief `i` when `0 <= i < n`, else the nearest valid index `0` or `n - 1`. One unsigned compare
 * is the whole in-range test (a negative index wraps past `n`), so an in-range read pays one
 * predictable branch. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T clampIndex(T i, T n)
{
    using U = typename Traits<T>::U;
    return U(i) < U(n) ? i : (i < T(0) ? T(0) : n - T(1));
}
/** @brief `a / b` rounded toward negative infinity; 0 for a zero divisor. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T floorDiv(T a, T b)
{
    if (b == T(0))
        return T(0);
    if constexpr (Traits<T>::is_signed) {
        if (b == T(-1))
            return neg(a);
        const T q = static_cast<T>(a / b);
        return (a % b != T(0) && ((a < T(0)) != (b < T(0)))) ? static_cast<T>(q - T(1)) : q;
    } else {
        return static_cast<T>(a / b);
    }
}
/** @brief The remainder of `floorDiv`, with the sign of `b`; 0 for a zero divisor. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T mod(T a, T b)
{
    if (b == T(0))
        return T(0);
    if constexpr (Traits<T>::is_signed) {
        if (b == T(-1))
            return T(0);
        const T r = static_cast<T>(a % b);
        return (r != T(0) && ((r < T(0)) != (b < T(0)))) ? static_cast<T>(r + b) : r;
    } else {
        return static_cast<T>(a % b);
    }
}
/** @brief `a & b`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T bitAnd(T a, T b)
{
    return static_cast<T>(a & b);
}
/** @brief `a | b`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T bitOr(T a, T b)
{
    return static_cast<T>(a | b);
}
/** @brief `a ^ b`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T bitXor(T a, T b)
{
    return static_cast<T>(a ^ b);
}
/** @brief `~a`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T bitNot(T a)
{
    return static_cast<T>(~a);
}
/** @brief `a << b`; 0 when `b` is the width or more (or negative). */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T shiftLeft(T a, T b)
{
    using U = typename Traits<T>::U;
    if (static_cast<U>(b) < U(Traits<T>::bits))
        return static_cast<T>(static_cast<U>(a) << static_cast<U>(b));
    return T(0);
}
/** @brief `a >> b`; when `b` is the width or more (or negative) 0, or -1 for a negative `a`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T shiftRight(T a, T b)
{
    using U = typename Traits<T>::U;
    if (static_cast<U>(b) < U(Traits<T>::bits))
        return static_cast<T>(a >> b);
    if constexpr (Traits<T>::is_signed)
        return a < T(0) ? T(-1) : T(0);
    else
        return T(0);
}
/** @brief The integer of type `T` nearest toward zero to the real `x`, saturating at the limits of
 * `T`; NaN gives 0. */
template<class T, class R>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T fromReal(R x)
{
    if (math::isnan(x))
        return T(0);
    if (x <= static_cast<R>(Traits<T>::lo))
        return Traits<T>::lo;
    if (x >= static_cast<R>(Traits<T>::hi))
        return Traits<T>::hi;
    return static_cast<T>(x);
}

} // namespace integer
} // namespace aether
