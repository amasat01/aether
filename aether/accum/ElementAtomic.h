// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ElementAtomic.h
 * @brief `aether::accum::atomic`: one named atomic update per operation on a single array element, on the host
 *        (`std::atomic_ref`) and on the device (the atomic intrinsics, a compare-exchange loop where none exists), behind
 *        one call. Includes nothing a runtime-compiled unit lacks.
 *
 * Every helper takes the element by reference and returns the value the element held before the update. The folds
 * (`minAt`, `maxAt`, `andAt`, `xorAt`, and the sum of `fetchAddAt`) leave the element at the same value in any order the
 * threads arrive in; the old values, and which thread wins an `exchangeAt` or a `compareExchangeAt`, depend on that order.
 */

#include <type_traits>

#include "aether/macros.h"

#ifndef __CUDA_ARCH__
#include <atomic>
#endif

namespace aether {
namespace accum {
namespace atomic {

namespace detail {
/** @brief True when `v` should replace `cur` under the keyed minimum: smaller, or a NaN (a NaN absorbs, as in
 *         `np.minimum`, so the fold does not depend on the order values arrive in). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool takesMin_(E v, E cur)
{
    if constexpr (std::is_floating_point_v<E>)
        return v < cur || v != v;
    else
        return v < cur;
}

/** @brief The keyed maximum's twin of `takesMin_`. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool takesMax_(E v, E cur)
{
    if constexpr (std::is_floating_point_v<E>)
        return v > cur || v != v;
    else
        return v > cur;
}

/** @brief The unsigned integer type that has the size of `E` (4 or 8 bytes), the word an atomic compare-exchange acts on. */
template<class E>
using WordOf_ = std::conditional_t<sizeof(E) == 8, unsigned long long, unsigned int>;

/** @brief The bits of `value` as the word type of its size. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() WordOf_<E> wordOf_(E value)
{
    static_assert(sizeof(E) == 4 || sizeof(E) == 8, "an atomic element update acts on a 4 or 8 byte element");
    WordOf_<E> word;
    AETHER_BITCOPY(&word, &value, sizeof(E));
    return word;
}

/** @brief The element the bits `word` hold. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E fromWord_(WordOf_<E> word)
{
    E value;
    AETHER_BITCOPY(&value, &word, sizeof(E));
    return value;
}

#if defined(__CUDA_ARCH__)
/** @brief The device's compare-exchange loop: replaces the element by `next(current)` until no other thread changed it in
 *         between, and returns the value it replaced. Used where no native atomic exists (floating min and max). */
template<class E, class Next>
__device__ inline E casLoop_(E* address, Next next)
{
    auto* word         = reinterpret_cast<WordOf_<E>*>(address);
    WordOf_<E> old     = *reinterpret_cast<volatile WordOf_<E>*>(address);
    WordOf_<E> assumed;
    do {
        assumed = old;
        old     = atomicCAS(word, assumed, wordOf_<E>(next(fromWord_<E>(assumed))));
    } while (assumed != old);
    return fromWord_<E>(old);
}
#else
/** @brief The host's compare-exchange loop: replaces the element by `next(current)` while `wanted(current)` holds and no
 *         other thread changed it in between; returns the value it saw last. */
template<class E, class Wanted>
inline E casWhile_(E& slot, E replacement, Wanted wanted)
{
    std::atomic_ref<E> ref(slot);
    E cur = ref.load(std::memory_order_relaxed);
    while (wanted(replacement, cur)
        && !ref.compare_exchange_weak(cur, replacement, std::memory_order_relaxed, std::memory_order_relaxed)) { }
    return cur;
}
#endif

} // namespace detail

/**
 * @brief Atomic minimum of `value` into the element `slot`; the element ends as the minimum of its initial value and every
 *        value any thread offered, in any order (a NaN absorbs, as in `np.minimum`). Integers and reals.
 * @return The value the element held before this thread's update.
 */
template<class E, class T>
AETHER_DEVICEHOST() inline E minAt(E& slot, T value)
{
    const E v = static_cast<E>(value);
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_integral_v<E> && sizeof(E) == 4 && std::is_signed_v<E>)
        return atomicMin(reinterpret_cast<int*>(&slot), static_cast<int>(v));
    else if constexpr (std::is_integral_v<E> && sizeof(E) == 4)
        return atomicMin(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(v));
    else if constexpr (std::is_integral_v<E> && std::is_signed_v<E>)
        return atomicMin(reinterpret_cast<long long*>(&slot), static_cast<long long>(v));
    else if constexpr (std::is_integral_v<E>)
        return atomicMin(reinterpret_cast<unsigned long long*>(&slot), static_cast<unsigned long long>(v));
    else
        return detail::casLoop_(&slot, [v](E cur) { return detail::takesMin_(v, cur) ? v : cur; });
#else
    return detail::casWhile_(slot, v, [](E x, E cur) { return detail::takesMin_(x, cur); });
#endif
}

/** @brief Atomic maximum of `value` into the element `slot`: the twin of `minAt` (a NaN absorbs, as in `np.maximum`). */
template<class E, class T>
AETHER_DEVICEHOST() inline E maxAt(E& slot, T value)
{
    const E v = static_cast<E>(value);
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_integral_v<E> && sizeof(E) == 4 && std::is_signed_v<E>)
        return atomicMax(reinterpret_cast<int*>(&slot), static_cast<int>(v));
    else if constexpr (std::is_integral_v<E> && sizeof(E) == 4)
        return atomicMax(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(v));
    else if constexpr (std::is_integral_v<E> && std::is_signed_v<E>)
        return atomicMax(reinterpret_cast<long long*>(&slot), static_cast<long long>(v));
    else if constexpr (std::is_integral_v<E>)
        return atomicMax(reinterpret_cast<unsigned long long*>(&slot), static_cast<unsigned long long>(v));
    else
        return detail::casLoop_(&slot, [v](E cur) { return detail::takesMax_(v, cur) ? v : cur; });
#else
    return detail::casWhile_(slot, v, [](E x, E cur) { return detail::takesMax_(x, cur); });
#endif
}

/** @brief Atomic bitwise and of `bits` into the integer element `slot`; the element ends as the and of its initial value and
 *         every value offered, in any order. @return The value the element held before this thread's update. */
template<class E, class T>
AETHER_DEVICEHOST() inline E andAt(E& slot, T bits)
{
    static_assert(std::is_integral_v<E>, "aether::accum::atomic::andAt: a bitwise update acts on an integer element");
#if defined(__CUDA_ARCH__)
    using W = detail::WordOf_<E>;
    return static_cast<E>(atomicAnd(reinterpret_cast<W*>(&slot), static_cast<W>(static_cast<E>(bits))));
#else
    return std::atomic_ref<E>(slot).fetch_and(static_cast<E>(bits), std::memory_order_relaxed);
#endif
}

/** @brief Atomic bitwise xor of `bits` into the integer element `slot`: the twin of `andAt`. */
template<class E, class T>
AETHER_DEVICEHOST() inline E xorAt(E& slot, T bits)
{
    static_assert(std::is_integral_v<E>, "aether::accum::atomic::xorAt: a bitwise update acts on an integer element");
#if defined(__CUDA_ARCH__)
    using W = detail::WordOf_<E>;
    return static_cast<E>(atomicXor(reinterpret_cast<W*>(&slot), static_cast<W>(static_cast<E>(bits))));
#else
    return std::atomic_ref<E>(slot).fetch_xor(static_cast<E>(bits), std::memory_order_relaxed);
#endif
}

/** @brief Atomic add of `term` into the element `slot`, returning the value the element held before it. The sum of the
 *         terms does not depend on the order, the returned value does. */
template<class E, class T>
AETHER_DEVICEHOST() inline E fetchAddAt(E& slot, T term)
{
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_integral_v<E> && sizeof(E) == 8 && std::is_signed_v<E>)
        return static_cast<E>(atomicAdd(reinterpret_cast<unsigned long long*>(&slot), static_cast<unsigned long long>(static_cast<E>(term))));
    else if constexpr (std::is_integral_v<E> && sizeof(E) == 4 && std::is_signed_v<E>)
        return static_cast<E>(atomicAdd(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(static_cast<E>(term))));
    else
        return atomicAdd(&slot, static_cast<E>(term));
#else
    return std::atomic_ref<E>(slot).fetch_add(static_cast<E>(term), std::memory_order_relaxed);
#endif
}

/** @brief Atomic exchange: the element `slot` takes `value` and the call returns what it held. Each element is exchanged
 *         atomically; which thread receives which old value depends on the order the threads arrive in. */
template<class E, class T>
AETHER_DEVICEHOST() inline E exchangeAt(E& slot, T value)
{
    const E v = static_cast<E>(value);
#if defined(__CUDA_ARCH__)
    using W = detail::WordOf_<E>;
    return detail::fromWord_<E>(atomicExch(reinterpret_cast<W*>(&slot), detail::wordOf_<E>(v)));
#else
    return std::atomic_ref<E>(slot).exchange(v, std::memory_order_relaxed);
#endif
}

/** @brief Atomic compare-exchange: when the element `slot` holds the bits of `expected` it takes `desired`. Returns the value
 *         the element held, so the call won exactly when the returned bits equal those of `expected`. */
template<class E, class T, class U>
AETHER_DEVICEHOST() inline E compareExchangeAt(E& slot, T expected, U desired)
{
    E seen = static_cast<E>(expected);
#if defined(__CUDA_ARCH__)
    using W = detail::WordOf_<E>;
    return detail::fromWord_<E>(atomicCAS(reinterpret_cast<W*>(&slot), detail::wordOf_<E>(seen), detail::wordOf_<E>(static_cast<E>(desired))));
#else
    std::atomic_ref<E>(slot).compare_exchange_strong(
        seen, static_cast<E>(desired), std::memory_order_relaxed, std::memory_order_relaxed);
    return seen;
#endif
}

} // namespace atomic
} // namespace accum
} // namespace aether
