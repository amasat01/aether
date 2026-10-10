// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file atomic.h
 * @brief `aether::accum::atomic`: lightweight atomic primitives used by
 *        producers that aggregate per-sample results across concurrent
 *        contributors, on both CUDA and host paths behind a single call.
 */

#include <cmath>
#include <cstdint>
#include <cstring>

#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/view/View.h"

#ifndef __CUDA_ARCH__
#include <atomic>
#endif

namespace aether {
namespace accum {
namespace atomic {

namespace detail {

/** @brief Atomic minimum on the magnitude of a signed real value at
 *         `address`, preserving the sign already stored there. */
AETHER_DEVICEHOST() inline double minAbsReal_(double* address, double val)
{
#if defined(__CUDA_ARCH__)
    auto* p                = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *p;
    unsigned long long assumed;
    const double absVal = fabs(val);
    do {
        assumed          = old;
        const double cur = __longlong_as_double(assumed);
        if (absVal >= fabs(cur))
            return cur;
        const double newVal             = copysign(absVal, cur);
        const unsigned long long newRep = __double_as_longlong(newVal);
        old                             = atomicCAS(p, assumed, newRep);
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    auto* p                = reinterpret_cast<std::uint64_t*>(address);
    auto* atomicP          = reinterpret_cast<std::atomic<std::uint64_t>*>(p);
    std::uint64_t expected = atomicP->load(std::memory_order_relaxed);
    const double absVal    = std::abs(val);
    while (true) {
        double cur;
        std::memcpy(&cur, &expected, sizeof(cur));
        if (absVal >= std::abs(cur))
            return cur;
        const double newVal = std::copysign(absVal, cur);
        std::uint64_t desired;
        std::memcpy(&desired, &newVal, sizeof(desired));
        if (atomicP->compare_exchange_weak(
                expected, desired, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return cur;
        }
    }
#endif
}

/** @brief Monotone OR-store of `true` into a one-byte `bool` slot at
 *         `address`.
 *
 *  Invariant: the only operation is an idempotent set-to-1, performed as a
 *  byte-granular store (`ST.U8`), so it never touches a neighbouring byte and
 *  stays inside a one-byte allocation. The result is visible at the kernel
 *  boundary. A future need to clear a slot or read it back within a kernel
 *  must switch the slot type to 32 bits and use a word atomic. */
AETHER_DEVICEHOST() inline void orBool_(bool* address, bool val)
{
    if (!val)
        return;
#if defined(__CUDA_ARCH__)
    *reinterpret_cast<volatile unsigned char*>(address) = 1;
#else
    auto* atomicP = reinterpret_cast<std::atomic<unsigned char>*>(address);
    atomicP->store(static_cast<unsigned char>(1), std::memory_order_relaxed);
#endif
}

/**
 * @brief Knuth two-sum error term: given `s == fl(a + b)`, returns the
 *        residual `(a + b) - s` exactly, for any ordering of `|a|`, `|b|`
 *        and any signs (no `fast2Sum` precondition). Six FP64 flops,
 *        branch-free.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() double twoSumErr_(double a, double b, double s)
{
    // The residual is exact only for the ROUNDED a, b and s: once inlined, a
    // product feeding any of them would otherwise be contracted into these
    // subtractions (host -ffp-contract=fast, the gcc/clang default).
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const double bb = s - a;
    return (a - (s - bb)) + (b - bb);
}

/**
 * @brief Compensated accumulation of `term` into a (value, comp) slot pair.
 *        Delivered quantity is `value[i] + comp[i]`, independent of
 *        accumulation order. Requires `atomicAdd(double*, double)` (sm_60+;
 *        aether's lowest configured arch is sm_61).
 */
AETHER_DEVICEHOST() inline void compensatedSum_(double* value, double* comp, double term)
{
    // `term` is rounded once, here: a product passed in may not be fused
    // into `old + term` below (see twoSumErr_).
    AETHER_FP_BARRIER(term);
#if defined(__CUDA_ARCH__)
    const double old = atomicAdd(value, term);
    const double s    = old + term;
    atomicAdd(comp, twoSumErr_(old, term, s));
#else
    const double old = *value;
    const double s    = old + term;
    *value            = s;
    *comp             = *comp + twoSumErr_(old, term, s);
#endif
}

/** @brief Atomic maximum on a signed 32-bit integer at `address`. */
AETHER_DEVICEHOST() inline int maxInt_(int* address, int val)
{
#if defined(__CUDA_ARCH__)
    return atomicMax(address, val);
#else
    auto* atomicP = reinterpret_cast<std::atomic<int>*>(address);
    int expected  = atomicP->load(std::memory_order_relaxed);
    while (val > expected
        && !atomicP->compare_exchange_weak(
            expected, val, std::memory_order_relaxed, std::memory_order_relaxed)) { }
    return expected;
#endif
}

} // namespace detail

/**
 * @brief Atomic minimum on the magnitude of a signed real value.
 *
 * Compares `|val|` against `|h[i]|`; if smaller, stores `copysign(|val|,
 * h[i])` so the sign already at the slot is preserved.
 *
 * @tparam ViewT  a scalar (`extents<dyn>`) `double` view.
 * @return The value previously stored at the slot.
 */
template<class ViewT>
AETHER_DEVICEHOST() inline double minAbsReal(ViewT& h, const SampleIndex& i, double val)
{
    return detail::minAbsReal_(&h(i.global()), val);
}
/** @brief Raw-pointer overload for callers that already hold a raw address. */
inline double minAbsReal(double* h, double val) { return detail::minAbsReal_(h, val); }

/**
 * @brief Monotone OR-store of `true` into a one-byte `bool` slot.
 *
 * Because the underlying operation is an OR with the constant `true`, the
 * result is independent of the order concurrent writers execute in — a
 * plain monotone store of `1` suffices on both CUDA and host paths. A
 * `false` call is a no-op.
 *
 * @tparam ViewT  a scalar `bool` view.
 */
template<class ViewT>
AETHER_DEVICEHOST() inline void orBool(ViewT& h, const SampleIndex& i, bool val)
{
    detail::orBool_(&h(i.global()), val);
}
/** @brief Raw-pointer overload for callers that already hold a raw address. */
inline void orBool(bool* h, bool val) { detail::orBool_(h, val); }

/**
 * @brief Atomic maximum on a signed 32-bit integer slot.
 *
 * @tparam ViewT  a scalar `int` view.
 * @return The value previously stored at the slot.
 */
template<class ViewT>
AETHER_DEVICEHOST() inline int maxInt(ViewT& h, const SampleIndex& i, int val)
{
    return detail::maxInt_(&h(i.global()), val);
}
/** @brief Raw-pointer overload for callers that already hold a raw address. */
inline int maxInt(int* h, int val) { return detail::maxInt_(h, val); }

/**
 * @brief Compensated accumulation of a `double` term into a value slot and
 *        its companion residual (`comp`) slot. Delivered quantity is
 *        `value[i] + comp[i]`, independent of accumulation order.
 *
 * View-only: no raw-pointer overload. `aether/accum/AccumPlane.h` is its
 * only consumer.
 *
 * @tparam ViewT  scalar `double` views (value lane and companion lane).
 */
template<class ViewT>
AETHER_DEVICEHOST() inline void compensatedSum(ViewT& value, ViewT& comp, const SampleIndex& i, double term)
{
    detail::compensatedSum_(&value(i.global()), &comp(i.global()), term);
}

} // namespace atomic
} // namespace accum
} // namespace aether
