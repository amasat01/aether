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

/** @brief True when `x` is neither infinite nor NaN. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool isFinite_(T x)
{
    return x - x == T(0);
}

/** @brief `twoSumErr_` for `float` and `double`: the exact residual
 *         `(a + b) - s` given `s == fl(a + b)`. Same barriers as the
 *         `double` form. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() T twoSumErrT_(T a, T b, T s)
{
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const T bb = s - a;
    return (a - (s - bb)) + (b - bb);
}

/**
 * @brief Neumaier step: add one `term` to the (hi, comp) pair. `hi + comp`
 *        stays the running sum to second order. A non-finite `hi` drops the
 *        correction, so the pair then follows IEEE through `hi` (lone +Inf
 *        gives +Inf, +Inf with -Inf gives NaN, NaN stays NaN).
 */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void neumaierAdd_(T& hi, T& comp, T term)
{
    AETHER_FP_BARRIER(term);
    const T s = hi + term;
    const T e = twoSumErrT_(hi, term, s);
    hi        = s;
    comp      = isFinite_(s) ? comp + e : T(0);
}

/**
 * @brief Combine two (hi, comp) partials into `(oh, oc)`: TwoSum of the hi
 *        parts, comp parts added. Symmetric: swapping the operands gives the
 *        same bits. The non-finite rule of `neumaierAdd_` applies.
 */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void pairCombine_(T ah, T ac, T bh, T bc, T& oh, T& oc)
{
    const T s = ah + bh;
    const T e = twoSumErrT_(ah, bh, s);
    oh        = s;
    oc        = isFinite_(s) ? (ac + bc) + e : T(0);
}

/**
 * @brief Host twin of the warp butterfly for one column: the canonical
 *        32-slot pair tree over per-lane partials (xor 16, 8, 4, 2, 1).
 *        Returns the fully reduced (hi, comp) the butterfly leaves on the
 *        column's owning lane.
 */
template<class T>
inline void reduce32Host_(const T (&hi)[32], const T (&comp)[32], T& oh, T& oc)
{
    T h[32], c[32];
    for (int l = 0; l < 32; ++l) {
        h[l] = hi[l];
        c[l] = comp[l];
    }
    for (int off = 16; off >= 1; off >>= 1) {
        T nh[32], nc[32];
        for (int l = 0; l < 32; ++l)
            pairCombine_(h[l], c[l], h[l ^ off], c[l ^ off], nh[l], nc[l]);
        for (int l = 0; l < 32; ++l) {
            h[l] = nh[l];
            c[l] = nc[l];
        }
    }
    oh = h[0];
    oc = c[0];
}

/**
 * @brief Columns left on `lane` after `butterfly32_` over `n` columns: the
 *        contiguous run `[lo, lo + m)`, held in slots `0 .. m-1`.
 */
AETHER_DEVICEHOST() inline void butterflyOwned_(int lane, int n, int& lo, int& m)
{
    // Rounds split the slot count `width` the same way on every lane; a lane
    // may hold fewer live columns than slots (the tail is padding).
    lo        = 0;
    m         = n;
    int width = n;
    for (int off = 16; off >= 1; off >>= 1) {
        const int lower = width / 2;
        if (lane & off) {
            lo += lower;
            m = m > lower ? m - lower : 0;
        } else {
            m = m < lower ? m : lower;
        }
        width -= lower;
    }
}

#if defined(__CUDACC__)
/** @brief One butterfly round over the leading `M` slots (device only). */
template<int M, class T>
__device__ AETHER_FORCEINLINE() void butterflyRound_(T* h, T* c, int off, bool up)
{
    constexpr int LO = M / 2, UP = M - LO;
#pragma unroll
    for (int j = 0; j < UP; ++j) {
        const T sh = up ? (j < LO ? h[j] : T(0)) : h[LO + j];
        const T sc = up ? (j < LO ? c[j] : T(0)) : c[LO + j];
        const T rh = __shfl_xor_sync(0xffffffffu, sh, off);
        const T rc = __shfl_xor_sync(0xffffffffu, sc, off);
        if (up) {
            pairCombine_(h[LO + j], c[LO + j], rh, rc, h[LO + j], c[LO + j]);
        } else if (j < LO) {
            pairCombine_(h[j], c[j], rh, rc, h[j], c[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < UP; ++j) {
        h[j] = up ? h[LO + j] : (j < LO ? h[j] : T(0));
        c[j] = up ? c[LO + j] : (j < LO ? c[j] : T(0));
    }
}

/**
 * @brief Warp butterfly transpose-reduce of `N` (hi, comp) columns per lane
 *        (device only, full mask: all 32 lanes must call it). Rounds xor
 *        16, 8, 4, 2, 1; each round a lane keeps a contiguous half of its
 *        columns. Afterwards lane `l` holds, in slots `0 .. m-1`, the fully
 *        reduced columns `[lo, lo + m)` given by `butterflyOwned_`; each
 *        equals `reduce32Host_` of that column. Lanes past a ragged end
 *        must carry +0.0 pairs.
 */
template<int N, class T>
__device__ AETHER_FORCEINLINE() void butterfly32_(T (&h)[N], T (&c)[N], int lane)
{
    constexpr int M2 = N - N / 2, M3 = M2 - M2 / 2, M4 = M3 - M3 / 2, M5 = M4 - M4 / 2;
    butterflyRound_<N>(h, c, 16, lane & 16);
    butterflyRound_<M2>(h, c, 8, lane & 8);
    butterflyRound_<M3>(h, c, 4, lane & 4);
    butterflyRound_<M4>(h, c, 2, lane & 2);
    butterflyRound_<M5>(h, c, 1, lane & 1);
}
#endif

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
