// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file run.h
 * @brief `aether::accum::run`: add per-lane terms into a keyed accumulator
 *        with ONE atomic per run of lanes that share a slot, on both CUDA and
 *        host paths behind a single call.
 *
 * Three forms, each with a plain and a compensated call:
 *
 *  - **aligned run** (`add<W>`): the caller PROVES that the slot is the same
 *    over every aligned group of `W` lanes. One vote checks that the warp
 *    holds 32 consecutive samples starting on a multiple of 32.
 *  - **tested run** (`addTested`): no proof; one vote checks that the full
 *    warp shares one slot.
 *  - **sorted run** (`addSorted`): equal slots are adjacent in lane order
 *    (non-decreasing keys); each maximal run is found with a head flag.
 *
 * Device: the run is reduced with a plain shuffle tree and its first lane
 * issues the atomic. A warp the vote rejects, or one that is not full,
 * commits lane by lane: the call is correct for any key; the facts only
 * make it fast. Compensated forms keep the same plain tree and make the ONE
 * commit compensated: the run's sum is added into `target` and the two-sum
 * residual of that add into `comp` (the delivered value is `target + comp`).
 *
 * Host: every form sums into an open run held by the caller (`Open`, made
 * by `open()`), in lane order with a Neumaier step, and commits it when the
 * slot changes and at `close()`. The caller keeps one `Open` per target for a serial range of
 * lanes and closes it at the end of the range; the device ignores it.
 *
 * Tree order (the ULP contract's, never bit identity across shapes): the
 * aligned and tested runs use an xor butterfly, offsets W/2 ... 1; the
 * sorted run a segmented down-shuffle tree, offsets 1, 2, 4, 8, 16, the
 * lower lane on the left.
 *
 * `AETHER_RUN_COMMIT_HOOK(run)` is called once per atomic commit (`run`
 * true for a run's commit, false for a lane's own); it does nothing unless a
 * test defines it before including this header. Defining
 * `AETHER_RUN_COUNT_COMMITS` instead makes this header define the C-linkage
 * counters `aether_run_commits[2]` (run, lane) and count into them: a debug
 * build of ONE translation unit only.
 *
 * Includes only what an NVRTC-compiled kernel already has.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

#ifndef __CUDA_ARCH__
#include <atomic>
#include <bit>
#endif

#if defined(AETHER_RUN_COUNT_COMMITS) && !defined(AETHER_RUN_COMMIT_HOOK)
#if defined(__CUDACC__)
extern "C" {
__device__ unsigned long long aether_run_commits[2] = { 0, 0 };
}
#if defined(__CUDA_ARCH__)
#define AETHER_RUN_COMMIT_HOOK(run) atomicAdd(&aether_run_commits[(run) ? 0 : 1], 1ull)
#else
#define AETHER_RUN_COMMIT_HOOK(run) ((void)(run))
#endif
#else
extern "C" {
__attribute__((visibility("default"))) unsigned long long aether_run_commits[2] = { 0, 0 };
}
#define AETHER_RUN_COMMIT_HOOK(run)                                                                \
    __atomic_fetch_add(&aether_run_commits[(run) ? 0 : 1], 1ull, __ATOMIC_RELAXED)
#endif
#endif

#ifndef AETHER_RUN_COMMIT_HOOK
#define AETHER_RUN_COMMIT_HOOK(run) ((void)(run))
#endif

namespace aether {
namespace accum {
namespace run {

/** @brief The host's open run: the slot being summed, its companion (null
 *         for a plain run) and the partial `(s, c)`. Unused on the device. */
template<class E>
struct Open {
    E* slot = nullptr;
    E* comp = nullptr;
    E s     = E(0);
    E c     = E(0);
};

namespace detail {

/** @brief Element type behind a view's slot reference. */
template<class ViewT>
using slot_t = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<ViewT&>()(offset_t{ 0 }))>>;

/** @brief One atomic add of `t` into `slot`; returns the value it replaced. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E atomicAdd_(E* slot, E t)
{
#if defined(__CUDA_ARCH__)
    return atomicAdd(slot, t);
#else
#if defined(__cpp_lib_atomic_ref)
    return std::atomic_ref<E>(*slot).fetch_add(t, std::memory_order_relaxed);
#else // Apple libc++ has no std::atomic_ref: a compare-exchange loop on the value's bits
    using Bits = std::conditional_t<sizeof(E) == 4, unsigned int, unsigned long long>;
    static_assert(sizeof(E) == sizeof(Bits), "a 4- or 8-byte element");
    auto* bits = reinterpret_cast<Bits*>(slot);
    Bits seen  = __atomic_load_n(bits, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(bits, &seen, std::bit_cast<Bits>(std::bit_cast<E>(seen) + t),
                                        true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return std::bit_cast<E>(seen);
#endif
#endif
}

/** @brief Knuth two-sum residual `(a + b) - s` for `s == fl(a + b)`, any
 *         order and signs (`atomic::detail::twoSumErrT_`). */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErr_(E a, E b, E s)
{
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const E bb = s - a;
    return (a - (s - bb)) + (b - bb);
}

/** @brief The plain commit of `s` into `slot`. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void commit_(E* slot, E s, bool run)
{
    atomicAdd_(slot, s);
    AETHER_RUN_COMMIT_HOOK(run);
}

/** @brief The compensated commit of the partial `(s, c)`: ONE atomic add of
 *         `s` into `slot`, its two-sum residual plus `c` into `compSlot`. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void commitComp_(E* slot, E* compSlot, E s, E c, bool run)
{
    AETHER_FP_BARRIER(s);
    const E old = atomicAdd_(slot, s);
    const E sum = old + s;
    atomicAdd_(compSlot, twoSumErr_(old, s, sum) + c);
    AETHER_RUN_COMMIT_HOOK(run);
}

/** @brief Host: commit the open run. A plain run adds its partial
 *         `s + c` in one atomic; a compensated one commits the pair. */
template<class E>
inline void flush_(Open<E>& run)
{
    if (run.slot == nullptr)
        return;
    if (run.comp != nullptr)
        commitComp_(run.slot, run.comp, run.s, run.c, true);
    else
        commit_(run.slot, run.s + run.c, true);
}

/** @brief Host: add `t` into the open run on `slot` with a Neumaier step,
 *         first committing the run it replaces. */
template<class E>
inline void openAdd_(Open<E>& run, E* slot, E* comp, E t)
{
    if (slot != run.slot) {
        flush_(run);
        run = Open<E>{ slot, comp, t, E(0) };
        return;
    }
    AETHER_FP_BARRIER(t);
    const E s = run.s + t;
    const E e = twoSumErr_(run.s, t, s);
    run.s     = s;
    run.c     = s - s == E(0) ? run.c + e : E(0);
}

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ unsigned laneId_()
{
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

/** @brief The full warp holds 32 consecutive samples starting on a multiple
 *         of 32 (one vote). */
template<class I>
__device__ __forceinline__ bool warpAligned_(unsigned m, const I& own)
{
    return m == 0xffffffffu && __all_sync(m, (static_cast<unsigned>(own.global()) & 31u) == laneId_());
}

/** @brief The full warp's keys all equal lane 0's (one vote). */
template<class K>
__device__ __forceinline__ bool warpUniform_(unsigned m, K key)
{
    return m == 0xffffffffu && __all_sync(m, key == __shfl_sync(m, key, 0));
}

/** @brief Plain xor butterfly over aligned groups of `W` lanes. */
template<int W, class E>
__device__ __forceinline__ E runSum_(E t)
{
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1)
        t += __shfl_xor_sync(0xffffffffu, t, o);
    return t;
}

/** @brief Plain segmented sum of a full warp's sorted runs: each run's head
 *         gets the run's sum. `heads` has bit `l` set when lane `l` starts a
 *         run; a warp of heads only skips the tree. */
template<class E>
__device__ __forceinline__ E sortedSum_(E t, unsigned heads, unsigned lane)
{
    if (heads == 0xffffffffu)
        return t;
    const unsigned later = lane == 31u ? 0u : heads >> (lane + 1u);
    const unsigned end   = later ? lane + 1u + static_cast<unsigned>(__ffs(later) - 1) : 32u;
#pragma unroll
    for (unsigned o = 1; o < 32u; o <<= 1) {
        const E v = __shfl_down_sync(0xffffffffu, t, o);
        if (lane + o < end)
            t += v;
    }
    return t;
}

/** @brief Head flags of a full warp: lane 0, or a key that differs from the
 *         lane below. */
template<class K>
__device__ __forceinline__ unsigned sortedHeads_(K key, unsigned lane)
{
    const K below = __shfl_up_sync(0xffffffffu, key, 1);
    return __ballot_sync(0xffffffffu, lane == 0u || key != below);
}
#endif

} // namespace detail

/** @brief A closed host run for `target`'s element type. */
template<class ViewT>
AETHER_DEVICEHOST() inline Open<detail::slot_t<ViewT>> open(const ViewT&)
{
    return {};
}

/** @brief Commit the host's open run (the end of the caller's range). The
 *         device has none: a no-op there. */
template<class E>
AETHER_DEVICEHOST() inline void close(Open<E>& run)
{
#if !defined(__CUDA_ARCH__)
    detail::flush_(run);
    run = Open<E>{};
#else
    (void)run;
#endif
}

/**
 * @brief Add `term` into `target(at.global())`, one atomic per aligned run of
 *        `W` lanes.
 *
 * @tparam W   run length, a power of two in `[2, 32]`; the caller's fact is
 *             that the slot is equal over every aligned group of `W` lanes.
 * @param own  the lane's own sample index (the alignment vote reads it).
 * @param run  the host's open run (`open(target)`); unused on the device.
 */
template<int W, class ViewT, class T, class I, class E>
AETHER_DEVICEHOST() inline void add(ViewT& target, const SampleIndex& at, T term, const I& own, Open<E>& run)
{
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "a run is 2..32 lanes");
    E* slot = &target(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (detail::warpAligned_(__activemask(), own)) {
        const E s = detail::runSum_<W>(static_cast<E>(term));
        if ((detail::laneId_() & (W - 1)) == 0u)
            detail::commit_(slot, s, true);
        return;
    }
    detail::commit_(slot, static_cast<E>(term), false);
#else
    (void)own;
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#endif
}

/** @brief As `add<32>` with no proof: a warp whose 32 slot indices one vote
 *         finds equal commits once, any other lane by lane. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addTested(ViewT& target, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot = &target(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (detail::warpUniform_(__activemask(), static_cast<unsigned long long>(at.global()))) {
        const E s = detail::runSum_<32>(static_cast<E>(term));
        if (detail::laneId_() == 0u)
            detail::commit_(slot, s, true);
        return;
    }
    detail::commit_(slot, static_cast<E>(term), false);
#else
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#endif
}

/** @brief Add `term` into `target(at.global())` for keys that are
 *         non-decreasing in lane order: one atomic per maximal run of equal
 *         keys. Keys out of order only make the runs shorter. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addSorted(ViewT& target, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot = &target(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (__activemask() == 0xffffffffu) {
        const unsigned lane  = detail::laneId_();
        const unsigned heads = detail::sortedHeads_(static_cast<unsigned long long>(at.global()), lane);
        const E s            = detail::sortedSum_(static_cast<E>(term), heads, lane);
        if ((heads >> lane) & 1u)
            detail::commit_(slot, s, true);
        return;
    }
    detail::commit_(slot, static_cast<E>(term), false);
#else
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#endif
}

/** @brief Compensated `add<W>`: the plain run sum, then ONE compensated
 *         commit into `target` and `comp`. */
template<int W, class ViewT, class T, class I, class E>
AETHER_DEVICEHOST() inline void addCompensated(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, const I& own, Open<E>& run)
{
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "a run is 2..32 lanes");
    E* slot     = &target(at.global());
    E* compSlot = &comp(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (detail::warpAligned_(__activemask(), own)) {
        const E s = detail::runSum_<W>(static_cast<E>(term));
        if ((detail::laneId_() & (W - 1)) == 0u)
            detail::commitComp_(slot, compSlot, s, E(0), true);
        return;
    }
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#else
    (void)own;
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#endif
}

/** @brief Compensated `addTested`. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addTestedCompensated(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot     = &target(at.global());
    E* compSlot = &comp(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (detail::warpUniform_(__activemask(), static_cast<unsigned long long>(at.global()))) {
        const E s = detail::runSum_<32>(static_cast<E>(term));
        if (detail::laneId_() == 0u)
            detail::commitComp_(slot, compSlot, s, E(0), true);
        return;
    }
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#else
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#endif
}

/** @brief Compensated `addSorted`. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addSortedCompensated(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot     = &target(at.global());
    E* compSlot = &comp(at.global());
#if defined(__CUDA_ARCH__)
    (void)run;
    if (__activemask() == 0xffffffffu) {
        const unsigned lane  = detail::laneId_();
        const unsigned heads = detail::sortedHeads_(static_cast<unsigned long long>(at.global()), lane);
        const E s            = detail::sortedSum_(static_cast<E>(term), heads, lane);
        if ((heads >> lane) & 1u)
            detail::commitComp_(slot, compSlot, s, E(0), true);
        return;
    }
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#else
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#endif
}

} // namespace run
} // namespace accum
} // namespace aether
