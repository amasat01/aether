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
#include "aether/accum/TwoSum.h"
#include "aether/macros.h"

#include "aether/accum/RunArm.h" // variant-supplied: the atomics, the commit hook and every device/host body below

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
AETHER_RUN_ATOMIC_ADD_BODY(E, slot, t)
}

/** @brief The exact residual `(a + b) - s` for `s == fl(a + b)`: `accum::twoSumErr`. */
template<class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() E twoSumErr_(E a, E b, E s)
{
    return accum::twoSumErr(a, b, s);
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
    atomicAdd_(compSlot, accum::twoSumErrAtomic(old, s, sum) + c);
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
    const E e = accum::twoSumErrAccum(run.s, t, s);
    run.s     = s;
    run.c     = s - s == E(0) ? run.c + e : E(0);
}


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
AETHER_RUN_CLOSE_BODY(run)
}

/**
 * @brief Add `term` into `target(at.global())`, one atomic per aligned run of
 *        `W` lanes.
 *
 * @tparam W   run length, a power of two in `[2, 32]`; the caller's fact is
 *             that the slot is equal over every aligned group of `W` lanes.
 * @param target  the array the run sums into.
 * @param at      the sample being added; its global index picks the slot.
 * @param term    the value to add.
 * @param own  the lane's own sample index (the alignment vote reads it).
 * @param run  the host's open run (`open(target)`); unused on the device.
 */
template<int W, class ViewT, class T, class I, class E>
AETHER_DEVICEHOST() inline void add(ViewT& target, const SampleIndex& at, T term, const I& own, Open<E>& run)
{
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "a run is 2..32 lanes");
    E* slot = &target(at.global());
AETHER_RUN_ADD_BODY(W, slot, term, own, run)
}

/** @brief As `add<32>` with no proof: a warp whose 32 slot indices one vote
 *         finds equal commits once, any other lane by lane. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addTested(ViewT& target, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot = &target(at.global());
AETHER_RUN_ADD_TESTED_BODY(slot, at, term, run)
}

/** @brief Add `term` into `target(at.global())` for keys that are
 *         non-decreasing in lane order: one atomic per maximal run of equal
 *         keys. Keys out of order only make the runs shorter. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addSorted(ViewT& target, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot = &target(at.global());
AETHER_RUN_ADD_SORTED_BODY(slot, at, term, run)
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
AETHER_RUN_ADD_COMP_BODY(W, slot, compSlot, term, own, run)
}

/** @brief True for a block whose every thread is live and whose samples
 *         `[base + blockIdx.x * blockDim.x, ... + blockDim.x)` a caller has
 *         proven to share one key (the key moves in runs of 1024 aligned
 *         lanes, and `blockDim.x` divides 1024). Block-uniform. Always false
 *         on the host. */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool blockFull(long long base, long long count)
{
AETHER_RUN_BLOCKFULL_BODY(base, count)
}

/** @brief `addCompensated<32>` for a key that is uniform over the block: when
 *         `full` (`blockFull`), the warps' run sums are folded through shared
 *         memory by a fixed tree and ONE compensated commit is issued per
 *         BLOCK, not per warp. Every thread of the block must reach the call
 *         (it holds two barriers); `full` is block-uniform, and a block that is
 *         not full takes `addCompensated<32>` unchanged. */
template<class ViewT, class T, class I, class E>
AETHER_DEVICEHOST() inline void addCompensatedBlock(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, const I& own, bool full, Open<E>& run)
{
AETHER_RUN_ADD_COMP_BLOCK_BODY(target, comp, at, term, own, full, run)
    addCompensated<32>(target, comp, at, term, own, run);
}

/** @brief Compensated `addTested`. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addTestedCompensated(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot     = &target(at.global());
    E* compSlot = &comp(at.global());
AETHER_RUN_ADD_TESTED_COMP_BODY(slot, compSlot, at, term, run)
}

/** @brief Compensated `addSorted`. */
template<class ViewT, class T, class E>
AETHER_DEVICEHOST() inline void addSortedCompensated(
    ViewT& target, ViewT& comp, const SampleIndex& at, T term, Open<E>& run)
{
    E* slot     = &target(at.global());
    E* compSlot = &comp(at.global());
AETHER_RUN_ADD_SORTED_COMP_BODY(slot, compSlot, at, term, run)
}

} // namespace run
} // namespace accum
} // namespace aether
