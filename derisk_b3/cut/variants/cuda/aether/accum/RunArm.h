// Variant cuda: NOT PARSED here (no nvcc); the device arm of accum/run.h, text moved verbatim. The `__CUDA_ARCH__` split below
// is the one nvcc itself makes between its device pass and its host pass of one translation unit.
#pragma once
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

#if defined(__CUDA_ARCH__)
namespace aether {
namespace accum {
namespace run {
namespace detail {
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
} // namespace detail
} // namespace run
} // namespace accum
} // namespace aether
#define AETHER_RUN_ATOMIC_ADD_BODY(E, slot, t) \
    return atomicAdd(slot, t);
#define AETHER_RUN_CLOSE_BODY(run) \
    (void)run;
#define AETHER_RUN_ADD_BODY(W, slot, term, own, run) \
    (void)run; \
    if (detail::warpAligned_(__activemask(), own)) { \
    const E s = detail::runSum_<W>(static_cast<E>(term)); \
    if ((detail::laneId_() & (W - 1)) == 0u) \
    detail::commit_(slot, s, true); \
    return; \
    } \
    detail::commit_(slot, static_cast<E>(term), false);
#define AETHER_RUN_ADD_TESTED_BODY(slot, at, term, run) \
    (void)run; \
    if (detail::warpUniform_(__activemask(), static_cast<unsigned long long>(at.global()))) { \
    const E s = detail::runSum_<32>(static_cast<E>(term)); \
    if (detail::laneId_() == 0u) \
    detail::commit_(slot, s, true); \
    return; \
    } \
    detail::commit_(slot, static_cast<E>(term), false);
#define AETHER_RUN_ADD_SORTED_BODY(slot, at, term, run) \
    (void)run; \
    if (__activemask() == 0xffffffffu) { \
    const unsigned lane  = detail::laneId_(); \
    const unsigned heads = detail::sortedHeads_(static_cast<unsigned long long>(at.global()), lane); \
    const E s            = detail::sortedSum_(static_cast<E>(term), heads, lane); \
    if ((heads >> lane) & 1u) \
    detail::commit_(slot, s, true); \
    return; \
    } \
    detail::commit_(slot, static_cast<E>(term), false);
#define AETHER_RUN_ADD_COMP_BODY(W, slot, compSlot, term, own, run) \
    (void)run; \
    if (detail::warpAligned_(__activemask(), own)) { \
    const E s = detail::runSum_<W>(static_cast<E>(term)); \
    if ((detail::laneId_() & (W - 1)) == 0u) \
    detail::commitComp_(slot, compSlot, s, E(0), true); \
    return; \
    } \
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#define AETHER_RUN_BLOCKFULL_BODY(base, count) \
    const long long b = blockDim.x; \
    return (1024 % b == 0) && (base % b == 0) && ((static_cast<long long>(blockIdx.x) + 1) * b <= count);
#define AETHER_RUN_ADD_COMP_BLOCK_BODY(target, comp, at, term, own, full, run) \
    if (full) { \
    __shared__ E part[32]; \
    const unsigned lane = detail::laneId_(), warp = threadIdx.x >> 5, warps = blockDim.x >> 5; \
    const E s           = detail::runSum_<32>(static_cast<E>(term)); \
    if (lane == 0u) \
    part[warp] = s; \
    __syncthreads(); \
    if (warp == 0u) { \
    const E v = detail::runSum_<32>(lane < warps ? part[lane] : E(0)); \
    if (lane == 0u) \
    detail::commitComp_(&target(at.global()), &comp(at.global()), v, E(0), true); \
    } \
    __syncthreads(); \
    return; \
    }
#define AETHER_RUN_ADD_TESTED_COMP_BODY(slot, compSlot, at, term, run) \
    (void)run; \
    if (detail::warpUniform_(__activemask(), static_cast<unsigned long long>(at.global()))) { \
    const E s = detail::runSum_<32>(static_cast<E>(term)); \
    if (detail::laneId_() == 0u) \
    detail::commitComp_(slot, compSlot, s, E(0), true); \
    return; \
    } \
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#define AETHER_RUN_ADD_SORTED_COMP_BODY(slot, compSlot, at, term, run) \
    (void)run; \
    if (__activemask() == 0xffffffffu) { \
    const unsigned lane  = detail::laneId_(); \
    const unsigned heads = detail::sortedHeads_(static_cast<unsigned long long>(at.global()), lane); \
    const E s            = detail::sortedSum_(static_cast<E>(term), heads, lane); \
    if ((heads >> lane) & 1u) \
    detail::commitComp_(slot, compSlot, s, E(0), true); \
    return; \
    } \
    detail::commitComp_(slot, compSlot, static_cast<E>(term), E(0), false);
#else
#if defined(__cpp_lib_atomic_ref)
#define AETHER_RUN_ATOMIC_ADD_BODY(E, slot, t) \
    return std::atomic_ref<E>(*slot).fetch_add(t, std::memory_order_relaxed);
#else // Apple libc++ has no std::atomic_ref: a compare-exchange loop on the value's bits
#define AETHER_RUN_ATOMIC_ADD_BODY(E, slot, t) \
    using Bits = std::conditional_t<sizeof(E) == 4, unsigned int, unsigned long long>; \
    static_assert(sizeof(E) == sizeof(Bits), "a 4- or 8-byte element"); \
    auto* bits = reinterpret_cast<Bits*>(slot); \
    Bits seen  = __atomic_load_n(bits, __ATOMIC_RELAXED); \
    while (!__atomic_compare_exchange_n(bits, &seen, std::bit_cast<Bits>(std::bit_cast<E>(seen) + t), \
                                        true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {} \
    return std::bit_cast<E>(seen);
#endif
#define AETHER_RUN_CLOSE_BODY(run) \
    detail::flush_(run); \
    run = Open<E>{};
#define AETHER_RUN_ADD_BODY(W, slot, term, own, run) \
    (void)own; \
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#define AETHER_RUN_ADD_TESTED_BODY(slot, at, term, run) \
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#define AETHER_RUN_ADD_SORTED_BODY(slot, at, term, run) \
    detail::openAdd_(run, slot, static_cast<E*>(nullptr), static_cast<E>(term));
#define AETHER_RUN_ADD_COMP_BODY(W, slot, compSlot, term, own, run) \
    (void)own; \
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#define AETHER_RUN_BLOCKFULL_BODY(base, count) \
    (void)base; \
    (void)count; \
    return false;
#define AETHER_RUN_ADD_COMP_BLOCK_BODY(target, comp, at, term, own, full, run) \
    (void)full;
#define AETHER_RUN_ADD_TESTED_COMP_BODY(slot, compSlot, at, term, run) \
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#define AETHER_RUN_ADD_SORTED_COMP_BODY(slot, compSlot, at, term, run) \
    detail::openAdd_(run, slot, compSlot, static_cast<E>(term));
#endif
