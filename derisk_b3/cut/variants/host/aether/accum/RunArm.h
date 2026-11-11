// Variant host: the host arm of accum/run.h. The commit hook, the atomic add and every run body are spelled here once.
#pragma once
#include <atomic>
#include <bit>

// Commit counting (a test option): a visible counter pair bumped once per commit.
#if defined(AETHER_RUN_COUNT_COMMITS) && !defined(AETHER_RUN_COMMIT_HOOK)
extern "C" {
__attribute__((visibility("default"))) unsigned long long aether_run_commits[2] = { 0, 0 };
}
#define AETHER_RUN_COMMIT_HOOK(run) __atomic_fetch_add(&aether_run_commits[(run) ? 0 : 1], 1ull, __ATOMIC_RELAXED)
#endif
#ifndef AETHER_RUN_COMMIT_HOOK
#define AETHER_RUN_COMMIT_HOOK(run) ((void)(run))
#endif

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
