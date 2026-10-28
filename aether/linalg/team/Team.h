// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Team.h
 * @brief `aether::linalg::team::Team`: the handle a team kernel runs under, and
 *        the sizes and loop helpers shared by the team family.
 *
 * A team is the set of threads that work on ONE matrix: a 1-D thread block on
 * the device (`rank = threadIdx.x`, `size = blockDim.x`, `sync() = __syncthreads()`),
 * or on the host a team of size 1 (`rank 0`, `sync()` a no-op) or a team of host
 * threads that bring their own barrier. The team size is a performance
 * constant: every result in this family is bit-independent of it, and the
 * size-1 team is the CPU twin of the same code.
 *
 * Every team function is COLLECTIVE: all threads of the team call it with the
 * same arguments, and the matrices live in caller-provided scratch (global
 * memory, one slice per team). A function returns with its results visible to
 * the whole team. Scratch views alias freely only where a function says so.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/macros.h"

namespace aether {
namespace linalg {
namespace team {

/** @brief Largest supported dimension of the team family (memory is the real bound). */
inline constexpr std::size_t max_dim = 4096;

/** @brief Rows per tile of the QR reductions: fixed, so a sum never depends on the team size. */
inline constexpr std::size_t qrTile = 64;

namespace detail {

/** @brief `T` is `float` or `double` and `1 <= N <= max_dim`. */
template<class T, std::size_t N>
inline constexpr bool supported_v = (std::is_same_v<T, float> || std::is_same_v<T, double>) && N >= 1 && N <= max_dim;

} // namespace detail

/**
 * @brief The calling thread's place in its team.
 *
 * Plain data, passed by value. A host team of several threads sets `barrier`
 * and `ctx`; on the device `sync()` is the block barrier whatever they hold.
 */
struct Team {
    int rank = 0;                      ///< This thread's index in the team, `0 <= rank < size`.
    int size = 1;                      ///< Threads in the team.
    void (*barrier)(void*) = nullptr;  ///< Host only: the team barrier; null for a team of 1.
    void* ctx = nullptr;               ///< Host only: argument of `barrier`.

    /** @brief Barrier over the whole team, with the memory effects of every thread visible after it. */
    AETHER_DEVICEHOST() void sync() const
    {
#if defined(__CUDA_ARCH__)
        __syncthreads();
#else
        if (barrier != nullptr) {
            barrier(ctx);
        }
#endif
    }

    /** @brief The team of the current thread block (1-D blocks) on the device; the team of 1 on the host. */
    AETHER_DEVICEHOST() static Team here()
    {
        Team t;
#if defined(__CUDA_ARCH__)
        t.rank = static_cast<int>(threadIdx.x);
        t.size = static_cast<int>(blockDim.x);
#endif
        return t;
    }
};

/** @brief Scratch elements `qr` and `qrSolve` need as `work` for `cols` columns of `m` rows (`cols = max(N, k)`). */
AETHER_DEVICEHOST() constexpr std::size_t qrWorkSize(std::size_t m, std::size_t cols)
{
    return cols * ((m + qrTile - 1) / qrTile) + 1;
}

/** @brief Scratch elements `eigh` needs as `work`. */
AETHER_DEVICEHOST() constexpr std::size_t eighWorkSize(std::size_t n)
{
    return n * n + 4;
}

} // namespace team
} // namespace linalg
} // namespace aether
