// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Philox.h
 * @brief `aether::random::detail` — aether's OWN Philox4x32-10 counter-based
 *        generator, bit-compatible with cuRAND's device Philox state machine.
 *
 * One generator, both arms: the same code runs on both host and device, so
 * a seed names one stream everywhere (unconditional cross-mode
 * bit-identity). Two independent reasons force it to be aether's own code
 * rather than a `<curand_kernel.h>` include:
 *
 *  1. `<curand_kernel.h>` is a CUDA-toolkit header. The JIT/NVRTC path, the
 *     CPU-only arm and the tier-2 (non-NVIDIA) backends all need aether's
 *     random surface to compile with NO CUDA toolchain present — so no
 *     cuRAND dependency may exist anywhere in aether.
 *  2. cuRAND's device draws are `__device__`-only, so a host arm could never
 *     have reused them even where the toolkit IS installed.
 *
 * The algorithm below is therefore a from-scratch, portable-C++ reproduction
 * of the state machine in the CUDA toolkit's `curand_philox4x32_x.h` +
 * `curand_kernel.h` (READ-ONLY reference; those headers are NOT included and
 * NOT redistributed here — this is a reimplementation of the published
 * Philox4x32-10 algorithm of Salmon et al. (D. E. Shaw Research, SC'11) with
 * cuRAND's *positioning* and *output ordering* conventions):
 *
 *  - **Key/counter layout**, mirroring `curand_init(seed, subsequence,
 *    offset, &st)`: `key = (lo32(seed), hi32(seed))`, `ctr = (0,0,0,0)`, then
 *    `skipahead_sequence(subsequence)` bumps the HIGH counter half
 *    (`ctr.z`/`ctr.w`) and `skipahead(offset)` bumps the LOW half
 *    (`ctr.x`/`ctr.y`) by `offset/4` while parking the intra-quad position
 *    `STATE = offset & 3`. Net: `ctr = (lo32(offset/4), hi32(offset/4),
 *    lo32(subsequence), hi32(subsequence))`.
 *  - **Rounds**: 10 rounds, key bumped by the golden-ratio / sqrt(3)-1
 *    Weyl constants between rounds; `mulhilo32` products with
 *    `0xD2511F53`/`0xCD9E8D57`.
 *  - **Output ordering**: `next()` walks `output.{x,y,z,w}` and regenerates
 *    on wrap (cuRAND's `curand()`); `next4()` reproduces `curand4()`'s
 *    quad-straddling behaviour EXACTLY (including that it does not advance
 *    `STATE`) — `standardNormal<double>` depends on it, because the
 *    reference cuRAND device path reaches Box-Muller through
 *    `curand_normal_double`, which is a `curand4()` consumer.
 *
 * Verified against the real cuRAND host API (`curandGenerate` on
 * `CURAND_RNG_PSEUDO_PHILOX4_32_10`) by `tools/random/curand_host_probe.cpp`.
 *
 * Note: the counters here are 64-bit values (they are RNG stream
 * coordinates, not addresses) — sample addressing elsewhere in this library
 * stays on the narrower `offset_t` (32-bit); no address ever routes through
 * these.
 */

#include <cstdint>

#include "aether/macros.h"

/// @cond INTERNAL

namespace aether {
namespace random {
namespace detail {

/** @brief A quad of 32-bit words — the Philox counter/key/output block.
 *         (aether's own POD; deliberately NOT CUDA's `uint4`, which does not
 *         exist on the CPU-only arm.) */
struct Uint4 {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint32_t w;
};

/** @brief Philox key-bump Weyl constant for the low key word (`0x9E3779B9`,
 *         the golden ratio). */
inline constexpr std::uint32_t kPhiloxW32_0 = 0x9E3779B9u;
/** @brief Philox key-bump Weyl constant for the high key word (`0xBB67AE85`,
 *         sqrt(3) - 1). */
inline constexpr std::uint32_t kPhiloxW32_1 = 0xBB67AE85u;
/** @brief Philox round multiplier applied to counter word 0. */
inline constexpr std::uint32_t kPhiloxM4x32_0 = 0xD2511F53u;
/** @brief Philox round multiplier applied to counter word 2. */
inline constexpr std::uint32_t kPhiloxM4x32_1 = 0xCD9E8D57u;

/**
 * @brief Full 64-bit product of two 32-bit words, split into low (returned)
 *        and high (`hi`) halves.
 *
 * The device leg uses `__umulhi` + a 32-bit multiply (what cuRAND emits);
 * the host leg widens to 64 bits. The two are mathematically identical, so
 * this branch is a code-generation choice, never a numerical one.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::uint32_t mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t& hi)
{
#if defined(AETHER_DEVICE_COMPILE)
    hi = __umulhi(a, b);
    return a * b;
#else
    const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    hi = static_cast<std::uint32_t>(product >> 32);
    return static_cast<std::uint32_t>(product);
#endif
}

/** @brief One Philox4x32 round (Feistel-like word shuffle + two mulhilo). */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Uint4 philoxRound(const Uint4& ctr, std::uint32_t keyX, std::uint32_t keyY)
{
    std::uint32_t hi0 = 0;
    std::uint32_t hi1 = 0;
    const std::uint32_t lo0 = mulhilo32(kPhiloxM4x32_0, ctr.x, hi0);
    const std::uint32_t lo1 = mulhilo32(kPhiloxM4x32_1, ctr.z, hi1);
    return Uint4{ hi1 ^ ctr.y ^ keyX, lo1, hi0 ^ ctr.w ^ keyY, lo0 };
}

/**
 * @brief The Philox4x32-10 bijection: 10 rounds, 9 key bumps between them.
 *
 * 128 bits of counter + 64 bits of key in, 128 bits of output out — a pure
 * function, which is exactly what makes every draw in this module
 * reproducible and independent of thread / launch order.
 */
AETHER_DEVICEHOST() AETHER_FORCEINLINE() Uint4 philox4x32_10(Uint4 c, std::uint32_t keyX, std::uint32_t keyY)
{
    c = philoxRound(c, keyX, keyY); // 1
    for (int round = 1; round < 10; ++round) {
        keyX += kPhiloxW32_0;
        keyY += kPhiloxW32_1;
        c = philoxRound(c, keyX, keyY); // 2 .. 10
    }
    return c;
}

/**
 * @brief A Philox4x32-10 stream positioned at `(seed, subsequence, offset)`.
 *
 * Reproduces cuRAND's `curandStatePhilox4_32_10_t` + `curand_init` +
 * `curand`/`curand4` state machine (see the file docstring). Every draw in
 * `detail/Backend.h` constructs one of these fresh, consumes a fixed number
 * of words, and drops it — the object is a positioning device, never
 * persisted state, so nothing here is shared between samples or threads.
 *
 * The Box-Muller carry slots of cuRAND's own state (`boxmuller_extra`,
 * `boxmuller_flag`, ...) are deliberately ABSENT: they only matter to a
 * generator that is called repeatedly, and every aether draw re-positions
 * from scratch, so the carried second normal is never observable.
 */
class PhiloxStream {
public:
    /** @brief Position the stream, mirroring `curand_init(seed, subsequence, offset, &st)`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE()
        PhiloxStream(std::uint64_t seed, std::uint64_t subsequence, std::uint64_t offset)
        : ctr_{ 0u, 0u, 0u, 0u }
        , out_{ 0u, 0u, 0u, 0u }
        , keyX_{ static_cast<std::uint32_t>(seed) }
        , keyY_{ static_cast<std::uint32_t>(seed >> 32) }
        , state_{ 0u }
    {
        skipaheadSequence_(subsequence);
        skipahead_(offset);
    }

    /** @brief One 32-bit word — cuRAND's `curand(&state)`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::uint32_t next()
    {
        std::uint32_t ret = 0u;
        switch (state_++) {
        case 1: ret = out_.y; break;
        case 2: ret = out_.z; break;
        case 3: ret = out_.w; break;
        default: ret = out_.x; break;
        }
        if (state_ == 4u) {
            increment_();
            out_   = philox4x32_10(ctr_, keyX_, keyY_);
            state_ = 0u;
        }
        return ret;
    }

    /**
     * @brief Four 32-bit words — cuRAND's `curand4(&state)`.
     *
     * Note the deliberate quirk faithfully reproduced from cuRAND: `curand4`
     * regenerates the block and splices across the boundary according to the
     * CURRENT `STATE`, but never changes `STATE` itself. Any deviation here
     * would silently move `standardNormal<double>` off the real cuRAND device stream.
     */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() Uint4 next4()
    {
        const Uint4 tmp = out_;
        increment_();
        out_ = philox4x32_10(ctr_, keyX_, keyY_);
        switch (state_) {
        case 1: return Uint4{ tmp.y, tmp.z, tmp.w, out_.x };
        case 2: return Uint4{ tmp.z, tmp.w, out_.x, out_.y };
        case 3: return Uint4{ tmp.w, out_.x, out_.y, out_.z };
        default: return tmp;
        }
    }

private:
    /** @brief `++ctr` with carry — cuRAND's one-argument `Philox_State_Incr`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void increment_()
    {
        if (++ctr_.x)
            return;
        if (++ctr_.y)
            return;
        if (++ctr_.z)
            return;
        ++ctr_.w;
    }

    /** @brief `ctr += n` on the LOW half — cuRAND's two-argument `Philox_State_Incr`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void incrementLow_(std::uint64_t n)
    {
        const std::uint32_t nlo = static_cast<std::uint32_t>(n);
        std::uint32_t nhi       = static_cast<std::uint32_t>(n >> 32);

        ctr_.x += nlo;
        if (ctr_.x < nlo)
            ++nhi;

        ctr_.y += nhi;
        if (nhi <= ctr_.y)
            return;
        if (++ctr_.z)
            return;
        ++ctr_.w;
    }

    /** @brief `ctr += n` on the HIGH half — cuRAND's `Philox_State_Incr_hi`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void incrementHigh_(std::uint64_t n)
    {
        const std::uint32_t nlo = static_cast<std::uint32_t>(n);
        std::uint32_t nhi       = static_cast<std::uint32_t>(n >> 32);

        ctr_.z += nlo;
        if (ctr_.z < nlo)
            ++nhi;

        ctr_.w += nhi;
    }

    /** @brief cuRAND's `skipahead_sequence(n, &st)` — each subsequence is 2^66 words. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void skipaheadSequence_(std::uint64_t n)
    {
        incrementHigh_(n);
        out_ = philox4x32_10(ctr_, keyX_, keyY_);
    }

    /** @brief cuRAND's `skipahead(n, &st)` — n WORDS, so the intra-quad remainder parks in `state_`. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void skipahead_(std::uint64_t n)
    {
        state_ += static_cast<std::uint32_t>(n & 3ull);
        n /= 4ull;
        if (state_ > 3u) {
            n += 1ull;
            state_ -= 4u;
        }
        incrementLow_(n);
        out_ = philox4x32_10(ctr_, keyX_, keyY_);
    }

    Uint4 ctr_;
    Uint4 out_;
    std::uint32_t keyX_;
    std::uint32_t keyY_;
    std::uint32_t state_;
};

} // namespace detail
} // namespace random
} // namespace aether

/// @endcond
