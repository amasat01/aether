// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file f32cReference.h
 * @brief TEST-ONLY reference implementation of the F-HI32 codec, plus two
 *        deliberately defective twins.
 *
 * NOT PRODUCTION CODE. Nothing in this file is installed, nothing in the
 * shipped headers includes it, and no consumer may name it. It exists so that
 * `BandCell8`'s codec can be compared against an INDEPENDENT body rather than
 * against itself. It has no library include of any kind — by construction,
 * since a reference that compiled against the subject would not be one.
 *
 * `namespace f32c` below is a standalone transcription of the codec this
 * repo's depth, cost and window numbers were measured on. The unused arms
 * (`dfEncode`, `dfDecode`, `decodeCF`, `pow2`) were dropped, the CUDA
 * `__forceinline__` decoration was dropped because this header is host-only,
 * and the `Band3` struct kept its own name so that a production type can
 * never be substituted for it by accident.
 *
 * `decodeEscaped`'s two exponent BASES are `126` and `-94`, closing the
 * two-binade gap and the stolen canonical-zero code point that the naive
 * bases (`128`/`-95`) leave behind. The tags, the field widths, the
 * significand split, the zero case and the sign handling are a direct
 * transcription. **The TIER-1 codec below -- `encode`, `decode`,
 * `decodeHiOnly`, `isEscape`, `encodeRebias`, `decodeRebias` -- is the
 * load-bearing bit-identity oracle for the production codec.**
 *
 * WHY A COPY AND NOT AN INCLUDE
 * -----------------------------
 * The reference must never compile against the library it certifies, so it
 * is a self-contained copy rather than a shared header.
 *
 * THE DEFECTIVE TWINS
 * -------------------
 * `namespace f32cdefect` holds two copies of the encoder and the escape decoder
 * with one contract-bearing line removed from each:
 *
 *   S1  `encodeNoBorrowFixup`  -- drops the binade-borrow shift. This is the
 *       defect the `double` round-trip gate originally found, on 484 rows of
 *       3872: when the leading limb is an exact power of two and the lower
 *       limbs cancel, stepping one float down halves the unit in the last
 *       place, and the remainder has to be re-expressed in the halved unit.
 *   S2  `decodeEscapedNoZeroCase` -- drops the all-zero-word special case, so
 *       canonical zero decodes as `2^-94` instead.
 *
 * They are not commented-out code and they are not a build variant: they are
 * INPUTS to `SeededDefectsAreCaughtByThisCorpus`, which asserts that the
 * battery's corpus contains rows on which each of them is visibly wrong. A
 * corpus that cannot see a known defect certifies nothing, so the count of rows
 * that catch each defect is asserted to be non-zero and printed.
 */

#include <cstdint>

namespace aether_tests {
namespace f32c {

// ---------------------------------------------------------------------------
//  Verbatim probe codec (see PROVENANCE above)
// ---------------------------------------------------------------------------

inline int32_t f2i(float f)
{
    int32_t r;
    __builtin_memcpy(&r, &f, 4);
    return r;
}
inline float i2f(int32_t i)
{
    float r;
    __builtin_memcpy(&r, &i, 4);
    return r;
}

/// The metadata-free 3-limb carrier, `aether::banded::Band` field for field,
/// redeclared locally so the reference never compiles against the library it is
/// the oracle for.
struct Band3 {
    float hi, lo, tail;
};

/// A Band plus the power of two it has to be multiplied by -- what an
/// out-of-range value MUST decode to.
struct BiasedBand {
    Band3 b;
    int32_t bias;
};

inline int32_t rnI32(float x)
{
    return static_cast<int32_t>(__builtin_lrintf(x));
}
inline int64_t rnI64(float x)
{
    return static_cast<int64_t>(__builtin_llrintf(x));
}

/// 2^55 as a float, the scale that puts significand bit 55 at integer unit 1.
inline float kScale55() { return i2f((127 + 55) << 23); }

inline uint64_t encode(const Band3& b)
{
    const int32_t bh  = f2i(b.hi);
    const int32_t sgn = bh & static_cast<int32_t>(0x80000000);
    const int32_t ex  = bh & 0x7F800000;
    const float sig   = i2f((0x7F000000 - ex) | sgn);

    const float K  = kScale55();
    const float ul = b.lo * sig;
    const float ut = b.tail * sig;

    int64_t S = rnI64(ul * K) + static_cast<int64_t>(rnI32(ut * K));

    const uint32_t bw = static_cast<uint32_t>(S < 0);
    const uint32_t cross
        = bw & static_cast<uint32_t>((bh & 0x7FFFFF) == 0);
    S <<= cross;

    const uint32_t w1 = static_cast<uint32_t>(bh) - bw;
    return (static_cast<uint64_t>(w1) << 32) | static_cast<uint32_t>(S);
}

inline Band3 decode(uint64_t w)
{
    const uint32_t w1 = static_cast<uint32_t>(w >> 32);
    const uint32_t w0 = static_cast<uint32_t>(w);

    const int32_t se = static_cast<int32_t>(w1) & static_cast<int32_t>(0xFF800000);
    const int32_t eL = se - (23 << 23);
    const int32_t eT = se - (32 << 23);

    const int32_t L = static_cast<int32_t>(w0 >> 9);
    const int32_t T = static_cast<int32_t>(w0 & 0x1FF);

    Band3 b;
    b.hi   = i2f(static_cast<int32_t>(w1));
    b.lo   = i2f(eL | L) - i2f(eL);
    b.tail = i2f(eT | T) - i2f(eT);
    return b;
}

inline float decodeHiOnly(uint64_t w) { return i2f(static_cast<int32_t>(w >> 32)); }

inline bool isEscape(uint32_t w1)
{
    return static_cast<uint32_t>((w1 & 0x7F800000u) - 0x00800000u) >= 0x7F000000u;
}

inline Band3 decodeChecked(uint64_t w, uint32_t& esc)
{
    esc = static_cast<uint32_t>(isEscape(static_cast<uint32_t>(w >> 32)));
    return decode(w);
}

inline BiasedBand decodeEscaped(uint64_t w)
{
    const uint32_t w1      = static_cast<uint32_t>(w >> 32);
    const uint32_t w0      = static_cast<uint32_t>(w);
    const uint32_t tagHigh = (w1 >> 23) & 0xFF;
    const int32_t extExp   = static_cast<int32_t>((w1 >> 12) & 0x7FF);
    const uint32_t sigHi   = w1 & 0xFFF;

    BiasedBand r;
    // The finalized bases are 126 / -94 (an earlier probe used 128 / -95). @see the
    // header block; everything else on this line's either side is the probe's.
    r.bias = (tagHigh == 0xFF) ? (extExp + 126) : (-extExp - 94);
    const int32_t m0 = static_cast<int32_t>(0x800000u | (sigHi << 11) | (w0 >> 21));
    const int32_t m1 = static_cast<int32_t>(((w0 >> 0) & 0x1FFFFF) << 2);
    r.b.hi   = i2f((127 << 23) | (m0 & 0x7FFFFF));
    r.b.lo   = i2f(((127 - 23) << 23) | (m1 & 0x7FFFFF)) - i2f((127 - 23) << 23);
    r.b.tail = 0.0f;
    if ((w1 | w0) == 0u) {
        r.b.hi = 0.0f;
        r.b.lo = 0.0f;
        r.bias = 0;
    }
    if (static_cast<int32_t>(w1) < 0) {
        r.b.hi = -r.b.hi;
        r.b.lo = -r.b.lo;
    }
    return r;
}

inline Band3 decodeRebias(uint64_t w, int32_t arrayBias)
{
    const uint32_t w1
        = static_cast<uint32_t>(w >> 32) + (static_cast<uint32_t>(arrayBias) << 23);
    const uint32_t w0 = static_cast<uint32_t>(w);
    const int32_t se  = static_cast<int32_t>(w1) & static_cast<int32_t>(0xFF800000);
    const int32_t eL  = se - (23 << 23);
    const int32_t eT  = se - (32 << 23);
    const int32_t L   = static_cast<int32_t>(w0 >> 9);
    const int32_t T   = static_cast<int32_t>(w0 & 0x1FF);
    Band3 b;
    b.hi   = i2f(static_cast<int32_t>(w1));
    b.lo   = i2f(eL | L) - i2f(eL);
    b.tail = i2f(eT | T) - i2f(eT);
    return b;
}

inline uint64_t encodeRebias(const Band3& b, int32_t arrayBias)
{
    const uint64_t w = encode(b);
    return w
        - (static_cast<uint64_t>(static_cast<uint32_t>(arrayBias) << 23) << 32);
}

} // namespace f32c

// ---------------------------------------------------------------------------
//  The defective twins -- inputs to the corpus's own non-vacuity gate
// ---------------------------------------------------------------------------
namespace f32cdefect {

using f32c::Band3;
using f32c::BiasedBand;
using f32c::f2i;
using f32c::i2f;

/// S1: the binade-borrow fixup removed. Everything else is `f32c::encode`.
inline uint64_t encodeNoBorrowFixup(const Band3& b)
{
    const int32_t bh  = f2i(b.hi);
    const int32_t sgn = bh & static_cast<int32_t>(0x80000000);
    const int32_t ex  = bh & 0x7F800000;
    const float sig   = i2f((0x7F000000 - ex) | sgn);

    const float K  = f32c::kScale55();
    const float ul = b.lo * sig;
    const float ut = b.tail * sig;

    int64_t S = f32c::rnI64(ul * K) + static_cast<int64_t>(f32c::rnI32(ut * K));

    const uint32_t bw = static_cast<uint32_t>(S < 0);
    // DEFECT S1: `S <<= bw & ((bh & 0x7FFFFF) == 0);` is missing here.
    const uint32_t w1 = static_cast<uint32_t>(bh) - bw;
    return (static_cast<uint64_t>(w1) << 32) | static_cast<uint32_t>(S);
}

/// S2: the canonical-zero special case removed. Everything else is
/// `f32c::decodeEscaped`.
inline BiasedBand decodeEscapedNoZeroCase(uint64_t w)
{
    const uint32_t w1      = static_cast<uint32_t>(w >> 32);
    const uint32_t w0      = static_cast<uint32_t>(w);
    const uint32_t tagHigh = (w1 >> 23) & 0xFF;
    const int32_t extExp   = static_cast<int32_t>((w1 >> 12) & 0x7FF);
    const uint32_t sigHi   = w1 & 0xFFF;

    BiasedBand r;
    r.bias = (tagHigh == 0xFF) ? (extExp + 126) : (-extExp - 94);
    const int32_t m0 = static_cast<int32_t>(0x800000u | (sigHi << 11) | (w0 >> 21));
    const int32_t m1 = static_cast<int32_t>(((w0 >> 0) & 0x1FFFFF) << 2);
    r.b.hi   = i2f((127 << 23) | (m0 & 0x7FFFFF));
    r.b.lo   = i2f(((127 - 23) << 23) | (m1 & 0x7FFFFF)) - i2f((127 - 23) << 23);
    r.b.tail = 0.0f;
    // DEFECT S2: `if ((w1 | w0) == 0u) { ... }` is missing here.
    if (static_cast<int32_t>(w1) < 0) {
        r.b.hi = -r.b.hi;
        r.b.lo = -r.b.lo;
    }
    return r;
}

} // namespace f32cdefect
} // namespace aether_tests
