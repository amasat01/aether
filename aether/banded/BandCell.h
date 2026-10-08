// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandCell.h
 * @brief `BandCell` — a 16-byte biased-`Band` storage cell.
 *
 * @section why_bandcell Relationship to `BandCell8`
 * `BandCell8` (`BandCell8.h`) is the 8-byte, tier-1-windowed storage word
 * (`[2^-94, 2^126)`, fixed-point continuation encoding). `BandCell` is a
 * different, wider-range storage shape: three raw `Band` limbs re-centred
 * so `|hi|` sits in `[1, 2)`, plus an `int32_t` base-2 bias — twice the
 * memory footprint (16 bytes vs 8) in exchange for FP32's full exponent
 * range (`|value| < 2^128`, no tier-1 window) at rest.
 *
 * @section scope_bandcell Placement
 * `BandCell` (the type) lives at `aether::banded` scope; its primitives
 * (`cellFromBand`/`bandFromCell`/`loadCell`/`storeCell`) live in
 * `aether::banded::detail`, alongside `cellFromBand`'s sibling
 * `scalePow2f`, matching the split `Band`/`BandedReal` already use.
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

#ifdef AETHER_DEBUG_MODE
#include <cassert>
#endif

namespace aether {
namespace banded {

// =====================================================================
//  BandCell -- the storage face of the banded carrier (wide-range variant)
// =====================================================================

/// @brief 16-byte storage cell: three signed `Band` limbs plus a base-2 bias.
///
/// Logical value is `Band{hi, lo, tail} * 2^bias`. Produced by
/// `detail::cellFromBand`, consumed by `detail::bandFromCell`, and moved in
/// and out of memory by `detail::loadCell`/`detail::storeCell`.
struct alignas(16) BandCell {
    float hi;      ///< leading limb at rest; `|hi|` in `[1, 2)`, sign native
    float lo;      ///< second limb at rest, ~`2^-24`
    float tail;    ///< residue limb at rest, ~`2^-48`
    std::int32_t bias; ///< base-2 exponent the limbs were re-centred by
};

static_assert(sizeof(BandCell) == 16, "BandCell must be exactly one 128-bit memory transaction");
static_assert(alignof(BandCell) == 16, "BandCell must be naturally aligned for a single 128-bit access");
static_assert(std::is_trivially_copyable_v<BandCell>,
    "BandCell must stay trivially copyable (SoA buffers, cudaMemcpy, bit-cast punning)");

namespace detail {

// =====================================================================
//  Punned 128-bit access
// =====================================================================

/// @brief Opaque 16-byte view of a `BandCell`, used only to move one.
struct alignas(16) BandCellRaw {
    std::uint32_t x; ///< bits of `hi`
    std::uint32_t y; ///< bits of `lo`
    std::uint32_t z; ///< bits of `tail`
    std::uint32_t w; ///< bits of `bias`
};

static_assert(sizeof(BandCellRaw) == 16 && alignof(BandCellRaw) == 16,
    "BandCellRaw must have BandCell's exact footprint");
static_assert(std::is_trivially_copyable_v<BandCellRaw>);

/// @brief Read one cell from memory as a single 128-bit transaction. Going
/// through the raw view rather than a direct `memcpy` from the pointer keeps
/// the guarantee independent of layout coincidence (a plain `memcpy` from the
/// pointer can emit sixteen 1-byte loads on sm_61).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell loadCell(const BandCell* p)
{
    const BandCellRaw raw = *reinterpret_cast<const BandCellRaw*>(p);
    BandCell c;
    std::memcpy(&c, &raw, sizeof(BandCell));
    return c;
}

/// @brief Write one cell to memory as a single 128-bit transaction. @see loadCell.
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void storeCell(BandCell* p, BandCell c)
{
    BandCellRaw raw;
    std::memcpy(&raw, &c, sizeof(BandCellRaw));
    *reinterpret_cast<BandCellRaw*>(p) = raw;
}

// =====================================================================
//  Re-centring
// =====================================================================

/// @brief Multiply every limb by `2^k`, exactly, for any `k` in `[-252, 254]`.
/// Two exact power-of-two multiplies (via the shipped `scalePow2f`), split so
/// neither intermediate factor can be flushed to zero by a subnormal-hostile
/// build even at this file's largest `|k| = 127`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandScalePow2Split(Band b, int k)
{
    const int k1    = k / 2;
    const int k2    = k - k1;
    const float s1 = intAsFloat((127 + k1) << 23);
    const float s2 = intAsFloat((127 + k2) << 23);
    return scalePow2f(scalePow2f(b, s1), s2);
}

/// @brief `Band` -> `BandCell`: derive the bias from `hi` and re-centre.
///
/// @par Exactly two outcomes
/// The zero cell `{0, 0, 0, 0}`, or `|hi|` in `[1, 2)`. Nothing else.
///
/// @par Input contract
/// `b` is a certified `Band` (`|lo| <= ulp(hi)`); a zero `hi` under that
/// invariant means the whole band is zero. Run `normalize` first if a chain
/// can produce an un-normalized band with a zero `hi` over nonzero lower
/// limbs.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandCell cellFromBand(Band b)
{
    const int bexp = (floatAsInt(b.hi) >> 23) & 0xFF;
    if (bexp == 0)
        return BandCell{ 0.0f, 0.0f, 0.0f, 0 };
    const int e  = bexp - 127;
    const Band r = bandScalePow2Split(b, -e);
    return BandCell{ r.hi, r.lo, r.tail, e };
}

/// @brief `BandCell` -> `Band`: fold the bias back onto the limbs. Bit-exact
/// inverse of `cellFromBand` for every value inside the carrier's working
/// envelope.
///
/// @par Precondition
/// The logical value must land inside the `Band` working envelope — a
/// property of the consumer's declared exponent ranges (`BandedLimits.h`
/// admission), not re-tested per element. In `AETHER_DEBUG_MODE` builds the
/// bias is asserted to be one this file could have produced.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromCell(const BandCell& c)
{
#ifdef AETHER_DEBUG_MODE
    assert(c.bias >= -126 && c.bias <= 128
        && "BandCell::bias outside the range cellFromBand can produce -- "
           "uninitialized, byte-swapped, or not a BandCell at all");
#endif
    return bandScalePow2Split(Band{ c.hi, c.lo, c.tail }, c.bias);
}

} // namespace detail
} // namespace banded
} // namespace aether
