// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Format.h
 * @brief Host-only pretty-printing for `DType` (L2).
 *
 * Kept OUT of `DType.h` deliberately: `DType.h` is `AETHER_DEVICEHOST()`-safe
 * (no `std::` strings), and this header's `std::string`/`std::ostream` usage
 * must never be forced onto a device-compiled translation unit that only
 * wants the plain data type. Include this header explicitly when you need
 * `to_string()`/`operator<<`.
 */

#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>

#include "aether/dtype/DType.h"
#include "aether/dtype/dlpack.h"

namespace aether {

/** @brief Human-readable name for a DLPack type code (`"unknown"` for any code not listed). */
inline std::string to_string(DLDataTypeCode code)
{
    switch (code) {
    case kDLInt: return "int";
    case kDLUInt: return "uint";
    case kDLFloat: return "float";
    case kDLOpaqueHandle: return "opaque";
    case kDLBfloat: return "bfloat";
    case kDLComplex: return "complex";
    case kDLBool: return "bool";
    default: return "unknown";
    }
}

/** @brief Render a `DType` as `"<code><bits>"`, with an `"x<lanes>"` suffix when `lanes() > 1`. */
inline std::string to_string(const DType& dt)
{
    std::string s = to_string(static_cast<DLDataTypeCode>(dt.code())) + std::to_string(static_cast<unsigned>(dt.bits()));
    if (dt.lanes() > 1)
        s += "x" + std::to_string(static_cast<unsigned>(dt.lanes()));
    return s;
}

/** @brief Stream a `DType` via `to_string()`. */
inline std::ostream& operator<<(std::ostream& os, const DType& dt)
{
    return os << to_string(dt);
}

// ---------------------------------------------------------------------------
//  Codec — the SEMANTIC tag a bit-transparent dtype cannot carry
// ---------------------------------------------------------------------------

/**
 * @brief What the BYTES of an element mean, when the `DType` alone does not say.
 *
 * A `DType` answers "how wide is this element and how should a machine move
 * it". For every native scalar that is also the whole answer. It is NOT the
 * whole answer for an emulated real: `aether::banded::BandedReal` exports as
 * `(kDLUInt, 64, 1)` — the raw codec WORD, bit-transparently — because DLPack has
 * no code for a 56-bit three-limb carrier and fabricating one would hand every
 * importer a number it has to guess at (`aether/dtype/DType.h`'s `dtype_of`
 * docstring carries that decision).
 *
 * `Codec` is where the missing half lives. An exporter that CAN carry a semantic
 * tag reads it from `codec_of<T>` and passes it on; an importer that sees only
 * the `uint64` gets exactly what it should get — opaque words it will not
 * silently do arithmetic on. That is the documented contract, not a gap.
 *
 * Host-only, like everything else in this header: the tag is export metadata, it
 * is never consulted from a kernel, and `DType.h` stays device-safe by not
 * knowing about it.
 */
enum class Codec : std::uint8_t {
    /** @brief The element IS its dtype — every native scalar. */
    None = 0,
    /** @brief `aether::banded::BandedReal` / `BandCell8`: one 64-bit banded
     *         codec word, depth 56, tier-1 window `[2^-94, 2^126)`, at array
     *         bias 0. Decode with `aether::banded::detail::bandFromCell8`. */
    BandCell8 = 1,
};

/** @brief Human-readable name for a `Codec`. */
inline std::string to_string(Codec c)
{
    switch (c) {
    case Codec::None: return "none";
    case Codec::BandCell8: return "bandcell8";
    default: return "unknown";
    }
}

/**
 * @brief The semantic tag for element type `T` — `Codec::None` for every type
 *        whose dtype already says what it is.
 *
 * Deliberately a total function with a `None` default rather than a closed chain
 * with a `static_assert` else-branch: `dtype_of<T>` is the gate that refuses an
 * unsupported element type, and making a second gate out of the tag would mean
 * every new scalar has to be added in two places to compile at all.
 */
template<class T>
constexpr Codec codec_of()
{
    if constexpr (std::is_same_v<T, banded::BandedReal>
        || std::is_same_v<T, banded::BandCell8>) {
        return Codec::BandCell8;
    } else {
        return Codec::None;
    }
}

/** @brief Render a `DType` with its semantic tag appended when it has one, e.g.
 *         `"uint64[bandcell8]"`. A tagless element renders exactly as
 *         `to_string(dt)` does, so nothing about the existing spelling moves. */
inline std::string to_string(const DType& dt, Codec c)
{
    return (c == Codec::None) ? to_string(dt) : (to_string(dt) + "[" + to_string(c) + "]");
}

} // namespace aether
