// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DType.h
 * @brief `DType` (L2): a thin, `AETHER_DEVICEHOST()`-safe wrapper around
 *        DLPack's `DLDataType`, plus the `dtype_of<T>()` trait mapping a C++
 *        scalar type to its `DType`.
 *
 * No `std::` strings here by design (a lean device header) — host-only
 * pretty-printing lives in the separate `aether/dtype/Format.h`.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "aether/dtype/dlpack.h"
#include "aether/macros.h"

namespace aether {

/* Forward declarations ONLY — no definitions, no include of `aether/banded/`.
 * `dtype_of` needs to TELL these three types apart, and `std::is_same_v` on an
 * incomplete type is well-formed, so naming them costs this header (which the
 * umbrella DOES pull into every TU) nothing at all. Definitions live in
 * `aether/banded/`, whose umbrella is deliberately not reachable from
 * `aether/aether.h` — see `aether/banded/banded.h`'s diet section. */
namespace banded {
struct Band;
struct BandCell8;
struct BandedReal;
} // namespace banded

/**
 * @brief Thin wrapper around `DLDataType` (code/bits/lanes) — constexpr
 *        constructible and safe to use in `AETHER_DEVICEHOST()` code.
 */
struct DType {
    /** @brief The wrapped DLPack data-type descriptor. */
    DLDataType raw{};

    // No AETHER_DEVICEHOST() here: a `= default` special member on its
    // first declaration is automatically host+device eligible under nvcc,
    // which warns (#20012-D) that an explicit annotation is redundant.
    constexpr DType() = default;

    /** @brief Construct from a raw `(code, bits, lanes)` triple. */
    AETHER_DEVICEHOST() constexpr DType(
        std::uint8_t code, std::uint8_t bits, std::uint16_t lanes = 1)
        : raw{ code, bits, lanes }
    {
    }

    /** @brief Construct by wrapping an existing `DLDataType` verbatim. */
    AETHER_DEVICEHOST() constexpr explicit DType(DLDataType dl)
        : raw(dl)
    {
    }

    /** @brief DLPack type code (see `DLDataTypeCode` in dlpack.h). */
    AETHER_DEVICEHOST() constexpr std::uint8_t code() const { return raw.code; }
    /** @brief Number of bits per lane. */
    AETHER_DEVICEHOST() constexpr std::uint8_t bits() const { return raw.bits; }
    /** @brief Number of lanes (1 for a plain scalar). */
    AETHER_DEVICEHOST() constexpr std::uint16_t lanes() const { return raw.lanes; }

    /** @brief Total size in bytes: `bits() * lanes() / 8`. */
    AETHER_DEVICEHOST() constexpr std::size_t size_bytes() const
    {
        return (static_cast<std::size_t>(raw.bits) * static_cast<std::size_t>(raw.lanes)) / 8;
    }
};

AETHER_DEVICEHOST() constexpr bool operator==(const DType& a, const DType& b)
{
    return a.raw.code == b.raw.code && a.raw.bits == b.raw.bits && a.raw.lanes == b.raw.lanes;
}

AETHER_DEVICEHOST() constexpr bool operator!=(const DType& a, const DType& b)
{
    return !(a == b);
}

/**
 * @brief The codegen-facing Int value type: a plain `std::int64_t` alias
 *        for element data that happens to be integral (loop-carried
 *        counters, index arrays a kernel computes over, …).
 *
 * Deliberately distinct from `aether::offset_t` (`aether/index/Offset.h`),
 * which is the 32-bit address/offset-arithmetic width for register
 * pressure — `offset_t` is never a value a kernel computes with, only a
 * position it computes through. `int_t` carries no such constraint: it is
 * an ordinary element dtype like `double`/`float` (64-bit, matching
 * `dtype_of<int_t>()` below to `kDLInt`/64 — already a supported branch of
 * `dtype_of`, unchanged by this alias).
 */
using int_t = std::int64_t;

/**
 * @brief Map a C++ scalar type to its `DType`.
 *
 * Supported: `double`, `float`, `bool` (-> `kDLBool`, bits=8),
 * `std::int64_t`/`std::int32_t`, `std::uint64_t`/`std::uint32_t`/`std::uint8_t`,
 * plus the two banded STORAGE types (see below). Any other `T` is a
 * `static_assert` failure (L2) rather than a silent fallback.
 *
 * @par ★ The banded codec exports BIT-TRANSPARENTLY as `uint64`
 * `BandedReal` and `BandCell8` map to `(kDLUInt, 64, 1)` — the raw 64-bit codec
 * WORD, reinterpreted under a defined view-layer contract. This is deliberate
 * and it is the only honest answer available:
 *  - DLPack has no type code for a 56-bit three-limb emulated real, and
 *    FABRICATING one would hand every importer a code it must guess at. An
 *    importer that sees `uint64` and does not know the semantic tag gets exactly
 *    what it should get — opaque words it will not silently do arithmetic on.
 *  - The size is right (8 bytes, naturally aligned), the buffer is
 *    byte-transparent, and a round trip through any DLPack consumer that merely
 *    MOVES the buffer is lossless.
 * The SEMANTICS ride `aether/dtype/Format.h`'s `Codec` vocabulary
 * (`codec_of<T>`), which is where an exporter that can carry a semantic tag
 * reads what these words mean. That header is host-only and out of the umbrella,
 * exactly like the rest of its pretty-printing surface.
 *
 * @par `Band` has NO dtype, and says why
 * `Band` is a register carrier with no memory form at all, so there is nothing
 * to export. Asking for its dtype gets a message pointing at `BandedReal` rather
 * than the generic unsupported-type text — the mistake is specific enough to
 * deserve a specific answer.
 */
template<class T>
AETHER_DEVICEHOST() constexpr DType dtype_of()
{
    if constexpr (std::is_same_v<T, double>) {
        return DType(static_cast<std::uint8_t>(kDLFloat), 64, 1);
    } else if constexpr (std::is_same_v<T, float>) {
        return DType(static_cast<std::uint8_t>(kDLFloat), 32, 1);
    } else if constexpr (std::is_same_v<T, bool>) {
        return DType(static_cast<std::uint8_t>(kDLBool), 8, 1);
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return DType(static_cast<std::uint8_t>(kDLInt), 64, 1);
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DType(static_cast<std::uint8_t>(kDLInt), 32, 1);
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return DType(static_cast<std::uint8_t>(kDLUInt), 64, 1);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return DType(static_cast<std::uint8_t>(kDLUInt), 32, 1);
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return DType(static_cast<std::uint8_t>(kDLUInt), 8, 1);
    } else if constexpr (std::is_same_v<T, banded::BandedReal>
        || std::is_same_v<T, banded::BandCell8>) {
        // BIT-TRANSPARENT export of the codec word. See this function's
        // docstring, and `codec_of<T>` in aether/dtype/Format.h for the
        // semantic tag that says what the words mean.
        return DType(static_cast<std::uint8_t>(kDLUInt), 64, 1);
    } else if constexpr (std::is_same_v<T, banded::Band>) {
        static_assert(sizeof(T) == 0,
            "aether::dtype_of<banded::Band>: `Band` is a REGISTER carrier with no "
            "memory form — there is nothing to export. Export the STORAGE type "
            "`aether::banded::BandedReal` instead (it maps to uint64, the codec "
            "word, bit-transparently).");
    } else {
        static_assert(sizeof(T) == 0,
            "aether::dtype_of<T>: unsupported T — no DLPack DType mapping is defined for it");
    }
}

} // namespace aether
