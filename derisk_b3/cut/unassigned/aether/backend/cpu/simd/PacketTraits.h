// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketTraits.h
 * @brief Compile-time SIMD ISA detection + packet-width selection.
 *
 * Host-only header — no `AETHER_DEVICE` code — but must compile in
 * CUDA-mode builds (host side).
 */

#include <cstddef>
#include <type_traits>

// ISA detection headers (host-only; SIMD intrinsics are never used from
// device code in this library).
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h> // SSE4.1: rounding intrinsics some toolchains want pulled in here
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace aether {
namespace simd {

/** @brief SIMD ISA levels this library recognizes, ordered weakest-first. */
enum class ISA : unsigned { Scalar = 0, SSE2 = 1, AVX2 = 2, AVX512 = 3 };

/** @brief The highest ISA level available at compile time (`-march=...`). */
inline constexpr ISA DetectedISA =
#if defined(__AVX512F__)
    ISA::AVX512;
#elif defined(__AVX2__)
    ISA::AVX2;
#elif defined(__SSE2__)
    ISA::SSE2;
#else
    ISA::Scalar;
#endif

/** @brief SIMD register width in bytes for `DetectedISA` (0 for Scalar). */
inline constexpr std::size_t RegisterBytes = (DetectedISA == ISA::AVX512) ? std::size_t{ 64 }
    : (DetectedISA == ISA::AVX2)                                          ? std::size_t{ 32 }
    : (DetectedISA == ISA::SSE2)                                          ? std::size_t{ 16 }
                                                                           : std::size_t{ 0 };

/**
 * @brief Number of `DataT` elements that fit in one SIMD register — the
 *        packet width `Packet<DataT, PreferredWidth<DataT>>` (the "native"
 *        packet) uses. Falls back to `1` (the scalar-fallback `Packet`
 *        specialization) when no SIMD ISA is detected or `DataT` is not
 *        `float`/`double`.
 */
template<class DataT>
inline constexpr std::size_t PreferredWidth
    = (RegisterBytes > 0 && (std::is_same_v<DataT, double> || std::is_same_v<DataT, float>))
    ? static_cast<std::size_t>(RegisterBytes / sizeof(DataT))
    : std::size_t{ 1 };

/** @brief Compile-time SIMD traits bundle for `DataT`. */
template<class DataT>
struct PacketTraits {
    static constexpr std::size_t Width = PreferredWidth<DataT>;
    static constexpr ISA Level = DetectedISA;
    /** @brief `true` when a real (non-scalar-fallback) SIMD width is active. */
    static constexpr bool HasSIMD = (Width > 1);
};

} // namespace simd
} // namespace aether
