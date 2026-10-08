// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketMask.h
 * @brief SIMD lane masks for conditional (masked) load/store — the tail-
 *        packet mechanism.
 *
 * `PacketMask<T,Width>` wraps platform-specific mask storage (AVX-512:
 * `__mmask8`/`__mmask16`; AVX2/SSE2/scalar: an integer bitmask, converted
 * to a SIMD blend/gather mask at the `Packet` level). Host-only header —
 * no `AETHER_DEVICE` code — but must compile in CUDA-mode builds (host
 * side).
 */

#include <cstddef>
#include <type_traits>

#include "aether/backend/cpu/simd/PacketTraits.h"

namespace aether {
namespace simd {

/**
 * @brief SIMD lane mask — primary template for `Width > 1`.
 * @tparam DataT  The scalar type whose packet this mask accompanies.
 * @tparam Width  Number of SIMD lanes (default: `PreferredWidth<DataT>`).
 */
template<class DataT, std::size_t Width = PreferredWidth<DataT>>
struct PacketMask {
#if defined(__AVX512F__)
    using StorageT = std::conditional_t<(Width <= 8), __mmask8, __mmask16>;
    StorageT mask_;

    PacketMask()
        : mask_{}
    {
    }
    explicit PacketMask(StorageT m)
        : mask_(m)
    {
    }

    static PacketMask allTrue() { return PacketMask{ static_cast<StorageT>((1u << Width) - 1u) }; }
    static PacketMask allFalse() { return PacketMask{ StorageT{ 0 } }; }
    static PacketMask firstN(std::size_t n) { return PacketMask{ static_cast<StorageT>((1u << n) - 1u) }; }

    bool anyTrue() const { return mask_ != 0; }
    bool isAllTrue() const { return mask_ == static_cast<StorageT>((1u << Width) - 1u); }

    PacketMask operator&(const PacketMask& o) const { return PacketMask{ static_cast<StorageT>(mask_ & o.mask_) }; }
    PacketMask operator|(const PacketMask& o) const { return PacketMask{ static_cast<StorageT>(mask_ | o.mask_) }; }
    PacketMask operator~() const
    {
        return PacketMask{ static_cast<StorageT>(~mask_ & static_cast<StorageT>((1u << Width) - 1u)) };
    }

    bool lane(std::size_t k) const { return (mask_ >> k) & 1u; }

#else
    // SSE2/AVX2 and scalar fallback: store the mask as a plain integer
    // bitmask; conversion to a SIMD blend/gather mask happens at the
    // Packet level (maskLoad/maskStore).
    using StorageT = unsigned;
    StorageT mask_;

    PacketMask()
        : mask_{ 0 }
    {
    }
    explicit PacketMask(unsigned m)
        : mask_(m)
    {
    }

    static PacketMask allTrue() { return PacketMask{ (1u << Width) - 1u }; }
    static PacketMask allFalse() { return PacketMask{ 0u }; }
    static PacketMask firstN(std::size_t n) { return PacketMask{ (1u << n) - 1u }; }

    bool anyTrue() const { return mask_ != 0; }
    bool isAllTrue() const { return mask_ == (1u << Width) - 1u; }

    PacketMask operator&(const PacketMask& o) const { return PacketMask{ mask_ & o.mask_ }; }
    PacketMask operator|(const PacketMask& o) const { return PacketMask{ mask_ | o.mask_ }; }
    PacketMask operator~() const { return PacketMask{ ~mask_ & ((1u << Width) - 1u) }; }

    bool lane(std::size_t k) const { return (mask_ >> k) & 1u; }
#endif
};

/** @brief Scalar specialization (`Width == 1`): zero-overhead `bool` mask. */
template<class DataT>
struct PacketMask<DataT, 1> {
    bool mask_;

    PacketMask()
        : mask_{ false }
    {
    }
    explicit PacketMask(bool m)
        : mask_(m)
    {
    }

    static PacketMask allTrue() { return PacketMask{ true }; }
    static PacketMask allFalse() { return PacketMask{ false }; }
    static PacketMask firstN(std::size_t n) { return PacketMask{ n > 0 }; }

    bool anyTrue() const { return mask_; }
    bool isAllTrue() const { return mask_; }

    PacketMask operator&(const PacketMask& o) const { return PacketMask{ mask_ && o.mask_ }; }
    PacketMask operator|(const PacketMask& o) const { return PacketMask{ mask_ || o.mask_ }; }
    PacketMask operator~() const { return PacketMask{ !mask_ }; }

    bool lane(std::size_t) const { return mask_; }
};

} // namespace simd
} // namespace aether
