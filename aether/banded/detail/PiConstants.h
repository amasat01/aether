// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether/banded/detail/PiConstants.h — the one definition of pi/2 on the Band carrier.
//
// Cody-Waite limbs of pi/2 (bit patterns, so the value is exact by construction), shared by the forward trig
// reduction (`BandTrig.h`, which also uses the 4th limb) and the inverse family (`BandInvTrig.h`, which
// reconstructs with `bandPiO2()`), so both families use the same limbs and cannot drift onto two different circles.
#pragma once
#include <cstdint>
#include "aether/banded/Band.h"
#include "aether/macros.h"

namespace aether::banded::detail {

inline constexpr std::uint32_t kBandPiO2CW1Bits = 0x3FC90FDBu; ///< ~2^0.65
inline constexpr std::uint32_t kBandPiO2CW2Bits = 0xB33BBD2Eu; ///< ~-2^-24.45
inline constexpr std::uint32_t kBandPiO2CW3Bits = 0xA6F72CEDu; ///< ~-2^-49.05
inline constexpr std::uint32_t kBandPiO2CW4Bits = 0x194C5170u; ///< ~2^-76.33

/// @brief `pi/2` as a `Band`, from the same limbs `trigReduce` reduces
/// against, so the forward and inverse families cannot drift onto two
/// different circles. The full `pi` is this value scaled by a power of
/// two elsewhere, which is exact: `pi - pi/2 == pi/2` limb for limb.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandPiO2()
{
    return Band{ intAsFloat(static_cast<int>(kBandPiO2CW1Bits)),
        intAsFloat(static_cast<int>(kBandPiO2CW2Bits)),
        intAsFloat(static_cast<int>(kBandPiO2CW3Bits)) };
}

} // namespace aether::banded::detail
