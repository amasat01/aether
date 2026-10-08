// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file carrier_egress.h
 * @brief The WORKING-CARRIER `double` egress used by the cert TUs — the
 *        counterpart of `detail::bandFromIEEE`, and NOT the storage terminal.
 *
 * @section why Why this exists
 * `Ff1::toDouble()`/`Ff2::toDouble()` are the STORAGE terminals (`Ff2.h` ->
 * `toBandedReal()` -> `BandedReal(Band)` -> `detail::cell8FromBand`), whose
 * precondition is `cell8BandIsStorable`. Handing them a below-window `Band`
 * aborts a debug build on that assert and, in a release build, encodes
 * SILENT GARBAGE — `1e-34` comes back as `-7.884178e+35`. The DD-oracle
 * corpora probe magnitudes far below tier 1's storage window (@see
 * `ff1FromDouble`/`ff2FromDouble`), so the cert TUs need a different egress
 * than the storage terminal — `detail::bandFromIEEE` is the matching
 * unchecked INGEST side.
 *
 * `Band.h` already ships that egress as `detail::bandToIEEE` and documents
 * it in exactly those terms ("the CARRIER's own egress terminal, symmetric
 * with `bandFromIEEE` above"). It is TOTAL — every finite `Band` has an
 * answer, inside the window or below it — which is the whole property a
 * deep-underflow CHARACTERISATION arm needs and a storage codec, correctly,
 * refuses to provide.
 *
 * @section noop Deliberately a no-op where storage IS applicable
 * On every in-window value the two terminals agree BIT-EXACTLY, so nothing
 * that legitimately measures the store path changes by using this (the
 * `StoreRoundTrip` rows keep `toDouble()` for exactly that reason). That
 * agreement is not asserted here in prose — it is a test row:
 * `BandCell8Window.InsideTheWindowTheTwoEgressesAgreeBitExactly`.
 *
 * ★ REVERTING THIS FILE'S ONE FUNCTION to the storage terminal is RED: it
 * aborts `Ff1Cert.AdversarialCornersF0`, `Ff2Cert.AdversarialCornersF0` and
 * the whole `BandCell8Window` suite on the `cell8BandIsStorable` assert.
 */

#include "aether/banded/Band.h"

#include <cstdint>
#include <cstring>

namespace aether_tests {
namespace cert {

/// @brief HOST ONLY. `Band` -> `double` through the CARRIER egress
/// (`detail::bandToIEEE`), the exact inverse-in-spirit of
/// `detail::bandFromIEEE`.
///
/// TOTAL: defined for every finite `Band`, including the below-window
/// magnitudes tier-1 storage cannot hold. Specials and signed zeros are
/// carried through unlaundered by `bandToIEEE` itself.
[[nodiscard]] inline double carrierToDouble(aether::banded::Band b)
{
    std::uint32_t ieeeLo = 0, ieeeHi = 0;
    aether::banded::detail::bandToIEEE(b, ieeeLo, ieeeHi);
    const std::uint64_t w
        = (static_cast<std::uint64_t>(ieeeHi) << 32) | static_cast<std::uint64_t>(ieeeLo);
    double d;
    std::memcpy(&d, &w, sizeof(d));
    return d;
}

} // namespace cert
} // namespace aether_tests
