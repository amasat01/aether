// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Ladder.h
 * @brief The banded-domain working-carrier ladder: the ordered
 *        `BandedLadder`, the `IsBandedStorageDomain` predicate and the
 *        `BandedDemandType`/`BandedDemandT` demand->rung selector.
 *
 * This header is pure compile-time metadata: every entity below is a
 * type-level trait or alias (zero codegen).
 *
 * @section certbits The `carrierCertifiedBits_v` trait
 * Carrier-agnostic SFINAE metadata: a primary that defaults to 53 (the
 * width-tag-free "Full" carrier — `Band` carries no `certifiedBits`
 * member by design, @see `Ff2.h`'s own file header) and a
 * `void_t`-detected specialization for any carrier that does declare a
 * `static constexpr int certifiedBits` (`Ff1` = 24, `Ff2` = 45).
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandedReal.h"
#include "aether/banded/Ff1.h"
#include "aether/banded/Ff2.h"
#include "aether/macros.h"

#include <concepts>
#include <type_traits>

namespace aether {
namespace banded {

// =====================================================================
//  carrierCertifiedBits_v -- the width tag the ladder reads; @see the file
//  header's "certbits" section.
// =====================================================================

/// @brief Primary: 53 (the width-tag-free "Full" carrier — `Band` carries no
/// `certifiedBits` member by design).
template<typename E, typename = void>
struct CarrierCertifiedBits {
    static constexpr int value = 53;
};
/// @brief Specialization: any carrier that does declare a
/// `static constexpr int certifiedBits` reads its own tag (`Ff1` = 24,
/// `Ff2` = 45).
template<typename E>
struct CarrierCertifiedBits<E, std::void_t<decltype(std::remove_cvref_t<E>::certifiedBits)>> {
    static constexpr int value = static_cast<int>(std::remove_cvref_t<E>::certifiedBits);
};
template<typename E>
inline constexpr int carrierCertifiedBits_v = CarrierCertifiedBits<E>::value;

// =====================================================================
//  Rung metadata + the ordered ladder (carrier-agnostic)
// =====================================================================

/// @brief Compile-time metadata for one ladder rung: its working carrier and
/// the carrier's conservative certified effective width.
template<typename Carrier>
struct Rung {
    using carrier                      = Carrier;
    static constexpr int certifiedBits = carrierCertifiedBits_v<Carrier>;
};

/// @brief An ordered list of rungs (ascending `certifiedBits` = narrowest
/// first). Order is the cost model until the ladder stops being totally
/// ordered.
template<typename... Rungs>
struct Ladder {
    static constexpr int size = sizeof...(Rungs);
};

/// @brief The banded working ladder: `<Ff1 (24), Ff2 (45), Band (53)>`,
/// narrowest first.
using BandedLadder = Ladder<Rung<Ff1>, Rung<Ff2>, Rung<Band>>;

// ---------------------------------------------------------------------
//  Ladder selection: the narrowest rung whose certifiedBits >= Bits.
// ---------------------------------------------------------------------

/// @brief Hard-error sentinel: instantiated only when a ladder is exhausted
/// without a rung certifying `Bits` (value-dependent so it never fires at
/// definition, only at the offending instantiation).
template<int>
inline constexpr bool kNoRungCertifies = false;

template<int Bits, typename LadderT>
struct LadderSelect; // primary: undefined

// Match helper: instantiate the tail only when the head rung does not
// certify Bits.
template<int Bits, typename Head, typename Tail, bool HeadCertifies>
struct LadderPick;
template<int Bits, typename Head, typename Tail>
struct LadderPick<Bits, Head, Tail, true> {
    using T_ = typename Head::carrier; // narrowest certifying rung -- stop here
};
template<int Bits, typename Head, typename Tail>
struct LadderPick<Bits, Head, Tail, false> {
    using T_ = typename LadderSelect<Bits, Tail>::T_; // keep climbing
};

template<int Bits, typename R0, typename... Rs>
struct LadderSelect<Bits, Ladder<R0, Rs...>> {
    using T_ = typename LadderPick<Bits, R0, Ladder<Rs...>, (R0::certifiedBits >= Bits)>::T_;
};

template<int Bits>
struct LadderSelect<Bits, Ladder<>> {
    static_assert(
        kNoRungCertifies<Bits>, "LadderSelect: no ladder rung certifies the requested Bits demand.");
    using T_ = void;
};

// =====================================================================
//  BandedDemandT<Bits, StorageT> -- the Banded-domain demand -> rung selector
// =====================================================================

/// @brief The banded storage selection domain: the storage leaf whose
/// declared working carrier is `Band`. A domain of one today, expressed
/// as a concept rather than an inline `std::same_as` because the next
/// banded storage leaf joins here and nowhere else.
template<typename T>
concept IsBandedStorageDomain = std::same_as<T, aether::banded::BandedReal>;

/// @brief Primary: identity for all Bits (any type outside the banded
/// storage domain is unaffected by this selector).
template<int Bits, typename StorageT, typename Enable = void>
struct BandedDemandType {
    using T_ = StorageT;
};

/// @brief Banded-storage specialization: the narrowest `BandedLadder` rung
/// certifying `Bits`. The Bits demand is restricted to the ladder's
/// certified set `{24, 45, 53}`; any other value is a hard compile error
/// (no silent widening or narrowing).
template<int Bits, typename T>
struct BandedDemandType<Bits, T, std::enable_if_t<IsBandedStorageDomain<T>>> {
    static_assert(Bits == 24 || Bits == 45 || Bits == 53,
        "BandedDemandT<Bits, BandedReal>: Bits must be in the BandedLadder's "
        "certified set {24, 45, 53} (Ff1 = 24, Ff2 = 45, Band = 53). No other "
        "demand has a certified rung -- widen the ladder before requesting a "
        "new width. (A demand landing in a gap -- 25..44 or 46..52 -- has no "
        "certifying rung and is a hard error.)");
    using T_ = typename LadderSelect<Bits, BandedLadder>::T_;
};

/// @brief The declared working alias for a region that has been measured
/// admissible for the banded rung and opts into it.
///
/// @warning Naming this alias is a claim that the region's range has been
/// certified — `Band` has no scale field, so its ceiling is FP32's own and
/// the failure at that edge is catastrophic rather than gradual. Pair
/// adoption with `banded::validateBandedAdmission` (`BandedLimits.h`) on the
/// region's declared exponents.
template<int Bits, typename StorageT>
using BandedDemandT = typename BandedDemandType<Bits, std::remove_cvref_t<StorageT>>::T_;

// =====================================================================
//  ladder_asserts -- compile-time-only admission checks (Banded-domain rows
//  only)
// =====================================================================
namespace ladder_asserts {

// Rung metadata: each carrier's certified width.
static_assert(carrierCertifiedBits_v<Ff1> == 24, "Ff1 is the w=24 rung.");
static_assert(carrierCertifiedBits_v<Ff2> == 45, "Ff2 is the w=45 rung.");
static_assert(carrierCertifiedBits_v<Band> == 53, "Band is the metadata-free Full 53-bit carrier.");
static_assert(Rung<Ff1>::certifiedBits == 24);
static_assert(Rung<Ff2>::certifiedBits == 45);
static_assert(Rung<Band>::certifiedBits == 53);
static_assert(BandedLadder::size == 3, "the banded ladder has three rungs");

// Non-vacuity: the banded ladder actually selects each rung at its own
// certified width. Without this check, the rows below would still pass
// even if BandedLadder/BandedDemandT did not exist at all.
static_assert(std::is_same_v<BandedDemandT<53, BandedReal>, Band>);
static_assert(std::is_same_v<BandedDemandT<45, BandedReal>, Ff2>);
static_assert(std::is_same_v<BandedDemandT<24, BandedReal>, Ff1>);

// A 53-bit banded demand must not silently answer a narrower rung -- the
// one outcome a plausible one-line implementation could produce.
static_assert(!std::is_same_v<BandedDemandT<53, BandedReal>, Ff2>,
    "BandedDemandT<53, BandedReal> resolved to Ff2 -- the ladder is not "
    "selecting the widest (Band) rung at the top demand");

// The identity primary still applies to a type outside the banded storage
// domain.
static_assert(std::is_same_v<BandedDemandT<53, double>, double>);
static_assert(std::is_same_v<BandedDemandT<53, float>, float>);
static_assert(std::is_same_v<BandedDemandT<24, int>, int>);

// IsBandedStorageDomain: BandedReal is in the domain; Band (the carrier,
// not the storage leaf) and a native arm are not -- the domain is keyed
// to the storage type.
static_assert(IsBandedStorageDomain<BandedReal>);
static_assert(!IsBandedStorageDomain<Band>);
static_assert(!IsBandedStorageDomain<double>);
static_assert(!IsBandedStorageDomain<Ff2>);

} // namespace ladder_asserts

} // namespace banded
} // namespace aether
