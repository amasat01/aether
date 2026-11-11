// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file WorkingType.h
 * @brief `aether::WorkingType<T>` — the storage -> working carrier map, and
 *        the two value-level helpers the expression layer converts
 *        through.
 *
 * @section what What this answers
 * A scalar the library stores and the scalar it computes in are not always
 * the same type. For `float`/`double` they are (and this file must cost
 * exactly nothing there). For an emulated scalar they are deliberately
 * different: `aether::banded::BandedReal` is an 8-byte codec word with no
 * arithmetic of its own, and every operator over it returns
 * `aether::banded::Band`, the metadata-free 53-bit working carrier. One
 * `element_type` per expression tree would otherwise force an encode at
 * the exit of every intermediate node; instead, the tree computes in the
 * carrier and encodes once at the store.
 *
 * `WorkingType<T>` is the one place that mapping is declared. `Expression`
 * publishes `working_type`, every node's `eval()` returns it, and the
 * assignment terminal (`aether/expr/Assign.h`) is the single place
 * `fromWorking` is paid.
 *
 * @section shape_workingtype The shape, and why the primary is an identity
 * The primary template maps every type to itself with two identity
 * conversions. That is not a placeholder: it is the contract that makes
 * the whole mechanism free for the native dtypes — `working_type_t<double>`
 * is `double`, and `toWorking`/`fromWorking` are `constexpr` functions
 * returning their argument, which every compiler folds away (per-kernel
 * register/stack/local usage of the CUDA test binary stays identical to
 * the pre-change baseline).
 *
 * @section odr ODR
 * This header names nothing from the banded tree, and must not: it is
 * reached from `aether/expr/Expression.h`, i.e. from every translation
 * unit that builds an expression at all, and the banded codec has no
 * business in a TU that adds two doubles (`tests/headers/
 * check_header_diet.sh` polices that same boundary from the other side).
 * The banded specialization therefore lives at the end of the header that
 * defines `aether::banded::BandedReal`, where the type is complete — so no
 * TU can see the storage scalar without also seeing its carrier
 * declaration, and there is no ordering in which one is visible and the
 * other is not.
 */

#include <type_traits>

#include "aether/macros.h"

namespace aether {

/**
 * @brief The working (on-register) carrier for storage scalar `T`, plus the two
 *        conversions between them. PRIMARY = identity; specialize for a scalar
 *        whose arithmetic lives on a different type.
 *
 * A specialization must provide the same three names:
 *  - `using type = W;`                 the working carrier
 *  - `static type toWorking(const T&)` storage → working (the region ENTRY)
 *  - `static T fromWorking(const type&)` working → storage (the PACK terminal)
 *
 * Both conversions are `AETHER_DEVICEHOST()`. A specialization whose
 * `fromWorking` is not `constexpr` is fine — every caller in `expr/` is a
 * template, so it is simply never constant-evaluated for that instantiation.
 */
template<class T>
struct WorkingType {
    using type = T;

    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr type toWorking(const T& v) { return v; }
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr T fromWorking(const type& v) { return v; }
};

/** @brief The working carrier for `T` (cv/ref-stripped). `T` for every native dtype. */
template<class T>
using working_type_t = typename WorkingType<std::remove_cvref_t<T>>::type;

namespace detail {

/**
 * @brief Storage → working, IDEMPOTENT: applying it to a value that is already
 *        the working carrier is the identity (`WorkingType<W>` is the primary
 *        for every carrier that is not itself somebody's storage type). That is
 *        what lets a node convert its operand unconditionally instead of
 *        branching on `E::isLeaf` — a leaf hands back the storage scalar, a
 *        composite hands back the carrier, and this call is correct for both.
 */
template<class T>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr working_type_t<T> toWorkingValue(const T& v)
{
    return WorkingType<std::remove_cvref_t<T>>::toWorking(v);
}

/**
 * @brief Working → the DESTINATION SLOT's own type — the PACK, keyed on where
 *        the value is about to land rather than on the tree's `element_type`.
 *
 * Keying on the slot is what makes a register-resident temporary that already
 * holds the carrier (`aether/expr/nodes/Product.h`'s fused caches) cost nothing
 * to write: `Slot` is then the carrier itself, `WorkingType<Slot>` is the
 * identity primary, and no encode is emitted. Keying on `element_type` instead
 * would pack into every such buffer and unpack straight back out.
 */
template<class Slot, class W>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr std::remove_cvref_t<Slot> fromWorkingValue(const W& w)
{
    return WorkingType<std::remove_cvref_t<Slot>>::fromWorking(w);
}

} // namespace detail
} // namespace aether
