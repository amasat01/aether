// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandedFwd.h
 * @brief DECLARATION-ONLY view of the banded family, for
 *        `aether/math/detail/MathDispatch.h`.
 *
 * @section why_bandedfwd Why this header exists
 * `MathDispatch.h` is included by every `aether::math` facade header, so
 * anything it pulls in lands in every translation unit that computes
 * `sin(double)`. The banded routing it generates mentions two type names
 * (`Band`, `BandedReal`) and five certified entry points — and the definitions
 * of those live in `aether/banded/`, which is ~2 000 lines of codec and carrier
 * that no `sin(double)` TU has any use for. That cost is exactly what
 * `tests/headers/check_header_diet.sh` exists to police, and it is why the
 * banded umbrella is deliberately NOT included from `aether/aether.h`.
 *
 * So `MathDispatch.h` includes THIS instead: names only, no definitions, no
 * `aether/banded/`. The definitions arrive with the umbrella in the TUs that
 * actually instantiate a banded facade call — which is, by construction, every
 * TU that has a banded value to pass.
 *
 * @section contract_bandedfwd Contract
 *  - The companion macros return `auto`, NOT `Band`. A function DEFINITION may
 *    not return an incomplete class type, so a `Band`-returning companion would
 *    force the whole banded family into every facade header and defeat this file
 *    entirely. The deduced type IS `Band` in every instantiation, and
 *    `tests/test_BandedReal_common.h` asserts that so the deduction cannot
 *    drift.
 *  - Everything the companion bodies name is DEPENDENT on the template
 *    parameter — `BandedWorkingOfT<T>` for the cast, `BandedFacade<T>`'s own
 *    static member for the call. Both indirections exist for one measured
 *    reason, stated at each.
 *  - A TU that calls a banded facade entry point without the umbrella fails
 *    LOUDLY — "incomplete type `BandedFacade<…>` used in nested name specifier"
 *    — never silently, and never by resolving to something else.
 */

#include <concepts>
#include <type_traits>

#include "aether/macros.h"

namespace aether {
namespace banded {

// Names only. The definitions live in aether/banded/{Band,BandedReal}.h.
struct Band;
struct BandedReal;

/* ── The BANDED family ────────────────────────────────────────────────────
 *
 * Two concepts, not one, because the two roles are not interchangeable: `Band`
 * is a REGISTER-resident working carrier with no memory form, and `BandedReal`
 * is an 8-byte STORAGE word with no arithmetic of its own. The split is what
 * lets the namespace-scope operators (`BandedRealOps.h`) demand "at least one
 * STORAGE operand" and so stay clear of `Band`'s own hidden-friend operators,
 * and what lets the facade companions accept both while always RETURNING the
 * working carrier — a chain therefore stays UNPACKED and pays no pack per op.
 *
 * ★ `IsBandedFamily` is ALSO the exclusion the BINARY dispatch scaffold in
 * `MathDispatch.h` carries, and that is not optional and not a style choice: the
 * scaffold's `(T, T)` shape is MORE SPECIALIZED than an `(A, B)` companion under
 * partial ordering, so for `fmax(Band, Band)` it would win outright and land on
 * the unsupported-type `static_assert` — constraints never get a say once
 * ordering has decided. The UNARY scaffold needs no such edit: a constrained
 * unary companion wins over the unconstrained scaffold. */
template<typename T>
concept IsBandedWorking = std::same_as<std::remove_cvref_t<T>, Band>;
template<typename T>
concept IsBandedStorage = std::same_as<std::remove_cvref_t<T>, BandedReal>;
template<typename T>
concept IsBandedFamily = IsBandedWorking<T> || IsBandedStorage<T>;

namespace detail {

/**
 * @brief A DEPENDENT spelling of `Band`, and the reason the companion macros can
 *        be defined at all without the banded definitions.
 *
 * ★ MEASURED under nvcc, not stylistic. Writing `static_cast<Band>(x)` inside a
 * companion body names a NON-dependent incomplete type, which makes the
 * enclosing call non-dependent too — so nvcc's front end resolves the callee at
 * PARSE time and rejects the whole thing ("incomplete type … is not allowed",
 * "function … returns incomplete type"). g++ is more permissive here and accepts
 * it, which is precisely why this shape has to be preserved rather than
 * simplified: the failure only appears on the device compiler.
 * `BandedWorkingOfT<T>` depends on `T`, so the cast, the call and the `auto`
 * deduction are all deferred to instantiation — where the caller necessarily has
 * the complete type.
 */
template<typename T>
struct BandedWorkingOf {
    using T_ = Band;
};
template<typename T>
using BandedWorkingOfT = typename BandedWorkingOf<T>::T_;

/**
 * @brief The banded facade indirection: `abs`, `copysign`, `fmax`, `fmin` and
 *        `pow`, re-exposed as static members of a CLASS TEMPLATE.
 *
 * DECLARED here, DEFINED in `aether/banded/BandedRealOps.h` once `Band` and the
 * certified bodies are complete.
 *
 * ★ A CLASS TEMPLATE AND NOT FIVE FREE DECLARATIONS, and the difference is not
 * taste. A free declaration `Band abs(Band);` is a NON-DEPENDENT name with a
 * non-dependent return type, so a companion body that calls it is resolved by
 * nvcc's front end while PARSING the template — before any instantiation — and
 * rejected. Making the ARGUMENTS dependent does not help; the callee's own
 * return type is what is being checked. `BandedFacade<T>::%pow` is a member of a
 * DEPENDENT type, so lookup, the return type and the enclosing `auto` deduction
 * are all deferred to instantiation.
 *
 * `T` is deliberately unused by every member: it exists to make the NAME
 * dependent and nothing else.
 */
template<typename T>
struct BandedFacade;

} // namespace detail
} // namespace banded
} // namespace aether
