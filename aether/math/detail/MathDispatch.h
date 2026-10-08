// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file MathDispatch.h
 * @brief Dispatch-macro scaffold for `aether::math`'s device/host scalar
 *        wrappers.
 *
 * `float` and `double` route through the two-leg scaffold below — device
 * compiles to the CUDA math intrinsic (`fabsf`/`::%fabs`, ...), host compiles
 * to the caller-supplied `std::` expression. NO `std::` is ever emitted on
 * the device path — the `#if defined(AETHER_DEVICE_COMPILE)` arm never
 * mentions `std::`.
 *
 * A third leg adds the BANDED family (`aether::banded::Band`, the register
 * carrier, and `aether::banded::BandedReal`, the 8-byte storage word). It
 * arrives NOT as a branch inside the scaffold but as a pair of constrained
 * COMPANION macros — `AETHER_MATH_UNARY_BAND` / `AETHER_MATH_BINARY_BAND`
 * — plus one edit to the binary scaffold's own requires-clause. Both
 * halves are load-bearing and neither is a style choice; the reasons are
 * stated at each of them below. Only the FIVE certified entry points are
 * widened (`abs`, `copysign`, `fmax`, `fmin`, `pow`); every other
 * `aether::math` name on a banded operand stays a hard compile error,
 * which is the honest answer until its own transcendental counterpart
 * lands.
 *
 * ★ THE COMPANIONS SEE THE BANDED FAMILY THROUGH A DECLARATION-ONLY HEADER
 * (`aether/math/detail/BandedFwd.h`). This file is reached from every
 * `aether::math` call, so pulling the banded definitions in here would put
 * the whole codec into every TU that computes `sin(double)` — exactly the
 * cost `tests/headers/check_header_diet.sh` polices. The companions
 * therefore return `auto`, and everything their bodies name is DEPENDENT on
 * the template parameter. See BandedFwd.h's own contract section.
 *
 * A macro body cannot contain `#ifdef`, so the *definition* of each macro
 * is selected here by the compilation pass (device vs host); every
 * expansion is therefore either the device or the host scaffold, never a
 * runtime branch between them.
 *
 * Deliberately has NO `#pragma once`: paired with `MathDispatchUndef.h`,
 * included once by `aether/math/math.h` and undef'd at that header's end so
 * the macros never leak to consumers, even though aether currently has
 * just the one facade file (`math.h`); a future split into multiple facade
 * headers (rounding.h, explog.h, ...) can reuse this pair unchanged.
 */

#include "aether/macros.h"
#include "aether/math/detail/BandedFwd.h"
#include "aether/math/detail/HostVectorMathRoute.h"

#include <concepts>
#include <type_traits>

#if defined(AETHER_DEVICE_COMPILE)
#define AETHER_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                          \
    template<class T>                                                                                                \
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T x)                                                             \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        if constexpr (std::is_same_v<T, float>) {                                                                    \
            return FLOATFN(x);                                                                                       \
        } else {                                                                                                      \
            return DOUBLEFN(x);                                                                                      \
        }                                                                                                            \
    }
#else
#define AETHER_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                          \
    template<class T>                                                                                                \
    AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T x)                                                             \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        return STDEXPR;                                                                                              \
    }
#endif

// ★ THE `requires` CLAUSE IS MANDATORY, NOT A CONVENIENCE. The scaffold's
// (T, T) shape is MORE SPECIALIZED than the (A, B) companion below under
// template partial ordering, so for a same-type banded call — `fmax(Band,
// Band)` — it WINS outright and lands on the unsupported-type static_assert.
// Constraints never get a say once ordering has decided, and there is no
// spelling of a companion that fixes this from outside: the scaffold's own
// requires-clause is the only place it can be fixed. The UNARY scaffold needs
// no such edit — a constrained unary companion wins over an unconstrained
// scaffold. `float`/`double` resolution is unchanged (the constraint is
// trivially true for them).
#if defined(AETHER_DEVICE_COMPILE)
#define AETHER_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                         \
    template<class T>                                                                                                \
        requires(!aether::banded::IsBandedFamily<T>)                                                                  \
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T a, T b)                                                        \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        if constexpr (std::is_same_v<T, float>) {                                                                    \
            return FLOATFN(a, b);                                                                                    \
        } else {                                                                                                      \
            return DOUBLEFN(a, b);                                                                                   \
        }                                                                                                            \
    }
#else
#define AETHER_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                         \
    template<class T>                                                                                                \
        requires(!aether::banded::IsBandedFamily<T>)                                                                  \
    AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T a, T b)                                                        \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        return STDEXPR;                                                                                              \
    }
#endif

// --- The BANDED companions ----------------------------------------------
//
// Constrained overloads that keep a banded chain UNPACKED: both accept the
// STORAGE leaf (`BandedReal`) as well as the WORKING carrier (`Band`), and both
// RETURN the working carrier, so
//
//      BandedReal s = fmax(abs(a), abs(b));   // three calls, ONE pack
//
// packs once at the assignment terminal and never in between. A companion that
// returned `BandedReal` would pay an encode and a decode per link — which is the
// "no pack per op" doctrine, spelled in the dispatch layer rather than only in
// the operator set.
//
// ★ THE RETURN TYPE IS `auto` AND THAT IS LOAD-BEARING. Naming `Band` would
// require the COMPLETE type where these macros EXPAND — i.e. in `math.h` and
// `explog.h` — which means the banded definitions in every TU that computes
// `sin(double)`. `auto` defers the requirement to instantiation, where the
// caller necessarily has the complete type already. The deduced type IS `Band`
// in every instantiation, and `tests/test_BandedReal_common.h` asserts exactly
// that so the deduction cannot drift.
//
// Identical for the device and host passes — the banded backends are
// AETHER_DEVICEHOST() — so there is no #ifdef pair here.
#define AETHER_MATH_UNARY_BAND(NAME, BANDFN)                                                                         \
    template<class T>                                                                                                \
        requires aether::banded::IsBandedFamily<T>                                                                    \
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto NAME(T x)                                            \
    {                                                                                                                \
        return aether::banded::detail::BandedFacade<T>::BANDFN(                                                      \
            static_cast<aether::banded::detail::BandedWorkingOfT<T>>(x));                                            \
    }

#define AETHER_MATH_BINARY_BAND(NAME, BANDFN)                                                                        \
    template<class A, class B>                                                                                       \
        requires(aether::banded::IsBandedFamily<A> && aether::banded::IsBandedFamily<B>)                             \
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto NAME(A a, B b)                                       \
    {                                                                                                                \
        return aether::banded::detail::BandedFacade<A>::BANDFN(                                                      \
            static_cast<aether::banded::detail::BandedWorkingOfT<A>>(a),                                             \
            static_cast<aether::banded::detail::BandedWorkingOfT<B>>(b));                                            \
    }
