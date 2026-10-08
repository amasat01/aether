// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Band.h
 * @brief `Band` — the metadata-free 53-bit WORKING carrier, and the certified
 *        arithmetic over it.
 *
 * Three bare FP32 limbs and nothing else: the sign lives in the VALUE of each
 * (signed) limb and there is no scale field at all. That single structural
 * choice is the whole point of the rung.
 *
 *  - **Cost.** Sign/scale metadata is what generates the integer scaffolding
 *    that dominates an emulated instruction stream. Signed limbs fold negation
 *    into SASS operand modifiers; a missing scale field deletes the alignment
 *    branch, the sign selects and the reband entirely.
 *  - **Correctness.** A carrier with no DERIVED sign cannot get the derivation
 *    wrong. The whole class of defects where sign metadata goes stale relative
 *    to the limbs has nothing here to go stale.
 *
 * The precision class is unchanged: `Band` certifies **53 effective bits**.
 * This rung buys instructions, never accuracy.
 *
 * @par The price: a hard ceiling
 * The exponent lives on the float limbs, so `Band`'s range IS FP32's range.
 * `hi` overflows at `2^128`. Downward there are TWO distinct floors with
 * different mechanisms: the CARRIER floor (`bandIntermediateAdmits`, `2^-96`),
 * which binds INTERMEDIATES anywhere in a chain, and the storage window of the
 * codec (`BandCell8.h`), which binds what can be parked. Both are declared,
 * not discovered — see the admission predicates in `BandCell8.h`.
 *
 * @par What this file deliberately does NOT carry
 * The interior demand rungs (`Ff1`/`Ff2`), the `Ladder`/`Recurrence`
 * machinery and the transcendental family (`exp`/`log`/`sin`/…) live
 * elsewhere. This file lands the CERTIFIED CORE — `neg`, `add`, `sub`,
 * `mul`, `recip`, `div` — plus the four exact sign/order ops the
 * `aether::math` facade routes to (`abs`, `copysign`, `fmax`, `fmin`) and
 * the IEEE ingest the codec needs.
 *
 * Every error-free transform routes through `detail/Fp32.h`'s EXPLICIT
 * `fma.rn.f32` / barrier-guarded arithmetic, so the results are
 * independent of `-fmad`/`-ffp-contract` — which is what makes a
 * host/device bit-for-bit comparison decidable by construction.
 */

#include <bit>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "aether/banded/detail/Fp32.h"
#include "aether/macros.h"

namespace aether {
namespace banded {

// =====================================================================
//  Operator-surface pre-declarations
// =====================================================================
//
// `Band` is forward-declared here (still incomplete) so the wall concept and
// the `detail::` prototypes can name it before its full definition. Both exist
// purely so `struct Band`'s hidden-friend operators can stay FULLY INLINE: an
// out-of-class definition of a hidden friend would make that name visible to
// ordinary (non-ADL) lookup from that point on — "unhiding" it and widening
// overload resolution for unrelated types, which is precisely what the hidden-
// friend decision exists to avoid.
struct Band;

/**
 * @brief The types `Band` must NEVER mix with under an operator (the operator
 *        wall).
 *
 * Today the wall names the two NATIVE scalars, and that is the aether-relevant
 * half: a `Band` silently mixing with a `float` or a `double` is either a
 * catastrophic width demote (24 or 53 IEEE bits standing in for a 72-bit
 * carrier) or, worse, an FP64 instruction appearing inside a carrier whose
 * entire warrant is that it targets hardware with no FP64 at all.
 * There is no implicit conversion either way today, so nothing here is
 * closing a LIVE single-hop
 * path; what the deleted overloads buy is that the failure is a targeted
 * "call to deleted function" naming the wall instead of a generic "no matching
 * operator", and that the wall is already standing on the day a bridge
 * constructor is added.
 *
 * ★ aether carries no CompDD/CddRaw rung and never will, so that arm of the
 * wall simply has no members here. The `Ff1` (24-bit) and `Ff2` (45-bit)
 * interior rungs join this concept (forward-declared below so this header
 * need not include `Ff1.h`/`Ff2.h` — both of THOSE headers include
 * `Band.h`, so the dependency runs one way only): the concept was written
 * as a disjunction so adding them is the one-line change this docstring
 * anticipated.
 */
struct Ff1; ///< `banded/Ff1.h`; forward-declared only for the wall below.
struct Ff2; ///< `banded/Ff2.h`; forward-declared only for the wall below.

template<typename T>
concept BandWallFamily = std::same_as<std::remove_cvref_t<T>, float>
    || std::same_as<std::remove_cvref_t<T>, double>
    || std::same_as<std::remove_cvref_t<T>, Ff1>
    || std::same_as<std::remove_cvref_t<T>, Ff2>;

namespace detail {
// Forward declarations only. Full bodies are defined further down this file
// (Multiply / Add-subtract / Reciprocal sections). Declaring them here, ahead
// of `struct Band`, is what lets the operators below stay fully inline.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band neg(Band a);
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band add(Band a, Band b);
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sub(Band a, Band b);
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band mul(Band a, Band b);
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band recip(Band b);
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band div(Band a, Band b);
} // namespace detail

// =====================================================================
//  Band -- the 3-limb metadata-free carrier
// =====================================================================

/// @brief Certified 3-slot signed-limb carrier. No sign field, no scale field,
/// NO MEMORY FORM (that is `BandedReal`/`BandCell8`, `BandCell8.h`).
struct Band {
    float hi;   ///< leading limb; carries the overall sign natively
    float lo;   ///< second limb, `|lo| <= ulp(hi)` after each certified op
    float tail; ///< residue limb; what lifts the carrier past the df64 ceiling

    /// @brief The certified effective width of this carrier. Declared
    /// EXPLICITLY rather than inherited from any default — a default that
    /// happens to agree would give the right answer for the wrong reason and
    /// would survive a later change to that default.
    static constexpr int certifiedBits = 53;

    // No AETHER_DEVICEHOST() on the defaulted ctor: a `= default` special
    // member on its first declaration is automatically host+device eligible
    // under nvcc, which warns (#20012-D) that an explicit annotation is
    // redundant (see aether/dtype/DType.h's own note).
    Band() = default;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Band(float h, float l, float t)
        : hi(h)
        , lo(l)
        , tail(t)
    {
    }

    // ── Banded arithmetic (hidden friends) ──────────────────────────────────
    //
    // Hidden friends: ADL finds these ONLY when an operand is already a
    // `Band`, so plain float/double expressions never see them and overload
    // resolution for unrelated types is never widened. Each op delegates to
    // the certified `detail::` free function forward-declared above; this
    // surface adds NO new numerics.

    /// @brief Banded add. @see detail::add.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator+(Band a, Band b)
    {
        return detail::add(a, b);
    }
    /// @brief Banded unary negate. @see detail::neg.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(Band a)
    {
        return detail::neg(a);
    }
    /// @brief Banded sub. @see detail::sub.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(Band a, Band b)
    {
        return detail::sub(a, b);
    }
    /// @brief Banded mul. @see detail::mul.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator*(Band a, Band b)
    {
        return detail::mul(a, b);
    }
    /// @brief Banded div — ONE named primitive with its own body and its own
    /// battery, never the composition `mul(a, recip(b))`. @see detail::div.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator/(Band a, Band b)
    {
        return detail::div(a, b);
    }

    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band& operator+=(Band& a, Band b)
    {
        a = a + b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band& operator-=(Band& a, Band b)
    {
        a = a - b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band& operator*=(Band& a, Band b)
    {
        a = a * b;
        return a;
    }
    // There is deliberately NO `operator/=`: division is only ever the
    // free `div()` function below, never a compound-assignment operator.

    // ── The operator wall: NO cross-representation operators, EVER ──────────
    //
    // One constrained template per op and per operand order. @see
    // BandWallFamily for what is walled and why.
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator+(Band, const O&) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator+(const O&, Band) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator-(Band, const O&) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator-(const O&, Band) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator*(Band, const O&) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator*(const O&, Band) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator/(Band, const O&) = delete;
    template<typename O>
        requires BandWallFamily<O>
    friend Band operator/(const O&, Band) = delete;

    // ── Raw bit accessors — a metadata-free carrier's OWN codec
    // boundary, distinct from BandCell8's packed storage word. `fromBits`/
    // `toBits` are a bare per-limb reinterpret (`__float_as_uint`/
    // `__uint_as_float` on device, `std::bit_cast` on host) — no rounding,
    // no admission check, no normalization — so `toBits(fromBits(h,l,t))`
    // and `fromBits` applied to `toBits`'s own output are bit-exact BY
    // CONSTRUCTION, never merely by test. ─────────────────────────────────

    /** @brief Reinterpret three raw IEEE-754 FP32 bit patterns as a `Band`'s
     *  limbs — a bare per-limb `bit_cast`, no rounding, no normalization. */
    static AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fromBits(std::uint32_t hi, std::uint32_t lo, std::uint32_t tail)
    {
#ifdef __CUDA_ARCH__
        return Band{ __uint_as_float(hi), __uint_as_float(lo), __uint_as_float(tail) };
#else
        return Band{ std::bit_cast<float>(hi), std::bit_cast<float>(lo), std::bit_cast<float>(tail) };
#endif
    }

    /** @brief Reinterpret this `Band`'s three limbs as raw IEEE-754 FP32 bit
     *  patterns — the inverse of `fromBits`, equally bare. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void toBits(std::uint32_t& hi, std::uint32_t& lo, std::uint32_t& tail) const
    {
#ifdef __CUDA_ARCH__
        hi = __float_as_uint(this->hi);
        lo = __float_as_uint(this->lo);
        tail = __float_as_uint(this->tail);
#else
        hi = std::bit_cast<std::uint32_t>(this->hi);
        lo = std::bit_cast<std::uint32_t>(this->lo);
        tail = std::bit_cast<std::uint32_t>(this->tail);
#endif
    }
};

static_assert(sizeof(Band) == 12, "Band is three bare FP32 limbs and nothing else");
static_assert(std::is_trivially_copyable_v<Band>,
    "Band must stay trivially copyable — it is a register carrier passed by value");

namespace detail {

// =====================================================================
//  RAW and NORMALIZED are two TYPES, not one type and a naming convention
// =====================================================================

/**
 * @brief A three-limb value that has NOT restored the carrier postcondition.
 *
 * `Band` promises `|lo| <= ulp(hi)`: the limbs are ORDERED, so the leading one
 * really is the value to 24 bits and any consumer that reads them in order —
 * the codec, a comparison, a store terminal — gets the answer it thinks it is
 * getting. The raw ops (`mulRaw`, `sqrRaw`, `addRaw`) deliberately do NOT
 * restore it: that restoration is three `fast2Sum`es whose only customer is a
 * consumer that reads the limbs, and inside a chain the next operation opens
 * with an exact cascade that does not care. Skipping it is where this rung's
 * instructions come from.
 *
 * @par Why a TYPE and not a comment
 * A naming convention cannot be checked. The hazard is also invisible to the
 * instrument consumers actually run: a dropped normalize can leave every
 * delivered double identical on an ordinary workload and be catastrophic on an
 * amplifying one. The rules, all three enforced by the compiler:
 *  - `Band -> BandRaw` is IMPLICIT and free (a certified carrier satisfies
 *    everything a raw operand may assume);
 *  - `BandRaw -> Band` DOES NOT EXIST — `normalize`/`normalizeSafe` is the
 *    only bridge, and it is the operation that makes the promise true;
 *  - therefore a `BandRaw` cannot reach the codec, a comparison or any other
 *    `Band`-typed boundary without passing through that bridge.
 */
struct BandRaw {
    float hi;   ///< leading limb, NOT promised to dominate the other two
    float lo;   ///< second limb
    float tail; ///< residue limb

    BandRaw() = default;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandRaw(float h, float l, float t)
        : hi(h)
        , lo(l)
        , tail(t)
    {
    }

    /// @brief Widening, and the only implicit direction there is.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandRaw(Band b)
        : hi(b.hi)
        , lo(b.lo)
        , tail(b.tail)
    {
    }
};

/**
 * @brief What a CANCELLATION-CAPABLE add returns: a raw value so disordered
 *        that `normalize`/`normalizeSafe` is the only thing allowed to touch it.
 *
 * `addRaw` sums two carriers whose signs it does not know. When they oppose,
 * the leading limbs cancel and `|lo|` exceeds `ulp(hi)` by the amplification
 * factor — the most disordered carrier a chain ever holds, not merely an
 * unrestored one. `BandRawCancel` converts to nothing and is accepted by
 * nothing except the two normalize overloads.
 *
 * The one sanctioned escape is `addRawNoCancel`, for an adder that can rule
 * cancellation out STATICALLY (summing squares, a Newton residual bounded away
 * from cancellation by its own convergence). Every call site of it owes a
 * one-line argument, and "the inputs are usually positive" is not one.
 */
struct BandRawCancel {
    float hi;   ///< leading limb; the cancellation has already happened here
    float lo;   ///< second limb, possibly LARGER than `ulp(hi)`
    float tail; ///< residue limb
};

// =====================================================================
//  Metadata-free zero-cost ops
// =====================================================================

/// @brief Negation. Signed limbs fold it into SASS operand modifiers; there is
/// no sign FIELD to flip and therefore no downstream select to pay.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band neg(Band a)
{
    return Band{ -a.hi, -a.lo, -a.tail };
}

/// @brief Negation of a raw value, which STAYS raw. Negating three floats
/// cannot order limbs that were not ordered, so this overload keeps the state
/// visible through the sign flip rather than laundering it through `Band`.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw neg(BandRaw a)
{
    return BandRaw{ -a.hi, -a.lo, -a.tail };
}

/// @brief Exact power-of-two scaling: three exact FP32 multiplies. There is no
/// scale field to fold into, which is why this needs no banding rule.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band scalePow2f(Band a, float s)
{
    return Band{ a.hi * s, a.lo * s, a.tail * s };
}

// =====================================================================
//  Specials -- the IEEE-754 value rules, carried in the leading limb
// =====================================================================

/// @brief The FP32 exponent field, all ones: the encoding infinities and NaNs
/// share, and the one pattern a finite limb can never present.
inline constexpr std::uint32_t kBandFp32ExpMask = 0x7F800000u;

/// @brief The canonical quiet NaN this carrier delivers. Payloads are NOT
/// transported: IEEE 754 requires no payload propagation, nothing downstream
/// reads one, and a single code point is what the `BandCell8` specials encoding
/// can hold. Every NaN a guard sees collapses onto this one.
inline constexpr std::uint32_t kBandCanonicalNanBits = 0x7FC00000u;

/**
 * @brief Is this leading limb an infinity or a NaN?
 *
 * A `Band` is three bare floats with no metadata, so "what class of value does
 * this carrier hold" has exactly one honest answer: the class of its LEADING
 * limb. A normalized carrier whose `hi` is finite holds a finite value (the
 * lower limbs are bounded by `ulp(hi)`), and one whose `hi` is not finite holds
 * nothing else — there is no magnitude a lower limb could contribute to an
 * infinity. Two integer instructions.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool bandHiIsSpecial(float x)
{
    return (static_cast<std::uint32_t>(floatAsInt(x)) & kBandFp32ExpMask)
        == kBandFp32ExpMask;
}

/**
 * @brief Do EITHER of these two leading limbs hold a special?
 *
 * Spelled as a MAXIMUM rather than as two comparisons because the masked
 * exponent field of a float is at most `kBandFp32ExpMask` — that is the field's
 * own maximum, not a coincidence of these operands — so the larger of the two
 * fields equals the mask exactly when at least one of them does. One `IMNMX`
 * and one `ISETP` where the obvious spelling costs two `ISETP`s and a predicate
 * combine, and it is the SAME claim, not a cheaper approximation of one.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool bandHiIsSpecial(
    float a, float b)
{
    const std::uint32_t ea
        = static_cast<std::uint32_t>(floatAsInt(a)) & kBandFp32ExpMask;
    const std::uint32_t eb
        = static_cast<std::uint32_t>(floatAsInt(b)) & kBandFp32ExpMask;
    return ((ea > eb) ? ea : eb) == kBandFp32ExpMask;
}

/**
 * @brief The canonical carrier for a special: the value in the leading limb,
 *        both lower limbs CLEARED, every NaN collapsed onto the canonical one.
 *
 * ★ Clearing the lower limbs is the load-bearing half. An unguarded op does not
 * merely get the leading limb wrong — it contaminates `lo` and `tail` with NaNs
 * manufactured by the error terms (`twoProd(0.9, Inf)` evaluates
 * `fma(0.9, Inf, -Inf)`, an `Inf - Inf`). A carrier with a correct `hi` and a
 * NaN `lo` reads correctly and then poisons the NEXT operation.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandSpecialCarrier(float v)
{
    const std::uint32_t u = static_cast<std::uint32_t>(floatAsInt(v));
    const bool isNan      = (u & 0x7FFFFFFFu) > kBandFp32ExpMask;
    const float canon
        = isNan ? intAsFloat(static_cast<int>(kBandCanonicalNanBits)) : v;
    return Band{ canon, 0.0f, 0.0f };
}

// =====================================================================
//  Terminals
// =====================================================================

/**
 * @brief THE bridge: the only operation that turns a `BandRaw` into a `Band`,
 *        because it is the only one that makes the promise `Band` carries true.
 *        Three `fast2Sum`es.
 *
 * @par ★ The SPECIALS guard, and why it lives HERE rather than at each op
 * Every certified op in this file ends `return normalize(...)`, so one test
 * here is one test per op — at HALF the width, because a RESULT has one leading
 * limb where a binary op has two operands. It is also the only placement that
 * generalises: a primitive added tomorrow inherits the rule by construction
 * instead of owing a guard nobody remembers to write.
 *
 * It WORKS because, for every op whose raw body opens with a `twoProd`/`twoSum`
 * on the leading limbs, FP32 hardware has ALREADY put the right special in
 * `r.hi` (`0.9 * Inf` gives `Inf`, `Inf - Inf` gives `NaN`, `0 * Inf` gives
 * `NaN`). What an unguarded body then does is destroy it: the ERROR term of
 * that same step is an `Inf - Inf`, hence a NaN, and the first `fast2Sum` pulls
 * that NaN straight into `hi`. The leading limb is correct on ENTRY and wrong
 * on EXIT, so the fix is to stop before the cascade rather than to re-derive
 * the answer afterwards.
 *
 * The ops this does NOT cover are the ones that destroy the class BEFORE a
 * leading limb is formed — `recip` and `div`, whose bit-hack Newton seed turns
 * every special into an all-NaN carrier. Those carry their own OPERAND guards,
 * at their own entries, and say so there.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalize(BandRaw r)
{
    if (bandHiIsSpecial(r.hi))
        return bandSpecialCarrier(r.hi);

    float h1, l1;
    fast2Sum(r.hi, r.lo, h1, l1);
    float lc, lb;
    fast2Sum(l1, r.tail, lc, lb);
    float h, l;
    fast2Sum(h1, lc, h, l);
    return Band{ h, l, lb };
}

/**
 * @brief The half of the bridge a CANCELLING sum needs: the opening `twoSum`
 *        does not assume `|hi| >= |lo|`, which is exactly the assumption a
 *        cancellation has destroyed. Carries the same specials guard, for the
 *        same reason and at the same price. @see normalize.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalizeSafe(BandRaw r)
{
    if (bandHiIsSpecial(r.hi))
        return bandSpecialCarrier(r.hi);

    float h1, l1;
    twoSum(r.hi, r.lo, h1, l1);
    float lc, lb;
    fast2Sum(l1, r.tail, lc, lb);
    float h, l;
    fast2Sum(h1, lc, h, l);
    return Band{ h, l, lb };
}

/// @brief Normalize the output of a cancellation-capable add. @see normalize.
/// These two overloads are the ONLY functions that accept a `BandRawCancel`,
/// and that is the whole content of the type.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalize(BandRawCancel r)
{
    return normalize(BandRaw{ r.hi, r.lo, r.tail });
}

/// @brief Order-free normalize of a cancellation-capable add's output — the
/// pairing this type exists for. @see normalizeSafe.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band normalizeSafe(BandRawCancel r)
{
    return normalizeSafe(BandRaw{ r.hi, r.lo, r.tail });
}

// =====================================================================
//  Multiply
// =====================================================================

/// @brief `a * b`, raw: three `twoProd`s, an ordered `twoSum` cascade and a
/// residual that folds the remaining cross terms in with explicit FMAs.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw mulRaw(BandRaw a, BandRaw b)
{
    float p_hi, p_lo;
    twoProd(a.hi, b.hi, p_hi, p_lo);
    float q_hi, q_lo;
    twoProd(a.hi, b.lo, q_hi, q_lo);
    float t_hi, t_lo;
    twoProd(a.lo, b.hi, t_hi, t_lo);

    float sA, eA;
    twoSum(p_lo, q_hi, sA, eA);
    float sB, eB;
    twoSum(sA, t_hi, sB, eB);

    float resid = addRN(eA, eB);
    resid       = resid + q_lo;
    resid       = resid + t_lo;
    resid       = fmaRN(a.lo, b.lo, resid);
    resid       = fmaRN(a.tail, b.hi, resid);
    resid       = fmaRN(b.tail, a.hi, resid);

    return BandRaw{ p_hi, sB, resid };
}

/**
 * @brief `a * a`, raw: the square, WITHOUT the duplicated cross term a general
 *        multiply is obliged to compute.
 *
 * `mulRaw(a, a)` forms `a.hi*a.lo` twice (FP multiplication is commutative and
 * `twoProd` is symmetric) and then spends a `twoSum` combining a number with
 * itself. Doubling a float is EXACT, so the square forms the cross term once
 * and doubles it; 24 source operations become 15.
 *
 * `hi` is always identical to `mulRaw(a, a)`'s (both are `fl(a.hi * a.hi)`), and
 * on a SINGLE-LIMB band the two are bit-identical in all three limbs. Elsewhere
 * they are NOT bit-equal, and that is the contract rather than a defect: the
 * same exact quantity `p_lo + 2*q_hi` is SPLIT differently between the `lo` limb
 * and the residual, so the guard bits below the delivered 53 move.
 *
 * Kept here because it is the shape later composites (`sumSq3`,
 * `rsqrtCore`, `rsqrtCube`) are built on; it has no in-library call site yet.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw sqrRaw(BandRaw a)
{
    float p_hi, p_lo;
    twoProd(a.hi, a.hi, p_hi, p_lo);
    float q_hi, q_lo;
    twoProd(a.hi, a.lo, q_hi, q_lo);

    // Doubling a float is EXACT, so the cross term is formed once and doubled
    // rather than computed twice and summed.
    const float d_hi = q_hi + q_hi;

    float sA, eA;
    twoSum(p_lo, d_hi, sA, eA);

    // `addRN(eA, q_lo + q_lo)` is two instructions computing one rounding;
    // `fmaRN(2, q_lo, eA)` is ONE instruction computing the SAME value, because
    // `2*q_lo` is exact and both spellings therefore round exactly once, in the
    // same place. Bit-identical by construction, not "within a bound".
    float resid = fmaRN(2.0f, q_lo, eA);
    resid       = fmaRN(a.lo, a.lo, resid);
    resid       = fmaRN(a.tail + a.tail, a.hi, resid);

    return BandRaw{ p_hi, sA, resid };
}

/// @brief `a * b` as a certified carrier — `mulRaw` plus the exit barrier.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band mul(Band a, Band b)
{
    return normalize(mulRaw(a, b));
}

// =====================================================================
//  Add / subtract
// =====================================================================

/**
 * @brief `a + b`, raw. There is no scale-alignment branch, no sign select and
 *        no sign-enforcement negation: deleting the last of those is
 *        numerically free, not a shortcut, because
 *        `fast2Sum(-a,-b) == -fast2Sum(a,b)` exactly.
 *
 * Returns `BandRawCancel`: this function does not know its operands' signs, so
 * it must be ASSUMED to cancel. @see BandRawCancel.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawCancel addRaw(
    BandRaw a, BandRaw b)
{
    float s_hi, e_hi;
    twoSum(a.hi, b.hi, s_hi, e_hi);
    float s_lo, e_lo;
    twoSum(a.lo, b.lo, s_lo, e_lo);
    float s_lo2, e_lo2;
    twoSum(s_lo, e_hi, s_lo2, e_lo2);

    float resid = addRN(e_lo, e_lo2);
    resid       = addRN(resid, a.tail);
    resid       = addRN(resid, b.tail);

    return BandRawCancel{ s_hi, s_lo2, resid };
}

/**
 * @brief `addRaw` for a caller who can prove, at COMPILE TIME, that these two
 *        operands cannot cancel — the one sanctioned exit from
 *        `BandRawCancel`.
 *
 * Bit-for-bit the same sum: this delegates to `addRaw` and re-labels the
 * result, so there is exactly one add body in this file and no way for two
 * spellings to drift apart. What differs is the PROMISE, and the promise is the
 * CALLER's — it is not checkable here.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw addRawNoCancel(
    BandRaw a, BandRaw b)
{
    const BandRawCancel r = addRaw(a, b);
    return BandRaw{ r.hi, r.lo, r.tail };
}

/// @brief `a + b` as a certified carrier.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band add(Band a, Band b)
{
    return normalize(addRaw(a, b));
}

/// @brief `a - b` as a certified carrier — the add of the negation, which
/// costs nothing extra because negation is three sign bits.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band sub(Band a, Band b)
{
    return add(a, neg(b));
}

// =====================================================================
//  Reciprocal / divide
// =====================================================================

/// @brief The magic constant of the reciprocal bit hack, DERIVED.
///
/// For `x = m 2^k` with `m` in `[1,2)` the IEEE-754 fields of `1/x` are, up to
/// the mantissa's nonlinearity, the fields of `x` SUBTRACTED from a constant:
/// `B(1/x) = 253*2^23 + (m + 2/m - 2)*2^23 - B(x)`, and `m + 2/m - 2` ranges
/// over `[0.8284, 1]`. Any `K` in that window is a seed; the one MINIMISING the
/// worst relative error over the mantissa circle is `0x7EF311C2`, worst
/// `5.051e-02 = 2^-4.307`.
///
/// ★ The SIGN comes out for free, and that is not luck: for a negative `x` the
/// pattern is `2^31 + a`, so `K - B(x)` in 32-bit wrap-around arithmetic is
/// `(K - a) - 2^31 == (K - a) | 2^31` exactly while `K - a` stays under `2^31`,
/// which it does for every admitted operand. ONE subtract, both signs, no mask
/// and no branch.
inline constexpr std::uint32_t kBandRecipSeedMagic = 0x7EF311C2u;

/**
 * @brief The reciprocal START POINT: one integer subtract and three FP32 Newton
 *        steps, IDENTICAL on the host and on the device.
 *
 * ```
 *     y  = intAsFloat(K - floatAsInt(b))     worst |dy/y| = 2^-4.307
 *     r  = fma(-b, y, 1);  y = fma(y, r, y)               2^-8.615
 *     r  = fma(-b, y, 1);  y = fma(y, r, y)               2^-17.217
 *     r  = fma(-b, y, 1);  y = fma(y, r, y)               2^-24.000
 * ```
 *
 * ★ THREE steps and not two, and the reason is a gate rather than taste. At two
 * steps the seed is `2^-17.2`, the refinement pair lands near `2^-34` instead
 * of `2^-46`, and the Newton residual's dropped `O(e^2)` term rises ABOVE the
 * carrier's `2^-72` resolution — the two-step seed is not a cheaper variant but
 * a defect.
 *
 * ★ ONE seed for both arms is what makes a HOST/DEVICE BIT-IDENTITY claim
 * possible at all for `recip`, `div` and everything downstream. Seeding from
 * `rcp.approx.f32` on the device and a correctly-rounded `1.0f/b` on the host
 * gives two ~24-bit seeds that are DIFFERENT NUMBERS, and a bit-identity claim
 * across arms is the only correctness statement available without a
 * reference implementation to compare against.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float recipSeed(float b)
{
    float y = intAsFloat(static_cast<int>(
        kBandRecipSeedMagic - static_cast<std::uint32_t>(floatAsInt(b))));
    float r = fmaRN(-b, y, 1.0f);
    y       = fmaRN(y, r, y);
    r       = fmaRN(-b, y, 1.0f);
    y       = fmaRN(y, r, y);
    r       = fmaRN(-b, y, 1.0f);
    return fmaRN(y, r, y);
}

/**
 * @brief `1/b` WITHOUT the exit barrier: the `recipSeed` start point refined
 *        into an unevaluated pair, then one Newton step in CORRECTION form.
 *
 * Two phases, both of which exist to stop paying for bits that cannot be kept:
 *
 *  - **An unevaluated pair is not capped by `float`.** An FP32 Newton step
 *    computes 22 new bits and then STORES them in a `float`, which keeps 24 in
 *    total, so a step that doubles the seed's correct bits delivers ONE. Written
 *    as the pair `y = (y0, y0*r)` with `r = 1 - b*y0` the same information costs
 *    six FP32 operations:
 *    ```
 *        twoProd(b.hi, y0) -> (s, se)   s + se = b.hi*y0 EXACTLY
 *        t = 1 - s                      EXACT by Sterbenz (s in [1/2, 2])
 *        w = fma(b.lo, y0, se)          the 2^-24-scale remainder
 *        r = t - w                      about 2^-22
 *    ```
 *    Dropped and STATED: `b.tail*y0`, about `2^-48` absolute against an `r` of
 *    `2^-22` — 24 binades under the carrier's own `2^-72` resolution.
 *  - **The residual is a SCALAR, not a carrier.** `y_new = y*(1 + e)` with
 *    `e = 1 - b*y` staged so its first subtraction is exact (Sterbenz); every
 *    later subtraction has a result no larger than its operands, so the results
 *    shrink monotonically and `e` reaches about `2^-44` carrying about `2^-68`
 *    absolute. The correction enters as TWO FP32 limbs, not one: one limb is
 *    rounded at `2^-24` of itself, which is invisible on a good seed and costs
 *    headroom on a bad one.
 *
 * Dropped, stated rather than assumed: the Newton second order `e^2`, about
 * `2^-88` relative — 34 binades below the 53 bits the carrier certifies.
 *
 * The body is the RAW one and the barrier lives in the wrapper because two
 * consumers want different things: `recip` is the public 53-bit reciprocal and
 * owes the barrier, while `div` consumes the result through `mulRaw`, an exact
 * cascade that never reads limb ORDER and for which the barrier is dead weight.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRaw recipRaw(Band b)
{
    // ONE seed, host and device. @see recipSeed.
    const float y0 = recipSeed(b.hi);

    // FP32 refinement of the bit-hack seed into a 2-limb root. The pair is
    // UNEVALUATED on purpose: `y.lo` holds the 22 bits a `float` could not.
    float s, se;
    twoProd(b.hi, y0, s, se);
    const float t = subRN(1.0f, s); // EXACT (Sterbenz, s in [1/2, 2])
    const float w = fmaRN(b.lo, y0, se);
    const float r = subRN(t, w); // r = 1 - b*y0, about 2^-22
    const BandRaw y{ y0, mulRN(y0, r), 0.0f };

    // One Band Newton residual as a SCALAR, then `y_new = y*(1 + e)` with
    // the correction carried as TWO FP32 limbs.
    const BandRaw by = mulRaw(b, y);
    const float a1   = subRN(1.0f, by.hi); // EXACT (Sterbenz)
    const float a2   = subRN(a1, by.lo);
    const float e    = subRN(a2, by.tail);

    const float cHi = mulRN(y.hi, e);
    const float cLo = addRN(fmaRN(y.hi, e, -cHi), mulRN(y.lo, e));
    float lo2, t2;
    fast2Sum(y.lo, cHi, lo2, t2);
    return BandRaw{ y.hi, lo2, addRN(t2, cLo) };
}

/**
 * @brief `1/b` as a certified carrier — `recipRaw` plus the exit barrier.
 *
 * @par ★ The SPECIALS guard is an OPERAND guard here, not the exit guard
 * `recipSeed` is a bit hack on the exponent field, so by the time a leading limb
 * exists the operand's CLASS is already gone — the seed turns every special into
 * an all-NaN carrier. `normalize`'s exit guard therefore has nothing correct
 * left to keep, and the classification has to happen BEFORE the seed. The zero
 * test rides along because a zero DIVISOR is a finite operand with an infinite
 * answer — the one row where "is the operand special" is not the same question
 * as "is the result".
 *
 * The answer itself is `1.0f / b.hi` in FP32, which IS the IEEE special table
 * for this row (`1/±0 = ±Inf`, `1/±Inf = ±0`, `1/NaN = NaN`, with the standard's
 * own signs). Spelling those five cases out in integer tests would be a second
 * implementation of the standard to keep in step with the first.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band recip(Band b)
{
    if (bandHiIsSpecial(b.hi) || b.hi == 0.0f)
        return bandSpecialCarrier(1.0f / b.hi);

    return normalize(recipRaw(b));
}

/**
 * @brief `a / b` as ONE primitive: `mulRaw(a, recipRaw(b))` with a SINGLE exit
 *        barrier.
 *
 * @par The licence for removing the inner barrier
 * A barrier is mandatory at a storage terminal, at a comparison or branch, at an
 * add/sub whose operands may have opposite signs, and at a demand-tier
 * transition. The consumer of the reciprocal HERE is `mulRaw`, which opens with
 * `twoProd`/`twoSum` cascades that are exact for ANY limb split and never read
 * limb ORDER — so it is none of the four. The barrier that a `Band` return type
 * would impose is a default, not a numerical requirement, and naming `recipRaw`
 * is what lets the site say so.
 *
 * @par The error cost, DERIVED
 * `mulRaw` drops `a.lo*b.tail`, `a.tail*b.lo` and `a.tail*b.tail`, whose bound
 * scales with `|b.lo| / ulp(b.hi)`. A NORMALIZED reciprocal has
 * `|lo| <= ulp(hi)/2`; the un-normalized one leaving `recipRaw` has
 * `|lo| <~ 2 ulp(hi)`, so those terms grow by at most 4x — from `2^-72` to
 * `2^-70` relative, sixteen binades below the `2^-54` half-ulp a 53-bit delivery
 * demands.
 *
 * @par Specials
 * An OPERAND guard, for the same reason `recip`'s is one (this body reaches
 * `recipRaw` too) and one row wider: `a.hi` can be the special while the divisor
 * is ordinary. `a.hi / b.hi` in FP32 answers every row of IEEE 754-2019's
 * division table at once, sign included.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band div(Band a, Band b)
{
    if (bandHiIsSpecial(a.hi, b.hi) || b.hi == 0.0f)
        return bandSpecialCarrier(a.hi / b.hi);

    return normalize(mulRaw(a, recipRaw(b)));
}

// =====================================================================
//  Sign and order -- the EXACT ops (the four certified facade entries)
// =====================================================================
//
//  1. The sign lives in the LIMBS, so sign work is free. A carrier with a sign
//     FIELD makes `abs` one store and then makes every downstream consumer pay
//     a select to interpret it; `Band` has the sign in the value of each limb,
//     so `abs`/`copysign` are three XORs against ONE mask and nothing
//     downstream pays anything.
//  2. The carrier has NO cheap total order, and that is the honest cost here.
//     `Band` cannot compare limb-by-limb, because two normalized carriers with
//     adjacent leading limbs can be ordered the other way by their `lo`s. So
//     `fmax`/`fmin` route the comparison through the certified `sub` and read
//     the sign of its leading limb: the ordering the carrier itself certifies,
//     over its whole ~72-bit content rather than over a 53-bit collapse.

/// @brief The three limbs XORed with ONE sign-bit pattern — the whole of this
/// carrier's sign machinery. `s` is either `0` or `0x80000000`; XORing that mask
/// into all three limbs negates the VALUE exactly, and unlike a comparison
/// against zero it carries the sign of a negative zero correctly.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band applySignMask(Band a, int s)
{
    return Band{ intAsFloat(floatAsInt(a.hi) ^ s),
        intAsFloat(floatAsInt(a.lo) ^ s), intAsFloat(floatAsInt(a.tail) ^ s) };
}

/// @brief `|a|`. Three XORs against the leading limb's own sign bit.
///
/// The mask form, not `(a.hi < 0) ? neg(a) : a`: `-0.0f < 0.0f` is FALSE, so the
/// comparison form returns `-0` for `abs(-0)` where IEEE says `+0`. It is the
/// sign BIT that is cleared, not a sign that is derived. Exact, total, and
/// defined on every carrier including the non-finite ones.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band abs(Band a)
{
    return applySignMask(a, floatAsInt(a.hi) & static_cast<int>(0x80000000u));
}

/// @brief `|a|` with the sign of `b`. One XOR of two sign bits, then the mask —
/// including both signed zeros on either argument, at the same three-XOR cost.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band copysign(Band a, Band b)
{
    return applySignMask(
        a, (floatAsInt(a.hi) ^ floatAsInt(b.hi)) & static_cast<int>(0x80000000u));
}

/**
 * @brief The larger of two carriers.
 *
 * @par The comparison is a `sub`, and the reason is a counterexample
 * The tempting body is a lexicographic limb compare, and it is WRONG: two
 * normalized carriers can have `a.hi > b.hi` and still satisfy `a < b`, e.g.
 * `a = (1, -2^-24, 0)` against `b = (1 - 2^-24, +2^-25, 0)`. The certified `sub`
 * answers the question the carrier can actually answer, over its whole content.
 *
 * @par The type system already enforces the barrier rule
 * A comparison may only happen on a NORMALIZED carrier. Nothing here checks
 * that: these take `Band`, and `BandRaw` does not convert to `Band` — only
 * `normalize` produces one. `fmax(BandRaw, BandRaw)` does not compile.
 *
 * @par NaN and ties
 * A NaN operand LOSES to a non-NaN one; both NaN gives a NaN. On `a == b` the
 * FIRST operand is returned, so `fmax(+0, -0) == +0` and `fmax(-0, +0) == -0`;
 * IEEE 754 leaves that unspecified and the choice is enumerated in the battery
 * rather than assumed.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmax(Band a, Band b)
{
    if (a.hi != a.hi)
        return b;
    if (b.hi != b.hi)
        return a;
    return (sub(a, b).hi < 0.0f) ? b : a;
}

/// @brief The smaller of two carriers. @see fmax — same comparison, same NaN
/// rule, same tie behaviour.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmin(Band a, Band b)
{
    if (a.hi != a.hi)
        return b;
    if (b.hi != b.hi)
        return a;
    return (sub(a, b).hi < 0.0f) ? a : b;
}

/**
 * @brief Positive difference: `a > b ? a - b : +0`.
 *
 * ONE subtraction, not a comparison AND a subtraction: on a banded carrier the
 * comparison IS the subtraction's sign, so the difference is formed once and its
 * own leading limb decides whether to deliver it or a zero. The explicit NaN
 * guard is on the OPERANDS rather than on the difference precisely so that two
 * equal infinities answer `+0` rather than the `Inf - Inf` NaN.
 *
 * Not a `aether::math` facade entry (the five certified entries are
 * `abs`/`copysign`/`fmax`/`fmin`/`pow`); kept because it is the shape the
 * consumer step-controllers reach for and it costs one function.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fdim(Band a, Band b)
{
    if (a.hi != a.hi || b.hi != b.hi)
        return Band{ intAsFloat(static_cast<int>(kBandCanonicalNanBits)), 0.0f, 0.0f };
    const Band d = sub(a, b);
    return (d.hi > 0.0f) ? d : Band{ 0.0f, 0.0f, 0.0f };
}

// =====================================================================
//  IEEE double -> Band  (the ingest the codec is built on)
// =====================================================================

/**
 * @brief The nearest float to `n * 2^k`, for a small unsigned `n` and ANY `k`.
 *
 * Splitting the scale into two exact factors is what keeps the multiplier
 * itself from wrapping out of the exponent field: `127 + k1` lands in `[1, 231]`
 * and `127 + k2` in `[1, 254]`, both encodable and neither ever zero, so neither
 * factor can be a subnormal or an infinity. The clamp only stops the MULTIPLIER
 * from wrapping — a genuinely unrepresentable product still overflows to the
 * same infinity it should. Two comparisons resolved as selects, no
 * data-dependent branch.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() float limbFromScaledInt(
    std::uint32_t n, int k)
{
    const int kc = (k < -252) ? -252 : ((k > 231) ? 231 : k);
    int k1       = kc / 2;
    k1           = (k1 > 104) ? 104 : k1;
    const int k2 = kc - k1;
    const float s1 = intAsFloat((127 + k1) << 23);
    const float s2 = intAsFloat((127 + k2) << 23);
    return (static_cast<float>(n) * s1) * s2;
}

/**
 * @brief IEEE double (as its two 32-bit halves) -> `Band`.
 *
 * The exact 24+24+5 mantissa split, with the exponent going STRAIGHT ONTO THE
 * FLOAT LIMBS instead of into a scale field. NO BANDING — those two lines are
 * both the whole saving and the whole ceiling exposure.
 *
 * ★ ONE unsigned range test covers every exit: `fexp_hi` outside `[1, 254]` is
 * exactly the union of `bexp == 0` (zero and every subnormal), `bexp == 2047`
 * (infinities and NaNs), a leading limb below FP32's normal range, and one above
 * it. Folding them together is what makes the tiny-head correction cost the fast
 * path NOTHING.
 *
 * @par Deliberately unchecked: the CEILING
 * Above `e = 127` the leading limb's exponent field leaves range upward and the
 * assembly wraps. That is the carrier's DECLARED ceiling obligation, enforced at
 * configuration time by the admission predicates in `BandCell8.h`; moving it
 * here would move an envelope. inf/NaN propagate through `hi` by IEEE FP32
 * rules.
 *
 * @par The sign survives underflow, deliberately
 * `-0` comes back as `-0` and a negative subnormal flushes to `-0`, which is
 * what IEEE gradual underflow does and what `recip`/`div` need in order to
 * answer `1 / -0 = -Inf` at all. The `(limb == 0) ? 0.0f : …` selects keep an
 * ABSENT lower limb a POSITIVE zero on a negative input — and they select the
 * LITERAL `0.0f`, never `x * 0.0f`, because an arithmetic "no-op" is not a no-op
 * on a NaN or an infinity.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFromIEEE(
    std::uint32_t ieee_lo, std::uint32_t ieee_hi)
{
    const std::uint32_t sign_bit = ieee_hi & 0x80000000u;
    const int bexp               = static_cast<int>((ieee_hi >> 20) & 0x7FFu);

    const std::uint32_t top24
        = 0x800000u | ((ieee_hi & 0xFFFFFu) << 3) | (ieee_lo >> 29);
    const std::uint32_t lo_int24 = (ieee_lo >> 5) & 0xFFFFFFu;
    const std::uint32_t tail_int = ieee_lo & 0x1Fu;

    // NO BANDING. These two lines are the difference from a scale-field ingest,
    // and they are both the whole saving and the whole ceiling exposure.
    const int e       = bexp - 1023;
    const int fexp_hi = e + 127;

    // The sign is folded in as a BIT, so it costs one XOR and carries no
    // arithmetic — the property the rung exists for.
    const int sign_flip = static_cast<int>(sign_bit);

    if (static_cast<std::uint32_t>(fexp_hi - 1) >= 254u) {
        if (bexp == 0)
            return Band{ intAsFloat(static_cast<int>(sign_bit)), 0.0f, 0.0f };

        if (bexp == 2047) {
            const std::uint32_t mant_nz = (ieee_hi & 0xFFFFFu) | ieee_lo;
            const std::uint32_t inf_bits
                = sign_bit | (mant_nz ? kBandCanonicalNanBits : kBandFp32ExpMask);
            return Band{ intAsFloat(static_cast<int>(inf_bits)), 0.0f, 0.0f };
        }

        if (fexp_hi <= 0) {
            // ★ THE TINY HEAD. The leading limb is subnormal as a float, or
            // below the smallest subnormal altogether, so it cannot be assembled
            // by shifting a field — the field would be zero or negative. It is
            // SCALED instead, by the same helper the two lower limbs use.
            // `top24` is never zero (the implicit bit is set), so there is no
            // absent-limb select on it; an underflow to zero KEEPS the sign.
            Band t;
            t.hi = intAsFloat(
                floatAsInt(limbFromScaledInt(top24, e - 23)) ^ sign_flip);
            const float t_lo = intAsFloat(
                floatAsInt(limbFromScaledInt(lo_int24, e - 47)) ^ sign_flip);
            t.lo = (lo_int24 == 0) ? 0.0f : t_lo;
            const float t_tl = intAsFloat(
                floatAsInt(limbFromScaledInt(tail_int, e - 52)) ^ sign_flip);
            t.tail = (tail_int == 0) ? 0.0f : t_tl;
            return t;
        }

        // `fexp_hi >= 255`, i.e. `e >= 128`: the DECLARED ceiling. Falls through
        // to the body below and takes it unchanged, deliberately.
    }

    Band r;

    const std::uint32_t hi_bits = sign_bit
        | (static_cast<std::uint32_t>(fexp_hi) << 23) | (top24 & 0x7FFFFFu);
    r.hi = intAsFloat(static_cast<int>(hi_bits));

    const float lo_else = intAsFloat(
        floatAsInt(limbFromScaledInt(lo_int24, e - 47)) ^ sign_flip);
    r.lo = (lo_int24 == 0) ? 0.0f : lo_else;

    const float tail_else = intAsFloat(
        floatAsInt(limbFromScaledInt(tail_int, e - 52)) ^ sign_flip);
    r.tail = (tail_int == 0) ? 0.0f : tail_else;

    return r;
}

/**
 * @brief `Band` -> IEEE double halves. HOST ONLY, the CARRIER's own egress
 * terminal, symmetric with `bandFromIEEE` above.
 *
 * @par Why egress must be total
 * `BandSpecialsCert`'s claim #3 is that egress delivers the infinity or
 * NaN it was handed, never laundering it into a finite value.
 *
 * @par Totality, not laundering
 * A special in `hi` (a canonical NaN or an infinity -- `bandFromIEEE` and
 * the certified arithmetic only ever PRODUCE one canonical NaN bit pattern,
 * `kBandCanonicalNanBits`) is read from `hi`'s own bits and delivered
 * unchanged, never summed. An exact zero is likewise read by VALUE, its sign
 * taken from `hi`'s bit pattern directly -- summing a `-0.0f` `hi` against
 * `+0.0f` lower limbs would round to `+0` (IEEE 754's "a sum of zeros with
 * opposite signs is +0 under round-to-nearest" rule, confirmed empirically
 * for this exact expression) and launder the sign this terminal exists to
 * carry.
 *
 * @par The finite path: the exact limb sum
 * For a finite value the contract is simply "the correctly-rounded double
 * nearest hi+lo+tail" -- and summing three ALREADY-EXACT `double` limb
 * conversions, smallest magnitude first, is exactly that:
 * `double(lo)+double(tail)` is exact (disjoint, far smaller than `hi`),
 * and the ONE addition of `hi` last is the correctly-rounded finish. Same
 * technique `detail::bandToDouble` (`BandCell8.h`) already uses for the
 * STORAGE egress terminal (which is deliberately sign-destroying at
 * zero) -- this is that technique's carrier-level sibling, made
 * sign-total instead because the carrier's own ingest is.
 */
inline void bandToIEEE(
    Band b, std::uint32_t& ieee_lo, std::uint32_t& ieee_hi)
{
    const std::uint32_t hiBits = static_cast<std::uint32_t>(floatAsInt(b.hi));
    const std::uint32_t sign   = hiBits & 0x80000000u;

    if ((hiBits & kBandFp32ExpMask) == kBandFp32ExpMask) {
        const std::uint32_t mant = hiBits & 0x7FFFFFu;
        ieee_hi = mant ? 0x7FF80000u : (sign | 0x7FF00000u);
        ieee_lo = 0u;
        return;
    }
    if (b.hi == 0.0f && b.lo == 0.0f && b.tail == 0.0f) {
        ieee_hi = sign;
        ieee_lo = 0u;
        return;
    }

    const double d = static_cast<double>(b.hi)
        + (static_cast<double>(b.lo) + static_cast<double>(b.tail));
    std::uint64_t bits64;
    std::memcpy(&bits64, &d, sizeof(bits64));
    ieee_lo = static_cast<std::uint32_t>(bits64);
    ieee_hi = static_cast<std::uint32_t>(bits64 >> 32);
}

// =====================================================================
//  Admission guard -- the DECLARED working envelope
// =====================================================================

/// @brief FP32's maximum normal exponent. `Band` is three bare floats, so this
/// — not a nominal design figure — is where the carrier actually dies.
inline constexpr int kBandHardCeilingExp = 128;

/// @brief The declared working ceiling, below the hard limit so admission has
/// room before the cliff.
inline constexpr int kBandNominalCeilingExp = 115;

/// @brief The CARRIER floor: the FP32 subnormal quantum entering the 53-bit
/// window. Carrier-resident, so no LATER operation repairs it — it binds
/// INTERMEDIATES anywhere in a chain, not only outputs.
inline constexpr int kBandCarrierFloorExp = -96;

/// @brief The default admission margin, in binades.
inline constexpr int kBandAdmissionMargin = 8;

/// @brief Does a consumer's declared largest magnitude fit under the ceiling,
/// with margin?
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandCeilingAdmits(
    int maxAbsExp, int margin = kBandAdmissionMargin)
{
    return maxAbsExp + margin <= kBandNominalCeilingExp;
}

/// @brief Does a consumer's declared smallest INTERMEDIATE magnitude stay above
/// the carrier floor, with margin?
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandIntermediateAdmits(
    int minIntermediateExp, int margin = kBandAdmissionMargin)
{
    return minIntermediateExp - margin >= kBandCarrierFloorExp;
}

} // namespace detail
} // namespace banded
} // namespace aether
