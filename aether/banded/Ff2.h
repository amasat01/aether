// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Ff2.h
 * @brief `Ff2` — the certified 8-byte float-float (df64) working-carrier
 *        rung, `certifiedBits == 45`.
 *
 * The layout is `{float hi, lo}`, sign native, no scale/tail field. Every
 * arithmetic op body (`+ - * /`, the fused `fma`, the `sqrt`/`rsqrt`
 * Newton pair) and the `Ff2Accum`/`Ff2AccumVec<N>` accumulator siblings
 * live here. Every FP32 elementary op routes through
 * `aether::banded::detail`'s `twoSum`/`fast2Sum`/`twoProd`/`fmaRN`/
 * `addRN` (`detail/Fp32.h`), the same error-free transforms `Band`'s own
 * certified core uses, so this rung costs nothing new at the primitive
 * level.
 *
 * @section entry The `Band` -> `Ff2` demote
 * `Band` already carries its sign natively in the limbs and has no
 * external scale field, so the demote is a plain drop of the `tail` limb
 * (the residue below `Ff2`'s 45-bit certified width — `Band`'s own
 * docstring calls `tail` "what lifts the carrier past the df64 ceiling").
 * No sign resolve, no scale fold, no rounding: `Ff2{b.hi, b.lo}` is the
 * whole body.
 *
 * @section egress_ff2 Two egress names, no third
 * `toBandedReal()` (`AETHER_DEVICEHOST()`) is the widen-then-pack chain
 * `Ff2 -> Band{hi,lo,0} -> BandedReal`: an exact widen, one round-to-
 * nearest-even pack. `toDouble()` is host-only and composes
 * `toBandedReal().toDouble()`, so there is only one encode path.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandedReal.h"
#include "aether/banded/detail/Fp32.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace aether {
namespace banded {

// =====================================================================
//  Ff2 -- certified 2-slot float-float df64 carrier (certifiedBits = 45)
// =====================================================================

/// @brief The carrier `Ff2` must never mix with under an operator. aether
/// carries only one wide carrier (`Band`), so the family is a single
/// name. `Ff1` is deliberately not in this family: `Ff1.h`'s own wall
/// (`Ff1WallFamily`, which does name `Ff2`) already covers both operand
/// orders of `ff1 <op> ff2` via ADL on the `Ff1` hidden friend, so only
/// one side of a cross-rung pair needs to declare the wall.
template<typename T>
concept Ff2WallFamily = std::same_as<std::remove_cvref_t<T>, Band>;

/**
 * @brief 8-byte float-float df64 working carrier (certifiedBits = 45).
 *
 * value = hi + lo, with the running invariant `|lo| <= ulp(hi)` after every
 * primitive (the terminal Fast2Sum in each op re-establishes it). The overall
 * sign lives natively in `hi` (and `lo`) — no external sign word, no scale
 * field, no memory form (like `Band`, `Ff2` is chain-resident only).
 */
struct Ff2 {
    float hi; ///< df64 high limb (carries the overall sign natively)
    float lo; ///< df64 low limb (|lo| <= ulp(hi) after each certified op)

    /// @brief The certified effective width of this carrier.
    static constexpr int certifiedBits = 45;

    /// @brief Default: uninitialized (trivial -- preserves POD-ness).
    Ff2() = default;

    /// @brief Construct directly from the two limbs (the df64 producer form).
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Ff2(float h, float l)
        : hi(h)
        , lo(l)
    {
    }

    /// @brief Region-entry demote: `Band` (53-bit, 3-limb) -> `Ff2` (45-bit,
    /// 2-limb). @see the file header's "entry" section — an exact limb drop,
    /// no sign/scale work needed (`Band` already carries neither externally).
    /// Implicit, so region-entry reads stay terse; the hidden-friend
    /// arithmetic below keeps `Band <op> Band` off this type entirely (a
    /// demote ctor is only a viable conversion when an operand is already
    /// `Ff2`).
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Ff2(Band b)
        : hi(b.hi)
        , lo(b.lo)
    {
    }

    /// @brief Store terminal (widen-then-pack half): `Ff2` -> `Band`, exact
    /// (`tail = 0`, no rounding). @see `toBandedReal()` for the full pack.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Band toBand() const
    {
        return Band{ hi, lo, 0.0f };
    }

    /// @brief Store terminal: `Ff2` -> `BandedReal`, through the exact widen
    /// above then `Band`'s own certified RNE pack. @see the file header's
    /// egress section.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return BandedReal(toBand());
    }

    /// @brief Host-only terminal: `Ff2` -> `double`, via `toBandedReal()`'s
    /// own exact inverse. @see the file header's egress section. No second
    /// encode path to drift.
    [[nodiscard]] inline double toDouble() const { return toBandedReal().toDouble(); }

    // ── df64 arithmetic (hidden friends; FP32-only, op counts in comments) ──
    //
    // Hidden friends: ADL finds an Ff2 operator only when an operand is Ff2,
    // so plain Band expressions never see these (the demote ctor would
    // otherwise make `band + band` ambiguous). Certs keep arithmetic pure-Ff2
    // (inputs demoted at entry).

    /// @brief Order-free df64 add (robust through hi-limb cancellation).
    /// @par Cost: ~19 FP32 (3x twoSum + 2 add + 1 fast2Sum).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 operator+(Ff2 a, Ff2 b)
    {
        float sh, sl;
        detail::twoSum(a.hi, b.hi, sh, sl); // 6 FP32
        float th, tl;
        detail::twoSum(a.lo, b.lo, th, tl); // 6 FP32
        float c = detail::addRN(sl, th);    // 1 FP32
        float vh, vl;
        detail::twoSum(sh, c, vh, vl);   // 6 FP32 (order-free)
        float w = detail::addRN(vl, tl); // 1 FP32
        float hi, lo;
        detail::fast2Sum(vh, w, hi, lo); // 3 FP32
        return Ff2{ hi, lo };
    }

    /// @brief Unary negate: flip both limbs (2 FP32 negs; ptxas folds to XOR).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 operator-(Ff2 a)
    {
        return Ff2{ -a.hi, -a.lo };
    }

    /// @brief df64 sub: `a + (-b)` (order-free add of the negated operand).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 operator-(Ff2 a, Ff2 b)
    {
        return a + Ff2{ -b.hi, -b.lo };
    }

    /// @brief df64 mul: exact twoProd(hi) + FMA cross terms + Fast2Sum.
    /// @par Cost: ~10 FP32 (twoProd 2 + 3 FMA + fast2Sum 3 + neg-fold).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 operator*(Ff2 a, Ff2 b)
    {
        float p, e;
        detail::twoProd(a.hi, b.hi, p, e); // 2 FP32 (exact)
        e = detail::fmaRN(a.hi, b.lo, e);  // 1 FP32
        e = detail::fmaRN(a.lo, b.hi, e);  // 1 FP32
        e = detail::fmaRN(a.lo, b.lo, e);  // 1 FP32 (tail, +margin)
        float hi, lo;
        detail::fast2Sum(p, e, hi, lo); // 3 FP32
        return Ff2{ hi, lo };
    }

    /// @brief df64 div: `a/b` via float-div seed + two df-Newton corrections.
    /// @par Cost: ~40 FP32 + 3 float-div. Caller ensures `b != 0`.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 operator/(Ff2 a, Ff2 b)
    {
        float q1 = a.hi / b.hi; // ~24-bit seed
        Ff2 r    = a - (Ff2{ q1, 0.0f } * b); // df residual
        float q2 = r.hi / b.hi;               // correction 1
        Ff2 r2   = r - (Ff2{ q2, 0.0f } * b); // df residual 2
        float q3 = r2.hi / b.hi;              // correction 2
        float hi, lo;
        detail::fast2Sum(q1, q2, hi, lo); // ~47-bit quotient df
        return Ff2{ hi, lo } + Ff2{ q3, 0.0f };
    }

    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2& operator+=(Ff2& a, Ff2 b)
    {
        a = a + b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2& operator-=(Ff2& a, Ff2 b)
    {
        a = a - b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2& operator*=(Ff2& a, Ff2 b)
    {
        a = a * b;
        return a;
    }

    // ── Cross-rung operator wall: no cross-rung operators, ever ─────────────
    //
    // Mixing the Ff2 rung with `Band` under + - * / is a hard compile error.
    // Hidden friends (ADL-found only when an operand is Ff2), so
    // `Band <op> Band` stays untouched.
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator+(Ff2, const O&) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator+(const O&, Ff2) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator-(Ff2, const O&) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator-(const O&, Ff2) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator*(Ff2, const O&) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator*(const O&, Ff2) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator/(Ff2, const O&) = delete;
    template<typename O>
        requires Ff2WallFamily<O>
    friend Ff2 operator/(const O&, Ff2) = delete;
};
static_assert(sizeof(Ff2) == 8 && alignof(Ff2) == 4,
    "Ff2 must be 2 x 32-bit slots (the 8-byte, half-of-Band carrier)");
static_assert(std::is_trivially_copyable_v<Ff2>,
    "Ff2 must stay trivially copyable (chain-resident value type)");
static_assert(Ff2::certifiedBits == 45,
    "Ff2 realizes the w=45 rung (certifiedBits 45), the WorkingCarrier door");

// =====================================================================
//  Fused df64 FMA (single final rounding; addend enters UNROUNDED)
// =====================================================================

/// @brief Fused df64 multiply-add: `a*b + c`, without double-rounding the
/// addend path. @par Cost: ~30 FP32.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 fma(Ff2 a, Ff2 b, Ff2 c)
{
    float p, pe;
    detail::twoProd(a.hi, b.hi, p, pe); // exact hi product
    float pl = detail::fmaRN(a.hi, b.lo, pe);
    pl       = detail::fmaRN(a.lo, b.hi, pl);
    pl       = detail::fmaRN(a.lo, b.lo, pl);
    float sh, sl;
    detail::twoSum(p, c.hi, sh, sl);
    float th, tl;
    detail::twoSum(pl, c.lo, th, tl);
    float uh, ul;
    detail::twoSum(sl, th, uh, ul); // order-free low combine
    float vh, vl;
    detail::twoSum(sh, uh, vh, vl); // order-free
    float w = detail::addRN(detail::addRN(vl, ul), tl);
    float hi, lo;
    detail::fast2Sum(vh, w, hi, lo);
    return Ff2{ hi, lo };
}

// ── Cross-rung operator wall, fma entry point ───────────────────────────────
namespace ff2_wall_detail {
template<typename T>
concept IsFf2 = std::same_as<std::remove_cvref_t<T>, Ff2>;
} // namespace ff2_wall_detail

template<typename A, typename B, typename C>
    requires(
        (ff2_wall_detail::IsFf2<A> || Ff2WallFamily<A>)
        && (ff2_wall_detail::IsFf2<B> || Ff2WallFamily<B>)
        && (ff2_wall_detail::IsFf2<C> || Ff2WallFamily<C>)
        && (Ff2WallFamily<A> || Ff2WallFamily<B> || Ff2WallFamily<C>)
        && (ff2_wall_detail::IsFf2<A> || ff2_wall_detail::IsFf2<B>
            || ff2_wall_detail::IsFf2<C>))
Ff2 fma(const A&, const B&, const C&) = delete;

// =====================================================================
//  df64 sqrt / rsqrt (hardware/float sqrt seed + Newton pattern)
// =====================================================================

/// @brief df64 sqrt: hardware/float sqrt seed + one df-Newton correction.
/// `sqrt(a) ~ x + (a - x^2)/(2x)`. `sqrt(<=0)` returns 0.
/// @par Cost: ~14 FP32 + 1 hw-sqrt.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 sqrt(Ff2 a)
{
    if (a.hi <= 0.0f)
        return Ff2{ 0.0f, 0.0f };
#ifdef __CUDA_ARCH__
    float x = __fsqrt_rn(a.hi); // correctly-rounded SFU sqrt seed
#else
    float x = sqrtf(a.hi);
#endif
    Ff2 r      = a - (Ff2{ x, 0.0f } * Ff2{ x, 0.0f }); // df residual a - x^2
    float corr = r.hi / (2.0f * x);                     // df-Newton correction
    float hi, lo;
    detail::fast2Sum(x, corr, hi, lo);
    return Ff2{ hi, lo };
}

/// @brief df64 rsqrt: `1/sqrt(a)` via the df sqrt + df reciprocal.
/// @par Cost: ~55 FP32 (sqrt + div).
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 rsqrt(Ff2 a)
{
    return Ff2{ 1.0f, 0.0f } / sqrt(a);
}

// =====================================================================
//  WorkingCarrier concept (conformance door)
// =====================================================================

/// @brief The pluggable-working-carrier shape: a chain-resident value
/// type that advertises a certified effective width, stores to the
/// `BandedReal` terminal, and composes under `+`/`-`/`*`. `Ff2` (45)
/// conforms; `Band` (the widest carrier) is reached through a different
/// terminal (its own implicit `BandedReal` ctor) and deliberately does
/// not carry a `certifiedBits`-tagged demand door here.
template<typename W>
concept WorkingCarrier = requires(W a, W b) {
    { W::certifiedBits } -> std::convertible_to<int>;
    { a.toBandedReal() } -> std::same_as<BandedReal>;
    { a + b };
    { a - b };
    { a * b };
};

static_assert(WorkingCarrier<Ff2>,
    "Ff2 must conform to the WorkingCarrier concept");

// =====================================================================
//  Ff2Accum / Ff2AccumVec<N> -- additive siblings of BandAccum (Accum.h)
// =====================================================================
//
// Sibling of `BandAccum` at the same namespace level (`aether::banded`,
// not nested in `detail`). State is a single `Ff2` lane; egress follows
// the same two-name pattern as `BandAccum` (`toDouble()`/`toBandedReal()`).

/// @brief Ff2 running-sum accumulator lane (2-slot state).
///
/// Chains terms through the order-free df64 add (cancellation-safe at the
/// terminal). State is a single `Ff2` -- no other members, no dispatch, no
/// heap.
struct Ff2Accum {
    Ff2 s{ 0.0f, 0.0f };

    Ff2Accum() = default;

    /// @brief Add an Ff2 term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(Ff2 t) { s = s + t; }
    /// @brief Subtract an Ff2 term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(Ff2 t) { s = s - t; }

    /// @brief Terminal: the accumulated df64 value (still Ff2).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2 normalized() const { return s; }

    /// @brief Host-only terminal: the accumulated value as a `double`.
    /// @see the file header's egress section.
    [[nodiscard]] inline double toDouble() const { return s.toDouble(); }

    /// @brief The accumulated value packed into `BandedReal` (RNE, one pack).
    /// @see the file header's egress section.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return s.toBandedReal();
    }
};

/// @brief N independent Ff2Accum lanes (drop-in sibling of
/// `BandAccumVec<N>`).
///
/// Mirrors `BandAccumVec<N>`'s own `std::index_sequence` unroll idiom
/// (@see `Accum.h`'s file docstring "vec" section); only the lane element
/// type differs.
template<int N>
struct Ff2AccumVec {
    Ff2Accum lanes[N];

    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff2Accum& lane(int i) { return lanes[i]; }
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() const Ff2Accum& lane(int i) const
    {
        return lanes[i];
    }

private:
    template<class E, std::size_t... Is>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTermUnroll_(
        const E& e, std::index_sequence<Is...>)
    {
        ((lane(static_cast<int>(Is)).addTerm(e.template eval<Is>(SampleIndex::make(0)))), ...);
    }
    template<class E, std::size_t... Is>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTermUnroll_(
        const E& e, std::index_sequence<Is...>)
    {
        ((lane(static_cast<int>(Is)).subTerm(e.template eval<Is>(SampleIndex::make(0)))), ...);
    }

public:
    /// @brief Add a vector expression term across all N lanes, in lane order.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(const Expression<E, Ff2>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "Ff2AccumVec<N>::addTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "Ff2AccumVec<N>::addTerm: term vector dimension must match N lanes");
        addTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }

    /// @brief Subtract a vector expression term across all N lanes, in lane order.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(const Expression<E, Ff2>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "Ff2AccumVec<N>::subTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "Ff2AccumVec<N>::subTerm: term vector dimension must match N lanes");
        subTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }
};

} // namespace banded
} // namespace aether
