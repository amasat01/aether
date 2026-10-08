// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Ff1.h
 * @brief `Ff1` — the certified 4-byte single-float working-carrier rung,
 *        `certifiedBits == 24`.
 *
 * A single IEEE-754 binary32 limb, sign native, no lo limb, no scale/band
 * field. Sibling of `Ff2.h` (df64, w=45); this header includes it because
 * `Ff1`'s entry and exit bridges reuse `Ff2`'s certified pieces rather
 * than re-deriving them: entry demotes through `Ff2` to one FP32 add,
 * exit widens back to `Ff2{v,0}` (exact) before `Ff2`'s certified store.
 */

#include "aether/banded/Ff2.h"

namespace aether {
namespace banded {

// =====================================================================
//  Ff1 -- single-float working carrier (certifiedBits = 24, w=24 rung)
// =====================================================================

/// @brief The carriers `Ff1` must never mix with under an operator: `Ff2`
/// (the next rung up) and `Band` (the demote source). A deleted overload
/// is an exact match on the family operand and outranks the converting
/// pure-`Ff1` operator, so `Ff1 + Band` is a diagnosed error rather than
/// a silent implicit demote. Cross-rung `Ff1<->Ff2` mixing is covered
/// from this side only (ADL on the `Ff1` hidden friend catches both
/// `ff1 + ff2` and `ff2 + ff1`), so `Ff2.h`'s own wall does not
/// separately name `Ff1`.
template<typename T>
concept Ff1WallFamily
    = std::same_as<std::remove_cvref_t<T>, Ff2> || std::same_as<std::remove_cvref_t<T>, Band>;

/**
 * @brief 4-byte single-float working carrier (certifiedBits = 24).
 *
 * value = v (an IEEE-754 binary32; the sign is native). Chain-resident (like
 * Ff2/Band): built by demoting a wider `Band`/`Ff2` at region entry, flows
 * through the FP32 hidden-friend operators, and packs back to `BandedReal`
 * only at an explicit store terminal. It is never a storage element.
 */
struct Ff1 {
    float v; ///< the single binary32 limb (carries the overall sign natively)

    /// @brief The certified effective width of this carrier (the w=24 rung).
    static constexpr int certifiedBits = 24;

    /// @brief Default: uninitialized (trivial -- preserves POD-ness).
    Ff1() = default;

    /// @brief Construct directly from the single limb (the producer form).
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Ff1(float x)
        : v(x)
    {
    }

    // ── Bridges (op-surface): wider carrier -> Ff1 (demote) ─────────────────

    /// @brief Region-entry demote: `Band` -> `Ff1`, via `Ff2`'s exact
    /// tail-drop then the `Ff2 -> Ff1` limb collapse. Implicit (region-entry
    /// reads stay terse; the hidden-friend arithmetic below keeps `Band<op>
    /// Band`/`Ff2<op>Ff2` off this type regardless).
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1(Band b)
        : Ff1(Ff2(b))
    {
    }

    /// @brief Rung demote: `Ff2` (df64) -> `Ff1`, `v = RN(hi + lo)` (one FP32
    /// add).
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1(Ff2 f);

    /// @brief Store terminal (widen-then-pack half): `Ff1` -> `Ff2{v,0}`,
    /// exact (`v` widens into `hi`, `lo = 0`, no rounding).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr Ff2 toFf2() const
    {
        return Ff2{ v, 0.0f };
    }

    /// @brief Store terminal: `Ff1` -> `BandedReal`, through the exact
    /// `Ff1->Ff2` widen above then `Ff2`'s certified pack. @see `Ff2.h`'s
    /// egress section; `Ff1` inherits it by composition rather than
    /// re-deriving.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return toFf2().toBandedReal();
    }

    /// @brief Host-only terminal: `Ff1` -> `double`. @see `Ff2.h`'s egress
    /// section.
    [[nodiscard]] inline double toDouble() const { return toFf2().toDouble(); }

    // ── FP32 arithmetic (hidden friends; each ONE correctly-rounded op) ─────

    /// @brief Correctly-rounded binary32 add. @par Cost: 1 FP32.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 operator+(Ff1 a, Ff1 b)
    {
        return Ff1{ a.v + b.v };
    }
    /// @brief Unary negate: flip the sign bit. @par Cost: 1 FP32 (ptxas: XOR).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 operator-(Ff1 a)
    {
        return Ff1{ -a.v };
    }
    /// @brief Correctly-rounded binary32 sub. @par Cost: 1 FP32.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 operator-(Ff1 a, Ff1 b)
    {
        return Ff1{ a.v - b.v };
    }
    /// @brief Correctly-rounded binary32 mul. @par Cost: 1 FP32.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 operator*(Ff1 a, Ff1 b)
    {
        return Ff1{ a.v * b.v };
    }
    /// @brief Correctly-rounded binary32 div. @par Cost: 1 FP32 (div.rn / `/`).
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 operator/(Ff1 a, Ff1 b)
    {
        return Ff1{ a.v / b.v };
    }

    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1& operator+=(Ff1& a, Ff1 b)
    {
        a = a + b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1& operator-=(Ff1& a, Ff1 b)
    {
        a = a - b;
        return a;
    }
    friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1& operator*=(Ff1& a, Ff1 b)
    {
        a = a * b;
        return a;
    }

    // ── Cross-rung operator wall: no cross-rung operators, ever ─────────────
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator+(Ff1, const O&) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator+(const O&, Ff1) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator-(Ff1, const O&) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator-(const O&, Ff1) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator*(Ff1, const O&) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator*(const O&, Ff1) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator/(Ff1, const O&) = delete;
    template<typename O>
        requires Ff1WallFamily<O>
    friend Ff1 operator/(const O&, Ff1) = delete;
};
static_assert(sizeof(Ff1) == 4 && alignof(Ff1) == 4,
    "Ff1 must be a single 32-bit slot (the 4-byte w=24 carrier)");
static_assert(std::is_trivially_copyable_v<Ff1>,
    "Ff1 must stay trivially copyable (chain-resident value type)");
static_assert(Ff1::certifiedBits == 24,
    "Ff1 realizes the w=24 rung (certifiedBits 24), the WorkingCarrier door");

// =====================================================================
//  Fused FMA (single correctly-rounded binary32 fma; 0-FP64)
// =====================================================================

/// @brief Fused multiply-add `a*b + c` as one correctly-rounded binary32 fma.
/// @par Cost: 1 FP32.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 fma(Ff1 a, Ff1 b, Ff1 c)
{
    return Ff1{ detail::fmaRN(a.v, b.v, c.v) };
}

// ── Cross-rung operator wall, fma entry point ───────────────────────────────
namespace ff1_wall_detail {
template<typename T>
concept IsFf1 = std::same_as<std::remove_cvref_t<T>, Ff1>;
} // namespace ff1_wall_detail

template<typename A, typename B, typename C>
    requires(
        (ff1_wall_detail::IsFf1<A> || Ff1WallFamily<A>)
        && (ff1_wall_detail::IsFf1<B> || Ff1WallFamily<B>)
        && (ff1_wall_detail::IsFf1<C> || Ff1WallFamily<C>)
        && (Ff1WallFamily<A> || Ff1WallFamily<B> || Ff1WallFamily<C>)
        && (ff1_wall_detail::IsFf1<A> || ff1_wall_detail::IsFf1<B>
            || ff1_wall_detail::IsFf1<C>))
Ff1 fma(const A&, const B&, const C&) = delete;

// =====================================================================
//  sqrt / rsqrt (single correctly-rounded binary32 seed)
// =====================================================================

/// @brief binary32 sqrt (device SFU `__fsqrt_rn`, host `sqrtf`). `sqrt(<=0)`
/// returns 0.
/// @par Cost: 1 hw-sqrt.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 sqrt(Ff1 a)
{
    if (a.v <= 0.0f)
        return Ff1{ 0.0f };
#ifdef __CUDA_ARCH__
    return Ff1{ __fsqrt_rn(a.v) };
#else
    return Ff1{ sqrtf(a.v) };
#endif
}

/// @brief binary32 rsqrt = `1/sqrt(a)` (two RN ops -> ~1 ulp of the 24-bit
/// carrier, so ~23 effective bits; characterised, not a defect).
/// @par Cost: 1 hw-sqrt + 1 FP32.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 rsqrt(Ff1 a)
{
    return Ff1{ 1.0f } / sqrt(a);
}

// =====================================================================
//  Bridge definitions (Ff2 complete at include site)
// =====================================================================

AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1::Ff1(Ff2 f)
    : v(f.hi + f.lo) // collapse the df pair to one RN binary32 (1 FP32)
{
}

// =====================================================================
//  WorkingCarrier conformance
// =====================================================================

static_assert(WorkingCarrier<Ff1>, "Ff1 must conform to the WorkingCarrier concept");

// =====================================================================
//  Ff1Accum / Ff1AccumVec<N> -- additive siblings of Ff2Accum/BandAccum
// =====================================================================
//
// The all-Ff1 accumulator surface (the w=24 rung's fold terminal). Same
// shape and egress as `Ff2Accum`/`Ff2AccumVec<N>` (`Ff2.h`) -- only the lane
// element type differs.

/// @brief Ff1 running-sum accumulator lane (1-slot state).
struct Ff1Accum {
    Ff1 s{ 0.0f };

    Ff1Accum() = default;

    /// @brief Add an Ff1 term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(Ff1 t) { s = s + t; }
    /// @brief Subtract an Ff1 term.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(Ff1 t) { s = s - t; }

    /// @brief Terminal: the accumulated value (still Ff1).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1 normalized() const { return s; }

    /// @brief Host-only terminal: the accumulated value as a `double`.
    [[nodiscard]] inline double toDouble() const { return s.toDouble(); }

    /// @brief The accumulated value packed into `BandedReal` (RNE, one pack).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal toBandedReal() const
    {
        return s.toBandedReal();
    }
};

/// @brief N independent Ff1Accum lanes (drop-in sibling of Ff2AccumVec<N>).
template<int N>
struct Ff1AccumVec {
    Ff1Accum lanes[N];

    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Ff1Accum& lane(int i) { return lanes[i]; }
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() const Ff1Accum& lane(int i) const
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
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void addTerm(const Expression<E, Ff1>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "Ff1AccumVec<N>::addTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "Ff1AccumVec<N>::addTerm: term vector dimension must match N lanes");
        addTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }

    /// @brief Subtract a vector expression term across all N lanes, in lane order.
    template<class E>
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void subTerm(const Expression<E, Ff1>& term)
    {
        static_assert(E::element_extents::Rank == 1,
            "Ff1AccumVec<N>::subTerm: term must be a rank-1 (vector) expression");
        static_assert(E::element_extents::static_extent(0) == static_cast<std::size_t>(N),
            "Ff1AccumVec<N>::subTerm: term vector dimension must match N lanes");
        subTermUnroll_(
            static_cast<const E&>(term), std::make_index_sequence<static_cast<std::size_t>(N)>{});
    }
};

} // namespace banded
} // namespace aether
