// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Chain.h
 * @brief Compile-time normalisation tracking for banded expression chains —
 *        the operators that keep a chain unrestored between ops and pay the
 *        normalise only where a consumer reads the limbs in order.
 *
 * @section chain_states The three states, all existing types but one
 *  - `Band` — normalised and certified (`|lo| <= ulp(hi)`): state N.
 *  - `BandRawT<k>` — three limbs, ordered but unrestored, `k` raw hops since
 *    the last normalise: state R<k>. `Band * Band` is R<1> (defined with
 *    `Band` in Band.h, with the types). Converts implicitly to
 *    `detail::BandRaw` (so every raw op accepts it) and to `Band` only through
 *    the normalising constructor (`Band r = a * b + c;` normalises there).
 *  - `detail::BandRawCancel` — a cancellation-capable sum: state C. Accepted
 *    by nothing but `normalize`/`normalizeSafe` and `Band`'s converting
 *    constructor. `detail::BandRawCancelN` is the C of normalised operands.
 *
 * @section chain_ops The operators
 *  - `(N|R) * (N|R)`, with at least one R (`Band * Band` itself is Band.h's), gives `R<max(i,j)+1>`. An operand already at the budget
 *    `kBandChainMaxHops` is `normalize`d first (a raw product is ordered, so
 *    the non-Safe bridge is enough) and the hop count restarts below it.
 *  - `(N|R|C) + (N|R|C)` and `-` give C: `addRaw`'s opening cascade is exact on
 *    any three-limb operand, and the result is cancel-capable whatever went in.
 *    `Band + Band` is C-N (Band.h): a C whose operands were all normalised.
 *  - A C operand is bridged before anything else touches it: C-N by `normalize`
 *    (what the certified per-op add always did), a C with an unrestored operand by
 *    `normalizeSafe`; so `C + x` is C and `C * x` is `R<1>`.
 *  - Unary minus keeps the state.
 *  - Terminals are the existing bridges: `normalize(R)` (R converts to
 *    `BandRaw`), `normalizeSafe(C)`, and the `BandedReal` constructors
 *    from R and C. Every op that reads limbs in order (`abs`, `fmax`, `fmin`,
 *    `copysign`, comparison, division, root, transcendentals) takes
 *    `Band`; division, root and the transcendentals normalise R or C
 *    implicitly, while for the four sign/order ops and the comparisons below
 *    the overload is deleted and names the fix.
 *
 * @section chain_specials Why one specials guard at the terminal is enough
 * The guard tests the leading limb only. A special in ANY limb of an operand
 * reaches the result's leading limb before the terminal, because every raw op
 * opens on the leading limbs with `twoProd`/`twoSum`: a special operand limb
 * is either the leading limb itself (the product or sum of it with anything is
 * an infinity or a NaN, and `0*Inf`, `Inf-Inf` give NaN) or it can only exist
 * because an earlier op had already put a special in ITS leading limb (the
 * error terms are formed from the leading ones, and `twoProd(0.9, Inf)`
 * evaluates `fma(0.9, Inf, -Inf)`). Once the leading limb is an infinity or a
 * NaN no later `*`, `+` or `-` makes it finite again: `Inf*x` is `Inf` or NaN,
 * `Inf+x` is `Inf` or NaN, NaN stays NaN. An overflow is the same case one op
 * later. The rows in `tests/test_BandChain_common.h` run the specials table
 * through raw chains with the special entering at the first and at the last
 * op.
 *
 * @section chain_budget Why a budget
 * Each raw product forms its cross terms from limbs that are no longer
 * ordered, so the neglected terms grow with the hop count; the constant is
 * set by the derived bound `bandChainErrorBound` below, pinned by
 * `static_assert`, and the accuracy rows (`BandChain.HopsEvidence`) record
 * the observation under it.
 */

#include "aether/banded/Band.h"
#include "aether/macros.h"

#include <concepts>
#include <type_traits>

namespace aether {
namespace banded {
namespace detail {

/// @brief Worst-case relative error of a chain of `K` raw multiplies with one terminal normalise, derived from
/// mulRaw's structure (Band.h): the twoSum/twoProd cascades are exact; what grows with hops is the MAGNITUDE of the
/// unrestored limbs. In units of ulp(hi) for `lo` and 2^-48*|hi| for `tail`, with normalised inputs (lo = 1/2, tail = 1):
///   lo_k   = 1/2 + 2*lo_{k-1}                      (p_lo plus the two cross products, each carrying an operand's lo)
///   tail_k = 2*lo_{k-1} + lo_{k-1}^2 + 2*tail_{k-1} (q_lo + t_lo, a.lo*b.lo, the two tail*hi terms)
///   e_k    = (2*lo_{k-1}*tail_{k-1} + tail_{k-1}^2*2^-24)*2^-72   dropped products (a.lo*b.tail, a.tail*b.lo, a.tail*b.tail)
///          + 6*2^-24*tail_k*2^-48                                   the six rounded FP32 ops of the residue chain
/// and the chain error is the sum of the e_k. Values: K=1 2^-67.6, K=2 2^-65.4, K=3 2^-63.2, K=4 2^-61.0, K=5 2^-58.7,
/// K=6 2^-56.1 against the 2^-52 contract. Measured (200k adversarial chains per K, __float128 reference): 2^-68.9 ...
/// 2^-65.0 at K = 1..6 -- the bound is conservative by 2^3..2^9, as a worst case over limb signs should be.
[[nodiscard]] constexpr double bandChainErrorBound(int hops) noexcept
{
    double lo = 0.5, tail = 1.0, sum = 0.0;
    for (int k = 1; k <= hops; ++k) {
        const double tailNext = 2.0 * lo + lo * lo + 2.0 * tail;
        sum += (2.0 * lo * tail + tail * tail * 0x1p-24) * 0x1p-72 + 6.0 * 0x1p-24 * tailNext * 0x1p-48;
        lo   = 0.5 + 2.0 * lo;
        tail = tailNext;
    }
    return sum;
}

/// @brief Raw hops a chain may accumulate before an operand is normalised: set to 3 by ruling (K = 4 also passes the
/// 2^8 margin; K+2 does not), justified by the two-sided pin below.
inline constexpr int kBandChainMaxHops = 3;
static_assert(bandChainErrorBound(kBandChainMaxHops) <= 0x1p-52 * 0x1p-8,
    "the raw-hop budget must keep a 2^8 margin under the 1 ULP53 contract");
static_assert(bandChainErrorBound(kBandChainMaxHops + 2) > 0x1p-52 * 0x1p-8,
    "the budget is the largest with that margin (K+2 fails it)");

/// @brief The same bound for a chain of `K` fused hops `fma(a, b, c)` with a NORMALISED addend: each hop is the raw
/// product above followed by `addRaw`'s cascade (the addend adds `lo_c = 1/2`, `tail_c = 1`; the cascade rounds three
/// FP32 sums of the residue):
///   lo'    = lo + lo_c + 1/2
///   resid  = (lo + lo_c) + lo' + tail + tail_c
///   e      = mul-hop terms + 3*2^-24*resid*2^-48
/// Values: K=1 2^-66.5, K=2 2^-63.4, K=3 2^-60.8, K=4 2^-58.4. An addend that is itself unrestored raises the K = 3
/// value to 2^-58.6 and breaks the 2^8 margin, which is why the fused hop takes a normalised addend only.
[[nodiscard]] constexpr double bandChainFmaErrorBound(int hops) noexcept
{
    double lo = 0.5, tail = 1.0, sum = 0.0;
    for (int k = 1; k <= hops; ++k) {
        const double tailNext = 2.0 * lo + lo * lo + 2.0 * tail;
        sum += (2.0 * lo * tail + tail * tail * 0x1p-24) * 0x1p-72 + 6.0 * 0x1p-24 * tailNext * 0x1p-48;
        const double loMul = 0.5 + 2.0 * lo;
        const double loNew = loMul + 0.5 + 0.5;
        const double resid = (loMul + 0.5) + loNew + tailNext + 1.0;
        sum += 3.0 * 0x1p-24 * resid * 0x1p-48;
        lo   = loNew;
        tail = resid;
    }
    return sum;
}
static_assert(bandChainFmaErrorBound(kBandChainMaxHops) <= 0x1p-52 * 0x1p-8,
    "the fused-hop budget must keep a 2^8 margin under the 1 ULP53 contract");
static_assert(bandChainFmaErrorBound(kBandChainMaxHops + 1) > 0x1p-52 * 0x1p-8,
    "the budget is the largest with that margin for the fused hop (K+1 fails it)");

template<typename T>
inline constexpr bool kIsChainRaw = false;
template<int K>
inline constexpr bool kIsChainRaw<BandRawT<K>> = true;

/// @brief Any raw state: R<k> or the cancel-capable C.
template<typename T>
concept IsChainRaw = kIsChainRaw<std::remove_cvref_t<T>>;
template<typename T>
concept IsChainCancel = std::same_as<std::remove_cvref_t<T>, BandRawCancel>
    || std::same_as<std::remove_cvref_t<T>, BandRawCancelN>;
/// @brief Operands that are ordered after their own bridge: a `Band`, or a C of normalised operands (C-N).
template<typename T>
inline constexpr bool kChainOrdered = std::same_as<std::remove_cvref_t<T>, Band>
    || std::same_as<std::remove_cvref_t<T>, BandRawCancelN>;
template<typename T>
concept IsChainOperand = std::same_as<std::remove_cvref_t<T>, Band> || IsChainRaw<T> || IsChainCancel<T>;

/// @brief A binary expression with at least one unrestored operand — so
/// `Band op Band` is never contended: that stays on Band's hidden friends.
template<typename A, typename B>
concept ChainPair = IsChainOperand<A> && IsChainOperand<B>
    && (IsChainRaw<A> || IsChainCancel<A> || IsChainRaw<B> || IsChainCancel<B>);

template<typename T>
inline constexpr int kChainHops = 0;
template<int K>
inline constexpr int kChainHops<BandRawT<K>> = K;

/// @brief The bridge a C takes before anything else touches it: C-N (normalised operands, the certified per-op add)
/// takes `normalize`; C with an unrestored operand takes the order-free `normalizeSafe`.
template<typename T>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band chainBridgeCancel(T x)
{
    if constexpr (std::same_as<std::remove_cvref_t<T>, BandRawCancelN>)
        return normalize(x);
    else
        return normalizeSafe(x);
}

/// @brief Hand a multiplicand to `mulRaw`: C is bridged, an R at
/// the budget is normalised, everything else passes untouched.
template<typename T>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto chainMulOperand(T x)
{
    if constexpr (IsChainCancel<T>)
        return chainBridgeCancel(x);
    else if constexpr (IsChainRaw<T> && kChainHops<T> >= kBandChainMaxHops)
        return normalize(static_cast<BandRaw>(x));
    else
        return x;
}

/// @brief Hand an addend to `addRaw`: only C needs a bridge first.
template<typename T>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto chainAddOperand(T x)
{
    if constexpr (IsChainCancel<T>)
        return chainBridgeCancel(x);
    else
        return x;
}

/// @brief The `mulRaw` hop with the state bookkeeping: `R<max(i,j)+1>`.
template<typename A, typename B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto chainMul(A a, B b)
{
    const auto pa = chainMulOperand(a);
    const auto pb = chainMulOperand(b);
    constexpr int ha = kChainHops<std::remove_cvref_t<decltype(pa)>>;
    constexpr int hb = kChainHops<std::remove_cvref_t<decltype(pb)>>;
    constexpr int k  = (ha > hb ? ha : hb) + 1;
    const BandRaw r  = mulRaw(BandRaw(pa), BandRaw(pb));
    return BandRawT<k>{ r.hi, r.lo, r.tail };
}

/// @brief Product through the chain: `R<1>` from a normalised operand.
template<typename A, typename B>
    requires ChainPair<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto operator*(A a, B b)
{
    return chainMul(a, b);
}

/// @brief Quotient through the chain: division is a door, so each unrestored operand is normalised once (its
/// converting constructor), and the quotient is one raw hop (`divRaw`), `R<1>`.
template<typename A, typename B>
    requires ChainPair<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawT<1> operator/(A a, B b)
{
    return divRaw(BandRaw(static_cast<Band>(a)), static_cast<Band>(b));
}

/// @brief The C kind a sum of `A` and `B` carries: C-N when both are ordered after their own bridge, else C.
template<typename A, typename B>
using ChainSumT = std::conditional_t<kChainOrdered<A> && kChainOrdered<B>, BandRawCancelN, BandRawCancel>;

/// @brief Sum through the chain: always C (C-N from ordered operands).
template<typename A, typename B>
    requires ChainPair<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() ChainSumT<A, B> operator+(A a, B b)
{
    const BandRawCancel r = addRaw(BandRaw(chainAddOperand(a)), BandRaw(chainAddOperand(b)));
    return ChainSumT<A, B>{ r.hi, r.lo, r.tail };
}

/// @brief Difference through the chain: always C (C-N from ordered operands).
template<typename A, typename B>
    requires ChainPair<A, B>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() ChainSumT<A, B> operator-(A a, B b)
{
    const BandRawCancel r = addRaw(BandRaw(chainAddOperand(a)), neg(BandRaw(chainAddOperand(b))));
    return ChainSumT<A, B>{ r.hi, r.lo, r.tail };
}

/// @brief Negation keeps the state: three sign flips order nothing.
template<int K>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawT<K> operator-(BandRawT<K> a)
{
    return BandRawT<K>{ -a.hi, -a.lo, -a.tail };
}
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawCancel operator-(BandRawCancel a)
{
    return BandRawCancel{ -a.hi, -a.lo, -a.tail };
}
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawCancelN operator-(BandRawCancelN a)
{
    return BandRawCancelN{ -a.hi, -a.lo, -a.tail };
}

/// @brief Fused multiply-add through the chain: `a * b + c` as ONE raw sequence (one `addRaw` cascade saved against
/// `a * b + c` spelled with operators), `R<max(i,j)+1>` like the product. The multiplicands follow the product's
/// budget rule; the addend must be NORMALISED (`Band`): the budget is certified for that case only.
template<typename A, typename B>
    requires(IsChainOperand<A> && IsChainOperand<B> && (IsChainRaw<A> || IsChainCancel<A> || IsChainRaw<B> || IsChainCancel<B>))
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto fma(A a, B b, Band c)
{
    const auto pa = chainMulOperand(a);
    const auto pb = chainMulOperand(b);
    constexpr int ha = kChainHops<std::remove_cvref_t<decltype(pa)>>;
    constexpr int hb = kChainHops<std::remove_cvref_t<decltype(pb)>>;
    constexpr int k  = (ha > hb ? ha : hb) + 1;
    const BandRawCancel r = fmaRaw(BandRaw(pa), BandRaw(pb), BandRaw(c));
    return BandRawT<k>{ r.hi, r.lo, r.tail };
}

/// @brief A raw addend is refused by name: the fused-hop bound holds for a normalised addend only.
template<typename A, typename B, typename C>
    requires(IsChainOperand<A> && IsChainOperand<B> && (IsChainRaw<C> || IsChainCancel<C>))
Band fma(A, B, C)
{
    static_assert(sizeof(C) == 0, "fma needs a normalised addend: normalize(R) or normalizeSafe(C) it first");
    return Band{};
}

/// @brief Product with a one-limb literal through the chain: `R<hops+1>`, the product's budget rule.
template<typename A>
    requires IsChainOperand<A>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto mulRawS(A a, float f)
{
    const auto pa = chainMulOperand(a);
    constexpr int k = kChainHops<std::remove_cvref_t<decltype(pa)>> + 1;
    const BandRaw r = mulRawS(BandRaw(pa), f);
    return BandRawT<k>{ r.hi, r.lo, r.tail };
}

/// @brief Sum with a one-limb literal through the chain: C-N from a normalised operand, else C.
template<typename A>
    requires IsChainOperand<A>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto addRawS(A a, float f)
{
    using R_ = ChainSumT<A, Band>;
    const BandRawCancel r = addRawS(BandRaw(chainAddOperand(a)), f);
    return R_{ r.hi, r.lo, r.tail };
}

/// @brief Exact power-of-two scaling of a raw value: three exact multiplies, the state (and hop count) unchanged.
template<int K>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandRawT<K> scalePow2f(BandRawT<K> a, float s)
{
    return BandRawT<K>{ a.hi * s, a.lo * s, a.tail * s };
}

/// @brief Square through the chain, the same state rule as the product.
template<typename A>
    requires IsChainOperand<A>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() auto chainSqr(A a)
{
    const auto pa    = chainMulOperand(a);
    constexpr int ha = kChainHops<std::remove_cvref_t<decltype(pa)>>;
    const BandRaw r  = sqrRaw(BandRaw(pa));
    return BandRawT<ha + 1>{ r.hi, r.lo, r.tail };
}

/// @brief The store terminal's bridge: R and C-N are ordered (`normalize`), C is not
/// (`normalizeSafe`).
template<int K>
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band chainNormalize(BandRawT<K> r)
{
    return normalize(static_cast<BandRaw>(r));
}
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band chainNormalize(BandRawCancel r)
{
    return normalizeSafe(r);
}
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band chainNormalize(BandRawCancelN r)
{
    return normalize(r);
}

// Order-reading ops refuse the unrestored states by name: the overload is
// viable only for an R or C operand and fails with the fix in the message.
template<typename T>
    requires(IsChainRaw<T> || IsChainCancel<T>)
Band abs(T)
{
    static_assert(sizeof(T) == 0, "abs reads the limbs in order: normalize(R) or normalizeSafe(C) the chain first");
    return Band{};
}
template<typename T, typename U>
    requires(IsChainRaw<T> || IsChainCancel<T> || IsChainRaw<U> || IsChainCancel<U>)
Band fmax(T, U)
{
    static_assert(sizeof(T) == 0, "fmax reads the limbs in order: normalize/normalizeSafe the chain first");
    return Band{};
}
template<typename T, typename U>
    requires(IsChainRaw<T> || IsChainCancel<T> || IsChainRaw<U> || IsChainCancel<U>)
Band fmin(T, U)
{
    static_assert(sizeof(T) == 0, "fmin reads the limbs in order: normalize/normalizeSafe the chain first");
    return Band{};
}
template<typename T, typename U>
    requires(IsChainRaw<T> || IsChainCancel<T> || IsChainRaw<U> || IsChainCancel<U>)
Band copysign(T, U)
{
    static_assert(sizeof(T) == 0, "copysign reads the limbs in order: normalize/normalizeSafe the chain first");
    return Band{};
}

} // namespace detail
} // namespace banded
} // namespace aether
