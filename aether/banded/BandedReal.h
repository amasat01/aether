// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandedReal.h
 * @brief `BandedReal` — the 8-byte storage value type of the banded carrier.
 *
 * `BandCell8` is a codec word and `Band` is a working carrier. Neither is
 * something a generic consumer can write `View<T, …>` or
 * `std::numeric_limits<T>::%epsilon()` against: the codec word carries a
 * bitwise identity (`+0 != -0`, deliberately — `BandCell8.h`'s
 * `operator==`), which is the right answer for "is this buffer still the
 * fill value" and the wrong one for arithmetic, and the working carrier
 * has no memory form at all.
 *
 * `BandedReal` is that missing face: one 64-bit cell at array bias 0,
 * with a numeric equality and total order over it, an explicit `double`
 * egress, a `std::numeric_limits` specialization (`BandedLimits.h`) and a
 * namespace-scope operator set (`BandedRealOps.h`) that returns the
 * working carrier so a chain never pays a pack per op.
 *
 * @par What it deliberately does not have
 *  - **No member arithmetic.** Chain arithmetic lives on `Band`; the
 *    namespace-scope operators return `Band`, so
 *    `BandedReal e = (a - b) * c / d;` packs once, at the assignment
 *    terminal, and never in between.
 *  - **No implicit `operator double()`.** Not a style matter: an implicit
 *    egress makes every `aether::math::*` entry point resolve silently on
 *    the host through native FP64 while the device arm `static_assert`s
 *    on the identical call — a mode-divergent trap, and a hole straight
 *    through the 0-FP64 portability warrant. Egress is by named function
 *    only, and `tests/compile_fail/check_bandedreal_typing_rejected.sh`
 *    pins that with a build that must fail.
 *  - **No `constexpr` construction from `double`.** The encode chain is
 *    not constexpr: the bit reinterprets go through `std::memcpy` and the
 *    roundings through `__builtin_lrintf`/`llrintf`. Every compile-time
 *    constant is instead built as an integer word expression over the
 *    `kBandCell8*` names — see `BandedLimits.h`.
 *
 * @par BandedReal is defined at bias 0
 * A `BandCell8`'s value is only defined relative to its array's bias. An
 * 8-byte value type has nowhere to put one, so every value-level
 * operation here — the comparisons, the `Band` decode, the limits —
 * assumes `arrayBias == 0`. Arrays needing a non-zero bias stay
 * `View<BandCell8, …>` plus an explicit bias.
 *
 * @par Include placement
 * Not reachable from `aether/aether.h` (header diet, policed by
 * `tests/headers/check_header_diet.sh`'s BANDED arms). Include the umbrella
 * `aether/banded/banded.h`, which pulls this, `BandedRealOps.h` and
 * `BandedLimits.h` in dependency order.
 */

#include <cstdint>
#include <type_traits>

#include "aether/banded/Band.h"
#include "aether/banded/BandCell8.h"
#include "aether/dtype/WorkingType.h"
#include "aether/macros.h"

namespace aether {
namespace banded {

// =====================================================================
//  BandedReal -- the value type
// =====================================================================

/// @brief The 8-byte banded storage scalar: a `BandCell8` word with numeric
/// semantics. @see the file header for what it deliberately lacks.
struct alignas(8) BandedReal {
    BandCell8 c; ///< the packed codec word, at array bias 0

    /// @brief Trivial — SoA buffers, `cudaMemcpy`, bit-cast punning. Value
    /// initialisation zero-initialises the word, i.e. the canonical positive
    /// zero.
    BandedReal() = default;

    // ── Construction ────────────────────────────────────────────────────────

    /// @brief Wrap a codec word. Explicit (a named factory, not a converting
    /// constructor): a `BandCell8` may carry a per-array bias frame and this
    /// type is defined at bias 0, so the conversion is a claim, not a
    /// coincidence.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandedReal
        fromCell(BandCell8 cell)
    {
        BandedReal r{};
        r.c = cell;
        return r;
    }

    /// @brief Construct from the raw 64-bit word — the only `constexpr`
    /// construction route (see the file header).
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandedReal
        fromBits(std::uint64_t w)
    {
        BandedReal r{};
        r.c = BandCell8{ w };
        return r;
    }

    /// @brief The raw 64-bit word.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr std::uint64_t
        toBits() const
    {
        return c.w;
    }

    /// @brief The stored cell, undecoded.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandCell8 cell() const
    {
        return c;
    }

    /// @brief Pack a working carrier at bias 0. Implicit: this is the
    /// assignment terminal a `Band` chain lands on, and it is the only place a
    /// chain is meant to pay an encode.
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal(Band b)
        : c(detail::cell8FromBand(b))
    {
    }

    /// @brief …the named spelling of the same thing.
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal fromBand(Band b)
    {
        return BandedReal(b);
    }

    /// @brief → `Band`, decoded at bias 0. Implicit: the sanctioned
    /// region-entry demote, which is why `BandedReal` is kept out of
    /// `BandWallFamily`.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() operator Band() const
    {
        return detail::bandFromCell8(c);
    }

    /// @brief …the named spelling. Also the name a coefficient-table seam
    /// detects a storage codec by (`requires { e.band(); }` rather than naming
    /// any codec type), so a type without this member silently leaves the
    /// certified decode path.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band band() const
    {
        return detail::bandFromCell8(c);
    }

    /// @brief Assignment from anything that converts (the pack terminal).
    ///
    /// Required for overload-resolution hygiene, not convenience. A template,
    /// so it is not a copy-assignment operator and trivial copyability survives.
    template<typename T>
        requires(std::is_convertible_v<T, BandedReal>
            && !std::is_same_v<std::remove_cvref_t<T>, BandedReal>)
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() BandedReal& operator=(T v)
    {
        return *this = static_cast<BandedReal>(v);
    }

    // ── Host-only `double` interop ───────────────────────────────────────────
    //
    // `double` must not appear in device code (the 0-FP64 portability warrant),
    // so these three are plain `inline` host functions and say so. Ingest routes
    // through `cell8FromDouble`, the terminal that carries the loud
    // out-of-window throw — there is no second copy of the encode to drift.

    /// @brief Host only. `double` -> `BandedReal` at bias 0, escape tier off.
    /// @throws aether::Error when the value is finite and outside tier 1's
    ///         exact window.
    [[nodiscard]] static inline BandedReal fromDouble(double x)
    {
        return fromCell(detail::cell8FromDouble(x, 0, false));
    }

    /// @brief Host only. The exact inverse of `fromDouble` on every tier-1
    /// value. Named, never an `operator double()`.
    [[nodiscard]] inline double toDouble() const
    {
        return detail::doubleFromCell8(c, 0, false);
    }

    /// @brief Host only. Explicit construction from a `double`.
    explicit inline BandedReal(double x)
        : c(detail::cell8FromDouble(x, 0, false))
    {
    }

    // ── Classification (the reserved code points) ───────────────────────────

    /// @brief True for the reserved NaN code point. There is exactly one.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool isNan() const
    {
        return detail::cell8IsSpecial(c)
            && ((static_cast<std::uint32_t>(c.w >> 32) & 0xFFFu)
                   | static_cast<std::uint32_t>(c.w))
            != 0u;
    }

    /// @brief True for either reserved infinity code point.
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool isInf() const
    {
        return detail::cell8IsSpecial(c) && !isNan();
    }

    /// @brief True for both signed zeros (`+0` is the all-zero word, `-0` the
    /// sign bit alone).
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool isZero() const
    {
        return (c.w & 0x7FFFFFFFFFFFFFFFull) == 0ull;
    }

    // ── Comparison ──────────────────────────────────────────────────────────
    //
    // The order is not the raw word order, checked by a test row
    // (`BandedRealCert.RawWordOrderIsNotValueOrder`). For a positive value
    // the 64-bit word is monotone in the value: the high half is an IEEE
    // binary32, whose bit pattern orders with its value, and the low half is
    // an unsigned continuation worth strictly less than one ulp of that
    // float ((2^32 - 1) * 2^(e-55) < 2^(e-23)), so it extends the order
    // without ever overtaking it. For a negative value the same monotonicity
    // runs backwards. `orderKey_` is the standard sign-magnitude-to-total-
    // order transform that repairs that, plus one extra fold the IEEE
    // version does not need: the two signed zeros are collapsed onto one key
    // so `-0 == +0`, which is what a numeric equality must answer and
    // precisely where `BandCell8::operator==` (bitwise, by design) gives the
    // other answer.
    //
    // Contract domain: tier 1 at bias 0, plus the three reserved code points
    // (+-Inf, NaN) and both zeros. Escape-tier words are out of contract:
    // the downward escape's extended exponent counts the wrong way in the
    // bit pattern, so an ordering claim over them would be false.

private:
    static constexpr std::uint64_t kSignBit_ = 0x8000000000000000ull;

    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr std::uint64_t
        orderKey_(std::uint64_t w)
    {
        // Collapse -0 onto +0 first: the two zeros must share one key.
        const std::uint64_t z = ((w & ~kSignBit_) == 0ull) ? 0ull : w;
        return (z & kSignBit_) ? ~z : (z | kSignBit_);
    }

public:
    /// @brief Numeric equality: `-0 == +0` is true, `NaN == NaN` is false.
    /// (`BandCell8::operator==` answers the storage question and gives the
    /// opposite answer on both — that is the reason this type is not an alias.)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator==(
        BandedReal a, BandedReal b)
    {
        if (a.isNan() || b.isNan())
            return false;
        return orderKey_(a.c.w) == orderKey_(b.c.w);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator!=(
        BandedReal a, BandedReal b)
    {
        return !(a == b);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator<(
        BandedReal a, BandedReal b)
    {
        if (a.isNan() || b.isNan())
            return false;
        return orderKey_(a.c.w) < orderKey_(b.c.w);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator>(
        BandedReal a, BandedReal b)
    {
        return b < a;
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator<=(
        BandedReal a, BandedReal b)
    {
        if (a.isNan() || b.isNan())
            return false;
        return orderKey_(a.c.w) <= orderKey_(b.c.w);
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator>=(
        BandedReal a, BandedReal b)
    {
        return b <= a;
    }
};

// =====================================================================
//  Layout claims -- the wrapper must cost nothing over the codec word
// =====================================================================

static_assert(sizeof(BandedReal) == sizeof(BandCell8),
    "BandedReal must cost exactly what the codec word costs — the whole premise "
    "of the storage port is an 8-byte element");
static_assert(sizeof(BandedReal) == 8, "one 64-bit memory transaction");
static_assert(alignof(BandedReal) == 8, "naturally aligned 64-bit access");
static_assert(std::is_trivially_copyable_v<BandedReal>,
    "BandedReal must stay trivially copyable (SoA buffers, cudaMemcpy, "
    "byte-transparent host<->device transfers)");
static_assert(std::is_standard_layout_v<BandedReal>,
    "BandedReal must stay standard-layout — the chunk memcpy contract and the "
    "texel punning both depend on it");
static_assert(std::is_trivially_default_constructible_v<BandedReal>,
    "aether's chunk/allocator fast path is gated on trivial default construction");

} // namespace banded

// =====================================================================
//  The working-carrier declaration
// =====================================================================

/**
 * @brief `BandedReal` stores as an 8-byte codec word and computes as `Band`.
 *
 * This specialization deletes the pack per node. Without it the identity
 * primary (`aether/dtype/WorkingType.h`) answers `BandedReal`, every
 * intermediate expression node encodes on the way out of its `eval()`,
 * and a three-op chain pays three RNE rounds to the codec's 56 bits where
 * one would do — at the store. Nothing about that is visible in a test
 * result (the carrier certifies 53 bits, so every packed value keeps its
 * contract); it is visible only in the instruction stream, which is why
 * the claim is checked by a row that asserts a codec-word inequality
 * against the per-node-packed reference (`tests/test_BandedReal_common.h`,
 * `BandedRealCert.ExpressionAlgebraOverBandedRealLeavesComputesInBandAndPacksOnce`).
 *
 * It lives here rather than in `WorkingType.h` because `WorkingType.h` is
 * reached from `aether/expr/Expression.h`, i.e. from every TU that builds
 * an expression at all, and it names nothing from `banded/` on purpose
 * (the header-diet boundary `tests/headers/check_header_diet.sh`
 * polices). Declaring the specialization at the end of the header that
 * defines `BandedReal` keeps the ODR question unaskable: no translation
 * unit can see the storage type without also seeing its carrier map, so
 * no TU can silently instantiate the identity primary for it and get a
 * differently-typed expression tree than its neighbour.
 *
 * The two conversions are the type's own sanctioned spellings — the
 * implicit region-entry `operator Band()` and the named pack terminal
 * `fromBand` — not a second copy of the codec bridge. Neither is
 * `constexpr` (the encode goes through `std::memcpy`/`__builtin_lrintf`,
 * which is not a constant expression; see this file's header), and
 * neither needs to be: every caller in `expr/` is a template, so this
 * instantiation is simply never constant-evaluated.
 */
template<>
struct WorkingType<banded::BandedReal> {
    using type = banded::Band;

    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() type toWorking(
        const banded::BandedReal& v)
    {
        return static_cast<banded::Band>(v);
    }
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() banded::BandedReal fromWorking(
        const type& w)
    {
        return banded::BandedReal::fromBand(w);
    }
};

} // namespace aether
