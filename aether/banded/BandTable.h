// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandTable.h
 * @brief `BandPair` — the fetchable (hi, lo) float-pair coefficient-table
 *        element, `Band`-typed half.
 *
 * @section scope_bandtable Scope
 * This file provides the layout (`{float hi, lo}`, 8-byte aligned, one
 * texel), `band()`'s zero-arithmetic decode, the `Band`-typed seam
 * operators, the identity comparison, and the host-only ingest
 * (`bandPairFromDouble`/`bandPairToDouble`/`bandPairAdmits`/
 * `bandPairCertifiedBits`/`bandPairTableFromDoubles`).
 *
 * @section texture What is not in this file
 * `fromTexel(float2)` and texture-array fetch wiring are not implemented
 * here: aether has no texture-array machinery yet. `BandPair` here is a
 * plain SoA-friendly value type; a consumer that fetches it out of a
 * texture needs that integration separately.
 *
 * @section throws The throw-loud guard
 * `bandPairTableFromDoubles` reports a bad coefficient through
 * `err::fail(operation, where, bytes, detail)` (`aether/err/Error.h`), the
 * same spelling `BandCell8.h`'s `cell8FromDouble` throw already uses in
 * this tree.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandedReal.h"
#include "aether/err/Error.h"
#include "aether/macros.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace aether {
namespace banded {

// =====================================================================
//  Envelope constants
// =====================================================================

/// @brief Significand bits a `BandPair` carries where both limbs are normal
/// floats. 24 from `hi` + 24 from `lo`; the 5-bit tail a `Band` carries has
/// no room left in eight bytes.
inline constexpr int kBandPairBits = 48;

/// @brief At and above this exponent the pair delivers all `kBandPairBits`.
/// Below it the second limb is subnormal and the depth declines one bit per
/// binade (`min(48, e + 150)`): `lo` sits ~24 binades under `hi`, and FP32's
/// smallest normal is `2^-126`.
inline constexpr int kBandPairFullDepthFloorExp = -102;

/// @brief `|x| < 2^128` is the hard ceiling — above it `(float)x` is an
/// infinity and there is no pair at all. This matches `kBandHardCeilingExp`
/// (`Band.h`), because it is the same limit for the same reason: the limbs
/// are floats.
inline constexpr int kBandPairHardCeilingExp = detail::kBandHardCeilingExp;

/**
 * @brief The relative bound on the ingest truncation, `2^-48`, derived
 * rather than calibrated against a measurement.
 *
 * With `hi = fl32(x)` and `lo = fl32(x - hi)`, and `x` in binade `e`:
 * `|x - hi| <= 2^(e-24)` (half an FP32 ulp of `hi`), and narrowing that
 * residue costs at most half of its own ulp, `2^-24 * |x - hi| <=
 * 2^(e-48)`. Since `|x| >= 2^e (1 - 2^-24)`, the relative bound is `2^-48`
 * with room to spare.
 */
inline constexpr double kBandPairRelBound = 3.5527136788005009e-15; // 2^-48

// =====================================================================
//  BandPair — the element (Band-typed half)
// =====================================================================

/**
 * @brief Eight-byte, one-texel coefficient-table element: a two-limb `Band`.
 *
 * Layout is `{ float hi; float lo; }`, 8-byte aligned, trivially copyable.
 * The words are the carrier's limbs, not an encoding of them — `band()`
 * emits zero arithmetic instructions.
 */
struct alignas(8) BandPair {
    float hi; ///< leading limb — the value to 24 bits, with the sign
    float lo; ///< second limb, `|lo| <= ulp32(hi)/2` by construction

    /// @brief Default: uninitialised (trivial -- keeps the type POD for SoA
    /// buffers and `cudaMemcpy`).
    BandPair() = default;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandPair(float h, float l)
        : hi(h)
        , lo(l)
    {
    }

    /// @brief Named construction from limbs already known to satisfy the
    /// pair invariant (a transcribed literal, for instance).
    [[nodiscard]] static AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr BandPair fromLimbs(
        float h, float l)
    {
        return BandPair{ h, l };
    }

    // ── Decode ─────────────────────────────────────────────────────────────

    /**
     * @brief -> `Band`. Zero arithmetic instructions.
     *
     * @par Exactness
     * Exact: the carrier's value is `hi + lo + 0`, precisely the value the
     * eight bytes hold; nothing is rounded, approximated or re-split. The
     * carrier postcondition `|lo| <= ulp(hi)` holds by construction of the
     * ingest split, so the result is a certified `Band` and not a
     * `BandRaw` (@see `bandPairFromDouble`). All error in this path
     * belongs to the ingest, bounded by `kBandPairRelBound`.
     */
    [[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band band() const
    {
        return Band{ hi, lo, 0.0f };
    }

    /// @brief Explicit, named conversion — the preferred spelling is
    /// `.band()`; this is the type-driven form for a generic call site.
    [[nodiscard]] explicit AETHER_DEVICEHOST() AETHER_FORCEINLINE() operator Band() const
    {
        return band();
    }

    // ── The seam operators (hidden friends), Band-typed only ────────────────
    //
    // ADL finds these only when an operand is a `BandPair`, so no unrelated
    // overload set is widened. `BandPair` is a storage leaf, not a chain
    // carrier: it has no arithmetic of its own, every operator here
    // immediately decodes to `Band`, and `BandPair op BandPair` does not
    // exist.

    /// @brief `pair - band`, decode fused into the certified subtract.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(
        BandPair a, Band b)
    {
        return detail::sub(a.band(), b);
    }
    /// @brief `band - pair`. @see operator-(BandPair, Band)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator-(
        Band a, BandPair b)
    {
        return detail::sub(a, b.band());
    }
    /// @brief `pair + band`. @see operator-(BandPair, Band)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator+(
        BandPair a, Band b)
    {
        return detail::add(a.band(), b);
    }
    /// @brief `band + pair`. @see operator-(BandPair, Band)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator+(
        Band a, BandPair b)
    {
        return detail::add(a, b.band());
    }
    /// @brief `pair * band`. @see operator-(BandPair, Band)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator*(
        BandPair a, Band b)
    {
        return detail::mul(a.band(), b);
    }
    /// @brief `band * pair`. @see operator-(BandPair, Band)
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band operator*(
        Band a, BandPair b)
    {
        return detail::mul(a, b.band());
    }

    // ── Identity ───────────────────────────────────────────────────────────
    //
    // Limb-wise, a storage identity rather than a numeric comparison: the
    // right answer for "is this buffer still the fill value", the wrong
    // one for arithmetic (which does not use it). Float semantics
    // otherwise: NaN limbs are unequal to themselves, `-0.0f` equals
    // `+0.0f`.
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator==(
        BandPair a, BandPair b)
    {
        return a.hi == b.hi && a.lo == b.lo;
    }
    [[nodiscard]] friend AETHER_DEVICEHOST() AETHER_FORCEINLINE() bool operator!=(
        BandPair a, BandPair b)
    {
        return !(a == b);
    }
};

static_assert(sizeof(BandPair) == 8 && alignof(BandPair) == 8, "BandPair must be one 8-byte texel");
static_assert(std::is_trivially_copyable_v<BandPair>,
    "BandPair must stay trivially copyable (SoA buffers, cudaMemcpy)");

// =====================================================================
//  Host-side ingest (host only -- `double` must not appear in device code)
// =====================================================================

/**
 * @brief `double` -> `BandPair`. The split, and the only place it happens.
 *
 * ```
 *   hi = fl32(x)                 one hardware narrowing conversion
 *   r  = x - (double)hi          EXACT: r needs at most 29 significand bits
 *   lo = fl32(r)                 one hardware narrowing conversion
 * ```
 *
 * @par Why not `bandFromIEEE`/`cell8FromDouble`
 * Those primitives' leading limb is built by shifting `e + 127` into a float
 * exponent field. `(float)x` is the hardware's own correctly-rounded
 * narrowing at every magnitude instead: full precision while normal, gradual
 * underflow through the subnormals, then zero.
 *
 * @par The carrier postcondition this makes true
 * `|lo| <= ulp32(hi)/2 * (1 + 2^-24) <= ulp32(hi)`, so `Band{hi, lo, 0}` is a
 * certified carrier the moment it is decoded and never needs a `normalize`.
 *
 * @par Zero and out of range
 * `+-0` both produce `{+0.0f, +0.0f}` (the sign of zero is not carried).
 * `|x| >= 2^128` yields `hi = inf`, `lo = NaN`; NaN and infinity propagate
 * into `hi`. Unchecked here, deliberately — the check belongs once per
 * table (`bandPairAdmits`/`bandPairTableFromDoubles`), not once per
 * coefficient.
 */
[[nodiscard]] inline BandPair bandPairFromDouble(double x)
{
    if (x == 0.0)
        return BandPair{ 0.0f, 0.0f };
    const float hi = static_cast<float>(x);
    const float lo = static_cast<float>(x - static_cast<double>(hi));
    return BandPair{ hi, lo };
}

/// @brief `BandPair` -> `double`, exactly. `hi + lo` is a sum of two floats
/// with no overlap, so it is exact in a double and equals the value a
/// device carrier would decode. Host-only; not used on any device path.
[[nodiscard]] inline double bandPairToDouble(BandPair p)
{
    return static_cast<double>(p.hi) + static_cast<double>(p.lo);
}

/// @brief Can this value be held at all? False for NaN, for infinity, and
/// for any `|x| >= 2^128` (where the leading limb would be an infinity).
/// True below `2^-126`: the pair degrades gracefully there and reports its
/// depth through `bandPairCertifiedBits`, it does not lie.
[[nodiscard]] inline bool bandPairAdmits(double x)
{
    return std::isfinite(x) && std::isfinite(static_cast<float>(x));
}

/// @brief Depth actually delivered at this magnitude: `min(48, e + 150)`,
/// clamped at 0. The declining part is FP32's subnormal staircase under the
/// second limb, one bit per binade, with no step and no cliff.
[[nodiscard]] inline int bandPairCertifiedBits(double x)
{
    if (x == 0.0)
        return kBandPairBits;
    int e = 0;
    (void)std::frexp(std::fabs(x), &e);
    e -= 1; // frexp returns [0.5,1) * 2^e; we want floor(log2|x|)
    const int b = e + 150;
    return (b >= kBandPairBits) ? kBandPairBits : ((b < 0) ? 0 : b);
}

/// @brief What one table conversion cost, measured: row count, exponent
/// span, the worst depth anywhere in it, and the worst relative ingest
/// error actually realised, checked against `kBandPairRelBound`.
struct BandPairTableStats {
    std::size_t count    = 0; ///< rows converted
    std::size_t zeros    = 0; ///< rows that were exactly zero
    int minExp           = 0; ///< floor(log2|x|) of the smallest non-zero
    int maxExp           = 0; ///< floor(log2|x|) of the largest
    int minCertifiedBits = kBandPairBits; ///< worst depth in the table
    double maxRelError   = 0.0;           ///< worst realised |dx| / |x|
};

/**
 * @brief Build a coefficient table from `double` data, failing loud on a
 * bad entry.
 *
 * Throws (`aether::Error`, via `err::fail`) on the first coefficient the
 * format cannot hold — NaN, infinity, or `|x| >= 2^128` — naming the
 * index. A table is built once, off the hot path, so a silent infinity in
 * a coefficient row would otherwise produce silently wrong results
 * downstream.
 *
 * Depth loss is not an error and does not throw; it is reported instead.
 * A caller that needs all 48 bits checks `minCertifiedBits`; a caller
 * feeding a `Band` chain additionally needs `bandCeilingAdmits(maxExp)`
 * and `bandFloorAdmits(minExp)` (`BandedLimits.h`), which describe the
 * chain's envelope and are narrower than this format's.
 *
 * @param src  `n` doubles, the coefficients in their natural order
 * @param dst  `n` `BandPair`s, caller-allocated
 * @param n    the number of coefficients (length of `src`/`dst`)
 * @param what a name for the table, used in the throw message
 */
inline BandPairTableStats bandPairTableFromDoubles(
    const double* src, BandPair* dst, std::size_t n, const char* what = "coefficient table")
{
    if (src == nullptr || dst == nullptr)
        err::fail("BandPair table ingest", "host", n * sizeof(BandPair),
            std::string("null buffer building ") + what);

    BandPairTableStats st;
    st.count        = n;
    bool anyNonZero = false;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = src[i];
        if (!bandPairAdmits(x))
            err::fail("BandPair table ingest", "host", n * sizeof(BandPair),
                std::string(what) + " entry " + std::to_string(i)
                    + " is outside the BandPair envelope (NaN, infinity, or "
                      "|x| >= 2^128): "
                    + std::to_string(x));
        const BandPair p = bandPairFromDouble(x);
        dst[i]           = p;
        if (x == 0.0) {
            st.zeros++;
            continue;
        }
        int e = 0;
        (void)std::frexp(std::fabs(x), &e);
        e -= 1;
        if (!anyNonZero) {
            st.minExp  = e;
            st.maxExp  = e;
            anyNonZero = true;
        } else {
            if (e < st.minExp)
                st.minExp = e;
            if (e > st.maxExp)
                st.maxExp = e;
        }
        const int b = bandPairCertifiedBits(x);
        if (b < st.minCertifiedBits)
            st.minCertifiedBits = b;
        const double rel = std::fabs(x - bandPairToDouble(p)) / std::fabs(x);
        if (rel > st.maxRelError)
            st.maxRelError = rel;
    }
    return st;
}

} // namespace banded
} // namespace aether
