// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandRound.h
 * @brief `floor` / `ceil` / `round` / `trunc` / `fmod` / the Band `fma`
 *        over `Band`.
 *
 * @section admission Admission
 * `floor`/`ceil`/`round`/`trunc` are 0-ULP exact: bit ops and exact
 * exact-sum cascades add no error at any magnitude, the same class
 * `abs`/`copysign`/`fmax`/`fmin` are in (`certifyExactTotal*`,
 * `tests/bandmath/BandMathCert.h`). `fdim` (`Band.h`), despite also being
 * domain-total (no operand it rejects), is not in that class: its body is
 * `sub(a,b)` or `+0` — the comparison is the subtraction's sign — so its
 * accuracy is `sub`'s accuracy, the certified 2-ULP spine bound, not 0.
 * `fdim`'s golden table and cert rows therefore use the `spine_admitted`
 * shape (bound 2, `certifySpineAdmittedBinary`), the same one
 * `add`/`sub`/`mul` use.
 *
 * None of floor/ceil/round/trunc/fdim carries a per-op named admission
 * predicate: there is nothing beyond the generic carrier envelope
 * (`bandCeilingAdmits`/`bandIntermediateAdmits`, `Band.h`) for either
 * accuracy class to admit past — no rounding, no reduction, no op-specific
 * degradation curve. `fma` is the same shape `add`/`sub`/`mul` already
 * have: spine-admitted (2 ULP), no per-op predicate either. `fmod` is the
 * one op in this file with a genuinely new predicate, `bandFmodAdmits`,
 * because its claim (exactness, not a ULP bound) turns on a third leg no
 * other op in the library has: the operands' limb span.
 */

#include "aether/banded/Band.h"
#include "aether/banded/detail/ExactSum.h"
#include "aether/banded/detail/FmaRaw.h"
#include "aether/macros.h"

#include <cmath> // host floorf/truncf/copysignf

namespace aether {
namespace banded {
namespace detail {

// =========================================================================
//  floor / ceil / round
// =========================================================================

/**
 * @brief `floor(a)`: the greatest integer <= `a`, delivered at full carrier
 * width (not `floor` of the value rounded to 53 bits first — see the
 * "Full carrier width" note below).
 *
 * @par Why the early-stop cascade is exact
 * A normalized carrier satisfies `|lo| <= ulp(hi)/2` and `|tail| <=
 * ulp(lo)/2`. If `hi` is not an integer, its fractional part `f` is a
 * nonzero multiple of `ulp(hi)`, so `f >= ulp(hi) >= 2|lo|` and
 * `f <= 1 - ulp(hi)`; therefore `f + lo + tail` stays strictly inside
 * `(0, 1)` and the lower limbs cannot move the floor. `floor(a) =
 * floorf(a.hi)`, exactly, in one instruction — the same argument one limb
 * down handles `hi` integer / `lo` not, and one more handles `tail`.
 *
 * @par The two guards
 * A normalized carrier whose `hi` is zero has `lo == tail == 0`
 * (`normalize` opens with a `fast2Sum` that moves any nonzero `lo` into
 * `hi`), so `hi == 0` means the value is exactly `+-0`, whose floor is
 * itself including its sign — the general path could not preserve that
 * (`-0 + 0 == +0`). A non-finite carrier takes the same exit (an infinity
 * stays `(inf,0,0)` rather than letting the exact-sum tail manufacture a
 * NaN); a NaN needs no guard (`floorf(NaN) != NaN` opens the exit itself).
 *
 * @par Full carrier width, not a 53-bit round first
 * This computes `floor(v)` directly on the carrier, rather than rounding
 * `v` to 53 bits and flooring that (which would answer `floor(RN53(v))`
 * instead). The two agree on every carrier built from a `double`; they
 * differ only on a chain-resident intermediate holding more than 53 bits
 * (e.g. `3 - 2^-60`: rounding-first answers 3, this answers 2). This form
 * is the defensible one — the floor of the number, idempotent on its own
 * output at full carrier width.
 *
 * @par Contract: 0 ULP, total (no admission predicate, see file header).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band floor(Band a)
{
    // Zero (either sign) and non-finite both leave with the limbs cleared.
    if (!(a.hi != 0.0f && a.hi - a.hi == 0.0f))
        return Band{ a.hi, 0.0f, 0.0f };

    const float fh = floorf(a.hi);
    if (fh != a.hi)
        return Band{ fh, 0.0f, 0.0f };

    const float fl = floorf(a.lo);
    if (fl != a.lo) {
        // `a.hi` is a nonzero integer and `fl` is an integer with
        // `|fl| <= max(1, ulp(a.hi)/2) <= |a.hi|`, so `fast2Sum`'s ordering
        // precondition holds and the pair is the exact sum.
        float rh, rl;
        fast2Sum(a.hi, fl, rh, rl);
        return Band{ rh, rl, 0.0f };
    }

    return exactSum3(a.hi, a.lo, floorf(a.tail));
}

/// @brief `ceil(a) = -floor(-a)` — sign-bit flips only. The returned lower
/// limbs may be negative zeros where `floor`'s were positive; value-identical
/// (`+0 == -0`, the store terminal reads the sign from `hi`) and it is what
/// keeps `ceil(-0) == -0` / `ceil(+0) == +0` without a special case.
/// Contract: 0 ULP, total.
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band ceil(Band a)
{
    return neg(floor(neg(a)));
}

/**
 * @brief `round(a)`: nearest integer, half away from zero (C's `round`; not
 * `nearbyint`/`rint`'s half-to-even).
 *
 * @par Why the half is added by growing the expansion, not by `add`
 * `add` delivers 53 correct bits of `|a| + 1/2`, not enough: whether the
 * fractional part sits above/below/exactly at one half is destroyed by a
 * rounded sum on operands near a tie. Shewchuk's grow-expansion adds a
 * scalar to a nonoverlapping expansion exactly (three `twoSum`s), and
 * flooring the resulting four-term expansion floors the true `|a| + 1/2`
 * — the tie lands away from zero by construction, no tie-detection branch
 * anywhere in the body.
 *
 * @par Why `|a|` and not `a`
 * `floor(a + 1/2)` rounds a half up — away from zero for positives, toward
 * zero for negatives. Taking the magnitude first (three XORs) and
 * reapplying the sign afterwards makes one body serve both signs and gets
 * `round(-0.5) == -1`, `round(-0) == -0` for free.
 *
 * @par The fourth limb
 * The grown expansion's smallest term is the residual of
 * `twoSum(1/2, tail)`, bounded by `1/2` in magnitude for any float operand
 * (`fl(x + 1/2)` is either `x` itself or the exact sum), so its floor is
 * `0` or `-1` — never wider, which is what keeps the four-term sum inside
 * a three-limb carrier.
 *
 * @par Contract: 0 ULP, total.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band round(Band a)
{
    if (!(a.hi != 0.0f && a.hi - a.hi == 0.0f))
        return Band{ a.hi, 0.0f, 0.0f };

    const int  sgn = floatAsInt(a.hi) & static_cast<int>(0x80000000u);
    const Band m   = applySignMask(a, sgn); // |a|, exact

    // Grow by 1/2: (tail, lo, hi) is nonoverlapping and increasing, so
    // three `twoSum`s produce the exact four-term expansion (g0, g1, g2,
    // q2) of `|a| + 1/2`, also nonoverlapping and increasing.
    float q0, g0;
    twoSum(0.5f, m.tail, q0, g0);
    float q1, g1;
    twoSum(q0, m.lo, q1, g1);
    float q2, g2;
    twoSum(q1, m.hi, q2, g2);

    // Floor the expansion, leading term first — same early-stop argument
    // as `floor`, one limb deeper.
    Band r;
    const float f2 = floorf(q2);
    if (f2 != q2) {
        r = Band{ f2, 0.0f, 0.0f };
    } else {
        const float f1 = floorf(g2);
        if (f1 != g2) {
            float rh, rl;
            fast2Sum(q2, f1, rh, rl);
            r = Band{ rh, rl, 0.0f };
        } else {
            const float f0 = floorf(g1);
            if (f0 != g1)
                r = exactSum3(q2, g2, f0);
            else
                r = exactSum4(q2, g2, g1, floorf(g0));
        }
    }
    return applySignMask(r, sgn);
}

/**
 * @brief `trunc(a)`: round toward zero, added over the already-existing
 * `floor` + `abs`/`copysign`: `trunc(a) = copysign(floor(|a|), a)`.
 *
 * @par Correctness, by cases (0 ULP, total: `trunc` is exact by the
 * composition itself)
 *  - `a >= 0`: `|a| == a`, so `trunc(a) == floor(a)` — correct (round
 *    toward zero and round down agree for non-negatives).
 *  - `a < 0`: the standard identity is `trunc(a) = -floor(-a)`.
 *    `copysign(floor(|a|), a)` computes exactly that: `|a| == -a` for
 *    `a < 0`, so `floor(|a|) == floor(-a)`, and reapplying `a`'s (negative)
 *    sign via `copysign` is the negation — `-floor(-a)`, matching the
 *    identity.
 *  - `a == -0`: `|a| == +0` (sign bit cleared), `floor(+0) == +0`
 *    (`floor`'s own zero guard), `copysign(+0, -0) == -0` — matches C99
 *    `trunc(-0.0) == -0.0`.
 *  - `a == NaN`/`+-inf`: `floor` already passes these through unchanged
 *    (its own non-finite/NaN guard), `abs`/`copysign` preserve the class —
 *    `trunc(NaN)` is a NaN, `trunc(+-inf) == +-inf`, matching C99.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band trunc(Band a)
{
    return copysign(floor(abs(a)), a);
}

// =========================================================================
//  fmod -- exact on span <= 72, a theorem about the carrier, not a ULP
//  claim.
// =========================================================================

/// @brief The quotient digit width, in bits.
inline constexpr int kBandFmodChunk = 20;

/// @brief Iteration ceiling: 253 binades / 19 bits of progress per step,
/// plus the three close-range fixups. Measured worst case is 12; this is a
/// hard stop, not a schedule.
inline constexpr int kBandFmodMaxIter = 20;

/// @brief The widest limb span — leading bit to lowest set bit, anywhere
/// in the three limbs — for which `fmod`'s exact remainder is guaranteed
/// to fit in a carrier. Derived (`r = x - n*y` is a multiple of
/// `2^min(g(x),g(y))`, bounded by `|y|`, so it occupies at most
/// `max(S(x),S(y))` bit positions and three limbs hold 72); measurement
/// puts the first inexact row at span 75, two positions of margin over
/// this derived edge.
inline constexpr int kBandFmodMaxSpan = 72;

/**
 * @brief Exact `a - n*B` for an integer-valued float `n`, delivered as a
 * carrier — the one primitive `fmod`'s reduction is built from.
 *
 * @par Why this cannot be `sub(a, mul(n, B))`
 * `mul` rounds, and one rounding anywhere in the reduction destroys the
 * whole exactness claim. `n*B` is exact only as a six-term expansion
 * (three `twoProd`s), and `a - (that)` is a nine-term exact sum known to
 * fit in three limbs; this body costs 72 FP32 by cascading the nine terms
 * level-by-level through four known magnitude bands (`~2^E(a)`,
 * `~2^(E(a)-23)`, `~2^(E(a)-46)`, `~2^(E(a)-69)`), each a `twoSum`, with
 * only the final level's plain adds assumed exact (derived from the
 * granularity argument, checked by `FmodReductionIsExactOnHost`).
 *
 * @warning `d0` (the L0 residual) is not optional. `a.hi - p1` looks like
 * a Sterbenz difference but `p1` may land one binade below `a.hi`, needing
 * 25 bits on that finer grid — measurement found 197/5112 rows go inexact
 * with the bare subtract.
 *
 * @par Postcondition: normalized via `normalizeSafe` (the cascade
 * cancels, so `normalize`'s ordering assumption is unavailable).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band bandFmodStep(
    Band a, float n, Band B)
{
    float p1, e1, p2, e2, p3, e3;
    twoProd(n, B.hi, p1, e1);
    twoProd(n, B.lo, p2, e2);
    twoProd(n, B.tail, p3, e3);

    // L0 — the leading cancellation. `d0` is the residual that must not
    // be dropped; see the warning above.
    float c0, d0;
    twoSum(a.hi, -p1, c0, d0);

    // L1
    float s1, t1, t2, t3, t4, H;
    twoSum(a.lo, -p2, s1, t1);
    twoSum(s1, -e1, s1, t2);
    twoSum(s1, d0, s1, t3);
    twoSum(c0, s1, H, t4);

    // L2
    float u, w1, w2, w3, w4, w5, w6, M;
    twoSum(a.tail, -p3, u, w1);
    twoSum(u, -e2, u, w2);
    twoSum(u, t1, u, w3);
    twoSum(u, t2, u, w4);
    twoSum(u, t3, u, w5);
    twoSum(u, t4, M, w6);

    // L3 — six adds, exact by the granularity argument in the doc block.
    float T = addRN(w1, w2);
    T       = addRN(T, w3);
    T       = addRN(T, w4);
    T       = addRN(T, w5);
    T       = addRN(T, w6);
    T       = subRN(T, e3);

    return normalizeSafe(BandRaw{ H, M, T });
}

/**
 * @brief Admission for a consumer that calls `fmod`.
 *
 * @param maxAbsExpDividend exponent of the largest `|x|`. The reduction
 *        scales the divisor up to the dividend's binade, so this — not
 *        the divisor — is what the ceiling leg is owed on.
 * @param minAbsExpDivisor exponent of the smallest `|y|`. The reduction
 *        reads `y`'s leading limb exponent field, so that limb must be a
 *        normal float; the carrier-floor leg implies it with 38 binades
 *        to spare.
 * @param maxSpan the widest limb span either operand can hold — 53 for
 *        anything ingested from a double, 72 for a full-width carrier,
 *        more for a hand-built one like `(1, 2^-100, 0)`. Above
 *        `kBandFmodMaxSpan` the exact remainder is not representable and
 *        the op is no longer an exact one — the error is then bounded
 *        absolutely at about `2^-54|y|` rather than relatively.
 * @param margin the certified admission margin (defaults to
 *        `kBandAdmissionMargin`).
 *
 * @note There is no output-floor leg, and its absence is derived: `fmod`
 * is exact, so a remainder near the carrier floor is exactly the right
 * answer rather than a degraded one — unlike every other op in this
 * family, where a tiny output means lost bits.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr bool bandFmodAdmits(
    int maxAbsExpDividend, int minAbsExpDivisor, int maxSpan,
    int margin = kBandAdmissionMargin)
{
    return bandCeilingAdmits(maxAbsExpDividend, margin)
        && bandIntermediateAdmits(minAbsExpDivisor, margin)
        && maxSpan <= kBandFmodMaxSpan;
}

// Two-sided: the guard must agree with the constants it is derived from,
// and must reject on each leg alone -- a leg nobody has seen say no is a
// leg that could have been deleted.
static_assert(bandFmodAdmits(60, -40, 53),
    "fmod must admit the shape every consumer actually has: operands "
    "ingested from doubles (span 53) at ordinary magnitudes");
static_assert(bandFmodAdmits(107, -88, kBandFmodMaxSpan),
    "fmod must admit exactly at all three of its declared edges at once");
static_assert(!bandFmodAdmits(60, -40, kBandFmodMaxSpan + 1),
    "fmod must REJECT one bit position past the span edge -- the leg "
    "that makes the op's claim EXACTNESS rather than a tolerance");
static_assert(!bandFmodAdmits(60, -40, 120),
    "fmod must REJECT a carrier whose limbs are 120 bit positions apart "
    "-- the exact remainder then needs four limbs");
static_assert(!bandFmodAdmits(108, -40, 53),
    "fmod must REJECT on the DIVIDEND ceiling alone");
static_assert(!bandFmodAdmits(60, -89, 53),
    "fmod must REJECT on the DIVISOR carrier floor alone");
static_assert(!bandFmodAdmits(108, -40, 53) && bandFmodAdmits(108, -40, 53, 0),
    "the margin must change fmod's verdict on its exponent legs");
static_assert(!bandFmodAdmits(60, -40, 73, 0),
    "and must NOT rescue the span leg: representability is not a margin "
    "question, and a consumer must never opt out of it");

/**
 * @brief `fmod(x, y)` — the IEEE remainder, truncated quotient, sign of
 * the dividend. Bit-exact on the admitted domain.
 *
 * @par Contract
 *  - `|fmod(x,y)| < |y|` and `sign(fmod(x,y)) == sign(x)`, including the
 *    zeros: `fmod(-12,4)` is `-0`.
 *  - `|x| < |y|` returns `x` bit for bit — the reduction never runs.
 *  - `fmod(x,y) == fmod(x,-y)` and `fmod(-x,y) == -fmod(x,y)`, limb for
 *    limb (the signs are XOR masks applied outside the arithmetic).
 *  - Exact whenever both operands satisfy `bandFmodAdmits`. Not "0.5
 *    ULP": zero.
 *
 * @par Terminals
 * `y == 0`, a NaN operand and an infinite `x` all return NaN. An infinite
 * `y` returns `x` (the mathematical limit). `x == +-0` returns `x` with
 * its sign. The order of these four tests is load-bearing: a NaN test
 * written as `!(v-v==0)` is also true for an infinity, so folding the NaN
 * and infinite-dividend cases together would swallow `fmod(x,+-inf)`
 * (whose answer is `x`, not NaN).
 *
 * @par Cost
 * Input-dependent: `1 + ceil((E(x)-E(y))/19)` steps of ~72 FP32 each.
 * Worst case over the whole window is 12 iterations; typical consumer
 * shapes (an angle wrap, a time-of-day fold) are one.
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fmod(Band x, Band y)
{
    const float kNaN = intAsFloat(0x7FC00000);
    const float kInf = intAsFloat(0x7F800000);

    if (x.hi != x.hi || y.hi != y.hi || y.hi == 0.0f)
        return Band{ kNaN, 0.0f, 0.0f };
    if (x.hi == kInf || x.hi == -kInf)
        return Band{ kNaN, 0.0f, 0.0f };
    if (y.hi == kInf || y.hi == -kInf)
        return x; // fmod(x, +-inf) = x
    if (x.hi == 0.0f)
        return x; // +-0, sign preserved

    const int  sgn = floatAsInt(x.hi) & static_cast<int>(0x80000000u);
    Band       a   = applySignMask(x, sgn); // |x|, three XORs
    const Band b   = abs(y);

    for (int it = 0; it < kBandFmodMaxIter; ++it) {
        if (a.hi == 0.0f)
            break; // the remainder is exactly zero
        const int ea = ((floatAsInt(a.hi) >> 23) & 0xFF) - 127;
        const int eb = ((floatAsInt(b.hi) >> 23) & 0xFF) - 127;
        const int d  = ea - eb;

        if (d > kBandFmodChunk) {
            // Coarse: scale the divisor to `kBandFmodChunk` binades below
            // the remainder and take a digit. The `2^k` is split into two
            // normal factors for the reason `log`'s reconstruction splits
            // its own — a single `2^k` with `k` past 127 has no float to
            // live in.
            const int  k  = d - kBandFmodChunk;
            const int  h1 = k >> 1;
            const Band B  = scalePow2f(
                scalePow2f(b, intAsFloat((127 + h1) << 23)), intAsFloat((127 + (k - h1)) << 23));
            a = bandFmodStep(a, truncf(a.hi / B.hi), B);
        } else {
            // Close range. The digit is at most `2^(kBandFmodChunk+1)`.
            // A digit of 0 or 1 on a non-negative remainder is one case,
            // and merging them is what makes `|x| < |y|` bit-exact.
            const float n = truncf(a.hi / b.hi);
            if (a.hi >= 0.0f && n <= 1.0f) {
                const Band diff = bandFmodStep(a, 1.0f, b);
                if (diff.hi < 0.0f)
                    break; // 0 <= a < b -- done, with a's own limbs
                a = diff;
            } else if (n == 0.0f) {
                a = bandFmodStep(a, -1.0f, b); // a < 0 and |a| < b: a += b
            } else {
                a = bandFmodStep(a, n, b);
            }
        }
    }

    return applySignMask(a, sgn);
}

// =========================================================================
//  fma -- the public checked entry point over the `fmaRaw` family
//  (`FmaRaw.h`).
// =========================================================================

/**
 * @brief `a*b + c`, single Band-level rounding barrier. Certified 53 bits
 * over the same spine envelope `add`/`sub`/`mul` share (no per-op named
 * predicate — @see file header "admission" section). Contract: 2 ULP
 * in-domain (matching `add`/`sub`/`mul`'s own bound; `tests/bandmath/
 * golden_fma.h`'s minted table, 509/24/8).
 *
 * @par Cost: about 54 FP32 — `mulRaw`+`addRaw`-equivalent minus one
 * `normalize` barrier (one barrier cheaper than `add(mul(a,b),c)`'s own
 * composition, which pays two).
 */
[[nodiscard]] AETHER_DEVICEHOST() AETHER_FORCEINLINE() Band fma(Band a, Band b, Band c)
{
    return normalize(fmaRaw(a, b, c));
}

} // namespace detail
} // namespace banded
} // namespace aether
