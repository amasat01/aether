#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_round.py — golden minter for the rounding
and remainder ops (floor, ceil, round, trunc, fdim, fmod).

floor/ceil/round/trunc/fdim/fmod are exactly representable ops — a
correctly-rounded double reference is admissible for them, so none of
these six ops needs MPFR: a native `double` IEEE-754 computation
(Python's own `math.floor`/`math.ceil`/`math.fmod`, plus a Fraction-exact
half-away-from-zero `round` and a hand-written `fdim`) already is the
correctly-rounded double reference — the same reasoning
`tools/bandmath/gen_corpus.py`'s own module docstring records for
abs/copysign/fmax/fmin/add/sub/mul(/fma): native double arithmetic on
this toolchain is the correctly-rounded IEEE-754 reference, and
`tools/ulp_oracle` itself stays untouched (consumed, not rebuilt).
`tools/ulp_oracle` does carry a `fmod` FuncId but is not invoked here:
MPFR at 256 bits would answer the same double once correctly rounded
(fmod's mathematical remainder for two finite doubles is always exactly
representable in double precision — no rounding step for MPFR to do
differently), so routing through it would add a subprocess dependency
for zero additional correctness. This script is therefore its own
separate, purpose-fit minter (following `gen_corpus.py`'s and
`gen_corpus_root.py`'s precedent for an op ulp_oracle does not cover
well) rather than an ulp_oracle consumer.

Per-op accuracy class (this package's own `BandRound.h` file header
"admission" section):
  - `floor`/`ceil`/`round`/`trunc`: exact, 0 ULP (`exact_total`, the same
    class `abs`/`copysign`/`fmax`/`fmin` use in gen_corpus.py).
  - `fdim`: 2 ULP (`spine_admitted`) — its body is `sub(a,b)` or `+0`, so
    its accuracy is `sub`'s certified 2-ULP spine bound, not 0. (A first
    draft of `BandRound.h`'s own docstring got this wrong and self-
    corrected — see that file's own note.)
  - `fmod`: exact, 0 ULP, but not `exact_total`'s generic envelope — its
    own `bandFmodAdmits` (ceiling on the dividend, carrier-floor on the
    divisor, span <= 72) governs in_domain/degraded/rejected instead.

Admission constants transcribed verbatim from `aether/banded/Band.h`
(`kBandNominalCeilingExp`/`kBandCarrierFloorExp`/`kBandAdmissionMargin`)
and `aether/banded/BandRound.h` (`kBandFmodMaxSpan`) — same convention
`gen_corpus.py`'s own "admission predicates — transcribed verbatim"
section established, reproduced here rather than imported so this script
has no import-time dependency on the other beyond `SplitMix64` (below).

Usage:
    python3 tools/bandmath/gen_corpus_round.py [--seed HEX] [--out-dir DIR]

Refuses to write when $CI is set (mirrors the project's re-mint convention:
a gate must never regenerate its own expectation).
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import argparse
from fractions import Fraction

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5  # matches the seed used across the golden_*.h corpora (see golden_random.h)


# =====================================================================
#  Bit helpers
# =====================================================================
def bits_of(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def double_of(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
QNAN = 0x7FF8000000000000
NAN_PAYLOAD_A = 0x7FF8000000000001
NAN_PAYLOAD_B = 0xFFF8000000012345

DBL_MAX = struct.unpack("<d", struct.pack("<Q", 0x7FEFFFFFFFFFFFFF))[0]
DBL_MIN_SUBNORMAL = struct.unpack("<d", struct.pack("<Q", 0x0000000000000001))[0]


def is_nan(x: float) -> bool:
    return x != x


# =====================================================================
#  Admission — TRANSCRIBED verbatim from aether/banded/Band.h /
#  aether/banded/BandRound.h. Same constants gen_corpus.py's own
#  "Admission predicates" section carries.
# =====================================================================
K_BAND_NOMINAL_CEILING_EXP = 115
K_BAND_CARRIER_FLOOR_EXP = -96
K_BAND_ADMISSION_MARGIN = 8
K_BAND_ADMITTED_LO = K_BAND_CARRIER_FLOOR_EXP + K_BAND_ADMISSION_MARGIN  # -88
K_BAND_ADMITTED_HI = K_BAND_NOMINAL_CEILING_EXP - K_BAND_ADMISSION_MARGIN  # 107
K_BAND_HARD_CEILING_EXP = 128
K_BAND_HARD_FLOOR_EXP = -149
K_BAND_FMOD_MAX_SPAN = 72  # BandRound.h::kBandFmodMaxSpan


def band_ceiling_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e + margin <= K_BAND_NOMINAL_CEILING_EXP


def band_intermediate_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_CARRIER_FLOOR_EXP


def spine_admits(e: int) -> bool:
    return band_ceiling_admits(e) and band_intermediate_admits(e)


def band_fmod_admits(max_exp_dividend, min_exp_divisor, max_span: int,
                      margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return band_ceiling_admits(max_exp_dividend, margin) \
        and band_intermediate_admits(min_exp_divisor, margin) \
        and max_span <= K_BAND_FMOD_MAX_SPAN


def exponent_of(x: float):
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    m, e = math.frexp(x)
    return e - 1


def label_for_exponent(e):
    if e is None:
        return 0  # zero/inf/nan: always in_domain (gen_corpus.py precedent)
    if e >= K_BAND_HARD_CEILING_EXP or e <= K_BAND_HARD_FLOOR_EXP:
        return 2
    if spine_admits(e):
        return 0
    return 1


# =====================================================================
#  splitmix64 — SAME stream gen_corpus.py's own SplitMix64 produces (a
#  local copy rather than a cross-import: this script's only shared
#  dependency with gen_corpus.py is the ALGORITHM, not the module, mirroring
#  gen_corpus_root.py's own choice not to import gen_corpus for anything but
#  rsqrtCube's corpus).
# =====================================================================
class SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform01(self) -> float:
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)


# =====================================================================
#  Golden reference — native double arithmetic (module docstring).
# =====================================================================
def ref_floor(a: float) -> float:
    if is_nan(a) or math.isinf(a):
        return a
    if a == 0.0:
        return a  # preserve the sign of a zero
    return float(math.floor(a))  # math.floor returns int in py3 -- cast back


def ref_ceil(a: float) -> float:
    if is_nan(a) or math.isinf(a):
        return a
    if a == 0.0:
        return a
    return float(math.ceil(a))


def ref_round(a: float) -> float:
    """Half away from zero, computed EXACTLY via Fraction so the reference
    itself never suffers the +0.5 tie-rounding a naive float computation
    would (the property `BandRound.h::round`'s own docstring derives via
    Shewchuk's grow-expansion; this mirrors it independently, in exact
    rational arithmetic rather than a floating grow-expansion)."""
    if is_nan(a) or math.isinf(a) or a == 0.0:
        return a
    fr = Fraction(a)  # EXACT: every finite double is an exact rational
    afr = abs(fr) + Fraction(1, 2)
    n = afr.numerator // afr.denominator  # floor, afr > 0
    return math.copysign(float(n), a)


def ref_trunc(a: float) -> float:
    if is_nan(a) or math.isinf(a) or a == 0.0:
        return a
    return float(math.trunc(a))


def ref_fdim(a: float, b: float) -> float:
    if is_nan(a) or is_nan(b):
        return float("nan")
    d = a - b  # native double subtraction, IEEE correctly-rounded
    return d if d > 0.0 else 0.0


def ref_fmod(a: float, b: float) -> float:
    if is_nan(a) or is_nan(b) or b == 0.0:
        return float("nan")
    if math.isinf(a):
        return float("nan")
    if math.isinf(b):
        return a
    if a == 0.0:
        return a
    return math.fmod(a, b)  # exact for finite double operands (C99 fmod)


# =====================================================================
#  Op table
# =====================================================================
class Op:
    def __init__(self, name, arity, category, ref, bound, provenance):
        self.name = name
        self.arity = arity
        self.category = category  # 'exact_total' | 'spine_admitted' | 'fmod'
        self.ref = ref
        self.bound = bound
        self.provenance = provenance


OPS = [
    Op("floor", 1, "exact_total", ref_floor, 0,
       "aether/banded/BandRound.h::floor — total, exact, 0 ULP "
       "(early-stop exact-sum cascade)."),
    Op("ceil", 1, "exact_total", ref_ceil, 0,
       "aether/banded/BandRound.h::ceil — neg(floor(neg(x))), 0 ULP."),
    Op("round", 1, "exact_total", ref_round, 0,
       "aether/banded/BandRound.h::round — half away from zero, "
       "Shewchuk grow-expansion, 0 ULP."),
    Op("trunc", 1, "exact_total", ref_trunc, 0,
       "aether/banded/BandRound.h::trunc — copysign(floor(|x|), x). "
       "0 ULP by construction over the already-exact floor/abs/"
       "copysign."),
    Op("fdim", 2, "spine_admitted", ref_fdim, 2,
       "aether/banded/Band.h::fdim — body is sub(a,b) or +0, so accuracy "
       "is sub's own certified 2-ULP spine bound, not 0 (BandRound.h's "
       "own self-corrected 'admission' note)."),
]

FMOD_OP = Op("fmod", 2, "fmod", ref_fmod, 0,
             "aether/banded/BandRound.h::fmod — exact (0 ULP) on span <= "
             "72 (bandFmodAdmits), not a tolerance; degrades past it "
             "(absolute error ~2^-54|y|).")


# =====================================================================
#  Corpus construction — mirrors gen_corpus.py's own generic build_rows
#  shape (specials + at_exp sweep + random bulk), independently written
#  (this script's own copy, per its module docstring).
# =====================================================================
def specials_pool():
    return {POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, NAN_PAYLOAD_A, NAN_PAYLOAD_B}


def at_exp(e, mant_seed_n=3):
    out = []
    for i in range(mant_seed_n):
        m = 1.0 + (i + 0.5) / mant_seed_n
        s = 1.0 if i % 2 == 0 else -1.0
        out.append(math.ldexp(m, e) * s)
    return out


def result_label(vals, ref: float):
    exps = [exponent_of(v) for v in vals] + [exponent_of(ref)]
    finite_exps = [e for e in exps if e is not None]
    if not finite_exps:
        return 0
    label = max(label_for_exponent(e) for e in finite_exps)
    return label


def build_rows_generic(op: Op, seed: int, bulk_n: int = 500):
    rng = SplitMix64(seed)
    rows = []
    seen = set()

    def emit(bits_tuple):
        if bits_tuple in seen:
            return
        seen.add(bits_tuple)
        vals = [double_of(b) for b in bits_tuple]
        ref = op.ref(*vals)
        label = result_label(vals, ref)
        rows.append(tuple(bits_tuple) + (bits_of(ref), label))

    def pad(a, b=None):
        if op.arity == 1:
            return (a,)
        return (a, bits_of(1.0) if b is None else b)

    # ---- specials: always in_domain (module note) ----
    if op.arity == 1:
        for a in specials_pool():
            emit(pad(a))
    else:
        vals_special = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.5, -3.5,
                         float("inf"), float("-inf"), float("nan")]
        for a in vals_special:
            for b in vals_special:
                emit((bits_of(a), bits_of(b)))
        for a in specials_pool():
            emit(pad(a, bits_of(1.0)))
            if op.arity == 2:
                emit((bits_of(1.0), a))

    # ---- integer boundaries (floor/ceil/round/trunc's own home turf) ----
    for k in range(-40, 41):
        for frac_num, frac_den in ((0, 1), (1, 2), (1, 4), (3, 4), (1, 1000)):
            v = k + frac_num / frac_den
            if op.arity == 1:
                emit((bits_of(v),))
            else:
                emit((bits_of(v), bits_of(1.0)))
                emit((bits_of(1.0), bits_of(v)))
    # half-way ties, both signs, several magnitudes (round's own edge)
    for base in (0.5, 1.5, 2.5, 3.5, 1000000.5, 4503599627370496.5):
        for s in (1.0, -1.0):
            v = base * s
            if op.arity == 1:
                emit((bits_of(v),))
            else:
                emit((bits_of(v), bits_of(1.0)))

    # ---- exponent sweep across the admitted envelope, past the margin
    # into DEGRADED territory (100..127, safe -- `bandFromIEEE`'s ingest is
    # a clean, unwrapped FP32 exponent field there, @see that function's
    # own "Deliberately unchecked: the CEILING" doc block). ★ The FLOOR
    # side (e <= -149) is DELIBERATELY NOT swept here, unlike gen_corpus.
    # py's own generic pattern: this whole op family (floor/ceil/round/
    # trunc/fdim) CLAMPS a tiny-magnitude operand to a small constant (0,
    # +-1) regardless of whether the Band ingest preserved its exact value,
    # so a "rejected" label past the hard floor would assert an escape
    # that provably CANNOT be observed — found by this package's own first
    # minter draft (26 spurious "rejected row silently passed" failures).
    # Every op in THIS package shares the property (it is what "exact,
    # total, no degradation curve" means); gen_corpus.py's own abs/
    # copysign/fmax/fmin/add/sub/mul do NOT, because their answer's
    # MAGNITUDE tracks the operand's, so a corrupted tiny operand there
    # really can surface as an observable mismatch.
    for e in list(range(-100, -85)) + list(range(-92, -84)) + list(range(100, 116)) \
            + list(range(-40, 41, 5)) + list(range(118, 128)):
        for v in at_exp(e):
            if op.arity == 1:
                emit((bits_of(v),))
            else:
                for w in at_exp(e - 3):
                    emit((bits_of(v), bits_of(w)))
                emit((bits_of(v), bits_of(1.0)))
                emit((bits_of(1.0), bits_of(v)))

    # ---- REJECTED coverage: exactly e = 128 (fexp_hi = 255), the ONE
    # exponent whose ingest is STILL a clean, PREDICTABLE IEEE encoding —
    # `hi_bits`'s exponent field lands EXACTLY on `kBandFp32ExpMask` (no
    # wrap into the sign bit yet, @see `bandFromIEEE`'s own doc block:
    # "above e=127 ... the assembly wraps"), so the leading limb becomes a
    # clean Inf (mantissa bits all zero) or a canonical NaN (any mantissa
    # bit set — the common case for a non-round `at_exp` mantissa), either
    # way an UNAMBIGUOUS escape against a finite `ref`. `e = 129` and
    # beyond genuinely DO wrap (verified empirically: a first draft swept
    # up to `e = 140` and got both spurious escapes AND spurious silent
    # passes from the wrapped bit pattern landing, by chance, back inside
    # a subnormal-exponent encoding) and are NOT swept — a narrower but
    # HONEST rejected corpus, not a wider unpredictable one.
    for v in at_exp(K_BAND_HARD_CEILING_EXP):
        if op.arity == 1:
            emit((bits_of(v),))
        else:
            for w in at_exp(K_BAND_HARD_CEILING_EXP - 3):
                emit((bits_of(v), bits_of(w)))
            emit((bits_of(v), bits_of(1.0)))
            emit((bits_of(1.0), bits_of(v)))

    # ---- random bulk over ordinary magnitudes ----
    for _ in range(bulk_n):
        e = -60 + int(rng.uniform01() * 150)
        m = 1.0 + rng.uniform01()
        s = 1.0 if (rng.next() & 1) else -1.0
        v = math.ldexp(m, e) * s
        if op.arity == 1:
            emit((bits_of(v),))
        else:
            e2 = -60 + int(rng.uniform01() * 150)
            m2 = 1.0 + rng.uniform01()
            s2 = 1.0 if (rng.next() & 1) else -1.0
            w = math.ldexp(m2, e2) * s2
            emit((bits_of(v), bits_of(w)))

    return rows


def build_rows_fmod(seed: int, bulk_n: int = 900):
    rng = SplitMix64(seed)
    rows = []
    seen = set()

    def emit(a_bits, b_bits):
        key = (a_bits, b_bits)
        if key in seen:
            return
        seen.add(key)
        a, b = double_of(a_bits), double_of(b_bits)
        ref = ref_fmod(a, b)
        ea, eb, eref = exponent_of(a), exponent_of(b), exponent_of(ref)
        # Every corpus row here is DOUBLE-sourced, so span(a)<=53,
        # span(b)<=53 always -- the span leg of bandFmodAdmits is therefore
        # satisfied for every ROW (the dedicated exponent-gap span LADDER
        # that pushes span itself past 72 is a separate, hand-built-Band
        # test — @see test_BandMathRound_common.h's own
        # FmodSpanLadderIsBitExactWithinTheDeclaredContract).
        if ea is None or eb is None:
            label = 0  # a special operand: terminals, always in_domain
        else:
            # ★ bandFmodAdmits (BandRound.h) owes NO leg on the DIVIDEND's
            # own carrier floor -- it only binds the dividend's ceiling and
            # the divisor's floor (the reduction's own two failure modes).
            # But when |x| < |y|, fmod returns `x` VERBATIM, so the
            # RESULT-BEARING operand is `x` itself, and a dividend far
            # below the carrier floor is an INGEST fidelity problem
            # `bandFromIEEE` cannot avoid -- the SAME "generic envelope
            # must cover every result-bearing operand AND the true
            # reference's own exponent" finding gen_corpus.py's own module
            # docstring records for abs/copysign (found here independently:
            # a first draft of this function let a dividend at exponent
            # -149 through as in_domain, measuring a genuine
            # 1.008e-45 -> 1.401e-45 (FLT_TRUE_MIN) ingest-rounding
            # mismatch -- correct carrier behaviour, wrong label). Every
            # exponent that could BEAR the result — both operands AND the
            # true mathematical remainder's own exponent — is therefore
            # checked against the generic spine envelope, IN ADDITION to
            # `bandFmodAdmits`'s own specific legs.
            exps = [e for e in (ea, eb, eref) if e is not None]
            worst_hi = max(exps) if exps else None
            worst_lo = min(exps) if exps else None
            if worst_hi is not None and (worst_hi >= K_BAND_HARD_CEILING_EXP
                                          or worst_lo <= K_BAND_HARD_FLOOR_EXP):
                label = 2
            elif band_fmod_admits(ea, eb, 53) and all(spine_admits(e) for e in exps):
                label = 0
            else:
                label = 1
        rows.append((a_bits, b_bits, bits_of(ref), label))

    # Terminals: y=0, x=inf, y=inf, x=+-0, NaN either side.
    vals_special = [0.0, -0.0, 1.0, -1.0, 3.5, -3.5, float("inf"), float("-inf"), float("nan")]
    for a in vals_special:
        for b in vals_special:
            emit(bits_of(a), bits_of(b))

    # Exponent-gap ladder over double-representable spans (0..53), both
    # coarse (d > kBandFmodChunk=20) and close-range reduction paths, plus
    # a mixed-exponent schedule (dividend/divisor exponents drawn from
    # different bands so the reduction alternates coarse/close across the
    # sweep), distinct from an all-worst-gap sweep: a mixed warp was found
    # to cost more than a uniform one.
    gaps = [0, 1, 3, 8, 15, 19, 20, 21, 30, 45, 60, 90, 130, 195]
    base_exps = [-70, -40, -10, 0, 10, 40, 70, 95]
    mants = [1.0, 1.2000002384185791, 1.9999998807907104, 1.0000001192092896]
    for be in base_exps:
        for g in gaps:
            for m in mants:
                x = math.ldexp(m, be)
                y = math.ldexp(1.0000001192092896, be - g)
                if math.isinf(x) or y == 0.0:
                    continue
                emit(bits_of(x), bits_of(-x))
                emit(bits_of(x), bits_of(y))
                emit(bits_of(-x), bits_of(y))
                emit(bits_of(x), bits_of(-y))

    # Random bulk, log-uniform magnitudes both signs, spanning the full
    # admitted envelope and past both edges, INCLUDING the hard FP32
    # ceiling/floor (128/-149) so the corpus enumerates genuinely REJECTED
    # rows (an ingest round-trip escape), not merely degraded ones.
    for _ in range(bulk_n):
        ex = -160 + int(rng.uniform01() * 300)
        ey = -160 + int(rng.uniform01() * 300)
        mx = 1.0 + rng.uniform01()
        my = 1.0 + rng.uniform01()
        sx = 1.0 if (rng.next() & 1) else -1.0
        sy = 1.0 if (rng.next() & 1) else -1.0
        x = math.ldexp(mx, ex) * sx
        y = math.ldexp(my, ey) * sy
        if y == 0.0 or math.isinf(x):
            continue
        emit(bits_of(x), bits_of(y))

    return rows


# =====================================================================
#  Header emission — matches golden_add.h's format exactly.
# =====================================================================
def emit_header(op: Op, rows, out_path: str, seed: int, minter_note: str):
    in_dom = sum(1 for r in rows if r[-1] == 0)
    deg = sum(1 for r in rows if r[-1] == 1)
    rej = sum(1 for r in rows if r[-1] == 2)

    row_lines = []
    for r in rows:
        ins = r[:op.arity]
        ref = r[op.arity]
        label = r[op.arity + 1]
        in_str = ", ".join(f"0x{v:016X}ULL" for v in ins)
        row_lines.append(f"    {{ {{ {in_str} }}, 0x{ref:016X}ULL, {label} }},")
    row_block = "\n".join(row_lines)
    md5 = hashlib.md5(("\n" + row_block + "\n").encode()).hexdigest()

    text = f"""#pragma once

// GENERATED by tools/bandmath/gen_corpus_round.py — golden reference +
// domain label table for `{op.name}`, DO NOT EDIT BY HAND. Regenerate with
// the mint command below and commit the diff in the SAME commit as
// whatever test consuming this header changed.
//
// mint command line: python3 tools/bandmath/gen_corpus_round.py
// seed:              0x{seed:016X}ULL
// reference:         {minter_note}
// provenance:        {op.provenance}
// bound (in_domain, ULP): {op.bound}
//
// label encoding: 0=in_domain 1=degraded 2=rejected
// corpus: {len(rows)} rows ({in_dom} in_domain / {deg} degraded / {rej} rejected)
//
// The row block below is md5-FENCED:
//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'
//   sed -n "/$BEGIN_RE/,/$END_RE/p" <this file> | sed '1d;$d' | md5sum
// and compare against the value on the BEGIN line.

#include <cstdint>
#include <cstddef>

namespace aether_tests {{
namespace bandmath_golden {{
namespace {op.name} {{

inline constexpr int kArity = {op.arity};
inline constexpr double kUlpBound = {op.bound};
struct Row {{
    std::uint64_t in[{op.arity}];  ///< input bit patterns
    std::uint64_t ref;         ///< golden reference, correctly-rounded double bits
    std::uint8_t  label;       ///< 0=in_domain 1=degraded 2=rejected
}};

// ---8<--- GOLDEN ROWS BEGIN (md5 = {md5}) ---8<---
inline constexpr Row kRows[] = {{
{row_block}
}};
// ---8<--- GOLDEN ROWS END ---8<---

inline constexpr std::size_t kCorpusSize = {len(rows)};
inline constexpr std::size_t kInDomainCount = {in_dom};
inline constexpr std::size_t kDegradedCount = {deg};
inline constexpr std::size_t kRejectedCount = {rej};

}} // namespace {op.name}
}} // namespace bandmath_golden
}} // namespace aether_tests
"""
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(rows)} rows ({in_dom}/{deg}/{rej})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=hex(DEFAULT_SEED))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "tests", "bandmath"))
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus_round.py: refusing to write under $CI", file=sys.stderr)
        return 1

    seed = int(args.seed, 16)
    out_dir = os.path.abspath(args.out_dir)

    for op in OPS:
        rows = build_rows_generic(op, seed)
        emit_header(op, rows, os.path.join(out_dir, f"golden_{op.name}.h"), seed,
                    "native IEEE-754 double arithmetic (this script's own minter -- "
                    "see module docstring DEVIATION note for why this is not "
                    "MPFR-routed)")

    fmod_rows = build_rows_fmod(seed)
    emit_header(FMOD_OP, fmod_rows, os.path.join(out_dir, "golden_fmod.h"), seed,
                "native IEEE-754 double arithmetic (math.fmod -- C99 guarantees "
                "an EXACT remainder for finite double operands, so no MPFR step "
                "changes the answer; see module docstring)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
