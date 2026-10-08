#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus.py — corpus generator and golden minter for
the exact and spine-admitted Band ops (abs, copysign, fmax, fmin, add,
sub, mul, fma).

Per op, enumerates (never samples) the boundary families implied by the
op's own admission predicate (transcribed below, with file:line
references on every constant), labels every row `in_domain | degraded |
rejected`, and mints a golden reference header per op.

The eight ops covered here are all IEEE-exact primitives — `Band`'s
certified `add`/`sub`/`mul` deliver a result within a
`Band::certifiedBits`-derived bound of the correctly-rounded double
answer, and abs/copysign/fmax/fmin are exact bit operations with no
rounding at all. Routing these through `tools/ulp_oracle`'s MPFR core
would (a) require extending it to a third arity (fma is ternary;
ulp_oracle only carries arity 1/2) and (b) introduce a needless
double-rounding risk MPFR's own 256-bit precision accepts only because
the alternative (a closed-form transcendental) has none. Native `double`
arithmetic on this toolchain is the correctly-rounded IEEE-754 reference
for +,-,*,fma, so this script computes the golden value directly rather
than through MPFR — `tools/ulp_oracle` itself is untouched (consumed,
not rebuilt); this is a separate, purpose-fit minter for the one class of
op ulp_oracle was never built to cover. The emitted header format/
md5-fence convention mirrors ulp_oracle's own (README.md "fence check")
so `tests/bandmath/BandMathCert.h` consumes both uniformly, and `ulp.h`
(tools/ulp_oracle/ulp.h) is reused unchanged as the one ULP-distance
comparison primitive.

Usage:
    python3 tools/bandmath/gen_corpus.py [--seed HEX] [--out-dir DIR]

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

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5  # matches the seed used across the golden_*.h corpora (see golden_random.h)


# =====================================================================
#  splitmix64 — matches ulp_oracle's own Rng (oracle_core.cpp) and the
#  project-wide seeded-stream convention.
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

    def mantissa(self) -> float:
        """Uniform in [1, 2)."""
        return 1.0 + self.uniform01()

    def sign(self) -> float:
        return -1.0 if (self.next() & 1) else 1.0


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

DBL_MIN_NORMAL = struct.unpack("<d", struct.pack("<Q", 0x0010000000000000))[0]
DBL_MAX = struct.unpack("<d", struct.pack("<Q", 0x7FEFFFFFFFFFFFFF))[0]
DBL_MIN_SUBNORMAL = struct.unpack("<d", struct.pack("<Q", 0x0000000000000001))[0]

# Tier edges: 2^-94 (past the carrier floor kBandCarrierFloorExp=-96) and 2^125 (short of the nominal ceiling
# kBandNominalCeilingExp=115 + hard FP32 overflow at 2^128).
TIER_EDGE_LO = math.ldexp(1.0, -94)
TIER_EDGE_HI = math.ldexp(1.0, 125)


def is_nan(x: float) -> bool:
    return x != x


def classify(b: int) -> int:
    exp = (b >> 52) & 0x7FF
    frac = b & 0xFFFFFFFFFFFFF
    if exp == 0x7FF:
        return 4 if frac else 3  # nan / inf
    if exp == 0:
        return 1 if frac else 0  # subnormal / zero
    return 2  # normal


# =====================================================================
#  Admission predicates — transcribed verbatim from
#  aether/banded/Band.h's admission-guard section, same constants, same formulas:
#    kBandNominalCeilingExp = 115
#    kBandCarrierFloorExp   = -96
#    kBandAdmissionMargin   = 8
#    bandCeilingAdmits(e, m=8)      = e + m <= 115
#    bandIntermediateAdmits(e, m=8) = e - m >= -96
#  For add/sub/mul: an operand exponent e is certified in-domain when
#  -88 <= e <= 107 (bandCeilingAdmits+bandIntermediateAdmits on
#  operands/result at the default margin).
# =====================================================================
K_BAND_NOMINAL_CEILING_EXP = 115
K_BAND_CARRIER_FLOOR_EXP = -96
K_BAND_ADMISSION_MARGIN = 8
K_BAND_ADMITTED_LO = K_BAND_CARRIER_FLOOR_EXP + K_BAND_ADMISSION_MARGIN  # -88
K_BAND_ADMITTED_HI = K_BAND_NOMINAL_CEILING_EXP - K_BAND_ADMISSION_MARGIN  # 107
K_BAND_HARD_CEILING_EXP = 128  # FP32 hard overflow (Band.h's "hard ceiling" doc note)
K_BAND_HARD_FLOOR_EXP = -149  # FP32 hard subnormal floor


def band_ceiling_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e + margin <= K_BAND_NOMINAL_CEILING_EXP


def band_intermediate_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_CARRIER_FLOOR_EXP


def spine_admits(e: int) -> bool:
    return band_ceiling_admits(e) and band_intermediate_admits(e)


# =====================================================================
#  Golden reference — native `double` arithmetic (bit-identical to the
#  C++ candidate's IEEE-754 hardware ops; see the module docstring for why
#  this is not MPFR-routed). fmax/fmin reproduce `Band`'s own documented
#  tie-break (`fmax`'s "NaN and ties" doc note): the first operand is returned on a tie
#  for fmax, the second for fmin — not the generic (unspecified) IEEE-754
#  tie behaviour, so this is a hand-written comparison, not Python's
#  builtin max()/min() or C's fmax()/fmin().
# =====================================================================
def ref_abs(a: float) -> float:
    # math.fabs is IEEE fabs — a bare sign-bit clear, including on NaN and
    # infinity, matching Band's own 3-XOR-mask implementation exactly.
    return math.fabs(a)


def ref_copysign(a: float, b: float) -> float:
    return math.copysign(a, b)


def ref_fmax(a: float, b: float) -> float:
    if is_nan(a):
        return b
    if is_nan(b):
        return a
    return b if (a - b) < 0.0 else a


def ref_fmin(a: float, b: float) -> float:
    if is_nan(a):
        return b
    if is_nan(b):
        return a
    return a if (a - b) < 0.0 else b


def ref_add(a: float, b: float) -> float:
    return a + b


def ref_sub(a: float, b: float) -> float:
    return a - b


def ref_mul(a: float, b: float) -> float:
    return a * b


def ref_fma(a: float, b: float, c: float) -> float:
    return math.fma(a, b, c) if hasattr(math, "fma") else _fma_fallback(a, b, c)


def _fma_fallback(a: float, b: float, c: float) -> float:
    import ctypes

    libm = ctypes.CDLL("libm.so.6")
    libm.fma.restype = ctypes.c_double
    libm.fma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    return libm.fma(a, b, c)


# =====================================================================
#  Op table
# =====================================================================
class Op:
    def __init__(self, name, arity, category, ref, bound, provenance):
        self.name = name
        self.arity = arity
        self.category = category  # 'exact_total' or 'spine_admitted'
        self.ref = ref
        self.bound = bound  # ULP bound for in_domain rows (int, ULP units)
        self.provenance = provenance


OPS = [
    Op("abs", 1, "exact_total", ref_abs, 0,
       "aether/banded/Band.h:830 — total, exact (3 sign-XOR mask), 0 ULP."),
    Op("copysign", 2, "exact_total", ref_copysign, 0,
       "aether/banded/Band.h:837 — total, exact (1 sign-XOR), 0 ULP."),
    Op("fmax", 2, "exact_total", ref_fmax, 0,
       "aether/banded/Band.h:863 — total, exact (certified sub's sign "
       "decides), 0 ULP; tie returns operand A."),
    Op("fmin", 2, "exact_total", ref_fmin, 0,
       "aether/banded/Band.h:874 — total, exact, 0 ULP; tie returns "
       "operand B (asymmetric vs fmax, both documented in-body)."),
    Op("add", 2, "spine_admitted", ref_add, 2,
       "aether/banded/Band.h:608 (add, one normalize) / :131 "
       "(certifiedBits=53). Bound derived from the S x 2 x eps formula "
       "the recip/rsqrt family's own bound rows use (kRecipUlpBound=2.0, "
       "'S x 2 x eps = 2 ULP'), applied with S=1 (one terminal-normalize "
       "rounding event, exactly recip's own S=1 case) since "
       "certifiedBits=53 promises double-equivalent precision out of a "
       "wider (72-bit) 3-limb carrier."),
    Op("sub", 2, "spine_admitted", ref_sub, 2,
       "aether/banded/Band.h:615 (sub = add of neg, inherits one "
       "normalize). Bound: same derivation as add (S=1 -> 2 ULP) — sub "
       "delegates to add, so the terminal-normalize count is identical."),
    Op("mul", 2, "spine_admitted", ref_mul, 2,
       "aether/banded/Band.h:555 (mul, one normalize). Bound: same "
       "derivation as add (S=1 -> 2 ULP) — mul's body is also exactly "
       "one terminal normalize(mulRaw(...))."),
    Op("fma", 3, "spine_admitted", ref_fma, 2,
       "Band-level fma is not implemented in aether. Confirmed absent "
       "from aether/banded/Band.h (add/sub/mul/abs/copysign/fmax/fmin/"
       "div/recip all present; fma is not — grep for 'Band fma(Band' "
       "returns zero hits) and from aether/banded/BandedRealOps.h's "
       "BandedFacade (documents four dispatch entries: abs/copysign/"
       "fmax/fmin only). aether::math::fma (aether/math/math.h:107) is "
       "restricted to T=float|double, no Band overload. This table's "
       "golden rows are minted so the harness is ready for a future fma "
       "port (touching only the generator's op table, the subject list, "
       "and its own test rows) but not wired into BandMathCert.h/the "
       "SASS subject list/the bench pending that port."),
]


# =====================================================================
#  Corpus construction per op — UNIFIED across both categories.
#
# A first draft of this generator labelled abs/copysign/fmax/fmin's entire
# corpus in_domain unconditionally (their contract reads "total, exact"
# over "all (incl. non-finite)"), and separately seeded its bulk/specials
# sample from arbitrary IEEE double magnitudes (DBL_MIN_NORMAL ~2^-1022,
# DBL_MAX ~2^1023), copying `tools/ulp_oracle`'s own generic-specials
# pattern — appropriate for a plain double->double MPFR reference, wrong
# here. `Band` is a lossy 3xFP32-limb
# encoding of a double (that is the whole point of the carrier — see
# `bandFromIEEE`'s "tiny head"/ceiling-wrap handling):
# "total, exact" describes the OP given a value the carrier ALREADY holds,
# never the round trip's fidelity against the ORIGINAL double outside the
# carrier's own admission window. Running the first draft surfaced this
# immediately as genuine cert failures (e.g. `abs(DBL_MIN_NORMAL)` reading
# `got=0`, `ulp=4503599627370496`) — a defect in the TEST's domain
# modelling, not in `Band::abs`. Fix: EVERY op (both categories) is now
# labelled from the SAME admission predicate applied to the RESULT-BEARING
# operand exponent(s); the categories differ only in the in_domain BOUND
# (0 ULP for abs/copysign/fmax/fmin, 2 ULP for add/sub/mul(/fma)) and in
# which operand(s) the result's exponent is derived from (see
# `EXPONENT_RULE` below). The two exact zero/inf/NaN-class specials rows
# are labelled in_domain UNCONDITIONALLY because `bandFromIEEE` round-trips
# them losslessly regardless of the OTHER operand's magnitude (its own
# sign-bit-only zero/inf/NaN paths) — this is
# the ONE place magnitude genuinely does not matter.
# =====================================================================
def specials_pool():
    return {POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, NAN_PAYLOAD_A, NAN_PAYLOAD_B}


def binary_specials_generic():
    """Curated combos crossing zero/inf/nan/+-1/+-2 — the ties/signed-zero
    rows Band.h documents in-body for fmax/fmin/copysign."""
    vals = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, float("inf"), float("-inf"), float("nan")]
    out = set()
    for a in vals:
        for b in vals:
            out.add((bits_of(a), bits_of(b)))
    return out


def exponent_of(x: float):
    """Base-2 exponent such that x == mantissa * 2**exponent_of(x) with
    1 <= |mantissa| < 2 (ldexp convention). None for 0/inf/nan (magnitude
    is not the governing property for those — see module note above)."""
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    m, e = math.frexp(x)  # x = m * 2**e, 0.5 <= |m| < 1
    return e - 1


def label_for_exponent(e):
    if e is None:
        return 0  # zero/inf/nan: always in_domain, see module note.
    if e >= K_BAND_HARD_CEILING_EXP or e <= K_BAND_HARD_FLOOR_EXP:
        return 2  # rejected: hard FP32 escape (overflow / subnormal collapse)
    if spine_admits(e):
        return 0
    return 1  # degraded: representable but past the certified margin


# Per-op rule for which operand exponent(s) the RESULT'S admission tracks —
# see module note: copysign's magnitude comes ONLY from its first operand
# (the sign source's own magnitude is irrelevant, sign survives collapse);
# fmax/fmin/add/sub select/combine operands so the WORSE of the two
# governs; mul(/fma)'s result exponent is the SUM of its factors' exponents
# (this is what a first draft got wrong for mul specifically — co-locating
# both operands at the SAME target exponent silently DOUBLES the product's
# exponent, which is why the first run's mul-degraded family measured
# genuine FP32 overflow (`got=inf`) instead of the intended gradual decay).
def result_label(op: Op, vals, ref: float):
    # ★ SECOND FINDING (also fixed here, not papered over): the first
    # per-op-rule draft of this function used max(EVERY OPERAND's own
    # exponent) for fmax/fmin too — wrong, because fmax/fmin DELIVER one of
    # their operands UNCHANGED. A row where operand `a` sits past the hard
    # ceiling but `b` (the one actually SELECTED, being the smaller/larger
    # as the op requires) is safely in range produces a perfectly correct,
    # in-bound, finite answer — labelling it "rejected" was asserting a
    # failure that does not occur. The general, uniform fix: include the
    # RESULT's OWN exponent (`exponent_of(ref)`, ref computed from the
    # SAME true `double` arithmetic the golden table already carries) in
    # the set every op's label is drawn from, alongside every INPUT
    # operand's own exponent (an operand that cannot itself be correctly
    # INGESTED is a real problem regardless of what the result turns out to
    # be). This one rule replaces every op-specific branch: abs/copysign's
    # result exponent already equals their magnitude-source operand's, so
    # the old special cases fall out as a special case of the general one;
    # mul(/fma)'s result exponent IS its true product exponent (no more
    # approximating it as a sum of ldexp() request targets, which is what
    # caused the FIRST finding above by an OFF-BY-A-FRACTION-OF-A-BINADE
    # miss right at the ceiling).
    exps = [exponent_of(v) for v in vals] + [exponent_of(ref)]
    finite_exps = [e for e in exps if e is not None]
    if not finite_exps:
        return 0
    return max(label_for_exponent(e) for e in finite_exps)


def build_rows(op: Op, seed: int, bulk_n: int = 500):
    rng = SplitMix64(seed)
    rows = []
    seen = set()

    def emit(bits_tuple):
        if bits_tuple in seen:
            return
        seen.add(bits_tuple)
        vals = [double_of(b) for b in bits_tuple]
        ref = op.ref(*vals)
        label = result_label(op, vals, ref)
        rows.append((bits_tuple, ref, label))

    def pad_arity(*bits):
        # fma (arity 3) reuses the binary specials/edge points with a THIRD
        # operand pinned to +1.0 (an additive identity-ish filler — fma's
        # own golden table is prepared-not-wired, see the OPS table entry,
        # so this only needs to be reasonable, not exhaustively tuned).
        bits = list(bits)
        while len(bits) < op.arity:
            bits.append(bits_of(1.0))
        return tuple(bits)

    # ---- zero/inf/NaN-class specials — always in_domain (module note) ----
    if op.arity == 1:
        for a in specials_pool():
            emit(pad_arity(a))
        for a in (bits_of(1.0), bits_of(-1.0), bits_of(TIER_EDGE_LO), bits_of(-TIER_EDGE_LO),
                  bits_of(TIER_EDGE_HI), bits_of(-TIER_EDGE_HI)):
            emit(pad_arity(a))
    else:
        for a, b in binary_specials_generic():
            emit(pad_arity(a, b))
        for a in specials_pool():
            emit(pad_arity(a, bits_of(1.0)))
            emit(pad_arity(bits_of(1.0), a))

    def at_exp(e, mant_seed_n=3):
        out = []
        for i in range(mant_seed_n):
            m = 1.0 + (i + 0.5) / mant_seed_n
            s = 1.0 if i % 2 == 0 else -1.0
            out.append(math.ldexp(m, e) * s)
        return out

    # `target_operand_exp(e)` is the exponent to place EACH operand at so
    # the op's OWN result lands near exponent `e` (module note: sum-typed
    # for mul/fma, identity-typed otherwise).
    def target_operand_exp(e):
        if op.name in ("mul", "fma"):
            return e // 2
        return e

    def make_points(e):
        # Independent sign choice on EVERY operand — a first draft derived
        # `b` from `a` (`b = a * 0.75`), which ALWAYS shares `a`'s sign and
        # silently starved every binary op's edge/degraded/rejected
        # families of same-sign-vs-opposite-sign diversity (surfaced as a
        # near-vacuous RED-first identity-twin arm for `copysign`: `return
        # a;` coincidentally equals `copysign(a,b)` whenever sign(a) ==
        # sign(b), which was EVERY generated row). Two mantissas x two
        # signs per operand now cover both cases explicitly.
        oe = target_operand_exp(e)
        mants = (1.25, 1.75)
        signs = (1.0, -1.0)
        out = []
        if op.arity == 1:
            for m in mants:
                for s in signs:
                    out.append((math.ldexp(m, oe) * s,))
        elif op.arity == 2:
            for ma in mants:
                for sa in signs:
                    for sb in signs:
                        a = math.ldexp(ma, oe) * sa
                        b = math.ldexp(ma * 0.83, oe) * sb
                        out.append((a, b))
        else:  # arity 3 (fma, prepared-not-wired)
            for ma in mants:
                for sa in signs:
                    a = math.ldexp(ma, oe) * sa
                    out.append((a, a * 0.75, a * 0.5))
        return out

    # ---- admitted edge: exactly at the declared boundary ----
    for e in (K_BAND_ADMITTED_LO, K_BAND_ADMITTED_HI):
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))

    # ---- bulk in_domain sweep, uniform over the admitted range ----
    for _ in range(bulk_n):
        e = K_BAND_ADMITTED_LO + rng.next() % (K_BAND_ADMITTED_HI - K_BAND_ADMITTED_LO + 1)
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))

    # ---- degraded: floor-side walk PAST the admitted edge, monotone family
    # (steps follow the measured curve, 2^-90 .. 2^-104: 1,2,4,6,8,12,16
    # binades past K_BAND_ADMITTED_LO) ----
    for step in (1, 2, 4, 6, 8, 12, 16):
        e = K_BAND_ADMITTED_LO - step
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))

    # ---- degraded: ceiling-side walk PAST the admitted edge (margin zone,
    # below the hard FP32 overflow) ----
    for step in (1, 2, 4, 8, 12, 18, 20):
        e = K_BAND_ADMITTED_HI + step
        if e >= K_BAND_HARD_CEILING_EXP:
            continue
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))

    # ---- rejected: hard escapes ----
    for e in (K_BAND_HARD_CEILING_EXP, K_BAND_HARD_CEILING_EXP + 2):
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))
    for e in (K_BAND_HARD_FLOOR_EXP, K_BAND_HARD_FLOOR_EXP - 4):
        for combo in make_points(e):
            emit(pad_arity(*[bits_of(v) for v in combo]))

    return rows


def build_corpus(op: Op, seed: int):
    return build_rows(op, seed)


# =====================================================================
#  Header emission — md5-fenced, mirrors `tools/ulp_oracle`'s own convention
#  (README.md "fence check").
# =====================================================================
def write_header(op: Op, rows, seed: int, out_path: str, argv_echo: str):
    in_domain = sum(1 for r in rows if r[2] == 0)
    degraded = sum(1 for r in rows if r[2] == 1)
    rejected = sum(1 for r in rows if r[2] == 2)

    head = []
    head.append(f"// GENERATED by tools/bandmath/gen_corpus.py — golden reference + domain")
    head.append(f"// label table for `{op.name}`, DO NOT EDIT BY HAND. Regenerate with the")
    head.append(f"// mint command line below and commit the diff in the SAME commit as")
    head.append(f"// whatever test consuming this header changed.")
    head.append(f"//")
    head.append(f"// mint command line: {argv_echo}")
    head.append(f"// seed:              0x{seed:016X}ULL")
    head.append(f"// reference:         native IEEE-754 double arithmetic (see")
    head.append(f"//                    tools/bandmath/gen_corpus.py's module docstring")
    head.append(f"//                    DEVIATION note for why this is not MPFR-routed)")
    head.append(f"// provenance:        {op.provenance}")
    head.append(f"// bound (in_domain, ULP): {op.bound}")
    head.append(f"//")
    head.append(f"// label encoding: 0=in_domain 1=degraded 2=rejected")
    head.append(f"// corpus: {len(rows)} rows ({in_domain} in_domain / {degraded} degraded / "
                 f"{rejected} rejected)")
    head.append(f"//")
    head.append("// The row block below is md5-FENCED:")
    head.append("//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'")
    head.append("//   sed -n \"/$BEGIN_RE/,/$END_RE/p\" <this file> | sed '1d;$d' | md5sum")
    head.append("// and compare against the value on the BEGIN line.")

    body = []
    body.append("#pragma once\n")
    body.append("\n".join(head) + "\n")
    body.append("#include <cstdint>\n#include <cstddef>\n")
    body.append(f"namespace aether_tests {{\nnamespace bandmath_golden {{\nnamespace {op.name} {{\n")
    body.append(f"inline constexpr int kArity = {op.arity};")
    body.append(f"inline constexpr double kUlpBound = {op.bound}.0;")
    body.append("struct Row {")
    body.append(f"    std::uint64_t in[{op.arity}];  ///< input bit patterns")
    body.append("    std::uint64_t ref;         ///< golden reference, correctly-rounded double bits")
    body.append("    std::uint8_t  label;       ///< 0=in_domain 1=degraded 2=rejected")
    body.append("};\n")

    rows_text_lines = ["inline constexpr Row kRows[] = {"]
    for bits_tuple, ref, label in rows:
        in_str = ", ".join(f"0x{b:016X}ULL" for b in bits_tuple)
        rows_text_lines.append(f"    {{ {{ {in_str} }}, 0x{bits_of(ref):016X}ULL, {label} }},")
    rows_text_lines.append("};")
    rows_text = "\n".join(rows_text_lines) + "\n"

    md5 = hashlib.md5(rows_text.encode("utf-8")).hexdigest()

    body.append(f"// ---8<--- GOLDEN ROWS BEGIN (md5 = {md5}) ---8<---")
    body.append(rows_text.rstrip("\n"))
    body.append("// ---8<--- GOLDEN ROWS END ---8<---\n")
    body.append(f"inline constexpr std::size_t kCorpusSize = {len(rows)};")
    body.append(f"inline constexpr std::size_t kInDomainCount = {in_domain};")
    body.append(f"inline constexpr std::size_t kDegradedCount = {degraded};")
    body.append(f"inline constexpr std::size_t kRejectedCount = {rejected};\n")
    body.append(f"}} // namespace {op.name}\n}} // namespace bandmath_golden\n}} // namespace aether_tests\n")

    with open(out_path, "w") as f:
        f.write("\n".join(body))

    return in_domain, degraded, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "bandmath"))
    ap.add_argument("--ops", default=None, help="comma-separated subset (default: all)")
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus.py: refusing to mint under $CI — a gate must never "
              "regenerate its own expectation.", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    argv_echo = "python3 tools/bandmath/gen_corpus.py " + " ".join(sys.argv[1:])

    want = set(args.ops.split(",")) if args.ops else None

    print(f"{'op':<10} {'total':>7} {'in_domain':>10} {'degraded':>9} {'rejected':>9}")
    for op in OPS:
        if want is not None and op.name not in want:
            continue
        rows = build_corpus(op, args.seed)
        out_path = os.path.join(out_dir, f"golden_{op.name}.h")
        in_d, deg, rej = write_header(op, rows, args.seed, out_path, argv_echo)
        print(f"{op.name:<10} {len(rows):>7} {in_d:>10} {deg:>9} {rej:>9}  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
