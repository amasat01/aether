#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_explog.py — golden minter for the ported
exp/log/pow ops.

`exp`/`log`/`pow` are minted against `tools/ulp_oracle`'s MPFR (256-bit,
correctly rounded) reference, unlike the exact/near-exact ops
`tools/bandmath/gen_corpus.py` covers with native `double` arithmetic.
This script therefore does not touch `gen_corpus.py` (which stays the
minter for those exact/near-exact primitives) — it is a separate,
purpose-fit two-stage minter for the transcendental class:

  Stage 1 — write a corpus-spec file per op under `tools/bandmath/specs/`
            (the boundary families below: reduction constants,
            admission-predicate edges, branch points, the always-on
            specials block ulp_oracle itself adds) and run the
            `ulp_oracle` CLI (built standalone, see
            tools/ulp_oracle/README.md) against it, producing a raw
            MPFR-referenced header (`aether_tests::ulp_golden::<fn>::kRows`,
            no domain label — ulp_oracle knows nothing about Band's
            admission predicates).
  Stage 2 — label each minted row `in_domain`/`degraded`/`rejected` from
            Band's own admission predicate (`bandExpAdmits`/`bandLogAdmits`/
            `bandPowAdmits`, transcribed below from
            `aether/banded/BandExpLog.h` — same constants, same formulas,
            checked by the "single-point interval" reading: a row's own
            input(s) are the consumer's declared interval, mirroring
            `gen_corpus.py`'s own per-row `result_label` convention) and
            re-emit in `gen_corpus.py`'s own header shape
            (`aether_tests::bandmath_golden::<op>::Row{in[N], ref, label}`,
            md5-fenced) so `tests/bandmath/BandMathCertTranscendental.h`'s
            drivers consume it exactly like every other golden table.

Usage:
    python3 tools/bandmath/gen_corpus_explog.py --oracle-bin <path/to/ulp_oracle>
        [--out-dir tests/bandmath] [--spec-dir tools/bandmath/specs]

Refuses to write when $CI is set (mirrors gen_corpus.py's own re-mint
convention).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import struct
import subprocess
import sys

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5  # matches the seed convention gen_corpus.py / ulp_oracle use

# =====================================================================
#  Bit helpers (same convention as gen_corpus.py)
# =====================================================================
def bits_of(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def double_of(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


LN2 = 0.6931471805599453
SQRT2 = 1.4142135623730951


def nextafter(x: float, direction: float) -> float:
    return math.nextafter(x, direction)


# =====================================================================
#  Admission predicates — transcribed verbatim from
#  aether/banded/BandExpLog.h. Same constants as aether/banded/Band.h's
#  own bandCeilingAdmits/bandIntermediateAdmits (K_BAND_ADMITTED_LO/HI
#  already match gen_corpus.py's own transcription of those two).
# =====================================================================
K_BAND_NOMINAL_CEILING_EXP = 115
K_BAND_CARRIER_FLOOR_EXP = -96
K_BAND_ADMISSION_MARGIN = 8
K_BAND_ADMITTED_LO = K_BAND_CARRIER_FLOOR_EXP + K_BAND_ADMISSION_MARGIN  # -88
K_BAND_ADMITTED_HI = K_BAND_NOMINAL_CEILING_EXP - K_BAND_ADMISSION_MARGIN  # 107
K_BAND_LOG_TAIL_NORMAL_EXP = -77
K_BAND_HARD_CEILING_EXP = 128
K_BAND_HARD_FLOOR_EXP = -149
K_BAND_LOG2E = 1.44269504088896340736
K_BAND_EXP_SAT_HIGH = 88.7228391  # 128*ln2
K_BAND_EXP_SAT_LOW = -103.972077  # -150*ln2


def band_ceiling_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e + margin <= K_BAND_NOMINAL_CEILING_EXP


def band_intermediate_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_CARRIER_FLOOR_EXP


def band_exp_ceil_exp(max_input: float) -> int:
    t = max_input * K_BAND_LOG2E
    i = math.trunc(t)
    return i + 1 if t > float(i) else i


def band_exp_floor_exp(min_input: float) -> int:
    t = min_input * K_BAND_LOG2E
    i = math.trunc(t)
    return i - 1 if t < float(i) else i


def band_exp_admits(max_input: float, min_input: float, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return band_ceiling_admits(band_exp_ceil_exp(max_input), margin) and band_intermediate_admits(
        band_exp_floor_exp(min_input), margin
    )


def exponent_of(x: float):
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    m, e = math.frexp(x)
    return e - 1


def band_log_admits(max_abs_exp_arg, min_abs_exp_arg, min_abs_exp_result, margin=K_BAND_ADMISSION_MARGIN) -> bool:
    return (
        band_ceiling_admits(max_abs_exp_arg, margin)
        and band_intermediate_admits(min_abs_exp_arg, margin)
        and band_intermediate_admits(min_abs_exp_result, margin)
    )


def band_pow_admits(max_t, min_t, max_abs_exp_base, min_abs_exp_base, min_abs_exp_log, margin=K_BAND_ADMISSION_MARGIN) -> bool:
    return (
        band_log_admits(max_abs_exp_base, min_abs_exp_base, min_abs_exp_log, margin)
        and band_exp_admits(max_t, min_t, margin)
        and (min_abs_exp_log - margin >= K_BAND_LOG_TAIL_NORMAL_EXP)
    )


# =====================================================================
#  Per-op row labeling (single-point reading: the row's own input(s) ARE
#  the "consumer interval" — mirrors gen_corpus.py's result_label).
# =====================================================================
def label_exp(x: float) -> int:
    if not math.isfinite(x):
        return 0  # specials: the guard's own exact IEEE rule (Band.h note, gen_corpus.py's "magnitude
        # genuinely does not matter" rule for the zero/inf/nan class)
    if x < K_BAND_EXP_SAT_LOW or x > K_BAND_EXP_SAT_HIGH:
        return 2  # rejected: past saturation -- the guard fires an EXACT 0/+inf/NaN
    if band_exp_admits(x, x):
        return 0
    return 1  # degraded: representable, past the certified edge, not yet saturated


def label_log(x: float) -> int:
    # Mirrors aether::banded::detail::log's own guard structure exactly
    # (BandExpLog.h): NaN, +-0 and +inf are EXACT IEEE answers (specials,
    # "magnitude genuinely does not matter" -- always in_domain); -inf and
    # any finite x<0 hit the "x < 0" branch, which returns NaN -- a LOUD
    # REJECT, not a degradation.
    if x != x:
        return 0
    if x == 0.0:
        return 0
    if math.isinf(x):
        return 0 if x > 0.0 else 2
    if x < 0.0:
        return 2
    r = math.log(x)
    if r == 0.0:
        return 0  # x == 1.0 exactly -> log(1) == +0 bit-exact (an "Exact point", not a near-one leg case)
    e = exponent_of(x)
    er = exponent_of(r)
    if e is None or er is None:
        return 0
    if e >= K_BAND_HARD_CEILING_EXP or e <= K_BAND_HARD_FLOOR_EXP or \
            er >= K_BAND_HARD_CEILING_EXP or er <= K_BAND_HARD_FLOOR_EXP:
        return 2
    if band_log_admits(e, e, er):
        return 0
    return 1


def label_pow(x: float, y: float) -> int:
    # Mirror aether's own pow() guard rows, in order: y==0 (->1 for EVERY x,
    # exact), y==1 (->x, exact), x<=0/+inf/NaN (loud reject NaN) are ALWAYS
    # in_domain (the guard delivers the exact IEEE answer, "magnitude
    # genuinely does not matter" -- same rule the specials rows get).
    if y == 0.0 or y == 1.0:
        return 0
    if not (x > 0.0 and x < float("inf")):
        # ★ FINDING (fixed here, not papered over): a first draft labelled
        # EVERY row that fails Band's x>0-finite guard `in_domain`, reasoning
        # "the guard's NaN is exact". That is true only where MPFR's own
        # answer is ALSO NaN (e.g. a NEGATIVE base to a non-integer power,
        # where IEEE pow itself is undefined) -- it is FALSE wherever MPFR's
        # full IEEE `pow` has a real answer Band's narrower guard does not
        # compute (e.g. `pow(-32, -32)` = a tiny finite double, an EVEN
        # negative-integer case; measured live as `ref=0 got=nan` failures
        # once run). Band's own doc block calls this "a LOUD REJECT of the
        # uncertified domain, not an IEEE pow" -- i.e. the DOCUMENTED escape,
        # which is exactly what `rejected` asserts (the row must NOT
        # silently agree with full IEEE `pow`).
        return 2
    if x == 1.0:
        # pow(1, y) == 1 for every FINITE y, exact ("falls out", not
        # special-cased in Band's guard) -- ★ FINDING: a first draft
        # dropped "finite" and labelled pow(1, inf)/pow(1, NaN) in_domain
        # too. Band's guard does NOT special-case x==1 at all; those two
        # rows fall through to `exp(mul(y, log(x)))` = `exp(mul(inf, +0))`
        # = `exp(NaN)` = NaN, where full IEEE `pow(1, inf) == 1` -- the SAME
        # "LOUD REJECT of the uncertified domain" the x<=0 branch documents,
        # caught live (`ref=1 got=nan`) once run.
        return 0 if math.isfinite(y) else 2
    ex = exponent_of(x)
    logx = math.log(x)
    elog = exponent_of(logx)
    if ex is None or elog is None:
        return 0
    t = y * logx
    if math.isnan(t):
        return 2
    if math.isinf(t):
        # ★ SECOND FINDING: a first draft treated ANY infinite `t` as a
        # rejected escape. Wrong -- when `t` is a genuine MATHEMATICAL
        # +-infinity (one factor finite, the other +-inf, product not
        # indeterminate), `exp(+-inf)` is a WELL-DEFINED limit (+inf / +0)
        # that Band's guard delivers EXACTLY, and MPFR's own `pow` uses the
        # SAME limit (e.g. `pow(0.5, +inf) == 0`, `pow(2, +inf) == +inf`) --
        # caught live (`ref=0 got=0` "silently passed", i.e. correctly, on
        # the curated {0.5,+inf}/{2,-inf} pow specials). Only a NaN `t`
        # (the indeterminate `0*inf` shape) is a genuine reject.
        return 0
    # exp's OWN saturation on t (NOT the generic 2^128/2^-149 hard-escape --
    # that bounds x/y's own admission; t feeds exp(), whose hard 0/+inf
    # clamp fires at kBandExpSat{Low,High}, well short of the FP32 ceiling).
    # A 1.0-wide margin around the threshold absorbs the fact that `t` here
    # is a PYTHON double approximation of what Band's own `mul(y, log(x))`
    # actually produces (its own log(x) already carries a tiny relative
    # error) -- a row landing WITHIN the margin is labelled `degraded`
    # (a weak finiteness-only claim) rather than `rejected` (a hard escape
    # assertion), so boundary jitter between the two computations of `t`
    # cannot manufacture a spurious "silently passed in-bound" failure.
    if t < K_BAND_EXP_SAT_LOW - 1.0 or t > K_BAND_EXP_SAT_HIGH + 1.0:
        return 2
    if band_pow_admits(t, t, ex, ex, elog):
        return 0
    return 1


# =====================================================================
#  Stage 1 — corpus-spec files (ulp_oracle grammar, README.md).
# =====================================================================
def build_exp_spec() -> str:
    lines = [f"seed 0x{DEFAULT_SEED:016X}"]
    lines.append("# reduction-boundary family: every admitted k*ln2 in/around [-88,107], +-1ULP")
    lines.append("reduction ln2 115")
    # extra depth (contract: "5 depths incl 1e-9/1e-15") at representative k
    pts = []
    for k in (-88, -60, -1, 0, 1, 60, 107):
        c = k * LN2
        for depth in (1e-15, 1e-9, 1e-6, 1e-3):
            pts.append(c + depth)
            pts.append(c - depth)
    # reduction-midpoint
    for k in (-88, -50, -1, 0, 1, 50, 107):
        pts.append((k + 0.5) * LN2)
    # domain edges (certified interval, both sides, tight neighbours)
    lo, hi = -60.997, 74.16679
    for e in (lo, hi):
        pts += [e, nextafter(e, e - 1), nextafter(e, e + 1)]
    # tiny |x|
    pts += [1e-300, -1e-300, 1e-30, -1e-30, 1e-10, -1e-10]
    lines.append("points " + " ".join(repr(v) for v in pts))
    lines.append("# pow2 input family")
    lines.append("reduction pow2 -30 6")
    lines.append("# bulk in-domain SAMPLE")
    lines.append(f"uniform {lo} {hi} 400")
    lines.append("# degraded bands (past the certified edge, short of saturation), both sides")
    lines.append(f"uniform {K_BAND_EXP_SAT_LOW} {lo} 150")
    lines.append(f"uniform {hi} {K_BAND_EXP_SAT_HIGH} 60")
    lines.append("# rejected: past saturation, both sides")
    lines.append(f"uniform {K_BAND_EXP_SAT_HIGH} 400.0 60")
    lines.append(f"uniform -700.0 {K_BAND_EXP_SAT_LOW} 60")
    return "\n".join(lines) + "\n"


def build_log_spec() -> str:
    lines = [f"seed 0x{DEFAULT_SEED:016X}"]
    lines.append("points " + " ".join(
        repr(v) for v in (
            1.0, nextafter(1.0, 2.0), nextafter(1.0, 0.0),
            -1.0, -2.0, -0.5, -1e300,
        )
    ))
    # near-one bulk: |x-1| in units of ULP(1) = 2^-52 -- ★ FINDING (not
    # papered over): the near-one ADMISSION leg bites at 2^-88, far BELOW
    # IEEE double's own resolution at 1 (2^-52 -- a double simply cannot
    # represent a value strictly between 1 and 1+2^-52). Every x this
    # harness's `bandFromDoubleBits` ingest can produce therefore has
    # |log x| >= ~2^-52, comfortably INSIDE the admitted [-88,107] window --
    # the near-one leg can only ever bite a Band value produced by CHAINED
    # Band arithmetic, never a fresh double-bit-pattern corpus row. So this
    # class exercises the near-one BULK (accuracy as x approaches 1, the
    # in_domain case) rather than a degraded/rejected near-one row -- that
    # degraded exercise is out of reach for THIS corpus format by construction,
    # not by omission.
    pts = []
    for k in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 1000, 10 ** 6, 10 ** 9, 10 ** 12):
        d = k * (2.0 ** -52)
        pts += [1.0 + d, 1.0 - d]
    lines.append("points " + " ".join(repr(v) for v in pts))
    # sqrt(2) centring seam, at several exponents
    pts2 = []
    for e in (-90, -50, -10, 0, 1, 10, 50, 100):
        m = SQRT2 * (2.0 ** e)
        pts2 += [m, nextafter(m, m * 1.01), nextafter(m, m * 0.99)]
    lines.append("points " + " ".join(repr(v) for v in pts2))
    lines.append("# pow2 input family, envelope-edge on both legs, both signs (negative -> loud reject)")
    lines.append("reduction pow2 -95 112")
    lines.append("# bulk positive SAMPLE, wide log-uniform sweep")
    lines.append("loguniform 1e-30 1e32 400")
    lines.append("# bulk negative / loud-reject SAMPLE")
    lines.append("points " + " ".join(repr(-v) for v in (1e-10, 1e-3, 1.0, 2.0, 1e3, 1e10, 1e30)))
    return "\n".join(lines) + "\n"


def build_pow_spec() -> str:
    lines = [f"seed 0x{DEFAULT_SEED:016X}"]
    # identities
    id_pts = []
    for x in (2.0, 0.5, 3.0, 1e-10, 1e10, -2.0, -0.0, float("nan"), float("inf")):
        id_pts += [(x, 0.0), (x, 1.0)]
    for y in (2.0, 0.5, -1.0, 100.0, -100.0, 0.25, 1e-10):
        id_pts.append((1.0, y))
    # pow(2,k) exact family
    for k in range(-100, 101, 5):
        id_pts.append((2.0, float(k)))
    # cross-checks
    for x in (3.0, 5.0, 1e10, 1e-10, 0.001, 1000.0):
        id_pts += [(x, 2.0), (x, 0.5)]
    lines.append("points2 " + " ".join(f"{repr(a)} {repr(b)}" for a, b in id_pts))
    # amplification corners: push |y*log x| toward exp's certified extremes
    # from bases as close to 1 as an IEEE double can represent (k ULPs away,
    # k=1..huge -- @see build_log_spec's near-one FINDING: pow's -69 leg,
    # like log's -88, is unreachable-by-REJECTION from a double-derived base,
    # so every row below is a genuine IN_DOMAIN amplification-corner
    # accuracy exercise, never a near-one REJECT).
    amp_pts = []
    for k in (1, 2, 4, 16, 256, 10 ** 6, 10 ** 12):
        d = k * (2.0 ** -52)
        for sign in (1.0, -1.0):
            x = 1.0 + sign * d
            logx = math.log(x)
            if logx == 0.0:
                continue
            for target_t in (74.0, -60.0, 10.0, -10.0):
                y = target_t / logx
                amp_pts.append((x, y))
    # exp-inherited ceiling/floor REJECTS: t = y*log(x) pushed WAY past exp's
    # certified interval from ordinary (non-near-one) bases -- genuinely
    # reachable, unlike the near-one leg.
    for base, big_y in ((2.0, 200.0), (2.0, -200.0), (0.5, 200.0), (10.0, 50.0),
                         (10.0, -50.0), (1.5, 300.0), (1.5, -300.0)):
        amp_pts.append((base, big_y))
    lines.append("points2 " + " ".join(f"{repr(a)} {repr(b)}" for a, b in amp_pts))
    lines.append("# base envelope edges crossed with 1 and with itself (reduction2's own pairing)")
    lines.append("reduction2 pow2 -90 109")
    lines.append("# bulk: log-uniform bases x moderate exponents")
    lines.append("loguniform2 1e-20 1e20 -40.0 40.0 300")
    return "\n".join(lines) + "\n"


# =====================================================================
#  Stage 1 runner + Stage 2 (parse raw MPFR header, relabel, re-emit).
# =====================================================================
UNARY_ROW_RE = re.compile(
    r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")
BINARY_ROW_RE = re.compile(
    r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")


def parse_unary_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16)) for a, b, _c in UNARY_ROW_RE.findall(text)]


def parse_binary_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16), int(c, 16)) for a, b, c, _c in BINARY_ROW_RE.findall(text)]


def write_labeled_header(op: str, arity: int, bound: float, rows, provenance: str, out_path: str, mint_cmd: str):
    in_domain = sum(1 for r in rows if r[-1] == 0)
    degraded = sum(1 for r in rows if r[-1] == 1)
    rejected = sum(1 for r in rows if r[-1] == 2)

    head = []
    head.append(f"// GENERATED by tools/bandmath/gen_corpus_explog.py -- golden reference + domain")
    head.append(f"// label table for `{op}`, DO NOT EDIT BY HAND. Regenerate with the mint command")
    head.append(f"// line below and commit the diff in the SAME commit as whatever test consuming")
    head.append(f"// this header changed.")
    head.append(f"//")
    head.append(f"// mint command line: {mint_cmd}")
    head.append(f"// reference:         MPFR @ 256 bits, correctly rounded to double (RNDN), via")
    head.append(f"//                    tools/ulp_oracle (mandatory for transcendentals);")
    head.append(f"//                    domain label added by THIS script")
    head.append(f"//                    from Band's own admission predicate (Stage 2).")
    head.append(f"// provenance:        {provenance}")
    head.append(f"// bound (in_domain, ULP): {bound}")
    head.append(f"//")
    head.append(f"// label encoding: 0=in_domain 1=degraded 2=rejected")
    head.append(f"// corpus: {len(rows)} rows ({in_domain} in_domain / {degraded} degraded / {rejected} rejected)")
    head.append(f"//")
    head.append("// The row block below is md5-FENCED:")
    head.append("//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'")
    head.append("//   sed -n \"/$BEGIN_RE/,/$END_RE/p\" <this file> | sed '1d;$d' | md5sum")
    head.append("// and compare against the value on the BEGIN line.")

    body = []
    body.append("#pragma once\n")
    body.append("\n".join(head) + "\n")
    body.append("#include <cstdint>\n#include <cstddef>\n")
    body.append(f"namespace aether_tests {{\nnamespace bandmath_golden {{\nnamespace {op} {{\n")
    body.append(f"inline constexpr int kArity = {arity};")
    body.append(f"inline constexpr double kUlpBound = {bound};")
    body.append("struct Row {")
    body.append(f"    std::uint64_t in[{arity}];  ///< input bit pattern(s)")
    body.append("    std::uint64_t ref;         ///< golden reference, correctly-rounded double bits")
    body.append("    std::uint8_t  label;       ///< 0=in_domain 1=degraded 2=rejected")
    body.append("};\n")

    rows_text_lines = ["inline constexpr Row kRows[] = {"]
    for r in rows:
        if arity == 1:
            in0, ref, label = r
            rows_text_lines.append(f"    {{ {{ 0x{in0:016X}ULL }}, 0x{ref:016X}ULL, {label} }},")
        else:
            in0, in1, ref, label = r
            rows_text_lines.append(f"    {{ {{ 0x{in0:016X}ULL, 0x{in1:016X}ULL }}, 0x{ref:016X}ULL, {label} }},")
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
    body.append(f"}} // namespace {op}\n}} // namespace bandmath_golden\n}} // namespace aether_tests\n")

    with open(out_path, "w") as f:
        f.write("\n".join(body))
    return in_domain, degraded, rejected


OPS = {
    "exp": (1, 0.5, build_exp_spec,
            "aether/banded/BandExpLog.h bandExp/exp; bound = kExpUlpBound=0.5 "
            "(BandLevers.ExpWithinDerivedUlpBoundOnHost)."),
    "log": (1, 0.5, build_log_spec,
            "aether/banded/BandExpLog.h log; bound = kLogUlpBound=0.5."),
    "pow": (2, 0.5, build_pow_spec,
            "aether/banded/BandExpLog.h pow; bound = kPowUlpBound=0.5."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-bin", required=True, help="path to the built ulp_oracle binary")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "bandmath"))
    ap.add_argument("--spec-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "specs"))
    ap.add_argument("--raw-dir", default="/tmp/p11c1_explog_raw")
    ap.add_argument("--ops", default=None)
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus_explog.py: refusing to mint under $CI.", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    spec_dir = os.path.abspath(args.spec_dir)
    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    want = set(args.ops.split(",")) if args.ops else None
    print(f"{'op':<6} {'total':>7} {'in_domain':>10} {'degraded':>9} {'rejected':>9}")
    for op, (arity, bound, spec_fn, provenance) in OPS.items():
        if want is not None and op not in want:
            continue
        spec_text = spec_fn()
        spec_path = os.path.join(spec_dir, f"{op}.spec")
        with open(spec_path, "w") as f:
            f.write(spec_text)

        raw_path = os.path.join(raw_dir, f"golden_{op}_mpfr_raw.h")
        cmd = [args.oracle_bin, op, spec_path, raw_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"ulp_oracle FAILED for {op}: rc={res.returncode}\n{res.stdout}\n{res.stderr}", file=sys.stderr)
            return 1

        if arity == 1:
            raw_rows = parse_unary_raw(raw_path)
            labeled = []
            label_fn = label_exp if op == "exp" else label_log
            for in0, ref in raw_rows:
                x = double_of(in0)
                labeled.append((in0, ref, label_fn(x)))
        else:
            raw_rows = parse_binary_raw(raw_path)
            labeled = []
            for in0, in1, ref in raw_rows:
                x = double_of(in0)
                y = double_of(in1)
                labeled.append((in0, in1, ref, label_pow(x, y)))

        mint_cmd = (f"python3 tools/bandmath/gen_corpus_explog.py --oracle-bin <ulp_oracle> "
                    f"--ops {op}  (Stage 1: {' '.join(cmd)})")
        out_path = os.path.join(out_dir, f"golden_{op}.h")
        in_d, deg, rej = write_labeled_header(op, arity, bound, labeled, provenance, out_path, mint_cmd)
        print(f"{op:<6} {len(labeled):>7} {in_d:>10} {deg:>9} {rej:>9}  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
