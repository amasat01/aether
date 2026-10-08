#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_trig.py — golden minter for the ported trig
ops (sin, cos, sincos).

`sin`/`cos`/`sincos` are minted against `tools/ulp_oracle`'s MPFR
(256-bit, correctly rounded) reference rather than against native double
arithmetic (same rule `gen_corpus_explog.py`'s own docstring cites).
Two-stage minter, identical shape to `gen_corpus_explog.py`:

  Stage 1 — write a corpus-spec file per op under `tools/bandmath/specs/`
            (enumerated reduction boundaries `k*pi/2` across the whole
            admitted `|k|` range up to the runtime guard `2^24`, the
            default-margin certified edge, a real large-argument worked
            example drawn from planetary nutation-precession theory,
            plus supplementary uniform bulk coverage) and run the
            `ulp_oracle` CLI against it.
  Stage 2 — label each minted row `in_domain`/`degraded`/`rejected` from
            `aether/banded/BandTrig.h`'s own `bandTrigAdmits` (transcribed
            below, same constant, same formula) and re-emit in
            `gen_corpus.py`'s own header shape so
            `tests/bandmath/BandMathCertTrig.h`'s drivers consume it
            exactly like every other golden table. `sincos` is dual-output
            at the oracle CLI level (`ref`=sin, `ref2`=cos in one row) and
            is re-emitted as two fields in `gen_corpus.py`'s Row shape
            extended with `ref2` (this file's own addition — no other op
            in this directory needed a second reference column).

Usage:
    python3 tools/bandmath/gen_corpus_trig.py --oracle-bin <path/to/ulp_oracle>
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

PI = math.pi
PI_HALF = math.pi / 2.0


# =====================================================================
#  Bit helpers (same convention as gen_corpus_explog.py)
# =====================================================================
def bits_of(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def double_of(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


def nextafter(x: float, direction: float) -> float:
    return math.nextafter(x, direction)


# =====================================================================
#  bandTrigAdmits -- transcribed verbatim from aether/banded/BandTrig.h.
#  One leg (ceiling only) -- see that file's own doc comment for why
#  there is no floor leg.
# =====================================================================
K_BAND_TRIG_MAX_ARG_EXP = 24  # kBandTrigKBits
K_BAND_ADMISSION_MARGIN = 8
K_BAND_TRIG_SAT_MAX = 16777216.0  # 2^24, the runtime guard


def band_trig_admits(max_abs_exp_arg: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return max_abs_exp_arg + margin <= K_BAND_TRIG_MAX_ARG_EXP


def exponent_of(x: float):
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    m, e = math.frexp(x)
    return e - 1


def round_to_float32(x: float) -> float:
    """Round-to-nearest-even a double to its float32 value (as a double),
    matching Band ingestion's `bandFromIEEE` (which stores `x.hi` as a
    `float`). Load-bearing for the guard/label decision right at the
    runtime cliff (2^24): a double one ULP past 2^24 is astronomically
    smaller than float32's OWN ulp there (2.0), so it rounds BACK DOWN to
    exactly 2^24 once it reaches `x.hi` -- the guard `x.hi <= kBandTrigSatMax`
    then ADMITS it, even though the raw double compares greater. A first
    draft compared the raw double directly and mislabeled two corpus rows
    `rejected` that Band actually computes (caught live: "rejected row ...
    silently passed in-bound: ref=... got=..." with ref==got exactly).
    A magnitude beyond float32's own finite range (the oracle's ALWAYS-ON
    specials block includes +-DBL_MAX, far past FLT_MAX) SATURATES to
    +-inf on a real cast; `struct.pack` raises instead, so that case is
    handled explicitly."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def label_trig(x: float) -> int:
    """Mirrors `bd::sincos`/`bd::sin`/`bd::cos`'s own guard structure
    (BandTrig.h): NaN and +-inf hit the FIRST guard (`!(x.hi >= -satMax &&
    x.hi <= satMax)`, false for NaN/inf) and return NaN -- MPFR's own
    `sin`/`cos` of NaN or +-inf is ALSO NaN (a genuine domain error, not a
    well-defined limit the way `pow`'s +-inf legs were), so this is an
    EXACT match, "magnitude genuinely does not matter" (in_domain, label 0),
    exactly like `label_exp`'s specials leg. `x == 0` is the second exact
    guard row (sin(+-0)==+-0, cos(+-0)==1), also in_domain. Beyond the
    runtime guard `2^24`, Band returns NaN while MPFR has a well-defined
    finite answer -- the documented escape (rejected, label 2). Inside the
    guard but past the DEFAULT-margin certified edge (`bandTrigAdmits`
    false): degraded (label 1) -- per BandTrig.h's own doc comment this
    band is NOT a gradual accuracy loss (the reduction's absolute error is
    uniform over the whole admitted range), it is simply UNCERTIFIED by the
    default declaration; the corpus still labels it the generic three-way
    way so the harness's degraded-arm driver runs, deliberately probing
    whether that "no real degradation" claim actually holds live.
    """
    if not math.isfinite(x):
        return 0
    if x == 0.0:
        return 0
    xhi = round_to_float32(x)  # what Band's guard actually compares
    if abs(xhi) > K_BAND_TRIG_SAT_MAX:
        return 2
    e = exponent_of(x)
    if e is None:
        return 0
    if band_trig_admits(e):
        return 0
    return 1


# =====================================================================
#  Stage 1 — corpus-spec (ulp_oracle grammar, README.md). Shared across
#  sin/cos/sincos: identical domain, identical reduction, so ONE spec
#  serves all three (mirrors how sincos's own body reuses trigReduce for
#  sin and cos separately).
# =====================================================================
def build_trig_spec() -> str:
    lines = [f"seed 0x{DEFAULT_SEED:016X}"]

    # --- dense near-origin reduction-boundary family: EVERY k*(pi/2) for
    # |k| <= 200 (401 boundaries), +-1ULP neighbors via the oracle's own
    # `reduction pi_over_2` directive (MPFR-derived pi, exact). Covers every
    # quadrant transition through the first ~314 rad -- the regime almost
    # every real consumer (unwrapped angles under a handful of orbits)
    # actually lands in.
    lines.append("# dense near-origin quadrant-boundary family: k*(pi/2), |k|<=200, MPFR pi")
    lines.append("reduction pi_over_2 200")

    # --- ENUMERATED boundaries across every binade up to and past the
    # runtime guard (2^24): k = +-2^e for e = 0..24 (49 values), each
    # x = k*(pi/2) via a plain double product (the ulp_oracle re-derives
    # its OWN MPFR reference for whatever double is fed in -- this point
    # only needs to sit NEAR the true boundary, not equal it, exactly the
    # convention gen_corpus_explog.py's own build_exp_spec/build_log_spec
    # use for their own large-k points). Each gets its own +-1ULP-of-double
    # neighbor pair, mirroring `addTriplet`'s own idiom.
    pts = []

    def add_triplet(v: float):
        pts.append(v)
        pts.append(nextafter(v, v - 1.0 if v != 0.0 else -1.0))
        pts.append(nextafter(v, v + 1.0 if v != 0.0 else 1.0))

    for e in range(0, 25):  # up to 2^24 inclusive
        for k in (float(2 ** e), -float(2 ** e)):
            add_triplet(k * PI_HALF)
    # midpoints between binade boundaries (probes the interior, not just the
    # power-of-two k itself)
    for e in range(0, 24):
        k = (2 ** e) * 1.5
        add_triplet(k * PI_HALF)
        add_triplet(-k * PI_HALF)
    # small-k fine coverage (every integer boundary through the first 20
    # quadrants beyond what `reduction pi_over_2 200` already gave, plus a
    # couple of PRIME k -- not needed since 200 already covers |k|<=200
    # exhaustively; kept here as a no-op comment for the reader).

    # --- the DEFAULT-margin certified edge (2^16 rad) and the runtime guard
    # (2^24 rad) themselves, both signs, tight neighbours -- the two label
    # TRANSITIONS the degraded/rejected buckets pivot on.
    for edge in (65536.0, 131072.0, K_BAND_TRIG_SAT_MAX):
        for s in (1.0, -1.0):
            add_triplet(s * edge)

    # --- a worked consumer example (see BandTrig.h's own `bandTrigAdmits`
    # doc comment): a real nutation-precession argument used in planetary
    # orientation theory, `192.93 + 41215163.19675*T` deg, T in +-1 Julian
    # century. Enumerated across the full declared range (not just the
    # endpoint) -- this is what "reduces and evaluates at that magnitude"
    # means.
    m2_const, m2_rate, deg2rad = 192.93, 41215163.19675, 1.74532925199432957692e-2
    for i in range(-100, 101, 4):
        T = i * 0.01
        add_triplet((m2_const + m2_rate * T) * deg2rad)

    lines.append("points " + " ".join(repr(v) for v in pts))

    # --- bulk supplementary coverage (SAMPLE, not enumerated -- same
    # "bulk in-domain SAMPLE" role gen_corpus_explog.py's own scripts give
    # `uniform`): in_domain interior, the (2^16,2^24] degraded band, and a
    # short stretch past the 2^24 guard (rejected).
    lines.append("# bulk in-domain SAMPLE, both signs")
    lines.append(f"uniform -65536.0 65536.0 500")
    lines.append("# bulk DEGRADED band (past the default-margin edge, short of the guard), both signs")
    lines.append(f"uniform 65536.0 {K_BAND_TRIG_SAT_MAX} 150")
    lines.append(f"uniform -{K_BAND_TRIG_SAT_MAX} -65536.0 150")
    lines.append("# bulk REJECTED (past the runtime guard), both signs")
    lines.append(f"uniform {K_BAND_TRIG_SAT_MAX} 200000000.0 60")
    lines.append(f"uniform -200000000.0 -{K_BAND_TRIG_SAT_MAX} 60")

    return "\n".join(lines) + "\n"


# =====================================================================
#  Stage 1 runner + Stage 2 (parse raw MPFR header, relabel, re-emit).
# =====================================================================
UNARY_ROW_RE = re.compile(
    r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")
# dual (sincos) raw rows: `{ 0x..ULL, 0x..ULL, <cls>, 0x..ULL, <cls> },` (NOT
# wrapped in an extra `{ }` around the input -- @see oracle_core.cpp's own
# `writeHeader`, the `dual` branch).
DUAL_ROW_RE = re.compile(
    r"\{\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*\d+\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*\d+\s*\}")


def parse_unary_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16)) for a, b, _c in UNARY_ROW_RE.findall(text)]


def parse_dual_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16), int(c, 16)) for a, b, c in DUAL_ROW_RE.findall(text)]


def write_labeled_header_unary(op: str, bound: float, rows, provenance: str, out_path: str, mint_cmd: str):
    in_domain = sum(1 for r in rows if r[-1] == 0)
    degraded = sum(1 for r in rows if r[-1] == 1)
    rejected = sum(1 for r in rows if r[-1] == 2)

    head = []
    head.append("// GENERATED by tools/bandmath/gen_corpus_trig.py -- golden reference + domain")
    head.append(f"// label table for `{op}`, DO NOT EDIT BY HAND. Regenerate with the mint command")
    head.append("// line below and commit the diff in the SAME commit as whatever test consuming")
    head.append("// this header changed.")
    head.append("//")
    head.append(f"// mint command line: {mint_cmd}")
    head.append("// reference:         MPFR @ 256 bits, correctly rounded to double (RNDN), via")
    head.append("//                    tools/ulp_oracle (mandatory for transcendentals);")
    head.append("//                    domain label added by THIS script")
    head.append("//                    from Band's own admission predicate (Stage 2).")
    head.append(f"// provenance:        {provenance}")
    head.append(f"// bound (in_domain, ULP, |ref|>=2^-13): {bound}")
    head.append("// below |ref|=2^-13 the ABSOLUTE bound governs instead (BandTrig.h's own")
    head.append("// kBandTrigAbsBound/kBandTrigAbsBoundExp) -- @see BandMathCertTrig.h's")
    head.append("// certifyTrigUnary, which checks BOTH legs.")
    head.append("//")
    head.append("// label encoding: 0=in_domain 1=degraded 2=rejected")
    head.append(f"// corpus: {len(rows)} rows ({in_domain} in_domain / {degraded} degraded / {rejected} rejected)")
    head.append("//")
    head.append("// The row block below is md5-FENCED:")
    head.append("//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'")
    head.append("//   sed -n \"/$BEGIN_RE/,/$END_RE/p\" <this file> | sed '1d;$d' | md5sum")
    head.append("// and compare against the value on the BEGIN line.")

    body = []
    body.append("#pragma once\n")
    body.append("\n".join(head) + "\n")
    body.append("#include <cstdint>\n#include <cstddef>\n")
    body.append(f"namespace aether_tests {{\nnamespace bandmath_golden {{\nnamespace {op} {{\n")
    body.append("inline constexpr int kArity = 1;")
    body.append(f"inline constexpr double kUlpBound = {bound};")
    body.append("struct Row {")
    body.append("    std::uint64_t in[1];  ///< input bit pattern")
    body.append("    std::uint64_t ref;    ///< golden reference, correctly-rounded double bits")
    body.append("    std::uint8_t  label;  ///< 0=in_domain 1=degraded 2=rejected")
    body.append("};\n")

    rows_text_lines = ["inline constexpr Row kRows[] = {"]
    for in0, ref, label in rows:
        rows_text_lines.append(f"    {{ {{ 0x{in0:016X}ULL }}, 0x{ref:016X}ULL, {label} }},")
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


def write_labeled_header_dual(bound: float, rows, provenance: str, out_path: str, mint_cmd: str):
    # rows: (in0, refSin, refCos, label) -- label computed from in0 alone
    # (sin/cos/sincos share ONE domain).
    in_domain = sum(1 for r in rows if r[-1] == 0)
    degraded = sum(1 for r in rows if r[-1] == 1)
    rejected = sum(1 for r in rows if r[-1] == 2)

    head = []
    head.append("// GENERATED by tools/bandmath/gen_corpus_trig.py -- golden reference + domain")
    head.append("// label table for `sincos`, DO NOT EDIT BY HAND. Regenerate with the mint")
    head.append("// command line below and commit the diff in the SAME commit as whatever test")
    head.append("// consuming this header changed.")
    head.append("//")
    head.append(f"// mint command line: {mint_cmd}")
    head.append("// reference:         MPFR @ 256 bits, correctly rounded to double (RNDN), via")
    head.append("//                    tools/ulp_oracle; refSin/refCos from ONE `sincos` MPFR call")
    head.append("//                    (sharing the same reduction, exactly as bd::sincos does).")
    head.append(f"// provenance:        {provenance}")
    head.append(f"// bound (in_domain, ULP, |ref|>=2^-13): {bound}")
    head.append("//")
    head.append("// label encoding: 0=in_domain 1=degraded 2=rejected")
    head.append(f"// corpus: {len(rows)} rows ({in_domain} in_domain / {degraded} degraded / {rejected} rejected)")
    head.append("//")
    head.append("// The row block below is md5-FENCED, same fence convention as every other")
    head.append("// golden_*.h in this directory.")

    body = []
    body.append("#pragma once\n")
    body.append("\n".join(head) + "\n")
    body.append("#include <cstdint>\n#include <cstddef>\n")
    body.append("namespace aether_tests {\nnamespace bandmath_golden {\nnamespace sincos {\n")
    body.append("inline constexpr int kArity = 1;")
    body.append(f"inline constexpr double kUlpBound = {bound};")
    body.append("struct Row {")
    body.append("    std::uint64_t in[1];   ///< input bit pattern")
    body.append("    std::uint64_t ref;     ///< golden sin(x), correctly-rounded double bits")
    body.append("    std::uint64_t refCos;  ///< golden cos(x), correctly-rounded double bits")
    body.append("    std::uint8_t  label;   ///< 0=in_domain 1=degraded 2=rejected")
    body.append("};\n")

    rows_text_lines = ["inline constexpr Row kRows[] = {"]
    for in0, refSin, refCos, label in rows:
        rows_text_lines.append(
            f"    {{ {{ 0x{in0:016X}ULL }}, 0x{refSin:016X}ULL, 0x{refCos:016X}ULL, {label} }},")
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
    body.append("} // namespace sincos\n} // namespace bandmath_golden\n} // namespace aether_tests\n")

    with open(out_path, "w") as f:
        f.write("\n".join(body))
    return in_domain, degraded, rejected


TRIG_BOUND = 0.5  # half-ULP-at-53, |ref| >= 2^-13 leg (BandTrig.h's own contract)
TRIG_PROVENANCE = (
    "aether/banded/BandTrig.h bd::sin/bd::cos/bd::sincos; "
    "bound = 0.5 ULP for |ref|>=2^-13, else the absolute leg (kBandTrigAbsBound=2^-68) -- "
    "BandMathCertTrig.h's certifyTrigUnary checks both."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-bin", required=True, help="path to the built ulp_oracle binary")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "bandmath"))
    ap.add_argument("--spec-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "specs"))
    ap.add_argument("--raw-dir", default="/tmp/p11c3_trig_raw")
    ap.add_argument("--ops", default=None, help="comma-separated subset of {sin,cos,sincos}")
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus_trig.py: refusing to mint under $CI.", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    spec_dir = os.path.abspath(args.spec_dir)
    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    want = set(args.ops.split(",")) if args.ops else {"sin", "cos", "sincos"}

    spec_text = build_trig_spec()
    spec_path = os.path.join(spec_dir, "trig.spec")
    with open(spec_path, "w") as f:
        f.write(spec_text)

    print(f"{'op':<8} {'total':>7} {'in_domain':>10} {'degraded':>9} {'rejected':>9}")

    if "sin" in want or "cos" in want:
        for op in ("sin", "cos"):
            if op not in want:
                continue
            raw_path = os.path.join(raw_dir, f"golden_{op}_mpfr_raw.h")
            cmd = [args.oracle_bin, op, spec_path, raw_path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"ulp_oracle FAILED for {op}: rc={res.returncode}\n{res.stdout}\n{res.stderr}",
                      file=sys.stderr)
                return 1
            raw_rows = parse_unary_raw(raw_path)
            labeled = [(in0, ref, label_trig(double_of(in0))) for in0, ref in raw_rows]
            mint_cmd = (f"python3 tools/bandmath/gen_corpus_trig.py --oracle-bin <ulp_oracle> "
                        f"--ops {op}  (Stage 1: {' '.join(cmd)})")
            out_path = os.path.join(out_dir, f"golden_{op}.h")
            in_d, deg, rej = write_labeled_header_unary(op, TRIG_BOUND, labeled, TRIG_PROVENANCE, out_path, mint_cmd)
            print(f"{op:<8} {len(labeled):>7} {in_d:>10} {deg:>9} {rej:>9}  -> {out_path}")

    if "sincos" in want:
        raw_path = os.path.join(raw_dir, "golden_sincos_mpfr_raw.h")
        cmd = [args.oracle_bin, "sincos", spec_path, raw_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"ulp_oracle FAILED for sincos: rc={res.returncode}\n{res.stdout}\n{res.stderr}", file=sys.stderr)
            return 1
        raw_rows = parse_dual_raw(raw_path)
        labeled = [(in0, refSin, refCos, label_trig(double_of(in0))) for in0, refSin, refCos in raw_rows]
        mint_cmd = (f"python3 tools/bandmath/gen_corpus_trig.py --oracle-bin <ulp_oracle> "
                    f"--ops sincos  (Stage 1: {' '.join(cmd)})")
        out_path = os.path.join(out_dir, "golden_sincos.h")
        in_d, deg, rej = write_labeled_header_dual(TRIG_BOUND, labeled, TRIG_PROVENANCE, out_path, mint_cmd)
        print(f"{'sincos':<8} {len(labeled):>7} {in_d:>10} {deg:>9} {rej:>9}  -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
