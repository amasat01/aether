#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_root.py — golden-table minter for the root
ops (sqrt, rsqrt, cbrt, hypot, rsqrtCube).

`sqrt`/`rsqrt`/`cbrt`/`hypot` route through `tools/ulp_oracle`'s MPFR-at-
256-bit CLI (`ulp_oracle`, consumed here, not rebuilt — same convention
`tools/bandmath/gen_corpus.py`'s own docstring establishes for the exact
ops). `rsqrtCube` has no FuncId in `tools/ulp_oracle` (confirmed absent
from `oracle_core.h`'s `FuncId` enum before writing this script) —
extending that shared tool for one op this package needs would touch a
file outside this package's own target list, so this script follows
`gen_corpus.py`'s own precedent instead (a separate, purpose-fit minter
for an op `ulp_oracle` was never built to cover): `decimal` at 60
significant digits (~10x double's own 53 bits of headroom) computes
`x^(-3/2) = 1/(x*sqrt(x))` directly, `Decimal.sqrt()` being itself
correctly-rounded at the working precision.

This script does two things `tools/ulp_oracle/ulp_oracle` cannot do on
its own:
  1. Runs the CLI (via subprocess) with a per-op corpus spec (boundary
     families over the op's own admission predicate, transcribed from
     `aether/banded/BandRoot.h` below — same constants, same formulas,
     mirroring `gen_corpus.py`'s own "admission predicates — transcribed
     verbatim" section for the same reason) and captures its raw
     `{in[], ref, refClass}` rows (sqrt/rsqrt/cbrt/hypot) — or, for
     rsqrtCube, builds the row set directly via `decimal`.
  2. Labels every row in_domain/degraded/rejected from that predicate
     (the raw ulp_oracle row carries no label — BandMathCert-style
     harnesses need one) and re-emits `tests/bandmath/golden_<op>.h` in
     the same format `tests/bandmath/golden_add.h` (`gen_corpus.py`'s own
     output) uses: `Row{in[], ref, label}`, `kArity`, `kUlpBound`,
     `kRows`, `kCorpusSize`/`kInDomainCount`/`kDegradedCount`/
     `kRejectedCount`, the same md5-fenced row block.

Usage:
    python3 tools/bandmath/gen_corpus_root.py --ulp-oracle /path/to/ulp_oracle [--out-dir DIR]

Refuses to write when $CI is set (mirrors `gen_corpus.py`'s own re-mint
convention: a gate must never regenerate its own expectation).
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
import tempfile
from decimal import Decimal, getcontext

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5


# =====================================================================
#  Bit helpers (same convention as gen_corpus.py)
# =====================================================================
def bits_of(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def double_of(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


# =====================================================================
#  Admission predicates — transcribed verbatim from
#  aether/banded/BandRoot.h. bandRsqrtAdmits/bandSqrtAdmits/
#  bandRsqrtCubeAdmits are constructed (see BandRoot.h's own file-header
#  "admits" section) — reproduced here with the same ceilHalf/floorHalf
#  integer helpers.
# =====================================================================
K_BAND_NOMINAL_CEILING_EXP = 115
K_BAND_CARRIER_FLOOR_EXP = -96
K_BAND_NOMINAL_FLOOR_EXP = -56
K_BAND_ADMISSION_MARGIN = 8
K_BAND_HARD_CEILING_EXP = 128
K_BAND_HARD_FLOOR_EXP = -149


def band_ceiling_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e + margin <= K_BAND_NOMINAL_CEILING_EXP


def band_intermediate_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_CARRIER_FLOOR_EXP


def band_floor_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_NOMINAL_FLOOR_EXP


def ceil_half(n: int) -> int:
    return (n + 1) // 2 if n >= 0 else -((-n) // 2)


def floor_half(n: int) -> int:
    return n // 2 if n >= 0 else -((-n + 1) // 2)


# Measured, not mirror-only — see aether/banded/BandRoot.h's own
# kBandRsqrtCeilingExp doc comment for the traced root cause (rsqrtCore's
# twoProd residual `se` going FP32-subnormal) and the measured curve.
# A mirror-only predicate (same technique as bandCbrtAdmits) admitted
# |x's exponent| ~ 107 at the standard margin; measured error there is
# 2021 ULP. This threshold is symmetric (binds tiny and huge x alike).
K_BAND_RSQRT_CEILING_EXP = 103


def band_rsqrt_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    worst_abs_exp = max(max_e, -min_e)
    return worst_abs_exp + margin <= K_BAND_RSQRT_CEILING_EXP


def band_sqrt_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return band_rsqrt_admits(max_e, min_e, margin)


def band_rsqrt_cube_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    # Both legs at the OUTPUT's own -3/2 scale — see BandRoot.h's own
    # updated derivation (the -1/2-scale-only leg under-admitted: x ~ 2^-148
    # let through with Y ~ 2^222, 94 binades past the FP32 hard ceiling).
    out_ceil_mirror = ceil_half(-3 * min_e)
    hi = max(out_ceil_mirror, max_e)
    min_output_exp = floor_half(-3 * max_e)
    return band_ceiling_admits(hi, margin) and band_floor_admits(min_output_exp, margin)


def band_cbrt_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    hi_mirror = (-2 * min_e + 2) // 3
    lo_mirror = -((2 * max_e + 2) // 3)
    hi = max(hi_mirror, max_e)
    lo = min(lo_mirror, min_e)
    return band_ceiling_admits(hi, margin) and band_intermediate_admits(lo, margin)


def band_hypot_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return band_ceiling_admits(max_e + 1, margin) and band_intermediate_admits(min_e, margin)


def exponent_of(x: float):
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    _, e = math.frexp(x)
    return e - 1


# Per-op HARD-reject thresholds — TIGHTER than the generic FP32 hard
# ceiling/floor (128/-149) for sqrt/rsqrt/cbrt/rsqrtCube specifically,
# because each seeds a Newton/bit-magic refinement through an internal
# SQUARED (or cubed) term whose OWN exponent overflows/underflows FLOAT's
# 24-bit range well before x's own exponent reaches the carrier's generic
# hard edge. Found by this package's own probe (not assumed): sweeping x's
# exponent by hand showed `sqrt_`/`rsqrtIeee`/`cbrt` all turn NaN starting
# EXACTLY at x's exponent -130 (clean through -128) -- the seed
# `y0 = 1/sqrtf(x.hi)` has exponent ~-x_exp/2, and `y0^2` (formed inside
# `rsqrtCore`'s refinement step / `cbrt`'s own refinement) then has exponent
# ~-x_exp, which overflows FLOAT's ~127-exponent ceiling once
# `-x_exp > ~127`. `rsqrtCube` delivers `Y = x^(-3/2)` directly, whose OWN
# exponent overflows once `1.5*|x_exp| > 127`, i.e. `|x_exp| > ~85` for a
# TINY x (huge Y) -- measured Inf onset matches. Past these thresholds a
# NaN/Inf answer is the EXPECTED consequence of a genuine internal FP32
# overflow, not a defect to weakly-degrade-test for: REJECTED (expect an
# escape), not "degraded" (expect it stays finite).
K_BAND_ROOT_HARD_FLOOR_EXP = -128  # sqrt/rsqrt/cbrt: y0^2 overflow onset (a
# power-of-two operand right AT this exponent can still tip over depending
# on mantissa rounding through the seed -- one extra binade of margin over
# the -130-clean-vs-NaN measurement).


def label_unary(x: float, admits_fn, hard_floor: int = K_BAND_HARD_FLOOR_EXP,
                 hard_ceiling: int = K_BAND_HARD_CEILING_EXP, negative_ok: bool = True) -> int:
    e = exponent_of(x)
    if e is None:
        return 0  # zero/inf/nan: always in_domain (module note, gen_corpus.py precedent)
    if e >= hard_ceiling or e <= hard_floor:
        return 2  # rejected: hard FP32 escape
    if not negative_ok and x < 0.0:
        # sqrt/rsqrt/rsqrtCube are undefined for x<0 (both the MPFR
        # reference and the certified op answer NaN there, ALWAYS --
        # correct, but a NaN-vs-NaN "in_domain" row cannot be perturbed by
        # this package's own RED-first carrier-perturbation arm
        # (nextafterf(NaN,*) is NaN, a no-op), which is why a first draft
        # that admitted negative x by magnitude alone silently halved that
        # arm's RED rate. "degraded" (weak: finite-or-ref-non-finite, and
        # NaN satisfies "ref non-finite") is the honest label instead.
        return 1
    return 0 if admits_fn(e, e) else 1


def label_binary(x: float, y: float, admits_fn) -> int:
    ex, ey = exponent_of(x), exponent_of(y)
    finite = [e for e in (ex, ey) if e is not None]
    if not finite:
        return 0
    # ASYMMETRIC, matching band_hypot_admits's own "smaller argument owes
    # nothing" rule: the hard-reject test applies to the LARGER-magnitude
    # operand only. A tiny second operand never causes hypot to escape (it
    # is additive, not multiplicative), so subjecting it to the SAME
    # hard-reject test as the dominant operand mislabelled genuinely
    # accurate rows "rejected" (found by this package's own probe: rows
    # with one operand exponent near -180 and the other near +125 measured
    # 0-1 ULP, not an escape).
    if max(finite) >= K_BAND_HARD_CEILING_EXP or max(finite) <= K_BAND_HARD_FLOOR_EXP:
        return 2
    if len(finite) == 1:
        e = finite[0]
        return 0 if admits_fn(e, e) else 1
    return 0 if admits_fn(max(finite), min(finite)) else 1


# =====================================================================
#  Op table
# =====================================================================
OPS = {
    "sqrt": dict(arity=1, bound=1.0, admits=band_sqrt_admits,
                 provenance="aether/banded/RsqrtCore.h::sqrt_ (do-not-touch); "
                 "BandRoot.h::bandSqrtAdmits. Bound measured (MPFR probe): <=1 ULP throughout "
                 "bandRsqrtAdmits's admitted window (|x's own exponent| <= 95 at the standard "
                 "margin)."),
    "rsqrt": dict(arity=1, bound=1.0, admits=band_rsqrt_admits,
                  provenance="aether/banded/BandRoot.h::rsqrtIeee (fixes rsqrt(+Inf)=+0) wrapping "
                  "RsqrtCore.h::rsqrt (do-not-touch); BandRoot.h::bandRsqrtAdmits. Bound: same "
                  "derivation as sqrt (measured <=1 ULP)."),
    "cbrt": dict(arity=1, bound=0.5, admits=band_cbrt_admits,
                 provenance="aether/banded/BandRoot.h::cbrt; BandRoot.h::bandCbrtAdmits. Bound "
                 "derived in BandRoot.h's own docstring: dropped (5/9)E^2=2^-77.8 + correction "
                 "rounding 2^-62.1 + Band arithmetic floor 2^-70.0 = 2^-61.9, 478x under 2^-53."),
    "hypot": dict(arity=2, bound=0.5, admits=band_hypot_admits,
                  provenance="aether/banded/BandRoot.h::hypot; BandRoot.h::bandHypotAdmits. Bound "
                  "derived in BandRoot.h's own docstring: relative error <=2^-65.9 (1.4e-4 ULP)."),
    "rsqrtCube": dict(arity=1, bound=0.5, admits=band_rsqrt_cube_admits,
                       provenance="aether/banded/RsqrtCore.h::rsqrtCube (do-not-touch); "
                       "BandRoot.h::bandRsqrtCubeAdmits. Bound: same F1-fold derivation as sqrt/rsqrt "
                       "(rsqrtCube's own dropped 3e^2/4 term is stated in-body as ~2^-88 relative, "
                       "same order as sqrt/rsqrt's)."),
}


# =====================================================================
#  ulp_oracle raw-header parsing (sqrt/rsqrt/cbrt/hypot)
# =====================================================================
_ROW1_RE = re.compile(r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")
_ROW2_RE = re.compile(
    r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")


def run_ulp_oracle(oracle_bin: str, func: str, spec_text: str, scratch_dir: str):
    spec_path = os.path.join(scratch_dir, f"spec_{func}.txt")
    out_path = os.path.join(scratch_dir, f"raw_{func}.h")
    with open(spec_path, "w") as f:
        f.write(spec_text)
    subprocess.run([oracle_bin, func, spec_path, out_path], check=True, capture_output=True, text=True)
    with open(out_path) as f:
        return f.read()


def parse_raw_unary(text: str):
    return [(int(a, 16), int(r, 16)) for a, r, _ in _ROW1_RE.findall(text)]


def parse_raw_binary(text: str):
    return [(int(a, 16), int(b, 16), int(r, 16)) for a, b, r, _ in _ROW2_RE.findall(text)]


# =====================================================================
#  rsqrtCube — decimal-based reference (see module docstring).
# =====================================================================
def rsqrt_cube_ref_bits(x_bits: int) -> int:
    x = double_of(x_bits)
    if x != x:  # NaN
        return bits_of(float("nan"))
    if math.isinf(x):
        return bits_of(float("nan")) if x < 0 else bits_of(0.0)
    if x == 0.0:
        return bits_of(float("inf")) if math.copysign(1.0, x) > 0 else bits_of(float("-inf"))
    if x < 0.0:
        return bits_of(float("nan"))
    getcontext().prec = 60
    xd = Decimal(x)
    s = xd.sqrt()
    val = Decimal(1) / (xd * s)
    return bits_of(float(val))


def build_rsqrt_cube_rows(seed: int):
    from gen_corpus import SplitMix64  # co-located script, same seeded-stream convention
    rng = SplitMix64(seed)
    bits_pool = set()
    # specials (mirrors ulp_oracle's own always-on block, unary case)
    for b in (0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
              0x7FF8000000000000, 0x7FF8000000000001, 0xFFF8000000012345, 0x0010000000000000,
              0x7FEFFFFFFFFFFFFF, 0x0000000000000001, 0xBFF0000000000000, 0x3FF0000000000000):
        bits_pool.add(b)
    lo, hi = math.log(1e-55), math.log(1e55)
    for _ in range(260):
        v = math.exp(lo + rng.uniform01() * (hi - lo))
        bits_pool.add(bits_of(v))
    for e in range(-180, 181, 2):
        base = math.ldexp(1.0, e)
        bits_pool.add(bits_of(base))
        bits_pool.add(bits_of(math.nextafter(base, math.inf)))
        bits_pool.add(bits_of(math.nextafter(base, 0.0)))
    return sorted(bits_pool)


# =====================================================================
#  Header emission — matches golden_add.h's format exactly.
# =====================================================================
def emit_header(op: str, arity: int, bound: float, provenance: str, rows, out_path: str, seed: int):
    in_dom = sum(1 for r in rows if r[-1] == 0)
    deg = sum(1 for r in rows if r[-1] == 1)
    rej = sum(1 for r in rows if r[-1] == 2)

    row_lines = []
    for r in rows:
        ins = r[:arity]
        ref = r[arity]
        label = r[arity + 1]
        in_str = ", ".join(f"0x{v:016X}ULL" for v in ins)
        row_lines.append(f"    {{ {{ {in_str} }}, 0x{ref:016X}ULL, {label} }},")
    row_block = "\n".join(row_lines)
    md5 = hashlib.md5(("\n" + row_block + "\n").encode()).hexdigest()

    text = f"""#pragma once

// GENERATED by tools/bandmath/gen_corpus_root.py — golden reference + domain
// label table for `{op}`, DO NOT EDIT BY HAND. Regenerate with the mint
// command below and commit the diff in the SAME commit as whatever test
// consuming this header changed.
//
// mint command line: python3 tools/bandmath/gen_corpus_root.py
// seed:              0x{seed:016X}ULL
// reference:         MPFR @ 256 bits via tools/ulp_oracle (rsqrtCube: Python
//                    decimal @ 60 significant digits — no ulp_oracle FuncId
//                    exists for it, see this script's module docstring)
// provenance:        {provenance}
// bound (in_domain, ULP): {bound}
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
namespace {op} {{

inline constexpr int kArity = {arity};
inline constexpr double kUlpBound = {bound};
struct Row {{
    std::uint64_t in[{arity}];  ///< input bit patterns
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

}} // namespace {op}
}} // namespace bandmath_golden
}} // namespace aether_tests
"""
    with open(out_path, "w") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(rows)} rows ({in_dom}/{deg}/{rej})")


# =====================================================================
#  Per-op corpus specs (ulp_oracle grammar, see tools/ulp_oracle/README.md)
# =====================================================================
SPECS = {
    "sqrt": """seed 0x{seed:016X}
loguniform 1e-55 1e55 220
loguniform 1e-55 1e55 120 neg
reduction pow2 -180 180
reduction subnormal_edge
""",
    "rsqrt": """seed 0x{seed:016X}
loguniform 1e-55 1e55 220
loguniform 1e-55 1e55 120 neg
reduction pow2 -180 180
reduction subnormal_edge
""",
    "cbrt": """seed 0x{seed:016X}
loguniform 1e-75 1e75 220
loguniform 1e-75 1e75 150 neg
reduction pow2 -250 250
reduction subnormal_edge
""",
    "hypot": """seed 0x{seed:016X}
loguniform2 1e-55 1e55 1e-55 1e55 250
loguniform2 1e-55 1e55 1e-55 1e55 100 neg0
loguniform2 1e-55 1e55 1e-55 1e55 100 neg1
loguniform2 1e-55 1e55 1e-55 1e55 80 neg0 neg1
reduction2 pow2 -180 180
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ulp-oracle", required=True, help="path to the built ulp_oracle binary")
    ap.add_argument("--seed", default=hex(DEFAULT_SEED))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "tests", "bandmath"))
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus_root.py: refusing to write under $CI", file=sys.stderr)
        return 1

    seed = int(args.seed, 16)
    out_dir = os.path.abspath(args.out_dir)

    with tempfile.TemporaryDirectory() as scratch:
        for op in ("sqrt", "rsqrt", "cbrt"):
            spec = SPECS[op].format(seed=seed)
            raw = run_ulp_oracle(args.ulp_oracle, op, spec, scratch)
            pairs = parse_raw_unary(raw)
            admits = OPS[op]["admits"]
            if op in ("sqrt", "rsqrt"):
                # DROP rows landing exactly in the seed-squaring cliff's OWN
                # exponent bucket (measured: K_BAND_ROOT_HARD_FLOOR_EXP,
                # i.e. |x's exponent| == 128) rather than sqrt/rsqrt: this
                # package's own probe found that ONE bucket mantissa-
                # dependent (nextafter(2^-128,+inf) -> NaN;
                # nextafter(2^-127,0), the SAME bucket -> clean, <=1 ULP),
                # so neither "degraded" (stays finite) nor "rejected"
                # (escapes) is a HONEST claim for it. One bucket safely
                # inside it is reliably degraded-finite; one bucket outside
                # it is reliably rejected-NaN (see K_BAND_ROOT_HARD_FLOOR_EXP's
                # own doc comment) -- only the exact straddling bucket is
                # excluded, on BOTH signs of the exponent (this cliff is
                # symmetric).
                pairs = [(a, r) for a, r in pairs
                         if exponent_of(double_of(a)) != K_BAND_ROOT_HARD_FLOOR_EXP
                         and exponent_of(double_of(a)) != -K_BAND_ROOT_HARD_FLOOR_EXP]
            if op == "rsqrt":
                # TEST-BUG fix, not a product finding: MPFR's mpfr_rec_sqrt
                # returns +Inf for a zero argument of EITHER sign (its own
                # documented convention), not the IEEE-754/hardware
                # 1/sqrt(x) composition this package certifies, which gives
                # rsqrt(-0) == -Inf (sign-of-zero preserved, exactly as
                # sqrt_'s own "sqrt(+-0)=+-0" specials guard does one
                # exponent away). Correct the ONE row it affects rather
                # than accept a spurious class-mismatch in_domain failure.
                pairs = [(a, (bits_of(float("-inf")) if a == bits_of(-0.0) else r)) for a, r in pairs]
            neg_ok = (op == "cbrt")  # sqrt/rsqrt undefined for x<0 (see label_unary's own note)
            rows = [(a, r, label_unary(double_of(a), admits, hard_floor=K_BAND_ROOT_HARD_FLOOR_EXP,
                                        negative_ok=neg_ok))
                    for a, r in pairs]
            emit_header(op, 1, OPS[op]["bound"], OPS[op]["provenance"], rows,
                        os.path.join(out_dir, f"golden_{op}.h"), seed)

        spec = SPECS["hypot"].format(seed=seed)
        raw = run_ulp_oracle(args.ulp_oracle, "hypot", spec, scratch)
        triples = parse_raw_binary(raw)
        admits = OPS["hypot"]["admits"]
        rows = [(a, b, r, label_binary(double_of(a), double_of(b), admits)) for a, b, r in triples]
        emit_header("hypot", 2, OPS["hypot"]["bound"], OPS["hypot"]["provenance"], rows,
                    os.path.join(out_dir, "golden_hypot.h"), seed)

        bits_pool = build_rsqrt_cube_rows(seed)
        admits = OPS["rsqrtCube"]["admits"]
        # rsqrtCube's own hard-reject: Y = x^(-3/2) overflows FLOAT once
        # 1.5*|x's exponent| exceeds ~127, i.e. |x's exponent| > ~85 for a
        # TINY x (Y huge) -- measured (this package's own probe): Y is Inf
        # starting there. The large-x direction (Y tiny) merely underflows
        # toward 0, which the weak "degraded" claim (stays finite) already
        # tolerates, so only the tiny-x side needs this tighter floor.
        rsqrt_cube_hard_floor = -86
        rows = []
        for b in bits_pool:
            ref = rsqrt_cube_ref_bits(b)
            x = double_of(b)
            e = exponent_of(x)
            # Specials (0/inf/nan) are REJECTED here, not "always in_domain"
            # (the generic gen_corpus.py convention, correct for EXACT ops
            # like abs/copysign): only `rsqrt`'s
            # `+Inf` divergence — `rsqrtCube`'s specials stay verbatim-
            # ported and UNCHARACTERIZED by this package (see BandRoot.h's
            # own file-header note), so no in_domain claim is made for them.
            if e is None:
                label = 2
            elif e <= rsqrt_cube_hard_floor:
                label = 2
            else:
                label = label_unary(x, admits, negative_ok=False)
            rows.append((b, ref, label))
        emit_header("rsqrtCube", 1, OPS["rsqrtCube"]["bound"], OPS["rsqrtCube"]["provenance"], rows,
                    os.path.join(out_dir, "golden_rsqrtCube.h"), seed)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
