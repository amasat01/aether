#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_invtrig.py -- golden-table minter for the
inverse trig ops (atan, asin, acos, atan2).

`atan`/`atan2`/`asin`/`acos` all route through `tools/ulp_oracle`'s MPFR-at-
256-bit CLI (`ulp_oracle`, consumed here, not rebuilt -- same convention
`gen_corpus.py`/`gen_corpus_root.py` use): the tool's own `FuncId` enum
already carries ATAN/ASIN/ACOS (unary) and ATAN2 (binary, native 2-arg
support -- `oracle_core.h`'s own docstring: "atan2's inputs are (y,x),
libm order"). atan2 therefore goes through the oracle's native 2-arg
FuncId (`uniform2`/`loguniform2`/`points2`/`reduction2` directives), not
the `decimal`-module workaround `gen_corpus_root.py` uses for `rsqrtCube`
(an op with no FuncId at all; atan2 needs no such workaround).

This script does two things `tools/ulp_oracle/ulp_oracle` cannot do on its
own:
  1. Runs the CLI (via subprocess) with a per-op corpus spec (boundary
     families over the op's own admission predicate, transcribed from
     `aether/banded/BandInvTrig.h` below -- same constants, same formulas,
     mirroring `gen_corpus_root.py`'s own "admission predicates --
     transcribed verbatim" section) and captures its raw `{in[], ref,
     refClass}` rows.
  2. Labels every row in_domain/degraded/rejected and re-emits
     `tests/bandmath/golden_<op>.h` in the same format `golden_add.h`/
     `golden_hypot.h` use: `Row{in[], ref, label}`, `kArity`, `kUlpBound`,
     `kRows`, `kCorpusSize`/`kInDomainCount`/`kDegradedCount`/
     `kRejectedCount`, the same md5-fenced row block.

`asin`/`acos` use a simpler labeling scheme than every other op in this
directory: `bandAsinAdmits` (`BandInvTrig.h`) is a one-leg, no-margin
predicate -- the domain `[-1,1]` is a mathematical limit, not a carrier
one, so there is no "degraded" zone at all (below the domain the op is
unconditionally exact, `asin(x)==x`; above it there is no answer to be
approximately right about). Rows are therefore labeled purely by value --
finite and `|x|<=1` is `in_domain`, everything else (`|x|>1`, `+-inf`,
`NaN`) is `rejected` -- not by the exponent-bucket `admits_fn(e,e)` scheme
`label_unary` below uses for `atan`/`atan2`.

Usage:
    python3 tools/bandmath/gen_corpus_invtrig.py --ulp-oracle /path/to/ulp_oracle [--out-dir DIR]

Refuses to write when $CI is set (mirrors `gen_corpus.py`'s/
`gen_corpus_root.py`'s own re-mint convention: a gate must never
regenerate its own expectation).
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

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5


# =====================================================================
#  Bit helpers (same convention as gen_corpus.py/gen_corpus_root.py)
# =====================================================================
def bits_of(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def double_of(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


def nextafter_bits(x: float, toward: float) -> int:
    return bits_of(math.nextafter(x, toward))


# =====================================================================
#  Admission predicates -- transcribed verbatim from
#  aether/banded/BandInvTrig.h (same constants, same formulas).
# =====================================================================
K_BAND_NOMINAL_CEILING_EXP = 115
K_BAND_CARRIER_FLOOR_EXP = -96
K_BAND_ADMISSION_MARGIN = 8
K_BAND_HARD_CEILING_EXP = 128
K_BAND_HARD_FLOOR_EXP = -149

K_BAND_INVTRIG_MAX_ARG_EXP = 0  # kBandInvTrigMaxArgExp

# atan2's own tighter hard-reject ceiling (measured, narrower than the
# generic K_BAND_HARD_CEILING_EXP=128): den = a*(1+T) can amplify the
# larger operand by up to ~1.84x (T up to 0.839), almost one full binade.
# Sweeping every sector's ratio (exponent 90..127) found the smallest
# exponent producing a real escape (NaN, or a silently wrong finite
# answer) is 117 -- reusing the same kBandNominalCeilingExp=115 every
# other op's own declared ceiling uses leaves 2 binades of measured
# slack, rather than inventing a new number.
K_BAND_ATAN2_HARD_CEILING_EXP = K_BAND_NOMINAL_CEILING_EXP


def band_ceiling_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e + margin <= K_BAND_NOMINAL_CEILING_EXP


def band_intermediate_admits(e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return e - margin >= K_BAND_CARRIER_FLOOR_EXP


def band_divisor_admits(max_e_divisor: int, min_e_divisor: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    return band_intermediate_admits(-max_e_divisor, margin) and band_ceiling_admits(-min_e_divisor, margin)


def band_atan2_admits(max_e: int, min_e: int, margin: int = K_BAND_ADMISSION_MARGIN) -> bool:
    # Third and fourth legs (derived from measurement, documented in
    # aether/banded/BandInvTrig.h's own bandAtan2Admits comment):
    # third, num/den are intermediates tracking the larger operand's own
    # magnitude; both operands tiny together underflows their lo/tail
    # limbs through the FP32 hard subnormal floor before the division
    # runs. Fourth (supersedes BandInvTrig.h's own "no floor leg" claim):
    # the reduced argument t's own worst-case exponent is min_e - max_e;
    # once t.hi itself drops into FP32 subnormal territory div()'s
    # correction limbs underflow to exactly zero. Measured clean through
    # t exponent -92, drifting from -98; -88 (kBandCarrierFloorExp+margin)
    # is 8 binades of slack.
    return (band_ceiling_admits(max_e + 1, margin) and band_divisor_admits(max_e + 1, min_e, margin)
            and band_intermediate_admits(max_e, margin)
            and band_intermediate_admits(min_e - max_e, margin))


def band_asin_admits(max_e: int) -> bool:
    return max_e <= K_BAND_INVTRIG_MAX_ARG_EXP


def exponent_of(x: float):
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    _, e = math.frexp(x)
    return e - 1


# =====================================================================
#  Op table
# =====================================================================
OPS = {
    "atan": dict(
        arity=1, bound=1.0, oracle_name="atan",
        provenance="aether/banded/BandInvTrig.h::atan (computed as atan2(x,1) "
        "internally); BandInvTrig.h::bandAtan2Admits stated on (max(e,0), min(e,0)). "
        "Bound: kBandInvTrigRelBoundExp = -53 (0.5 ULP, one leg, no absolute leg, "
        "oracle-derived 2^-60.2 total against the contract).",
    ),
    "asin": dict(
        arity=1, bound=1.0, oracle_name="asin",
        provenance="aether/banded/BandInvTrig.h::asin; BandInvTrig.h::bandAsinAdmits "
        "-- one leg, no margin, a hard mathematical domain [-1,1], not a carrier one. "
        "Bound: kBandInvTrigRelBoundExp = -53 (0.5 ULP, one leg).",
    ),
    "acos": dict(
        arity=1, bound=1.0, oracle_name="acos",
        provenance="aether/banded/BandInvTrig.h::acos; BandInvTrig.h::bandAsinAdmits, "
        "same hard domain as asin (both route through the same asinReducedRaw). "
        "Bound: kBandInvTrigRelBoundExp = -53 (0.5 ULP, one leg).",
    ),
    "atan2": dict(
        arity=2, bound=1.0, oracle_name="atan2",
        provenance="aether/banded/BandInvTrig.h::atan2 (built on AtanSector.h's "
        "matched-pair table); BandInvTrig.h::bandAtan2Admits (ceiling leg at maxE+1, "
        "divisor leg via bandDivisorAdmits). Bound: kBandInvTrigRelBoundExp = -53 "
        "(0.5 ULP, one leg, everywhere -- oracle-derived 2^-60.2 total). Inputs are "
        "(y,x), libm order, matching tools/ulp_oracle's own ATAN2 convention.",
    ),
}

# AtanSector's own four boundary tangents (aether/banded/AtanSector.h,
# do-not-touch) -- transcribed here (read, not re-derived) so the
# atan/atan2 corpus can enumerate sector transitions explicitly.
K_BAND_ATAN_SEC_B = [8.7488666177e-02, 2.6794919372e-01, 4.6630766988e-01, 7.0020753145e-01]

# The ambiguous zone near and past the (1+T)-amplified ceiling (see
# K_BAND_ATAN2_HARD_CEILING_EXP's own comment above): unlike every other
# hard-reject boundary in this file, the behaviour here is genuinely
# undefined, not uniform. Measurement found three distinct outcomes for
# operand exponents at and past 115: some rows measure perfectly clean
# (0 ULP), some measure NaN, and at least one measures a wrong finite
# answer (silently, no escape at all) -- and this does not resolve past
# the generic FP32 hard ceiling (128) either: bandFromIEEE (`Band.h`,
# do-not-touch) itself starts returning chaotic results once a double's
# magnitude exceeds what the carrier's own ingest logic was built for
# ("kBandHardCeilingExp=128 -- where the carrier actually dies", that
# file's own docstring) -- e.g. `x=3.40e38` (exponent 128) ingests to a
# value whose `atan` reads a correct pi/2, but `x=6.8e38` (exponent 129)
# ingests to something whose `atan` reads exactly `-0`, and `x=1.36e39`
# (exponent 130) reads `-inf`. The do-not-touch codec's own contract is
# silent past its declared window, so anything downstream is equally
# silent there. Neither "degraded" (needs finite) nor "rejected" (needs
# an escape) is an honest uniform claim, same class of problem
# gen_corpus_root.py's own K_BAND_ROOT_HARD_FLOOR_EXP straddling-bucket
# carve-out solves for sqrt/rsqrt: exclude the whole zone (return None ->
# the row is dropped, no claim made either way), unbounded above.
K_BAND_ATAN2_AMBIGUOUS_LO = K_BAND_ATAN2_HARD_CEILING_EXP  # 115


def _in_atan2_ambiguous_zone(e: int) -> bool:
    return e >= K_BAND_ATAN2_AMBIGUOUS_LO


def label_unary_atan(x: float):
    """atan's own exponent-bucket label, via atan(x) = atan2(x, 1). Returns
    None to exclude the row (see K_BAND_ATAN2_AMBIGUOUS_LO's own comment)."""
    e = exponent_of(x)
    if e is None:
        return 0  # 0/inf/nan: atan2's own specials guard answers exactly, always in_domain
    if _in_atan2_ambiguous_zone(e):
        return None  # see K_BAND_ATAN2_AMBIGUOUS_LO's own comment -- undefined past here, not just past 128
    if e <= K_BAND_HARD_FLOOR_EXP:
        # bandFromIEEE cannot represent a magnitude this tiny at all (rounds
        # to exact 0 on ingest) -- but atan(x)~=x near zero means the
        # rounded-to-zero answer is always within 1 double-ULP of the true
        # (equally tiny) reference, so this is never an observable "escape"
        # (certifyRootUnary's rejected branch wants non-finite or
        # out-of-bound; round-to-zero is neither). Degraded (weak: stays
        # finite), not rejected -- same reasoning gen_corpus_root.py's own
        # K_BAND_ROOT_HARD_FLOOR_EXP carve-out gives for a mantissa-
        # dependent bucket where neither strong label is honest.
        return 1
    max_e, min_e = max(e, 0), min(e, 0)
    return 0 if band_atan2_admits(max_e, min_e) else 1


def label_binary_atan2(y: float, x: float):
    """Returns None to exclude the row (see K_BAND_ATAN2_AMBIGUOUS_LO's own
    comment: the (1+T)-amplified ceiling zone is ratio-dependent, not a
    uniform escape)."""
    ey, ex = exponent_of(y), exponent_of(x)
    finite = [e for e in (ey, ex) if e is not None]
    if not finite:
        return 0
    if _in_atan2_ambiguous_zone(max(finite)):
        return None  # covers (and supersedes) the generic K_BAND_HARD_CEILING_EXP too -- unbounded above
    if max(finite) <= K_BAND_HARD_FLOOR_EXP:
        return 2
    if len(finite) == 1:
        e = finite[0]
        return 0 if band_atan2_admits(max(e, 0), min(e, 0)) else 1
    return 0 if band_atan2_admits(max(finite), min(finite)) else 1


def label_asin(x: float) -> int:
    """Hard 2-way domain: no degraded zone (see this file's own docstring)
    plus the carrier's own ingest floor (found by measurement, not stated
    in bandAsinAdmits, whose "no floor leg" claim is about the
    mathematical domain, not the codec): `bandFromIEEE` cannot represent a
    magnitude below FP32's hard floor (~2^-149) at all -- the hi limb
    truncates it to exact 0 on ingest, so `asin`'s own algorithm never even
    sees the true tiny value. A row down there is rejected for the same
    reason gen_corpus_root.py's own K_BAND_HARD_FLOOR_EXP gate exists: the
    escape is in the round trip, not in this op's own arithmetic. asin(x)~=x
    near zero, so the rounded-to-zero answer and the true (tiny nonzero)
    double reference do genuinely diverge in absolute terms -- but measured
    (at DBL_TRUE_MIN, the most extreme case), that divergence is exactly 1
    double-ULP, which is not an observable "escape" under
    certifyRootUnary's strict `d > bound` criterion (a rejected row that
    happens to read exactly `d == bound` "silently passes" and fails the
    assertion) -- same coincidental-non-escape shape label_unary_atan's own
    docstring records at its own hard floor. Degraded (weak: stays
    finite), not rejected."""
    if math.isnan(x) or math.isinf(x):
        return 2
    e = exponent_of(x)
    if e is not None and e <= K_BAND_HARD_FLOOR_EXP:
        return 1
    return 0 if abs(x) <= 1.0 else 2


def label_acos(x: float) -> int:
    """Same hard 2-way domain as asin -- but without asin's own carrier-floor
    carve-out (found by measurement): acos(x) = pi/2 - asin(x), and pi/2 is
    eight orders of magnitude (in exponent) away from any x tiny enough to
    hit the carrier floor -- `pi/2 - x` rounds to exactly pi/2 in double
    precision for any |x| below ~2^-53ish, let alone the FP32 hard floor
    (~2^-149). So a hard-floor-underflowed x (Band rounds it to exact 0)
    and the true double x give the identical correctly-rounded double
    answer (pi/2) either way -- genuinely in_domain, not a coincidence the
    way asin's own 1-ULP-away near-zero case is (see label_asin's own
    docstring)."""
    if math.isnan(x) or math.isinf(x):
        return 2
    return 0 if abs(x) <= 1.0 else 2


# =====================================================================
#  ulp_oracle raw-header parsing (same regex convention as gen_corpus_root.py)
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
#  Header emission — matches golden_add.h's/golden_hypot.h's format exactly.
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

// GENERATED by tools/bandmath/gen_corpus_invtrig.py — golden reference +
// domain label table for `{op}`, DO NOT EDIT BY HAND. Regenerate with the
// mint command below and commit the diff in the SAME commit as whatever
// test consuming this header changed.
//
// mint command line: python3 tools/bandmath/gen_corpus_invtrig.py
// seed:              0x{seed:016X}ULL
// reference:         MPFR @ 256 bits via tools/ulp_oracle (native FuncId, ATAN2 via the
//                    oracle's own 2-arg support -- @see this script's module docstring)
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
def _sector_boundary_points_unary():
    """Sector-boundary tangents (AtanSector.h) as `atan(x)` unary points, so
    the corpus exercises the sector transitions directly (atan(x)=atan2(x,1)
    puts a=1, b=x, so a boundary crossing on b/a is a crossing on x itself)."""
    pts = []
    for b in K_BAND_ATAN_SEC_B:
        for v in (b, math.nextafter(b, 0.0), math.nextafter(b, 1.0), -b):
            pts.append(f"{v!r}")
    return " ".join(pts)


SPECS = {
    "atan": f"""seed 0x{{seed:016X}}
loguniform 1e-40 1e40 220
loguniform 1e-40 1e40 120 neg
uniform -3.0 3.0 120
uniform 0.71 1.42 200
uniform -1.42 -0.71 200
reduction pow2 -140 140
points {_sector_boundary_points_unary()}
""",
    "asin": """seed 0x{seed:016X}
uniform -1.0 1.0 300
uniform -1.5 1.5 80
points 1.0 -1.0 0.5 -0.5 1.0000000001 -1.0000000001 2.0 -2.0 1e10 -1e10
points 1e-30 -1e-30 1e-160 -1e-160 1e-300 -1e-300
""",
    "acos": """seed 0x{seed:016X}
uniform -1.0 1.0 300
uniform -1.5 1.5 80
points 1.0 -1.0 0.5 -0.5 1.0000000001 -1.0000000001 2.0 -2.0 1e10 -1e10
points 1e-30 -1e-30 1e-160 -1e-160 1e-300 -1e-300
""",
    "atan2": """seed 0x{seed:016X}
uniform2 -10.0 10.0 -10.0 10.0 260
loguniform2 1e-40 1e40 1e-40 1e40 120
loguniform2 1e-40 1e40 1e-40 1e40 60 neg0
loguniform2 1e-40 1e40 1e-40 1e40 60 neg1
loguniform2 1e-40 1e40 1e-40 1e40 60 neg0 neg1
reduction2 pow2 -115 115
points2 1.0 0.087488666177 1.0 0.267949193720 1.0 0.466307669880 1.0 0.700207531450
points2 -1.0 0.087488666177 1.0 -0.267949193720 -1.0 -0.466307669880
points2 1.0 1e-110 1.0 -1e-110 1e-110 1.0 -1e-110 1.0
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ulp-oracle", required=True, help="path to the built ulp_oracle binary")
    ap.add_argument("--seed", default=hex(DEFAULT_SEED))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "tests", "bandmath"))
    args = ap.parse_args()

    if os.environ.get("CI"):
        print("gen_corpus_invtrig.py: refusing to write under $CI", file=sys.stderr)
        return 1

    seed = int(args.seed, 16)
    out_dir = os.path.abspath(args.out_dir)

    with tempfile.TemporaryDirectory() as scratch:
        for op in ("atan", "asin", "acos"):
            spec = SPECS[op].format(seed=seed)
            raw = run_ulp_oracle(args.ulp_oracle, OPS[op]["oracle_name"], spec, scratch)
            pairs = parse_raw_unary(raw)
            if op == "atan":
                rows = [(a, r, label_unary_atan(double_of(a))) for a, r in pairs]
            elif op == "asin":
                rows = [(a, r, label_asin(double_of(a))) for a, r in pairs]
            else:
                rows = [(a, r, label_acos(double_of(a))) for a, r in pairs]
            rows = [row for row in rows if row[-1] is not None]  # drop excluded (ambiguous-zone) rows
            emit_header(op, 1, OPS[op]["bound"], OPS[op]["provenance"], rows,
                        os.path.join(out_dir, f"golden_{op}.h"), seed)

        spec = SPECS["atan2"].format(seed=seed)
        raw = run_ulp_oracle(args.ulp_oracle, "atan2", spec, scratch)
        triples = parse_raw_binary(raw)
        rows = [(a, b, r, label_binary_atan2(double_of(a), double_of(b))) for a, b, r in triples]
        rows = [row for row in rows if row[-1] is not None]  # drop excluded (ambiguous-zone) rows
        emit_header("atan2", 2, OPS["atan2"]["bound"], OPS["atan2"]["provenance"], rows,
                    os.path.join(out_dir, "golden_atan2.h"), seed)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
