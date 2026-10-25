# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the tools/bandmath/gen_corpus*.py generators: bit helpers,
SplitMix64, the Band admission constants/predicates, ulp_oracle raw-header
parsing and the md5-fenced golden-header emitters.

Each generator imports this by sibling path (see its `sys.path` line), so it
runs standalone from any working directory.
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

DEFAULT_SEED = 0xFE7A5EED0C0FFEE5  # matches the seed used across the golden_*.h corpora (see golden_random.h)

OUT_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "bandmath")


def add_out_dir_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)


def refuse_under_ci(script: str) -> bool:
    """True (after saying so) when $CI is set: a gate must never regenerate its own expectation."""
    if os.environ.get("CI"):
        print(f"{script}: refusing to mint under $CI — a gate must never "
              "regenerate its own expectation.", file=sys.stderr)
        return True
    return False


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

def is_nan(x: float) -> bool:
    return x != x


def exponent_of(x: float):
    """Base-2 exponent such that x == mantissa * 2**exponent_of(x) with
    1 <= |mantissa| < 2 (ldexp convention). None for 0/inf/nan (magnitude
    is not the governing property for those)."""
    if x == 0.0 or math.isinf(x) or math.isnan(x):
        return None
    m, e = math.frexp(x)  # x = m * 2**e, 0.5 <= |m| < 1
    return e - 1


# =====================================================================
#  Admission predicates — transcribed verbatim from
#  aether/banded/Band.h's admission-guard section, same constants, same formulas:
#    kBandNominalCeilingExp = 115
#    kBandCarrierFloorExp   = -96
#    kBandAdmissionMargin   = 8
#    bandCeilingAdmits(e, m=8)      = e + m <= 115
#    bandIntermediateAdmits(e, m=8) = e - m >= -96
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


def label_for_exponent(e):
    if e is None:
        return 0  # zero/inf/nan: always in_domain
    if e >= K_BAND_HARD_CEILING_EXP or e <= K_BAND_HARD_FLOOR_EXP:
        return 2  # rejected: hard FP32 escape (overflow / subnormal collapse)
    if spine_admits(e):
        return 0
    return 1  # degraded: representable but past the certified margin


class Op:
    def __init__(self, name, arity, category, ref, bound, provenance):
        self.name = name
        self.arity = arity
        self.category = category
        self.ref = ref
        self.bound = bound
        self.provenance = provenance


def specials_pool():
    return {POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, NAN_PAYLOAD_A, NAN_PAYLOAD_B}


# =====================================================================
#  ulp_oracle raw-header parsing
# =====================================================================
UNARY_ROW_RE = re.compile(
    r"\{\s*\{\s*(0x[0-9A-Fa-f]+)ULL\s*\}\s*,\s*(0x[0-9A-Fa-f]+)ULL\s*,\s*(\d+)\s*\}")
BINARY_ROW_RE = re.compile(
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
    return [(int(a, 16), int(r, 16)) for a, r, _ in UNARY_ROW_RE.findall(text)]


def parse_raw_binary(text: str):
    return [(int(a, 16), int(b, 16), int(r, 16)) for a, b, r, _ in BINARY_ROW_RE.findall(text)]


def parse_unary_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16)) for a, b, _c in UNARY_ROW_RE.findall(text)]


# =====================================================================
#  Header emission — md5-fenced, mirrors `tools/ulp_oracle`'s own convention
#  (README.md "fence check").
# =====================================================================
FENCE_NOTE = [
    "// The row block below is md5-FENCED:",
    "//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'",
    "//   sed -n \"/$BEGIN_RE/,/$END_RE/p\" <this file> | sed '1d;$d' | md5sum",
    "// and compare against the value on the BEGIN line.",
]


def label_counts(rows):
    """(in_domain, degraded, rejected) over the trailing label of each row."""
    return (sum(1 for r in rows if r[-1] == 0),
            sum(1 for r in rows if r[-1] == 1),
            sum(1 for r in rows if r[-1] == 2))


def write_list_header(out_path: str, head, op: str, arity: int, bound, row_struct, row_lines, rows):
    """Header with `head` as its comment block and one `row_lines` entry per row."""
    in_domain, degraded, rejected = label_counts(rows)
    body = []
    body.append("#pragma once\n")
    body.append("\n".join(head) + "\n")
    body.append("#include <cstdint>\n#include <cstddef>\n")
    body.append(f"namespace aether_tests {{\nnamespace bandmath_golden {{\nnamespace {op} {{\n")
    body.append(f"inline constexpr int kArity = {arity};")
    body.append(f"inline constexpr double kUlpBound = {bound};")
    body.append("struct Row {")
    body.extend(row_struct)
    body.append("};\n")

    rows_text = "\n".join(["inline constexpr Row kRows[] = {"] + row_lines + ["};"]) + "\n"
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


def emit_golden_header(op: str, arity: int, bound, rows, out_path: str, preamble: str):
    """Header whose comment block is `preamble` (up to the label-encoding line)
    followed by the corpus summary and the fence note."""
    in_dom, deg, rej = label_counts(rows)

    row_lines = []
    for r in rows:
        in_str = ", ".join(f"0x{v:016X}ULL" for v in r[:arity])
        row_lines.append(f"    {{ {{ {in_str} }}, 0x{r[arity]:016X}ULL, {r[arity + 1]} }},")
    row_block = "\n".join(row_lines)
    md5 = hashlib.md5(("\n" + row_block + "\n").encode()).hexdigest()

    text = f"""#pragma once

{preamble}
// bound (in_domain, ULP): {bound}
//
// label encoding: 0=in_domain 1=degraded 2=rejected
// corpus: {len(rows)} rows ({in_dom} in_domain / {deg} degraded / {rej} rejected)
//
{chr(10).join(FENCE_NOTE)}

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
