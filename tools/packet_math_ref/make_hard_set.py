#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Builds the committed hard-case set tests/packetmath/faithful_hard.bin from
the full golden corpus (gen_golden.py) and the worst-row dump of
aether_faithful_check (--worst K FILE).

Per function: every special/extra row (the corpus tail after the --n random
rows; strided down to --tail rows when longer), --stride evenly spaced rows of
the random part, and the dumped worst rows (largest error at any width when
the dump was made). Rows are copied verbatim from the golden files, so the
set carries the same 128-bit mpmath reference without needing mpmath.

File layout (little endian): per function a 16-byte NUL-padded name, a
uint32 column count (3 unary / 4 binary), a uint32 row count, then the rows
as doubles (x, [y,] hi, frac). Functions in the order they appear.

Usage:
    python3 tools/packet_math_ref/make_hard_set.py --golden DIR --worst FILE --out tests/packetmath/faithful_hard.bin
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_golden import BINARY, DEFAULT_N, UNARY  # noqa: E402


def rows_of(path, cols):
    data = open(path, "rb").read()
    size = 8 * cols
    return [data[i:i + size] for i in range(0, len(data), size)]


def strided(rows, k):
    if len(rows) <= k:
        return list(rows)
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--worst", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--tail", type=int, default=384)
    a = ap.parse_args()

    # worst dump: "<label> <k>" then k lines "x y hi frac err" (hex floats)
    worst = {}
    with open(a.worst) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        label, k = lines[i].split()
        k = int(k)
        rows = []
        for ln in lines[i + 1:i + 1 + k]:
            x, y, hi, fr, _ = ln.split()
            rows.append((float.fromhex(x), float.fromhex(y), float.fromhex(hi), float.fromhex(fr)))
        stem = label.split(".")[-1] if label.startswith("sincos.") else label
        worst.setdefault(stem, []).extend(rows)
        i += 1 + k

    out = bytearray()
    total = 0
    for fn in UNARY + BINARY:
        cols = 4 if fn in BINARY else 3
        rows = rows_of(os.path.join(a.golden, fn + ".bin"), cols)
        pick = strided(rows[a.n:], a.tail) + strided(rows[:a.n], a.stride)
        for x, y, hi, fr in worst.get(fn, []):
            vals = (x, y, hi, fr) if cols == 4 else (x, hi, fr)
            pick.append(struct.pack("<%dd" % cols, *vals))
        seen, uniq = set(), []
        for r in pick:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        out += fn.encode().ljust(16, b"\0") + struct.pack("<II", cols, len(uniq)) + b"".join(uniq)
        total += len(uniq)
    with open(a.out, "wb") as f:
        f.write(out)
    print(f"{a.out}: {len(out)} bytes, {total} rows")


if __name__ == "__main__":
    main()
