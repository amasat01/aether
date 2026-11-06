#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_exp_table.py — the constants of `exp`'s table reduction (aether/banded/BandExpLog.h).

Computes, in exact rational arithmetic from a 120-digit ln 2:
  * `T_j = 2^(j/32)`, j = 0..31, as three FP32 limbs (round-to-nearest peeling), asserting the three limbs
    reproduce `T_j` to < 2^-72 relative;
  * `ln2/32` as six FP32 limbs: the first five carry 11 significant bits so `m * cw_i` is exact for every
    `|m| < 2^13` (asserted below for the extreme multipliers), the sixth is rounded;
and prints them as the C++ the header carries. With `--check <BandExpLog.h>` it instead verifies that the header's
table rows and reduction limbs equal the generated ones (exit 1 on any difference).

Usage: gen_exp_table.py [--check path/to/BandExpLog.h]
"""
import math
import re
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 120
N = 32
NLIMBS = 6
MMAX = 150 * N + 2  # |m| bound: |x| <= ~150 ln2-binades at saturation


def rn24(f):
    """Round `f` to the nearest FP32 value (normal range), as a Fraction."""
    if f == 0:
        return F(0)
    s = 1 if f > 0 else -1
    a = abs(f)
    e = 0
    while a >= 2 ** (e + 1):
        e += 1
    while a < 2 ** e:
        e -= 1
    q = a / F(2) ** (e - 23)
    n = math.floor(q)
    r = q - n
    if r > F(1, 2) or (r == F(1, 2) and n % 2 == 1):
        n += 1
    return s * n * F(2) ** (e - 23)


def trunc_bits(f, bits):
    s = 1 if f > 0 else -1
    a = abs(f)
    e = 0
    while a >= 2 ** (e + 1):
        e += 1
    while a < 2 ** e:
        e -= 1
    u = F(2) ** (e - bits + 1)
    return s * math.floor(a / u) * u


def lit(f):
    v = float(f)
    if v == 0:
        return "0.0f"
    t = "%.9g" % v
    return t + ("f" if ("e" in t or "." in t) else ".0f")


def exact_product_24(c, m):
    a = abs(c * m)
    e = 0
    while a >= 2 ** (e + 1):
        e += 1
    while a < 2 ** e:
        e -= 1
    return (a / F(2) ** (e - 23)).denominator == 1


def dec_to_frac(d):
    return F(d)


def reduction_limbs():
    ln2 = dec_to_frac(Decimal(2).ln())
    res = ln2 / N
    bits = 24 - MMAX.bit_length()
    limbs = []
    for _ in range(NLIMBS - 1):
        c = trunc_bits(res, bits)
        limbs.append(c)
        res -= c
    for c in limbs:
        for m in (1, 3, MMAX, MMAX - 1, 2 ** (MMAX.bit_length() - 1) + 1):
            assert exact_product_24(c, m), ("inexact zeroed-limb product", float(c), m)
    limbs.append(rn24(res))
    return limbs


def table():
    rows = []
    for j in range(N):
        t = dec_to_frac((Decimal(j) / N * Decimal(2).ln()).exp())
        limbs, rem = [], t
        for _ in range(3):
            c = rn24(rem)
            limbs.append(c)
            rem -= c
        assert abs(float(rem / t)) < 2 ** -72, ("table row below 2^-72", j)
        rows.append(limbs)
    return rows


def render_table(rows):
    dev = ",\n".join("    { " + ", ".join(lit(x) for x in t) + ", 0.0f }" for t in rows)
    host = ",\n".join("    " + ", ".join(lit(x) for x in t) + ", 0.0f" for t in rows)
    return dev, host


def main():
    rows = table()
    limbs = reduction_limbs()
    dev, host = render_table(rows)
    cw = "".join("    constexpr float cw%d = %s;\n" % (i + 1, lit(c)) for i, c in enumerate(limbs))
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        src = open(sys.argv[2]).read()
        bad = []
        if dev not in src:
            bad.append("device table rows")
        if host not in src:
            bad.append("host table rows")
        if cw not in src:
            bad.append("reduction limbs")
        if bad:
            print("gen_exp_table: header differs from the generated constants: " + ", ".join(bad))
            return 1
        print("gen_exp_table: header table and reduction limbs match the generated constants")
        return 0
    print("// device table (kBandExpTDev)\n" + dev + "\n// host table (kBandExpTHost)\n" + host + "\n// reduction limbs\n" + cw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
