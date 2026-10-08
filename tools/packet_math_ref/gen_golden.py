#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Golden-reference generator for the CPU packet math faithful-rounding gate.

For every function of ``aether/backend/cpu/simd/math/`` this writes one
binary file ``<out>/<fn>.bin`` of little-endian ``double`` records

    unary : x,    hi, frac
    binary: x, y, hi, frac

where ``hi`` is the exact result rounded to nearest (the double closest to the
exact value, subnormals and overflow included) and ``frac`` is the signed
distance ``(|exact| - |hi|) / gap`` in units of the gap between ``hi`` and its
neighbour on the side of the exact value, so ``|frac| <= 0.5``. ``frac == 0``
means the exact result IS ``hi`` (an exact case: the gate then accepts only
``hi``, with the sign of a zero). A result ``got`` is faithfully rounded
(error < 1 ULP) iff ``got == hi`` (error ``|frac|``) or ``got`` is the
neighbour of ``hi`` towards the exact value (error ``1 - |frac|``).

Exact values come from mpmath at 128 bits for finite inputs, and from the C
library (glibc, via ctypes) for special inputs (zeros, infinities, NaN) and
for the exactly-rounded functions (floor, ceil, trunc, round, rint, fabs,
fmax, fmin, fdim, copysign).

The corpus is a seeded mixture per function (``SPECS`` below): uniform in the
bit pattern over the whole domain, uniform in the bit pattern over the range
where the result is finite and non-trivial, uniform in value over the hard
ranges (near 0, near poles and branch points, near multiples of pi/2, the
overflow/underflow edges, subnormals), plus every special value. It is fully
determined by the function name and ``--n``; the md5 of every file is
committed in ``tests/packetmath/faithful_golden.md5`` so a regenerated corpus
can be checked byte for byte without trusting this script's environment
(``md5sum -c``). mpmath version used for the committed md5: 1.3.0.

Usage:
    python3 tools/packet_math_ref/gen_golden.py --out DIR [--n N] [--jobs J] [--only fn,fn]
"""

import argparse
import ctypes
import hashlib
import math
import os
import random
import struct
import sys
import zlib
from multiprocessing import Pool

import mpmath

PREC = 128
DEFAULT_N = 1 << 20

INF = float("inf")
NAN = float("nan")
DMAX = sys.float_info.max
DMIN = sys.float_info.min
DENORM = 5e-324

_libm = ctypes.CDLL("libm.so.6")
for _name in ("exp", "exp2", "exp10", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan",
              "atan", "asin", "acos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "sqrt",
              "floor", "ceil", "trunc", "round", "rint", "fabs"):
    getattr(_libm, _name).restype = ctypes.c_double
    getattr(_libm, _name).argtypes = [ctypes.c_double]
for _name in ("pow", "atan2", "hypot", "fmax", "fmin", "fdim", "copysign"):
    getattr(_libm, _name).restype = ctypes.c_double
    getattr(_libm, _name).argtypes = [ctypes.c_double, ctypes.c_double]


def libm(name, *a):
    if name == "rsqrt":
        return 1.0 / _libm.sqrt(a[0]) if a[0] != 0.0 else math.copysign(INF, a[0])
    return getattr(_libm, name)(*a)


# ─────────────────────────────── exact rounding ───────────────────────────────

def round_exact(v):
    """mpf -> (hi, frac) as described in the module docstring."""
    if v == 0:
        return 0.0, 0.0
    neg = v < 0
    a = -v if neg else v
    man, ex = mpmath.frexp(a)  # a = man * 2^ex, man in [0.5, 1)
    E = int(ex) - 1  # a in [2^E, 2^(E+1))
    sgn = -1.0 if neg else 1.0
    if E >= 1024:
        return sgn * INF, -0.5
    q = max(E - 52, -1074)
    t = mpmath.ldexp(a, -q)
    k = int(mpmath.floor(t))
    f = t - k
    if f > 0.5 or (f == 0.5 and (k & 1)):
        k += 1
        f = f - 1
    if k >= (1 << 53) and q == 1023 - 52:
        return sgn * INF, -0.5
    hi = math.ldexp(float(k), q)  # exact: k <= 2^53
    frac = float(f)
    if hi != 0.0:
        if abs(frac) < 2.0 ** -66:
            frac = 0.0  # exact case (mpmath at 128 bits is good to ~2^-72 ulp)
    elif frac == 0.0:
        frac = 2.0 ** -1000  # underflow to zero: not an exact case
    return sgn * hi if hi != 0.0 else math.copysign(0.0, sgn), frac


def finite(x):
    return x == x and abs(x) != INF


# ──────────────────────────── per-function references ─────────────────────────

def _m(x):
    return mpmath.mpf(x)


def ref_unary(fn, x):
    """Returns (hi, frac)."""
    if fn in ("floor", "ceil", "trunc", "round", "rint", "fabs"):
        return libm(fn, x), 0.0
    if not finite(x) or x == 0.0:
        return libm(fn, x), 0.0
    X = _m(x)
    if fn == "exp":
        v = mpmath.exp(X)
    elif fn == "exp2":
        v = mpmath.power(2, X)
    elif fn == "exp10":
        v = mpmath.power(10, X)
    elif fn == "expm1":
        v = mpmath.expm1(X)
    elif fn in ("log", "log2", "log10"):
        if x < 0:
            return NAN, 0.0
        if fn == "log":
            v = mpmath.log(X)
        else:
            b = 2 if fn == "log2" else 10
            # exact cases: integral powers of the base give an exact integer
            v = mpmath.log(X, b)
            r = mpmath.nint(v)
            if abs(v - r) < mpmath.mpf(2) ** -100 and mpmath.power(b, r) == X:
                v = r
    elif fn == "log1p":
        if x < -1:
            return NAN, 0.0
        if x == -1:
            return -INF, 0.0
        v = mpmath.log1p(X)
    elif fn in ("sin", "cos", "tan"):
        # Huge arguments: carry every bit of x through the reduction by pi/2,
        # plus 64 guard bits for the closest approaches to a multiple of pi/2.
        with mpmath.workprec(PREC + 64 + max(0, math.frexp(x)[1])):
            v = {"sin": mpmath.sin, "cos": mpmath.cos, "tan": mpmath.tan}[fn](X)
            return round_exact(v)
    elif fn == "atan":
        v = mpmath.atan(X)
    elif fn in ("asin", "acos"):
        if abs(x) > 1:
            return NAN, 0.0
        v = mpmath.asin(X) if fn == "asin" else mpmath.acos(X)
    elif fn == "sinh":
        v = mpmath.sinh(X)
    elif fn == "cosh":
        v = mpmath.cosh(X)
    elif fn == "tanh":
        v = mpmath.tanh(X)
    elif fn == "asinh":
        v = mpmath.asinh(X)
    elif fn == "acosh":
        if x < 1:
            return NAN, 0.0
        v = mpmath.acosh(X)
    elif fn == "atanh":
        if abs(x) > 1:
            return NAN, 0.0
        if abs(x) == 1:
            return math.copysign(INF, x), 0.0
        v = mpmath.atanh(X)
    elif fn == "cbrt":
        v = mpmath.cbrt(abs(X))
        v = -v if x < 0 else v
    elif fn == "sqrt":
        if x < 0:
            return NAN, 0.0
        v = mpmath.sqrt(X)
    elif fn == "rsqrt":
        if x < 0:
            return NAN, 0.0
        v = 1 / mpmath.sqrt(X)
    else:
        raise ValueError(fn)
    return round_exact(v)


def ref_binary(fn, x, y):
    if fn in ("fmax", "fmin", "fdim", "copysign"):
        return libm(fn, x, y), 0.0
    if not (finite(x) and finite(y)) or x == 0.0 or y == 0.0:
        return libm(fn, x, y), 0.0
    X, Y = _m(x), _m(y)
    if fn == "pow":
        if x == 1.0:
            return 1.0, 0.0
        if x < 0 and y != math.floor(y):
            return NAN, 0.0
        v = mpmath.power(X, Y)
        if isinstance(v, mpmath.mpc):
            return NAN, 0.0
    elif fn == "atan2":
        v = mpmath.atan2(X, Y)
    elif fn == "hypot":
        v = mpmath.hypot(X, Y)
    else:
        raise ValueError(fn)
    return round_exact(v)


# ─────────────────────────────────── corpus ───────────────────────────────────

def d2b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def b2d(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5, 2.5, -2.5, 10.0, -10.0, 0.1, -0.1,
    INF, -INF, NAN, -NAN, DENORM, -DENORM, DMIN, -DMIN, b2d(d2b(DMIN) - 1), -b2d(d2b(DMIN) - 1), DMAX, -DMAX,
    1e-300, -1e-300, 1e300, -1e300, 709.78, 709.79, 710.0, 710.4758600739439, 710.476, -708.4, -745.1,
    -745.13321910194110842, -745.2, -746.0, 1023.99, 1024.0, -1074.0, -1075.0, -1022.0, 307.0, 308.25, 308.26,
    -323.3, -323.5, -324.0, 22.0, -22.0, 0.49999999999999994, -0.49999999999999994, 0.9999999999999999,
    -0.9999999999999999, 1.0000000000000002, -1.0000000000000002, 2.0 ** -27, -(2.0 ** -27), 2.0 ** -28,
    2.0 ** -54, 2.0 ** -60, 2.0 ** 28, 2.0 ** 30, 2.0 ** 52, -(2.0 ** 52), 2.0 ** 53, 4503599627370495.5,
    -4503599627370495.5, 4503599627370497.0, 9007199254740993.0, math.pi, -math.pi, math.pi / 2,
    -math.pi / 2, math.pi / 4, 3 * math.pi / 4, 524288.0, 524289.5, 1e6, -1e6, 1e22, 1e100, 0.41421356237309503,
    2.414213562373095, 0.4142135623730951, 2.4142135623730954, math.sqrt(2) / 2, math.sqrt(2), 0.6931471805599453,
    -0.6931471805599453, 0.34657359027997264, 40.0, -40.0, 44.0, float.fromhex("0x1.62e42fefa39efp+9"), 6381956970095103 * 2.0 ** 797,
]


class Rng:
    def __init__(self, name, salt=0):
        self.r = random.Random(zlib.crc32(name.encode()) * 1000003 + salt)

    def u(self):
        return self.r.random()

    def bits(self, n=64):
        return self.r.getrandbits(n)


def draw(rng, spec):
    """One value from a spec tuple."""
    kind = spec[0]
    if kind == "bits":  # uniform in the bit pattern of |x| over [lo, hi] (lo >= 0), sign 'pos'|'neg'|'both'
        _, lo, hi, sign = spec
        blo, bhi = d2b(lo), d2b(hi)
        v = b2d(blo + rng.bits() % (bhi - blo + 1))
        if sign == "neg" or (sign == "both" and rng.bits(1)):
            v = -v
        return v
    if kind == "val":
        _, lo, hi = spec
        return lo + rng.u() * (hi - lo)
    if kind == "int":
        _, lo, hi = spec
        return float(lo + rng.bits() % (hi - lo + 1))
    if kind == "near":  # nearest doubles to k*c (+-1 ulp), k uniform integer in [1, K]
        _, c, K = spec
        k = 1 + rng.bits() % K
        v = float(mpmath.mpf(k) * c)
        return b2d(d2b(v) + (rng.bits() % 3) - 1)
    raise ValueError(kind)


PIO2 = None  # mpf, set per process
FULL = ("bits", 0.0, DMAX, "both")
POSALL = ("bits", 0.0, DMAX, "pos")

# (weight, spec) mixtures; weights are relative.
SPECS = {
    "exp": [(2, FULL), (3, ("bits", 0.0, 746.0, "both")), (2, ("val", -745.2, 709.8)), (1, ("val", -745.2, -708.0)),
            (1, ("val", 709.0, 709.8)), (1, ("val", -1.0, 1.0))],
    "exp2": [(2, FULL), (3, ("bits", 0.0, 1076.0, "both")), (2, ("val", -1075.0, 1024.0)),
             (1, ("val", -1075.0, -1021.0)), (1, ("val", 1023.0, 1024.0)), (1, ("val", -1.0, 1.0))],
    "exp10": [(2, FULL), (3, ("bits", 0.0, 324.0, "both")), (2, ("val", -323.7, 308.3)),
              (1, ("val", -323.7, -307.0)), (1, ("val", 307.0, 308.3)), (1, ("val", -1.0, 1.0))],
    "expm1": [(2, FULL), (3, ("bits", 0.0, 710.0, "both")), (2, ("val", -1.0, 1.0)), (1, ("val", -40.0, 40.0)),
              (1, ("val", -3.0, 3.0)), (1, ("val", 700.0, 709.8))],
    "log": [(4, POSALL), (2, ("val", 0.5, 2.0)), (1, ("val", 1 - 2.0 ** -20, 1 + 2.0 ** -20)),
            (1, ("val", 0.7, 1.42)), (1, ("bits", DENORM, DMIN, "pos")), (1, ("bits", 0.0, DMAX, "both"))],
    "log1p": [(3, POSALL), (2, ("bits", 0.0, 1.0, "neg")), (1, ("val", -1.0, -0.99)), (2, ("val", -0.5, 1.0)),
              (1, ("val", -2.0 ** -20, 2.0 ** -20)), (1, ("val", -0.3, 0.42))],
    "sin": [(2, FULL), (3, ("bits", 0.0, 524288.0, "both")), (1, ("val", -math.pi, math.pi)),
            (1, ("val", -524288.0, 524288.0)), (1, ("val", -100.0, 100.0)), (1, ("near", "pio2", 333772)),
            (1, ("val", 524288.0, 2.0 ** 25))],
    "atan": [(3, FULL), (2, ("val", -3.0, 3.0)), (1, ("val", 0.40, 0.43)), (1, ("val", 2.40, 2.43)),
             (1, ("bits", 0.0, 2.0 ** -20, "both")), (1, ("val", -100.0, 100.0))],
    "asin": [(3, ("bits", 0.0, 1.0, "both")), (2, ("val", -1.0, 1.0)), (1, ("val", 0.49, 0.51)),
             (1, ("val", 1 - 2.0 ** -20, 1.0)), (1, ("val", -1.0, -1 + 2.0 ** -20)), (1, ("val", -1.1, 1.1)),
             (1, ("val", 0.9, 1.0))],
    "sinh": [(2, FULL), (3, ("bits", 0.0, 711.0, "both")), (2, ("val", -1.0, 1.0)), (1, ("val", -22.0, 22.0)),
             (1, ("val", 21.0, 23.0)), (1, ("val", 709.0, 711.0)), (1, ("val", -3.0, 3.0))],
    "asinh": [(3, FULL), (2, ("val", -1.0, 1.0)), (1, ("val", -10.0, 10.0)), (1, ("val", 2.0 ** 27, 2.0 ** 29)),
              (1, ("bits", 0.0, 2.0 ** -20, "both")), (1, ("val", 0.4, 0.6))],
    "acosh": [(3, ("bits", 1.0, DMAX, "pos")), (2, ("val", 1.0, 1 + 2.0 ** -20)), (2, ("val", 1.0, 3.0)),
              (1, ("val", 2.0 ** 27, 2.0 ** 29)), (1, ("val", 1.0, 1.1)), (1, ("val", 0.0, 1.0))],
    "atanh": [(3, ("bits", 0.0, 1.0, "both")), (2, ("val", -0.5, 0.5)), (1, ("val", 0.49, 0.51)),
              (1, ("val", 1 - 2.0 ** -20, 1.0)), (1, ("val", -1.0, 1.0)), (1, ("val", 0.9, 1.0))],
    "cbrt": [(4, FULL), (2, ("val", -10.0, 10.0)), (1, ("val", 1.0, 8.0)), (1, ("bits", DENORM, DMIN, "both")),
             (1, ("int", -2097151, 2097151))],
    "rsqrt": [(4, POSALL), (2, ("val", 0.25, 4.0)), (1, ("bits", DENORM, DMIN, "pos")), (1, ("val", 1.0, 1.01))],
    "round": [(3, ("bits", 0.0, 2.0 ** 54, "both")), (2, ("val", -4.0, 4.0)), (1, FULL), (1, ("val", -1e6, 1e6))],
}
SPECS["log2"] = SPECS["log"]
SPECS["log10"] = SPECS["log"]
SPECS["cos"] = SPECS["sin"]
SPECS["tan"] = SPECS["sin"]
SPECS["acos"] = SPECS["asin"]
SPECS["cosh"] = SPECS["sinh"]
SPECS["tanh"] = SPECS["sinh"] + [(2, ("val", -20.0, 20.0))]
SPECS["sqrt"] = SPECS["rsqrt"]
for _f in ("floor", "ceil", "trunc", "rint", "fabs"):
    SPECS[_f] = SPECS["round"]

UNARY = ["exp", "exp2", "exp10", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "atan", "asin",
         "acos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "rsqrt", "sqrt", "floor", "ceil",
         "trunc", "round", "rint", "fabs"]
BINARY = ["pow", "atan2", "hypot", "fmax", "fmin", "fdim", "copysign"]

EXTRA_POINTS = {
    "exp2": [float(k) for k in range(-1080, 1030)] + [k + 0.5 for k in range(-1080, 1030)],
    "exp10": [float(k) for k in range(-330, 320)],
    "log2": [2.0 ** k for k in range(-1074, 1024)],
    "log10": [10.0 ** k for k in range(0, 23)] + [10.0 ** -k for k in range(1, 324)],
    "cbrt": [float(k) ** 3 for k in range(-200, 201)] + [2.0 ** k for k in range(-1074, 1024)],
    "rsqrt": [4.0 ** k for k in range(-537, 512)],
    "sqrt": [4.0 ** k for k in range(-537, 512)] + [float(k * k) for k in range(1, 3000)],
}


def pick(rng, mixture):
    tot = sum(w for w, _ in mixture)
    t = rng.u() * tot
    for w, s in mixture:
        t -= w
        if t < 0:
            return s
    return mixture[-1][1]


def unary_corpus(fn, n):
    rng = Rng(fn)
    mix = SPECS[fn]
    xs = []
    for _ in range(n):
        s = pick(rng, mix)
        if s[0] == "near":
            s = ("near", PIO2, s[2])
        xs.append(draw(rng, s))
    xs += SPECIALS + EXTRA_POINTS.get(fn, [])
    return [(x,) for x in xs]


def binary_corpus(fn, n):
    rng = Rng(fn)
    pairs = []
    grid = [(a, b) for a in SPECIALS for b in SPECIALS]
    if fn == "pow":
        for _ in range(n):
            t = rng.u() * 12
            if t < 3:  # |y log x| up to ~760: over/underflow range
                x = draw(rng, POSALL)
                lx = abs(math.log(x)) if x > 0 else 1.0
                y = (2 * rng.u() - 1) * 760.0 / max(lx, 1e-300)
            elif t < 5:
                x = 0.5 + 1.5 * rng.u()
                y = (2 * rng.u() - 1) * 760.0 / max(abs(math.log(x)), 1e-6)
            elif t < 6:
                x = 1 + (2 * rng.u() - 1) * 2.0 ** -20
                y = (2 * rng.u() - 1) * 760.0 / max(abs(math.log(x)), 1e-300)
            elif t < 7:
                x = -50 + rng.u() * (50 - 1e-3)
                y = float(rng.bits() % 401) - 200.0
            elif t < 8:
                x = -10 + 20 * rng.u()
                y = float(rng.bits() % 61) - 30.0
            elif t < 9:
                x = -10 + 20 * rng.u()
                y = float(rng.bits() % 61) - 30.5
            elif t < 10:
                x = draw(rng, POSALL)
                y = 0.5 if rng.bits(1) else -0.5
            elif t < 11:
                x = draw(rng, FULL)
                y = draw(rng, FULL)
            else:
                x = 1e-3 + 10 * rng.u()
                y = -50 + 100 * rng.u()
            pairs.append((x, y))
        pairs += [(float(b), float(e)) for b in (2, 3, 5, 7, 10) for e in range(-30, 31)]
        pairs += [(float(k * k), 0.5) for k in range(1, 200)] + [(2.0, k + 0.5) for k in range(-1080, 1030)]
    elif fn == "atan2":
        for _ in range(n):
            t = rng.u() * 8
            if t < 2:
                y, x = draw(rng, FULL), draw(rng, FULL)
            elif t < 4:
                y, x = -10 + 20 * rng.u(), -10 + 20 * rng.u()
            elif t < 5:
                x = -10 + 20 * rng.u()
                y = x * (2 * rng.u() - 1) * 2.0 ** -(rng.bits() % 60)
            elif t < 6:
                x = -(1e-3 + 10 * rng.u())
                y = x * (2 * rng.u() - 1) * 2.0 ** -(rng.bits() % 60)
            elif t < 7:
                x = -10 + 20 * rng.u()
                y = x * (1 + (2 * rng.u() - 1) * 2.0 ** -20)
            else:
                x = 1e-3 + 10 * rng.u()
                y = x * (0.4142135623730951 if rng.bits(1) else 2.414213562373095) * (1 + (2 * rng.u() - 1) * 1e-3)
                if rng.bits(1):
                    x = -x
            pairs.append((y, x))
    elif fn == "hypot":
        for _ in range(n):
            t = rng.u() * 6
            if t < 2:
                x, y = draw(rng, FULL), draw(rng, FULL)
            elif t < 4:
                x, y = -10 + 20 * rng.u(), -10 + 20 * rng.u()
            elif t < 5:
                x = draw(rng, FULL)
                y = x * (2 * rng.u() - 1) * 2.0 ** -(rng.bits() % 64)
            else:
                s = math.ldexp(1.0, int(rng.bits() % 2097) - 1074)
                x, y = (1 + rng.u()) * s, (2 * rng.u() - 1) * s
            pairs.append((x, y))
        trip = [(3, 4), (5, 12), (8, 15), (7, 24), (20, 21), (9, 40), (12, 35), (11, 60)]
        pairs += [(a * 2.0 ** k, b * 2.0 ** k) for a, b in trip for k in range(-1070, 1018, 7)]
    else:  # fmax, fmin, fdim, copysign
        for _ in range(n):
            pairs.append((draw(rng, FULL), draw(rng, FULL)))
    return pairs + grid


def _init():
    global PIO2
    mpmath.mp.prec = PREC
    PIO2 = mpmath.pi / 2


def _work(args):
    fn, chunk = args
    _init()
    out = bytearray()
    if fn in BINARY:
        for x, y in chunk:
            hi, fr = ref_binary(fn, x, y)
            out += struct.pack("<4d", x, y, hi, fr)
    else:
        for (x,) in chunk:
            hi, fr = ref_unary(fn, x)
            out += struct.pack("<3d", x, hi, fr)
    return bytes(out)


def generate(fn, n, jobs, outdir):
    _init()
    rows = binary_corpus(fn, n) if fn in BINARY else unary_corpus(fn, n)
    step = 8192
    chunks = [(fn, rows[i:i + step]) for i in range(0, len(rows), step)]
    path = os.path.join(outdir, fn + ".bin")
    h = hashlib.md5()
    with Pool(jobs, initializer=_init) as pool, open(path, "wb") as f:
        for blob in pool.imap(_work, chunks):
            f.write(blob)
            h.update(blob)
    return h.hexdigest(), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fns = [f for f in (a.only.split(",") if a.only else UNARY + BINARY) if f]
    for fn in fns:
        digest, rows = generate(fn, a.n, a.jobs, a.out)
        print(f"{digest}  {fn}.bin  rows={rows}", flush=True)


if __name__ == "__main__":
    main()
