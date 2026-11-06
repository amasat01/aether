#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/bandmath/gen_corpus_explog.py — golden minter for the ported
exp/expm1/exp2/log/log1p/log2/log10/tanh/sinh/cosh/asinh/acosh/atanh/tan/pow ops.

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
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_corpus_trig import label_trig
from _corpus_common import (
    BINARY_ROW_RE,
    DEFAULT_SEED,
    FENCE_NOTE,
    K_BAND_ADMISSION_MARGIN,
    K_BAND_HARD_CEILING_EXP,
    K_BAND_HARD_FLOOR_EXP,
    add_out_dir_arg,
    band_ceiling_admits,
    band_intermediate_admits,
    double_of,
    exponent_of,
    label_counts,
    parse_unary_raw,
    refuse_under_ci,
    write_list_header,
)

LN2 = 0.6931471805599453
SQRT2 = 1.4142135623730951


# =====================================================================
#  Admission predicates — transcribed verbatim from
#  aether/banded/BandExpLog.h. Same constants as aether/banded/Band.h's
#  own bandCeilingAdmits/bandIntermediateAdmits (K_BAND_ADMITTED_LO/HI
#  already match gen_corpus.py's own transcription of those two).
# =====================================================================
K_BAND_LOG_TAIL_NORMAL_EXP = -77
K_BAND_LOG2E = 1.44269504088896340736
K_BAND_EXP_SAT_HIGH = 88.7228391  # 128*ln2
K_BAND_EXP_SAT_LOW = -103.972077  # -150*ln2


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


def label_expm1(x: float) -> int:
    # Mirrors expm1's guard structure (BandExpLog.h): NaN/+-inf/+-0 are exact
    # IEEE answers; x past the high saturation, or outside the FP32 range the
    # ingest can carry, is the loud escape; x below the low saturation is
    # exactly -1, the correctly rounded answer (in_domain). expm1(x) ~ x for
    # tiny x (flushed to 0 below the FP32 floor, where the double metric
    # cannot call it a miss), so the carrier-floor leg degrades there; the
    # ceiling leg degrades the positive side.
    if not math.isfinite(x) or x == 0.0:
        return 0
    e = exponent_of(x)
    if x > K_BAND_EXP_SAT_HIGH or e >= K_BAND_HARD_CEILING_EXP:
        return 2
    if (x > 0.0 and not band_exp_admits(x, 0.0)) or not band_intermediate_admits(e):
        return 1
    return 0


def label_hyp(x: float, saturates: bool) -> int:
    # tanh/sinh/cosh over expm1 (BandHyp.h): +-0, +-inf, NaN are exact exits;
    # |x| outside the FP32 range is the loud escape, and so is sinh/cosh past
    # the FP32 envelope edge (the result is +-inf where the double is finite);
    # tiny |x| (carrier floor) and, for sinh/cosh, the ceiling margin degrade.
    if not math.isfinite(x) or x == 0.0:
        return 0
    e = exponent_of(x)
    a = abs(x)
    if e >= K_BAND_HARD_CEILING_EXP or (saturates and a > K_BAND_EXP_SAT_HIGH + LN2):
        return 2
    if not band_intermediate_admits(e) or (saturates and not band_exp_admits(max(a - LN2, 0.0), 0.0)):
        return 1
    return 0


def label_tanh(x: float) -> int:
    return label_hyp(x, False)


def label_sinh(x: float) -> int:
    return label_hyp(x, True)


def label_asinh(x: float) -> int:
    # asinh over log1p/log (BandInvHyp.h): +-0/+-inf/NaN exact; |x| outside the
    # FP32 range is the loud escape (the ingest rejects it); tiny |x| degrades.
    if not math.isfinite(x) or x == 0.0:
        return 0
    e = exponent_of(x)
    if e >= K_BAND_HARD_CEILING_EXP:
        return 2
    return 0 if band_intermediate_admits(e) else 1


def label_acosh(x: float) -> int:
    # x < 1 is the loud NaN reject; 1 + tiny (x == 1 exact is +0).
    if x != x or x == math.inf:
        return 0
    if x < 1.0:
        return 2
    if exponent_of(x) >= K_BAND_HARD_CEILING_EXP:
        return 2
    return 0


def label_atanh(x: float) -> int:
    # |x| > 1 is the loud NaN reject; +-1 are exact (+-inf); tiny |x| degrades.
    if x != x or x == 0.0 or abs(x) == 1.0:
        return 0
    if abs(x) > 1.0:
        return 2
    return 0 if band_intermediate_admits(exponent_of(x)) else 1


def label_tan(x: float):
    # tan = sin/cos (BandTrig.h): sin/cos's own guard and edge labels, then the
    # |cos x| bins -- relative claim at >= 2^-13, degraded below, and within
    # 2^-30 of a pole nothing is delivered, so those rows are not minted.
    base = label_trig(x)
    if base == 2 or not math.isfinite(x) or x == 0.0:
        return base
    c = abs(math.cos(x))
    if c < 2.0 ** -30:
        return None
    e = exponent_of(x)
    if not band_intermediate_admits(e):
        return 1  # carrier floor
    if c < 2.0 ** -13 or (abs(x) > 0.5 and abs(math.sin(x)) < 2.0 ** -13):
        return 1  # a leg is past its relative claim
    return base


def label_exp2(x: float) -> int:
    # exp2(x) = exp(x*ln2): exp's labels on the scaled argument, plus the FP32
    # overflow at x >= 128 (2^128 is not representable) as the loud escape.
    if not math.isfinite(x):
        return 0
    if x >= 128.0:
        return 2
    return label_exp(x * LN2)


def label_log1p(x: float) -> int:
    # Mirrors log1p's guards (BandExpLog.h): NaN/+-inf/+-0 and x == -1 are
    # exact answers; x < -1 is the loud NaN reject; x at or past 2^128 is out
    # of the FP32 range; tiny |x| (carrier floor, flushed values) and x past
    # the ceiling margin degrade.
    if not math.isfinite(x) or x == 0.0 or x == -1.0:
        return 0
    if x < -1.0:
        return 2
    e = exponent_of(x)
    if e >= K_BAND_HARD_CEILING_EXP:
        return 2
    if not band_intermediate_admits(e) or not band_ceiling_admits(e):
        return 1
    return 0


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
    # The IEEE points pow() answers exactly ahead of the log leg: x == 1 (any
    # y, NaN and +-inf included), x == +-0 and x == +-inf with y a non-NaN
    # nonzero (the sign of the answer is MPFR's own Annex F value).
    if x == 1.0:
        return 0
    if (x == 0.0 or math.isinf(x)) and not math.isnan(y):
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
        pts += [e, math.nextafter(e, e - 1), math.nextafter(e, e + 1)]
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


def build_expm1_spec() -> str:
    # exp's boundary families, plus the small-|x| decades where expm1 (unlike
    # exp) must keep full relative accuracy.
    return build_exp_spec() + "# small-|x| family, both signs\nreduction pow2 -60 -31\n"


def build_exp2_spec() -> str:
    pts = [float(k) for k in range(-150, 129)]
    pts += [k + d for k in (-60, -1, 0, 1, 60, 100) for d in (2.0 ** -30, -(2.0 ** -30), 0.5)]
    pts += [128.0, math.nextafter(128.0, 0.0), math.nextafter(128.0, 200.0), -150.0, math.nextafter(-150.0, -200.0)]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("reduction pow2 -30 6")
    lines.append("uniform -88.0 107.0 400")
    lines.append("uniform -150.0 -88.0 100")
    lines.append("uniform 107.0 128.0 60")
    lines.append("uniform 128.0 400.0 60")
    lines.append("uniform -700.0 -150.0 60")
    return "\n".join(lines) + "\n"


def build_log10_spec() -> str:
    pts = [10.0 ** k for k in range(0, 23)] + [10.0 ** -k for k in range(1, 23)]
    return build_log_spec() + "points " + " ".join(repr(v) for v in pts) + "\n"


def build_tanh_spec() -> str:
    pts = [22.0, math.nextafter(22.0, 0.0), math.nextafter(22.0, 30.0), 21.9999, 40.0, 1e30, -1e300, 1e300]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("reduction pow2 -60 6")
    lines.append("uniform -22.0 22.0 400")
    lines.append("uniform -60.0 60.0 100")
    return "\n".join(lines) + "\n"


def build_sinh_spec() -> str:
    pts = [22.0, math.nextafter(22.0, 0.0), math.nextafter(22.0, 30.0), 74.16679, 74.86, 88.7228391, 89.4, 89.5, 100.0,
           710.0, -1e300, 1e300, 0.34657359, 1e-30, 2.0 ** -149]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts + [-v for v in pts])]
    lines.append("reduction pow2 -60 6")
    lines.append("uniform -90.0 90.0 500")
    lines.append("uniform -700.0 700.0 60")
    return "\n".join(lines) + "\n"


def build_tan_spec() -> str:
    import random

    rng = random.Random(0x7A17)
    pio2 = math.pi / 2
    pts = []
    for _ in range(300):  # near the poles (2k+1)pi/2, k < 20000, at 2^-j offsets
        k = rng.randrange(0, 20000)
        pts.append((2 * k + 1) * pio2 + rng.choice((-1.0, 1.0)) * 2.0 ** -rng.randrange(1, 40) * (1 + rng.random()))
    pts += [pio2, -pio2, math.pi, 0.5, 1.0, 65535.0, 2.0 ** 24]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("reduction pow2 -60 24")
    lines.append("reduction pi_over_2 400")
    lines.append("uniform -65536.0 65536.0 500")
    lines.append("uniform -4.0 4.0 300")
    return "\n".join(lines) + "\n"


def build_asinh_spec() -> str:
    pts = [2.0, math.nextafter(2.0, 0.0), math.nextafter(2.0, 3.0), 2.0 ** 28, math.nextafter(2.0 ** 28, 0.0),
           math.nextafter(2.0 ** 28, 1e30), 1e30, 3e38, 1e300, 2.0 ** -19, 2.0 ** -149, 1e-300]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts + [-v for v in pts])]
    lines.append("reduction pow2 -60 120")
    lines.append("uniform -2.0 2.0 300")
    lines.append("loguniform 2.0 2e8 150")
    return "\n".join(lines) + "\n"


def build_acosh_spec() -> str:
    pts = [1.0, math.nextafter(1.0, 0.0), 0.5, 0.0, -1.0, 2.0, math.nextafter(2.0, 0.0), 2.0 ** 28, 1e30, 3e38, 1e300]
    pts += [1.0 + 2.0 ** -j for j in range(1, 53)]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("reduction pow2 -20 120")
    lines.append("uniform 1.0 2.0 300")
    lines.append("loguniform 2.0 1e38 200")
    return "\n".join(lines) + "\n"


def build_atanh_spec() -> str:
    pts = [1.0, -1.0, math.nextafter(1.0, 2.0), math.nextafter(-1.0, -2.0), 2.0, -2.0, 0.5, -0.5, 1e300, 2.0 ** -19, 1e-300]
    pts += [s * (1.0 - 2.0 ** -j) for j in range(1, 53) for s in (1.0, -1.0)]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("reduction pow2 -60 2")
    lines.append("uniform -1.0 1.0 400")
    return "\n".join(lines) + "\n"


def build_log1p_spec() -> str:
    pts = []
    for j in range(1, 41):  # the series/log seam, both signs, both branches
        for s in (1.0, -1.0):
            pts += [s * 2.0 ** -19 * (1 + 2.0 ** -j), s * 2.0 ** -19 * (1 - 2.0 ** -j)]
    for j in range(1, 53):  # the cancellation regime, -1 + 2^-j
        pts.append(-1.0 + 2.0 ** -j)
    pts += [-1.0, math.nextafter(-1.0, -2.0), -2.0, -1e300, 2.0 ** -149, -(2.0 ** -149), 1e-300, -1e-300, 1.0]
    lines = [f"seed 0x{DEFAULT_SEED:016X}", "points " + " ".join(repr(v) for v in pts)]
    lines.append("# pow2 families: series regime, then the log branch, both signs")
    lines.append("reduction pow2 -60 -20")
    lines.append("reduction pow2 -19 6")
    lines.append("# bulk: (-1, 8], and [8, 2^121)")
    lines.append("uniform -1.0 8.0 400")
    lines.append("loguniform 8.0 1e36 150")
    return "\n".join(lines) + "\n"


def build_log_spec() -> str:
    lines = [f"seed 0x{DEFAULT_SEED:016X}"]
    lines.append("points " + " ".join(
        repr(v) for v in (
            1.0, math.nextafter(1.0, 2.0), math.nextafter(1.0, 0.0),
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
        pts2 += [m, math.nextafter(m, m * 1.01), math.nextafter(m, m * 0.99)]
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
def parse_binary_raw(path: str):
    text = open(path).read()
    return [(int(a, 16), int(b, 16), int(c, 16)) for a, b, c, _c in BINARY_ROW_RE.findall(text)]


def write_labeled_header(op: str, arity: int, bound: float, rows, provenance: str, out_path: str, mint_cmd: str):
    in_domain, degraded, rejected = label_counts(rows)

    head = []
    head.append("// GENERATED by tools/bandmath/gen_corpus_explog.py -- golden reference + domain")
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
    head.append(f"// bound (in_domain, ULP): {bound}")
    head.append("//")
    head.append("// label encoding: 0=in_domain 1=degraded 2=rejected")
    head.append(f"// corpus: {len(rows)} rows ({in_domain} in_domain / {degraded} degraded / {rejected} rejected)")
    head.append("//")
    head.extend(FENCE_NOTE)

    row_struct = [
        f"    std::uint64_t in[{arity}];  ///< input bit pattern(s)",
        "    std::uint64_t ref;         ///< golden reference, correctly-rounded double bits",
        "    std::uint8_t  label;       ///< 0=in_domain 1=degraded 2=rejected",
    ]
    row_lines = []
    for r in rows:
        if arity == 1:
            in0, ref, label = r
            row_lines.append(f"    {{ {{ 0x{in0:016X}ULL }}, 0x{ref:016X}ULL, {label} }},")
        else:
            in0, in1, ref, label = r
            row_lines.append(f"    {{ {{ 0x{in0:016X}ULL, 0x{in1:016X}ULL }}, 0x{ref:016X}ULL, {label} }},")
    return write_list_header(out_path, head, op, arity, bound, row_struct, row_lines, rows)


OPS = {
    "exp": (1, 0.5, build_exp_spec,
            "aether/banded/BandExpLog.h bandExp/exp; bound = kExpUlpBound=0.5 "
            "(BandLevers.ExpWithinDerivedUlpBoundOnHost)."),
    "expm1": (1, 0.5, build_expm1_spec,
              "aether/banded/BandExpLog.h expm1; bound = 0.5 ULP53."),
    "log1p": (1, 1.0, build_log1p_spec,
              "aether/banded/BandExpLog.h log1p; bound = 1 ULP53."),
    "exp2": (1, 1.5, build_exp2_spec,
             "aether/banded/BandExpLog.h exp2; bound = 1.5 ULP53 (0.5 exp + 0.5 product + constant)."),
    "log2": (1, 1.5, build_log_spec,
             "aether/banded/BandExpLog.h log2; bound = 1.5 ULP53."),
    "log10": (1, 1.5, build_log10_spec,
              "aether/banded/BandExpLog.h log10; bound = 1.5 ULP53."),
    "tanh": (1, 2.0, build_tanh_spec,
             "aether/banded/BandHyp.h tanh; bound = 2 ULP53."),
    "sinh": (1, 2.0, build_sinh_spec,
             "aether/banded/BandHyp.h sinh; bound = 2 ULP53."),
    "cosh": (1, 2.0, build_sinh_spec,
             "aether/banded/BandHyp.h cosh; bound = 2 ULP53."),
    "tan": (1, 1.5, build_tan_spec, "aether/banded/BandTrig.h tan; bound = 1.5 ULP53 where |cos x| >= 2^-13."),
    "asinh": (1, 2.0, build_asinh_spec, "aether/banded/BandInvHyp.h asinh; bound = 2 ULP53."),
    "acosh": (1, 2.0, build_acosh_spec, "aether/banded/BandInvHyp.h acosh; bound = 2 ULP53."),
    "atanh": (1, 2.0, build_atanh_spec, "aether/banded/BandInvHyp.h atanh; bound = 2 ULP53."),
    "log": (1, 0.5, build_log_spec,
            "aether/banded/BandExpLog.h log; bound = kLogUlpBound=0.5."),
    "pow": (2, 0.5, build_pow_spec,
            "aether/banded/BandExpLog.h pow; bound = kPowUlpBound=0.5."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-bin", required=True, help="path to the built ulp_oracle binary")
    add_out_dir_arg(ap)
    ap.add_argument("--spec-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "specs"))
    ap.add_argument("--raw-dir", default="/tmp/p11c1_explog_raw")
    ap.add_argument("--ops", default=None)
    args = ap.parse_args()

    if refuse_under_ci("gen_corpus_explog.py"):
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
            label_fn = {"exp": label_exp, "expm1": label_expm1, "log1p": label_log1p, "exp2": label_exp2,
                                  "tanh": label_tanh, "sinh": label_sinh, "cosh": label_sinh,
                                  "tan": label_tan, "asinh": label_asinh, "acosh": label_acosh, "atanh": label_atanh}.get(op, label_log)
            for in0, ref in raw_rows:
                x = double_of(in0)
                lab = label_fn(x)
                if lab is not None:
                    labeled.append((in0, ref, lab))
        else:
            raw_rows = parse_binary_raw(raw_path)
            labeled = []
            for in0, in1, ref in raw_rows:
                x = double_of(in0)
                y = double_of(in1)
                labeled.append((in0, in1, ref, label_pow(x, y)))

        mint_cmd = (f"python3 tools/bandmath/gen_corpus_explog.py --oracle-bin <ulp_oracle> "
                    f"--ops {op}")
        out_path = os.path.join(out_dir, f"golden_{op}.h")
        in_d, deg, rej = write_labeled_header(op, arity, bound, labeled, provenance, out_path, mint_cmd)
        print(f"{op:<6} {len(labeled):>7} {in_d:>10} {deg:>9} {rej:>9}  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
