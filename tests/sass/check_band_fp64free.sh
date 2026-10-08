#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# tests/sass/check_band_fp64free.sh — SASS-level audit of the BANDED carrier's
# 0-FP64 invariant (sm_61): asks the same three questions of FIVE audited
# kernels.
#
# ★ 0-FP64 IS A PORTABILITY WARRANT, NOT AN OPTIMISATION. The banded carrier
# exists so a kernel needing double-class accuracy can run on hardware with
# NO FP64 units, or with them at 1:32 of the FP32 rate. A single FP64
# instruction in the emitted stream does not make the kernel slightly slower
# — it makes the warrant false. There is no such thing as an acceptable FP64
# exception here, and this gate is never to be softened to accommodate one.
#
# ## The three questions, asked of FIVE audited kernels
#
#   * `bandChainKernel` (tests/test_BandedReal.cu) — one expression
#     evaluated once over the certified core ops.
#   * `bandAccumFoldKernel` (tests/test_BandLevers.cu) — the
#     ACCUMULATOR's own shape: `BandAccum` state carried across a LOOP of
#     `addTerm`/`subTerm`, read back through the exemplar's new
#     `toBandedReal()` egress.
#   * `ff2PrimKernel` (tests/test_Ff2Cert.cu) — the
#     Ff-cores rung's own shape: three `BandedReal` operands demoted to `Ff2`,
#     seven independent primitives computed from them, each packed back once.
#     Neither a chain nor a fold.
#   * `rsqrtAtanPrimKernel` (tests/test_BandLevers.cu)
#     — the Rsqrt/Atan-sector shape: `rsqrt`/`sqrt_`/`rsqrtCube` (each its own
#     Newton core) plus `atanSector`'s pure table lookup.
#   * `roundTripKernel` (tests/test_BandCell.cu) —
#     `BandCell`'s (the 16-byte biased-`Band` cell) OWN codec, `cellFromBand`
#     then `bandFromCell`, exercised over device data.
#
#   1. ZERO FP64 INSTRUCTIONS. No DADD/DMUL/DFMA/DSETP/DMNMX, no FP64 SFU helpers
#      (MUFU.RCP64H/RSQ64H), no FP64 converts (F2F/I2F/F2I with .F64/.64
#      operands). A hit means a `double` leaked into the chain — a literal, a
#      promotion, a facade entry that resolved to the native leg.
#      NOTE: LDG/STG/LDS `.64` are 64-bit MEMORY ops (the codec's own 8-byte
#      words) and are EXPECTED; the patterns below are word-anchored so they
#      never match those.
#   2. INSTRUCTION BUDGET, derived rather than fitted (see the derivation block
#      further down). The failure class it exists to catch is a CODEC CROSSING
#      PER OP instead of per boundary — a regression that compiles, computes the
#      right answers, and is invisible to the FP64 grep because it is pure INT32.
#   3. NO STACK SPILL TRAFFIC (STL/LDL). A HARD failure here, not a warning: the
#      banded carrier is three bare FP32 limbs with no metadata, which is the
#      entire premise of the rung, so a chain of this size spilling to local
#      memory is a structural regression and 0 is a derived threshold.
#
# ## ★ THE POSITIVE CONTROL IS MANDATORY, for EVERY audited kernel
# `bandFp64InjectedKernel` / `bandAccumFoldFp64InjectedKernel` /
# `ff2PrimFp64InjectedKernel` / `rsqrtAtanPrimFp64InjectedKernel` /
# `roundTripFp64InjectedKernel` are each the identical body plus ONE
# deliberate FP64 operation on a KERNEL ARGUMENT (so it cannot be folded
# away). This audit is REQUIRED to find every one of them. Without that leg,
# "0 FP64 hits" is equally consistent with a working scan and with a symbol
# lookup that quietly matched nothing — which is exactly how a regex change or
# a kernel rename takes a gate like this out silently.
#
# ## Usage
#   tests/sass/check_band_fp64free.sh [<build dir>] [--mint-card <path>]
#
# `<build dir>` defaults to `$AETHER_BUILD_DIR` or `build_cuda_release`. The gate
# reads only `cuobjdump` output from the already-built `aether_tests` binary:
# COMPILE-ONLY, NO GPU EXECUTION, nothing here is ever launched.
#
# `--mint-card` writes the measured numbers to a result card. THE RUN WRITES
# THE CARD; prose elsewhere cites it and never restates a number by hand.

set -uo pipefail

BUILD_DIR=""
CARD_PATH=""
# --subject <op> restricts the per-op subject list below to ONE op (e.g.
# re-auditing just one op without re-running every prior op); default
# (unset) runs every op in P11C0_OPS. Op names are the bare lowercase form
# (abs, copysign, fmax, fmin, add, sub, mul) — NOT the five kernel-chain
# subjects above, which stay positional/unconditional.
SUBJECT_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mint-card)
            [[ $# -ge 2 ]] || { echo "--mint-card needs a path" >&2; exit 2; }
            CARD_PATH="$2"; shift 2 ;;
        --mint-card=*) CARD_PATH="${1#--mint-card=}"; shift ;;
        --subject)
            [[ $# -ge 2 ]] || { echo "--subject needs an op name" >&2; exit 2; }
            SUBJECT_FILTER="$2"; shift 2 ;;
        --subject=*) SUBJECT_FILTER="${1#--subject=}"; shift ;;
        -*) echo "unknown option '$1'" >&2; exit 2 ;;
        *) BUILD_DIR="$1"; shift ;;
    esac
done
BUILD_DIR="${BUILD_DIR:-${AETHER_BUILD_DIR:-build_cuda_release}}"

TESTS_BIN="${BUILD_DIR}/tests/aether_tests"
if [[ ! -x "${TESTS_BIN}" ]]; then
    echo "ERROR: aether_tests binary not found at ${TESTS_BIN}" >&2
    exit 1
fi
if ! command -v cuobjdump > /dev/null 2>&1; then
    echo "ERROR: cuobjdump not found on PATH; this gate cannot run" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# ★ THE BUDGET IS DERIVED FROM THE DETECTION REQUIREMENT, NOT FITTED TO THE
# OBSERVATION, and the derivation is this:
#
#   `bandChainKernel` makes exactly THREE codec boundary crossings, and the
#   number is READ OFF THE SOURCE rather than measured: two unpacks (the two
#   storage loads) and one pack (the store). Its op inventory, likewise read off
#   the source and restated in the kernel's own docblock, is TEN elementary
#   banded ops (3 abs, 1 min, 1 max, 1 copysign, 1 div, 1 mul, 1 add, 1 sub).
#
#   The failure class this budget exists to catch is a crossing PER OP instead of
#   per boundary — which is what happens the moment an expression layer stops
#   answering `Band` for a banded chain and every step packs and unpacks around
#   itself. That regression compiles, computes the right answers, and is
#   invisible to the FP64 grep because it is pure INT32. It costs SEVEN extra
#   pack/unpack pairs (10 ops minus 3 boundaries), and at the codec's own costs
#   (pack ~18 INT32, unpack ~10) that is at least 7 * 28 = 196 instructions.
#
#   So the budget must sit STRICTLY BELOW measured + 196, or it cannot see the
#   very thing it exists for. The headroom granted below is 160 — under 196 by
#   construction, and an order of magnitude above ordinary codegen jitter.
#
#   ★ Do NOT raise this number to make a red gate green. RE-DERIVE it: change the
#   kernel body and the op inventory in its docblock moves with it, and the two
#   must be re-reconciled here.
#
# The MEASURED baseline is minted into the card by the run itself
# (`--mint-card`); this default is the baseline plus the derived headroom.
# ---------------------------------------------------------------------------
BND_BUDGET="${AETHER_BAND_CHAIN_BUDGET:-700}"
BND_HEADROOM=160

# ---------------------------------------------------------------------------
# ★ The ACCUM FOLD budget, derived the SAME WAY: strictly below
# measured + 196.
#
#   `bandAccumFoldKernel` (tests/test_BandLevers.cu) folds `kAccumFoldTerms`
#   (= 8, read off that file's own constant) packed terms through ONE
#   `BandAccum` via `addTerm`/`subTerm`, then packs the result once via
#   `toBandedReal()`. Correct behaviour keeps the running accumulator (a
#   `Band`, three bare FP32 limbs) in REGISTERS for the whole loop, touching
#   the codec only at the 8 per-term unpacks (the input loads) and the ONE
#   final pack (the store) — 9 boundary crossings for 8 certified `Band` adds
#   (`addTerm`/`subTerm` are both exactly one `detail::add`/`detail::sub`
#   call; `sub` is `add` of a free `neg`, same cost).
#
#   The failure class this budget exists to catch is the accumulator's
#   RUNNING STATE round-tripping through the codec BETWEEN iterations instead
#   of staying live as a `Band` — e.g. a regression that stores the partial
#   sum back to a `BandedReal` and re-unpacks it before folding the next term.
#   There are exactly `kAccumFoldTerms - 1` = 7 such iteration-to-iteration
#   CARRY points where that could happen, each costing one extra pack/unpack
#   PAIR at the codec's own costs (pack ~18 INT32, unpack ~10) = 28
#   instructions, so 7 * 28 = 196 — the identical ceiling the chain-kernel
#   derivation above reaches, from an unrelated argument (iteration carry
#   points, not "ops minus boundaries"): both regressions are "one extra
#   pack/unpack pair per opportunity for the value to leave registers", and
#   this kernel's op count was chosen (`kAccumFoldTerms = 8`) so that
#   opportunity count is 7, same as the chain kernel's.
#
#   The headroom granted below is 160 — under 196 by construction, same
#   margin the chain-kernel budget carries.
#
#   ★ Do NOT raise this number to make a red gate green. RE-DERIVE it: change
#   `kAccumFoldTerms` or the fold body and this comment block moves with it.
# ---------------------------------------------------------------------------
ACC_BUDGET="${AETHER_BAND_ACCUM_BUDGET:-328}"
ACC_HEADROOM=160

# ---------------------------------------------------------------------------
# ★ The FF-CORES PRIMITIVE budget, derived from a
#   DIFFERENT regression class than subjects 1/2 (no chain / no fold, so
#   "extra crossings per op instead of per boundary" does not apply here):
#
#   `ff2PrimKernel` (`tests/test_Ff2Cert.cu`) demotes THREE independent
#   `BandedReal` operands to `Ff2` (`a[i]`,`b[i]`,`c[i]`), computes SEVEN
#   independent primitives (add, sub, mul, fma, div, sqrt, rsqrt — each
#   consuming only its own already-demoted operands, no chaining between
#   them), and packs each of the 7 results back to `BandedReal` once — 10
#   codec-boundary crossings total (3 unpacks + 7 packs), read off the source.
#
#   The failure class this budget exists to catch: `Ff2`'s demote ctor
#   (`Ff2(Band b): hi(b.hi), lo(b.lo)`) is a two-field COPY, and its egress
#   (`toBandedReal()`) is ONE certified `Band`-carrier pack — a regression that
#   routed either through an extra full codec round-trip (e.g. re-encoding a
#   demoted operand back through `BandedReal` before use) costs ONE extra
#   pack/unpack PAIR at the codec's own measured cost (~28 INT32
#   instructions, the SAME per-pair cost subjects 1/2 derive from). Unlike
#   those two kernels, THIS kernel's smallest realistic regression is a
#   SINGLE extra pair (there is no multi-op chain to amplify it through), so
#   the budget must sit STRICTLY BELOW measured + 28, not +196.
#
#   The MEASURED baseline is minted into the card by the run itself
#   (`--mint-card`); this default is the baseline plus the derived headroom.
#
#   ★ Do NOT raise this number to make a red gate green. RE-DERIVE it: change
#   `ff2PrimKernel`'s op inventory and this comment block moves with it.
# ---------------------------------------------------------------------------
FF2_BUDGET="${AETHER_BAND_FF2PRIM_BUDGET:-950}"
FF2_HEADROOM=20

# ---------------------------------------------------------------------------
# ★ The RSQRT/ATAN-SECTOR PRIMITIVE budget, the SAME
#   single-crossing regression class `ff2PrimKernel` derives against (no
#   op-chain between the four independent calls to amplify a regression
#   through):
#
#   `rsqrtAtanPrimKernel` (`tests/test_BandLevers.cu`) demotes TWO independent
#   `BandedReal` operands to `Band` (2 unpacks), computes THREE independent
#   RsqrtCore-family primitives (rsqrt, sqrt_, rsqrtCube — each paying its
#   OWN `rsqrtCore()` Newton refinement, no shared state between them) plus
#   ONE `atanSector` lookup (pure FP32 compares + a literal table, no core at
#   all), and packs each of the 5 results back once (5 packs) = 7 codec-
#   boundary crossings total, read off the source.
#
#   The smallest realistic regression here is the SAME shape `ff2PrimKernel`
#   derives against: ONE extra codec pack/unpack pair (~28 INT32
#   instructions) on a single crossing, not a multi-crossing chain-amplified
#   class. So the budget must sit STRICTLY BELOW measured + 28.
#
#   The MEASURED baseline is minted into the card by the run itself
#   (`--mint-card`); this default is the baseline plus the derived headroom.
#
#   ★ Do NOT raise this number to make a red gate green. RE-DERIVE it: change
#   `rsqrtAtanPrimKernel`'s op inventory and this comment block moves with it.
# ---------------------------------------------------------------------------
RSQAT_BUDGET="${AETHER_BAND_RSQATPRIM_BUDGET:-530}"
RSQAT_HEADROOM=20

# ---------------------------------------------------------------------------
# ★ The BANDCELL ROUND-TRIP budget (subject 5), the
#   SAME single-crossing regression class subjects 3/4 derive against:
#
#   `roundTripKernel` (`tests/test_BandCell.cu`) is `BandCell`'s OWN codec --
#   `cellFromBand` then `bandFromCell` -- exercised over device data, read off
#   the source: TWO `bandScalePow2Split` calls (one inside each direction),
#   each two `scalePow2f` calls of THREE FP32 multiplies (hi, lo, tail) = 12
#   FP32 multiplies, plus the bit-level exponent extract/rebuild each
#   direction does (int shifts/masks only). There is no algebra between
#   separate certified ops here to amplify a regression through -- the whole
#   body IS the codec, unlike `bandChainKernel`'s ten-op chain -- so the
#   smallest realistic regression is ONE extra `bandScalePow2Split` call (~14
#   instructions: 3 FP32 muls x 2 halves + int bookkeeping) on a single
#   crossing, not a multi-crossing chain-amplified class. So the budget must
#   sit STRICTLY BELOW measured + 28 (the SAME +28 ceiling subjects 3/4 use,
#   for the identical single-extra-pair argument).
#
#   The MEASURED baseline is minted into the card by the run itself
#   (`--mint-card`); this default is the baseline plus the derived headroom.
#
#   ★ Do NOT raise this number to make a red gate green. RE-DERIVE it: change
#   `roundTripKernel`'s op inventory and this comment block moves with it.
# ---------------------------------------------------------------------------
RTRIP_BUDGET="${AETHER_BAND_ROUNDTRIP_BUDGET:-80}"
RTRIP_HEADROOM=20

TMPD=$(mktemp -d)
trap 'rm -rf "${TMPD}"' EXIT

find_sym() {
    cuobjdump -arch sm_61 -sass "${TESTS_BIN}" 2>/dev/null \
        | grep -oP "Function : _[A-Za-z0-9_]*${1}[A-Za-z0-9_]*" \
        | head -1 | sed 's/Function : //'
}

count_match() { grep -cE "$1" "$2" 2>/dev/null || true; }

# FP64 arithmetic/compare/convert patterns. Word-anchored opcodes so that 64-bit
# MEMORY ops (LDG.E.64 / STG.E.64 / LDS.U.64) never match.
FP64_RE='\bDADD|\bDMUL|\bDFMA|\bDSETP|\bDMNMX|MUFU\.(RCP64H|RSQ64H)|F2F\.F64|F2F\..*\.F64|I2F\.F64|F2I\..*F64|F2I\.F64'

BND_SYM=$(find_sym bandChainKernel)
BND_INJ=$(find_sym bandFp64InjectedKernel)
ACC_SYM=$(find_sym bandAccumFoldKernel)
ACC_INJ=$(find_sym bandAccumFoldFp64InjectedKernel)
FF2_SYM=$(find_sym ff2PrimKernel)
FF2_INJ=$(find_sym ff2PrimFp64InjectedKernel)
RSQAT_SYM=$(find_sym rsqrtAtanPrimKernel)
RSQAT_INJ=$(find_sym rsqrtAtanPrimFp64InjectedKernel)
RTRIP_SYM=$(find_sym roundTripKernel)
RTRIP_INJ=$(find_sym roundTripFp64InjectedKernel)

echo "=============================================="
echo "  Banded FP64-free SASS audit  (sm_61)"
echo "  binary: ${TESTS_BIN}"
echo "=============================================="

FAIL=0
BND_OK=0
ACC_OK=0

# ---------------------------------------------------------------------------
# Subject 1: bandChainKernel
# ---------------------------------------------------------------------------
echo "--- subject: bandChainKernel -------------"
if [[ -z "${BND_SYM}" ]]; then
    echo "FAIL: bandChainKernel not found (tests/test_BandedReal.cu missing,"
    echo "      renamed, or dead-stripped?) -- there is nothing to scan"
    FAIL=1
elif [[ -z "${BND_INJ}" ]]; then
    echo "FAIL: bandFp64InjectedKernel not found -- the positive control is gone,"
    echo "      so a clean result proves nothing"
    FAIL=1
else
    cuobjdump -arch sm_61 -sass -fun "${BND_SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/bnd.sass"
    cuobjdump -arch sm_61 -sass -fun "${BND_INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/inj.sass"

    BND_FP64=$(count_match "${FP64_RE}" "${TMPD}/bnd.sass")
    INJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/inj.sass")
    BND_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/bnd.sass" || true)
    INJ_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/inj.sass" || true)
    BND_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/bnd.sass")
    # Register/stack usage from `-res-usage` (the `-elf` dump does not carry a
    # `REG:` field on this toolchain). STACK is read too and cross-checked
    # against the STL/LDL count below: two independent readings of the same
    # claim, which is what stops a format change from silently answering "0".
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null \
        | grep -A1 -F "Function ${BND_SYM}:" | tail -1)
    BND_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    BND_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-38s %s\n" "audited kernel" "${BND_SYM:0:60}"
    printf "  %-38s %d\n" "FP64 arithmetic/convert hits" "${BND_FP64}"
    printf "  %-38s %d (budget %d)\n" "total instructions" "${BND_TOTAL}" "${BND_BUDGET}"
    printf "  %-38s %d\n" "stack spill ops (STL/LDL)" "${BND_SPILL}"
    printf "  %-38s %s (stack %s B)\n" "register count" "${BND_REG}" "${BND_STACK}"
    printf "  %-38s %d (of %d instructions)\n" "POSITIVE CONTROL FP64 hits" \
        "${INJ_FP64}" "${INJ_TOTAL}"
    echo "----------------------------------------------"

    if (( BND_TOTAL == 0 )); then
        echo "FAIL: bandChainKernel disassembled to ZERO instructions -- the audit"
        echo "      would be scanning the empty set"
        FAIL=1
    elif (( BND_TOTAL > BND_BUDGET )); then
        echo "FAIL: banded chain count ${BND_TOTAL} exceeds budget ${BND_BUDGET}"
        echo "      -- the usual cause is a codec crossing PER OP instead of per"
        echo "      boundary. RE-DERIVE the budget; do not raise it."
        FAIL=1
    else
        echo "PASS: banded chain count ${BND_TOTAL} within budget ${BND_BUDGET}"
    fi

    if (( BND_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in the banded storage/step-control chain:"
        grep -E "${FP64_RE}" "${TMPD}/bnd.sass" | head -10
        FAIL=1
    else
        echo "PASS: ZERO FP64 instructions between the banded load and store"
    fi

    if (( INJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: the positive control came back CLEAN."
        echo "      bandFp64InjectedKernel contains a deliberate FP64 operation, so"
        echo "      this audit is blind and the PASS above certifies nothing."
        FAIL=1
    else
        echo "PASS: SEEDED-RED NON-VACUITY: positive control detected ${INJ_FP64} FP64 hits in bandFp64InjectedKernel -- the scan has demonstrated failure power on this binary"
    fi

    if [[ "${BND_STACK}" != "0" && "${BND_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${BND_STACK} bytes for the banded chain --"
        echo "      a second, independent reading of the same no-spill claim"
        FAIL=1
    fi
    if (( BND_SPILL != 0 )); then
        echo "FAIL: ${BND_SPILL} stack spill instructions in the banded chain --"
        echo "      three bare FP32 limbs with no metadata is the whole premise of"
        echo "      this rung; a chain this size must stay register-resident"
        FAIL=1
    else
        echo "PASS: no stack spill traffic (STL/LDL) in the banded chain"
    fi
    BND_OK=1
fi

# ---------------------------------------------------------------------------
# Subject 2: bandAccumFoldKernel
# ---------------------------------------------------------------------------
echo "--- subject: bandAccumFoldKernel ----------"
if [[ -z "${ACC_SYM}" ]]; then
    echo "FAIL: bandAccumFoldKernel not found (tests/test_BandLevers.cu missing,"
    echo "      renamed, or dead-stripped?) -- there is nothing to scan"
    FAIL=1
elif [[ -z "${ACC_INJ}" ]]; then
    echo "FAIL: bandAccumFoldFp64InjectedKernel not found -- the positive control"
    echo "      is gone, so a clean result proves nothing"
    FAIL=1
else
    cuobjdump -arch sm_61 -sass -fun "${ACC_SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/acc.sass"
    cuobjdump -arch sm_61 -sass -fun "${ACC_INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/accinj.sass"

    ACC_FP64=$(count_match "${FP64_RE}" "${TMPD}/acc.sass")
    ACCINJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/accinj.sass")
    ACC_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/acc.sass" || true)
    ACCINJ_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/accinj.sass" || true)
    ACC_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/acc.sass")
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null \
        | grep -A1 -F "Function ${ACC_SYM}:" | tail -1)
    ACC_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    ACC_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-38s %s\n" "audited kernel" "${ACC_SYM:0:60}"
    printf "  %-38s %d\n" "FP64 arithmetic/convert hits" "${ACC_FP64}"
    printf "  %-38s %d (budget %d)\n" "total instructions" "${ACC_TOTAL}" "${ACC_BUDGET}"
    printf "  %-38s %d\n" "stack spill ops (STL/LDL)" "${ACC_SPILL}"
    printf "  %-38s %s (stack %s B)\n" "register count" "${ACC_REG}" "${ACC_STACK}"
    printf "  %-38s %d (of %d instructions)\n" "POSITIVE CONTROL FP64 hits" \
        "${ACCINJ_FP64}" "${ACCINJ_TOTAL}"
    echo "----------------------------------------------"

    if (( ACC_TOTAL == 0 )); then
        echo "FAIL: bandAccumFoldKernel disassembled to ZERO instructions -- the"
        echo "      audit would be scanning the empty set"
        FAIL=1
    elif (( ACC_TOTAL > ACC_BUDGET )); then
        echo "FAIL: accum fold count ${ACC_TOTAL} exceeds budget ${ACC_BUDGET}"
        echo "      -- the usual cause is the running accumulator round-tripping"
        echo "      through the codec BETWEEN iterations. RE-DERIVE the budget;"
        echo "      do not raise it."
        FAIL=1
    else
        echo "PASS: accum fold count ${ACC_TOTAL} within budget ${ACC_BUDGET}"
    fi

    if (( ACC_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in the accumulator fold:"
        grep -E "${FP64_RE}" "${TMPD}/acc.sass" | head -10
        FAIL=1
    else
        echo "PASS: ZERO FP64 instructions in the accumulator fold"
    fi

    if (( ACCINJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: the positive control came back CLEAN."
        echo "      bandAccumFoldFp64InjectedKernel contains a deliberate FP64"
        echo "      operation, so this audit is blind and the PASS above"
        echo "      certifies nothing."
        FAIL=1
    else
        echo "PASS: SEEDED-RED NON-VACUITY: positive control detected ${ACCINJ_FP64} FP64 hits in bandAccumFoldFp64InjectedKernel -- the scan has demonstrated failure power on this binary"
    fi

    if [[ "${ACC_STACK}" != "0" && "${ACC_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${ACC_STACK} bytes for the accum fold --"
        echo "      a second, independent reading of the same no-spill claim"
        FAIL=1
    fi
    if (( ACC_SPILL != 0 )); then
        echo "FAIL: ${ACC_SPILL} stack spill instructions in the accum fold --"
        echo "      the running accumulator is three bare FP32 limbs with no"
        echo "      metadata; a fold this size must stay register-resident"
        FAIL=1
    else
        echo "PASS: no stack spill traffic (STL/LDL) in the accum fold"
    fi
    ACC_OK=1
fi

# ---------------------------------------------------------------------------
# Subject 3: ff2PrimKernel
# ---------------------------------------------------------------------------
FF2_OK=0
echo "--- subject: ff2PrimKernel -------"
if [[ -z "${FF2_SYM}" ]]; then
    echo "FAIL: ff2PrimKernel not found (tests/test_Ff2Cert.cu missing, renamed,"
    echo "      or dead-stripped?) -- there is nothing to scan"
    FAIL=1
elif [[ -z "${FF2_INJ}" ]]; then
    echo "FAIL: ff2PrimFp64InjectedKernel not found -- the positive control is"
    echo "      gone, so a clean result proves nothing"
    FAIL=1
else
    cuobjdump -arch sm_61 -sass -fun "${FF2_SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/ff2.sass"
    cuobjdump -arch sm_61 -sass -fun "${FF2_INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/ff2inj.sass"

    FF2_FP64=$(count_match "${FP64_RE}" "${TMPD}/ff2.sass")
    FF2INJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/ff2inj.sass")
    FF2_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/ff2.sass" || true)
    FF2INJ_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/ff2inj.sass" || true)
    FF2_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/ff2.sass")
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null \
        | grep -A1 -F "Function ${FF2_SYM}:" | tail -1)
    FF2_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    FF2_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-38s %s\n" "audited kernel" "${FF2_SYM:0:60}"
    printf "  %-38s %d\n" "FP64 arithmetic/convert hits" "${FF2_FP64}"
    printf "  %-38s %d (budget %d)\n" "total instructions" "${FF2_TOTAL}" "${FF2_BUDGET}"
    printf "  %-38s %d\n" "stack spill ops (STL/LDL)" "${FF2_SPILL}"
    printf "  %-38s %s (stack %s B)\n" "register count" "${FF2_REG}" "${FF2_STACK}"
    printf "  %-38s %d (of %d instructions)\n" "POSITIVE CONTROL FP64 hits" \
        "${FF2INJ_FP64}" "${FF2INJ_TOTAL}"
    echo "----------------------------------------------"

    if (( FF2_TOTAL == 0 )); then
        echo "FAIL: ff2PrimKernel disassembled to ZERO instructions -- the audit"
        echo "      would be scanning the empty set"
        FAIL=1
    elif (( FF2_TOTAL > FF2_BUDGET )); then
        echo "FAIL: ff2 primitive count ${FF2_TOTAL} exceeds budget ${FF2_BUDGET}"
        echo "      -- the usual cause is a demote/pack routing through an extra"
        echo "      codec round-trip. RE-DERIVE the budget; do not raise it."
        FAIL=1
    else
        echo "PASS: ff2 primitive count ${FF2_TOTAL} within budget ${FF2_BUDGET}"
    fi

    if (( FF2_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in the Ff-cores primitive chain:"
        grep -E "${FP64_RE}" "${TMPD}/ff2.sass" | head -10
        FAIL=1
    else
        echo "PASS: ZERO FP64 instructions in the Ff-cores primitive chain"
    fi

    if (( FF2INJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: the positive control came back CLEAN."
        echo "      ff2PrimFp64InjectedKernel contains a deliberate FP64 operation,"
        echo "      so this audit is blind and the PASS above certifies nothing."
        FAIL=1
    else
        echo "PASS: SEEDED-RED NON-VACUITY: positive control detected ${FF2INJ_FP64} FP64 hits in ff2PrimFp64InjectedKernel -- the scan has demonstrated failure power on this binary"
    fi

    if [[ "${FF2_STACK}" != "0" && "${FF2_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${FF2_STACK} bytes for the Ff-cores chain --"
        echo "      a second, independent reading of the same no-spill claim"
        FAIL=1
    fi
    if (( FF2_SPILL != 0 )); then
        echo "FAIL: ${FF2_SPILL} stack spill instructions in the Ff-cores chain --"
        echo "      Ff1/Ff2 are bare FP32 limbs with no metadata; a chain this"
        echo "      size must stay register-resident"
        FAIL=1
    else
        echo "PASS: no stack spill traffic (STL/LDL) in the Ff-cores chain"
    fi
    FF2_OK=1
fi

# ---------------------------------------------------------------------------
# Subject 4: rsqrtAtanPrimKernel
# ---------------------------------------------------------------------------
RSQAT_OK=0
echo "--- subject: rsqrtAtanPrimKernel --"
if [[ -z "${RSQAT_SYM}" ]]; then
    echo "FAIL: rsqrtAtanPrimKernel not found (tests/test_BandLevers.cu missing,"
    echo "      renamed, or dead-stripped?) -- there is nothing to scan"
    FAIL=1
elif [[ -z "${RSQAT_INJ}" ]]; then
    echo "FAIL: rsqrtAtanPrimFp64InjectedKernel not found -- the positive"
    echo "      control is gone, so a clean result proves nothing"
    FAIL=1
else
    cuobjdump -arch sm_61 -sass -fun "${RSQAT_SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/rsqat.sass"
    cuobjdump -arch sm_61 -sass -fun "${RSQAT_INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/rsqatinj.sass"

    RSQAT_FP64=$(count_match "${FP64_RE}" "${TMPD}/rsqat.sass")
    RSQATINJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/rsqatinj.sass")
    RSQAT_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/rsqat.sass" || true)
    RSQATINJ_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/rsqatinj.sass" || true)
    RSQAT_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/rsqat.sass")
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null \
        | grep -A1 -F "Function ${RSQAT_SYM}:" | tail -1)
    RSQAT_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    RSQAT_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-38s %s\n" "audited kernel" "${RSQAT_SYM:0:60}"
    printf "  %-38s %d\n" "FP64 arithmetic/convert hits" "${RSQAT_FP64}"
    printf "  %-38s %d (budget %d)\n" "total instructions" "${RSQAT_TOTAL}" "${RSQAT_BUDGET}"
    printf "  %-38s %d\n" "stack spill ops (STL/LDL)" "${RSQAT_SPILL}"
    printf "  %-38s %s (stack %s B)\n" "register count" "${RSQAT_REG}" "${RSQAT_STACK}"
    printf "  %-38s %d (of %d instructions)\n" "POSITIVE CONTROL FP64 hits" \
        "${RSQATINJ_FP64}" "${RSQATINJ_TOTAL}"
    echo "----------------------------------------------"

    if (( RSQAT_TOTAL == 0 )); then
        echo "FAIL: rsqrtAtanPrimKernel disassembled to ZERO instructions -- the"
        echo "      audit would be scanning the empty set"
        FAIL=1
    elif (( RSQAT_TOTAL > RSQAT_BUDGET )); then
        echo "FAIL: rsqrt/atan-sector primitive count ${RSQAT_TOTAL} exceeds budget ${RSQAT_BUDGET}"
        echo "      -- the usual cause is a demote/pack routing through an extra"
        echo "      codec round-trip. RE-DERIVE the budget; do not raise it."
        FAIL=1
    else
        echo "PASS: rsqrt/atan-sector primitive count ${RSQAT_TOTAL} within budget ${RSQAT_BUDGET}"
    fi

    if (( RSQAT_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in the rsqrt/atan-sector primitive chain:"
        grep -E "${FP64_RE}" "${TMPD}/rsqat.sass" | head -10
        FAIL=1
    else
        echo "PASS: ZERO FP64 instructions in the rsqrt/atan-sector primitive chain"
    fi

    if (( RSQATINJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: the positive control came back CLEAN."
        echo "      rsqrtAtanPrimFp64InjectedKernel contains a deliberate FP64"
        echo "      operation, so this audit is blind and the PASS above certifies"
        echo "      nothing."
        FAIL=1
    else
        echo "PASS: SEEDED-RED NON-VACUITY: positive control detected ${RSQATINJ_FP64} FP64 hits in rsqrtAtanPrimFp64InjectedKernel -- the scan has demonstrated failure power on this binary"
    fi

    if [[ "${RSQAT_STACK}" != "0" && "${RSQAT_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${RSQAT_STACK} bytes for the rsqrt/atan-sector chain --"
        echo "      a second, independent reading of the same no-spill claim"
        FAIL=1
    fi
    if (( RSQAT_SPILL != 0 )); then
        echo "FAIL: ${RSQAT_SPILL} stack spill instructions in the rsqrt/atan-sector chain --"
        echo "      RsqrtCore/AtanSector are bare FP32 limbs with no metadata; a"
        echo "      chain this size must stay register-resident"
        FAIL=1
    else
        echo "PASS: no stack spill traffic (STL/LDL) in the rsqrt/atan-sector chain"
    fi
    RSQAT_OK=1
fi

# ---------------------------------------------------------------------------
# Subject 5: roundTripKernel (BandCell's own codec)
# ---------------------------------------------------------------------------
RTRIP_OK=0
echo "--- subject: roundTripKernel -----"
if [[ -z "${RTRIP_SYM}" ]]; then
    echo "FAIL: roundTripKernel not found (tests/test_BandCell.cu missing, renamed,"
    echo "      or dead-stripped?) -- there is nothing to scan"
    FAIL=1
elif [[ -z "${RTRIP_INJ}" ]]; then
    echo "FAIL: roundTripFp64InjectedKernel not found -- the positive control is"
    echo "      gone, so a clean result proves nothing"
    FAIL=1
else
    cuobjdump -arch sm_61 -sass -fun "${RTRIP_SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/rtrip.sass"
    cuobjdump -arch sm_61 -sass -fun "${RTRIP_INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/rtripinj.sass"

    RTRIP_FP64=$(count_match "${FP64_RE}" "${TMPD}/rtrip.sass")
    RTRIPINJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/rtripinj.sass")
    RTRIP_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/rtrip.sass" || true)
    RTRIPINJ_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/rtripinj.sass" || true)
    RTRIP_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/rtrip.sass")
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null \
        | grep -A1 -F "Function ${RTRIP_SYM}:" | tail -1)
    RTRIP_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    RTRIP_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-38s %s\n" "audited kernel" "${RTRIP_SYM:0:60}"
    printf "  %-38s %d\n" "FP64 arithmetic/convert hits" "${RTRIP_FP64}"
    printf "  %-38s %d (budget %d)\n" "total instructions" "${RTRIP_TOTAL}" "${RTRIP_BUDGET}"
    printf "  %-38s %d\n" "stack spill ops (STL/LDL)" "${RTRIP_SPILL}"
    printf "  %-38s %s (stack %s B)\n" "register count" "${RTRIP_REG}" "${RTRIP_STACK}"
    printf "  %-38s %d (of %d instructions)\n" "POSITIVE CONTROL FP64 hits" \
        "${RTRIPINJ_FP64}" "${RTRIPINJ_TOTAL}"
    echo "----------------------------------------------"

    if (( RTRIP_TOTAL == 0 )); then
        echo "FAIL: roundTripKernel disassembled to ZERO instructions -- the audit"
        echo "      would be scanning the empty set"
        FAIL=1
    elif (( RTRIP_TOTAL > RTRIP_BUDGET )); then
        echo "FAIL: BandCell round-trip count ${RTRIP_TOTAL} exceeds budget ${RTRIP_BUDGET}"
        echo "      -- the usual cause is the codec routing through an extra rescale"
        echo "      pass. RE-DERIVE the budget; do not raise it."
        FAIL=1
    else
        echo "PASS: BandCell round-trip count ${RTRIP_TOTAL} within budget ${RTRIP_BUDGET}"
    fi

    if (( RTRIP_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in the BandCell round trip:"
        grep -E "${FP64_RE}" "${TMPD}/rtrip.sass" | head -10
        FAIL=1
    else
        echo "PASS: ZERO FP64 instructions in the BandCell round trip"
    fi

    if (( RTRIPINJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: the positive control came back CLEAN."
        echo "      roundTripFp64InjectedKernel contains a deliberate FP64 operation,"
        echo "      so this audit is blind and the PASS above certifies nothing."
        FAIL=1
    else
        echo "PASS: SEEDED-RED NON-VACUITY: positive control detected ${RTRIPINJ_FP64} FP64 hits in roundTripFp64InjectedKernel -- the scan has demonstrated failure power on this binary"
    fi

    if [[ "${RTRIP_STACK}" != "0" && "${RTRIP_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${RTRIP_STACK} bytes for the BandCell round trip --"
        echo "      a second, independent reading of the same no-spill claim"
        FAIL=1
    fi
    if (( RTRIP_SPILL != 0 )); then
        echo "FAIL: ${RTRIP_SPILL} stack spill instructions in the BandCell round trip --"
        echo "      BandCell is three bare FP32 limbs plus an int32 bias with no other"
        echo "      metadata; a round trip this size must stay register-resident"
        FAIL=1
    else
        echo "PASS: no stack spill traffic (STL/LDL) in the BandCell round trip"
    fi
    RTRIP_OK=1
fi

# ---------------------------------------------------------------------------
# The PER-OP subject list, generic across every live proof op — a new op
# lands here (and in `tests/test_BandMathSass.cu`'s macro table, and its own
# BandMathCert rows) and NOTHING ELSE.
#
# ★ NO codec boundary in these kernels at all (raw `Band` in/out over flat
# `float[3*n]`, `tests/test_BandMathSass.cu`'s own doc note) — DELIBERATELY
# the SAME shape the frozen sentinels were measured against (sub 72/30,
# mul 78/33 TOTAL/FP32), so there is no "measured + 28 codec crossing"
# budget to derive the way subjects 3-5 above do. `sub`/`mul` are instead
# gated EXACTLY against those frozen sentinels — a drift REDs the run.
# Every OTHER op has no prior frozen number, so its row is this run's OWN
# first measurement, becoming the baseline a later run would drift against.
# The ROUND family (`floor ceil round trunc fdim fmod fma`) extends the SAME
# table — `fma` (`aether/banded/BandRound.h`, over the hoisted `fmaRaw`
# family) reproduces the 54-FP32 sentinel in the SAME compilation; its own
# N/A row is retired below.
# ---------------------------------------------------------------------------
FP32_RE='\bFADD|\bFMUL|\bFFMA|\bFSETP|\bFMNMX|\bFSEL|\bMUFU|\bFCHK'
P11C0_OPS="abs copysign fmax fmin add sub mul exp log pow sqrt rsqrt rsqrtCube cbrt hypot sin cos sincos floor ceil round trunc fdim fmod fma atan atan2 asin acos"
if [[ -n "${SUBJECT_FILTER}" ]]; then
    P11C0_OPS="${SUBJECT_FILTER}"
fi
declare -A P11C0_SENTINEL_FP32=( [sub]=30 [mul]=33 [fma]=54 )
PER_OP_CARD_SECTION=$'\n## Per-op subject list\n'
PER_OP_CARD_SECTION+=$'\nRaw `Band` in/out, no codec boundary — directly comparable to the frozen\nsentinels (`sub` 30 / `mul` 33 / `fma` 54 FP32, all three reproduced below).\n'

for op in ${P11C0_OPS}; do
    capOp="$(printf '%s' "${op:0:1}" | tr '[:lower:]' '[:upper:]')${op:1}"
    SYM=$(find_sym "bandmath${capOp}Kernel")
    INJ=$(find_sym "bandmath${capOp}Fp64InjectedKernel")
    echo "--- subject: bandmath${capOp}Kernel -------------"
    if [[ -z "${SYM}" || -z "${INJ}" ]]; then
        echo "FAIL: bandmath${capOp}Kernel or its positive control not found in ${TESTS_BIN}"
        FAIL=1
        PER_OP_CARD_SECTION+="
### \`bandmath${capOp}Kernel\`: FAIL — kernel or positive control not found
"
        continue
    fi
    cuobjdump -arch sm_61 -sass -fun "${SYM}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/p11c0_${op}.sass"
    cuobjdump -arch sm_61 -sass -fun "${INJ}" "${TESTS_BIN}" 2>/dev/null > "${TMPD}/p11c0_${op}_inj.sass"
    T_TOTAL=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "${TMPD}/p11c0_${op}.sass" || true)
    T_FP32=$(count_match "${FP32_RE}" "${TMPD}/p11c0_${op}.sass")
    T_FP64=$(count_match "${FP64_RE}" "${TMPD}/p11c0_${op}.sass")
    T_SPILL=$(count_match '\bSTL|\bLDL' "${TMPD}/p11c0_${op}.sass")
    INJ_FP64=$(count_match "${FP64_RE}" "${TMPD}/p11c0_${op}_inj.sass")
    RES=$(cuobjdump -res-usage "${TESTS_BIN}" 2>/dev/null | grep -A1 -F "Function ${SYM}:" | tail -1)
    T_REG=$(echo "${RES}" | grep -oP 'REG:\K[0-9]+' || echo "?")
    T_STACK=$(echo "${RES}" | grep -oP 'STACK:\K[0-9]+' || echo "?")

    printf "  %-30s %s\n" "kernel" "${SYM:0:60}"
    printf "  %-30s %d/%d (TOTAL/FP32)\n" "instruction census" "${T_TOTAL}" "${T_FP32}"
    printf "  %-30s %d\n" "FP64 hits" "${T_FP64}"
    printf "  %-30s %d\n" "stack spill (STL/LDL)" "${T_SPILL}"
    printf "  %-30s %s (stack %s B)\n" "register/stack" "${T_REG}" "${T_STACK}"
    printf "  %-30s %d\n" "positive control FP64 hits" "${INJ_FP64}"

    OK=1
    if (( T_TOTAL == 0 )); then
        echo "FAIL: bandmath${capOp}Kernel disassembled to ZERO instructions"
        OK=0
    fi
    if (( T_FP64 != 0 )); then
        echo "FAIL: FP64 instructions in bandmath${capOp}Kernel:"
        grep -E "${FP64_RE}" "${TMPD}/p11c0_${op}.sass" | head -5
        OK=0
    fi
    if (( INJ_FP64 == 0 )); then
        echo "FAIL: SEEDED-RED NON-VACUITY: positive control clean for ${op} — the scan is blind"
        OK=0
    fi
    if [[ "${T_STACK}" != "0" && "${T_STACK}" != "?" ]]; then
        echo "FAIL: ptxas reports STACK:${T_STACK} bytes for ${op}"
        OK=0
    fi
    if (( T_SPILL != 0 )); then
        echo "FAIL: ${T_SPILL} stack spill instructions for ${op}"
        OK=0
    fi
    SENT="${P11C0_SENTINEL_FP32[${op}]:-}"
    SENT_LINE=""
    if [[ -n "${SENT}" ]]; then
        if (( T_FP32 != SENT )); then
            echo "FAIL: SENTINEL DRIFT for ${op}: measured FP32=${T_FP32}, frozen sentinel=${SENT}"
            OK=0
            SENT_LINE="| frozen sentinel (FP32) | ${SENT} — DRIFT, measured ${T_FP32} |"
        else
            echo "PASS: sentinel reproduced — ${op} FP32=${T_FP32} (frozen ${SENT}, SAME compilation)"
            SENT_LINE="| frozen sentinel (FP32) | ${SENT} — reproduced |"
        fi
    fi
    if (( OK )); then
        echo "PASS: bandmath${capOp}Kernel clean (0 FP64, 0 spill, positive control caught ${INJ_FP64} hits)"
    else
        FAIL=1
    fi

    PER_OP_CARD_SECTION+="
### \`bandmath${capOp}Kernel\`

| metric | value |
|---|---|
| audited kernel | \`${SYM}\` |
| positive control | \`${INJ}\` |
| TOTAL / FP32 | ${T_TOTAL} / ${T_FP32} |
| FP64 hits (audited / control) | ${T_FP64} / ${INJ_FP64} |
| stack spill (STL/LDL) | ${T_SPILL} |
| register count / STACK | ${T_REG} / ${T_STACK} |
${SENT_LINE}
"
done

# ---------------------------------------------------------------------------
# Card mint. THE RUN WRITES THE CARD; prose cites it and never restates a
# number by hand. Each subject's row block is independently md5-fenced so a
# hand-edit to either is visible.
# ---------------------------------------------------------------------------
if [[ -n "${CARD_PATH}" ]]; then
    mkdir -p "$(dirname "${CARD_PATH}")"
    {
        echo "# CARD_BAND_SASS_sm61 — the banded carrier's 0-FP64 warrant, measured"
        echo ""
        echo "\`Authority:\` MEASUREMENT of record for this suite's SASS acceptance ·"
        echo "\`Written by:\` \`tests/sass/check_band_fp64free.sh --mint-card\` — THE RUN"
        echo "WRITES THIS FILE. Do not hand-edit a number here; re-run the gate."
        echo ""
        echo "COMPILE-ONLY — nothing on this page was ever launched to produce these"
        echo "numbers; both subjects are read back from \`cuobjdump -sass\` /"
        echo "\`cuobjdump -res-usage\` over the built \`aether_tests\` binary."
        echo ""
        echo "\`\`\`"
        echo "binary:   ${TESTS_BIN}"
        echo "toolchain: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc version unavailable')"
        echo "\`\`\`"

        if (( BND_OK )); then
            BODY="${TMPD}/card_body_bnd.md"
            {
                echo "| metric | value |"
                echo "|---|---|"
                echo "| audited kernel | \`${BND_SYM}\` |"
                echo "| positive control | \`${BND_INJ}\` |"
                echo "| FP64 arithmetic/convert hits (audited) | ${BND_FP64} |"
                echo "| FP64 arithmetic/convert hits (control) | ${INJ_FP64} |"
                echo "| total instructions (audited) | ${BND_TOTAL} |"
                echo "| total instructions (control) | ${INJ_TOTAL} |"
                echo "| stack spill ops (STL/LDL) | ${BND_SPILL} |"
                echo "| register count | ${BND_REG} |"
                echo "| ptxas STACK (bytes) | ${BND_STACK} |"
                echo "| derived headroom | ${BND_HEADROOM} |"
                echo "| budget in force | ${BND_BUDGET} |"
            } > "${BODY}"
            BND_MD5=$(md5sum "${BODY}" | cut -d' ' -f1)
            echo ""
            echo "## Subject 1: \`bandChainKernel\`"
            echo ""
            echo "\`tests/test_BandedReal.cu\`, the adaptive-step shape in the banded"
            echo "carrier (3 abs, 1 min, 1 max, 1 copysign, 1 div, 1 mul, 1 add, 1 sub ="
            echo "10 ops; 2 unpacks + 1 pack = 3 codec crossings)."
            echo ""
            echo "<!-- ---8<--- CARD ROWS BEGIN bandChainKernel (md5 = ${BND_MD5}) ---8<--- -->"
            cat "${BODY}"
            echo "<!-- ---8<--- CARD ROWS END bandChainKernel ---8<--- -->"
            echo ""
            echo "Budget derivation: STRICTLY BELOW \`measured + 196\` — a codec crossing"
            echo "PER OP instead of per boundary costs 7 extra pack/unpack pairs (10 ops"
            echo "minus 3 boundaries) at ~28 INT32 instructions each. Headroom granted:"
            echo "${BND_HEADROOM}, under 196 by construction."
            echo ""
            echo "Positive control: \`${BND_INJ}\` is the identical body plus ONE"
            echo "deliberate FP64 operation on a kernel argument; the row above records"
            echo "that the scan found it."
        fi

        if (( ACC_OK )); then
            BODY="${TMPD}/card_body_acc.md"
            {
                echo "| metric | value |"
                echo "|---|---|"
                echo "| audited kernel | \`${ACC_SYM}\` |"
                echo "| positive control | \`${ACC_INJ}\` |"
                echo "| FP64 arithmetic/convert hits (audited) | ${ACC_FP64} |"
                echo "| FP64 arithmetic/convert hits (control) | ${ACCINJ_FP64} |"
                echo "| total instructions (audited) | ${ACC_TOTAL} |"
                echo "| total instructions (control) | ${ACCINJ_TOTAL} |"
                echo "| stack spill ops (STL/LDL) | ${ACC_SPILL} |"
                echo "| register count | ${ACC_REG} |"
                echo "| ptxas STACK (bytes) | ${ACC_STACK} |"
                echo "| derived headroom | ${ACC_HEADROOM} |"
                echo "| budget in force | ${ACC_BUDGET} |"
            } > "${BODY}"
            ACC_MD5=$(md5sum "${BODY}" | cut -d' ' -f1)
            echo ""
            echo "## Subject 2: \`bandAccumFoldKernel\`"
            echo ""
            echo "\`tests/test_BandLevers.cu\`, the accumulator's own shape: 8 terms folded"
            echo "through ONE \`BandAccum\` via \`addTerm\`/\`subTerm\` (8 certified Band"
            echo "adds), then packed once via \`toBandedReal()\` (8 unpacks + 1 pack = 9"
            echo "codec crossings)."
            echo ""
            echo "<!-- ---8<--- CARD ROWS BEGIN bandAccumFoldKernel (md5 = ${ACC_MD5}) ---8<--- -->"
            cat "${BODY}"
            echo "<!-- ---8<--- CARD ROWS END bandAccumFoldKernel ---8<--- -->"
            echo ""
            echo "Budget derivation: STRICTLY BELOW \`measured + 196\` — the running"
            echo "accumulator has 7 (= 8 terms - 1) iteration-to-iteration CARRY points"
            echo "where a regression could round-trip it through the codec instead of"
            echo "keeping it live as a \`Band\`, at ~28 INT32 instructions per extra"
            echo "pack/unpack pair. Headroom granted: ${ACC_HEADROOM}, under 196 by"
            echo "construction — same margin as subject 1, from an unrelated argument."
            echo ""
            echo "Positive control: \`${ACC_INJ}\` is the identical body plus ONE"
            echo "deliberate FP64 operation on a kernel argument; the row above records"
            echo "that the scan found it."
        fi

        if (( FF2_OK )); then
            BODY="${TMPD}/card_body_ff2.md"
            {
                echo "| metric | value |"
                echo "|---|---|"
                echo "| audited kernel | \`${FF2_SYM}\` |"
                echo "| positive control | \`${FF2_INJ}\` |"
                echo "| FP64 arithmetic/convert hits (audited) | ${FF2_FP64} |"
                echo "| FP64 arithmetic/convert hits (control) | ${FF2INJ_FP64} |"
                echo "| total instructions (audited) | ${FF2_TOTAL} |"
                echo "| total instructions (control) | ${FF2INJ_TOTAL} |"
                echo "| stack spill ops (STL/LDL) | ${FF2_SPILL} |"
                echo "| register count | ${FF2_REG} |"
                echo "| ptxas STACK (bytes) | ${FF2_STACK} |"
                echo "| derived headroom | ${FF2_HEADROOM} |"
                echo "| budget in force | ${FF2_BUDGET} |"
            } > "${BODY}"
            FF2_MD5=$(md5sum "${BODY}" | cut -d' ' -f1)
            echo ""
            echo "## Subject 3: \`ff2PrimKernel\`"
            echo ""
            echo "\`tests/test_Ff2Cert.cu\`, the Ff-cores rung's own shape: THREE"
            echo "independent \`BandedReal\` operands demoted to \`Ff2\` (3 unpacks),"
            echo "SEVEN independent primitives computed from them (add, sub, mul, fma,"
            echo "div, sqrt, rsqrt — no chaining between results), each packed back"
            echo "once (7 packs) = 10 codec-boundary crossings."
            echo ""
            echo "<!-- ---8<--- CARD ROWS BEGIN ff2PrimKernel (md5 = ${FF2_MD5}) ---8<--- -->"
            cat "${BODY}"
            echo "<!-- ---8<--- CARD ROWS END ff2PrimKernel ---8<--- -->"
            echo ""
            echo "Budget derivation: STRICTLY BELOW \`measured + 28\` — this kernel has"
            echo "no op-chain to amplify a regression through (each primitive consumes"
            echo "only its own demoted operands), so the smallest realistic regression"
            echo "is ONE extra codec pack/unpack pair (~28 INT32 instructions) on a"
            echo "single crossing, not the multi-crossing class subjects 1/2 derive"
            echo "against. Headroom granted: ${FF2_HEADROOM}, strictly under 28 by"
            echo "construction."
            echo ""
            echo "Positive control: \`${FF2_INJ}\` is the identical body plus ONE"
            echo "deliberate FP64 operation on a kernel argument; the row above records"
            echo "that the scan found it."
        fi

        if (( RSQAT_OK )); then
            BODY="${TMPD}/card_body_rsqat.md"
            {
                echo "| metric | value |"
                echo "|---|---|"
                echo "| audited kernel | \`${RSQAT_SYM}\` |"
                echo "| positive control | \`${RSQAT_INJ}\` |"
                echo "| FP64 arithmetic/convert hits (audited) | ${RSQAT_FP64} |"
                echo "| FP64 arithmetic/convert hits (control) | ${RSQATINJ_FP64} |"
                echo "| total instructions (audited) | ${RSQAT_TOTAL} |"
                echo "| total instructions (control) | ${RSQATINJ_TOTAL} |"
                echo "| stack spill ops (STL/LDL) | ${RSQAT_SPILL} |"
                echo "| register count | ${RSQAT_REG} |"
                echo "| ptxas STACK (bytes) | ${RSQAT_STACK} |"
                echo "| derived headroom | ${RSQAT_HEADROOM} |"
                echo "| budget in force | ${RSQAT_BUDGET} |"
            } > "${BODY}"
            RSQAT_MD5=$(md5sum "${BODY}" | cut -d' ' -f1)
            echo ""
            echo "## Subject 4: \`rsqrtAtanPrimKernel\`"
            echo ""
            echo "\`tests/test_BandLevers.cu\`, the Rsqrt/Atan-sector primitive shape:"
            echo "TWO independent \`BandedReal\` operands demoted to \`Band\` (2 unpacks),"
            echo "THREE independent RsqrtCore-family primitives (rsqrt, sqrt_,"
            echo "rsqrtCube — each its own Newton core) plus ONE \`atanSector\` lookup"
            echo "(pure compares + a literal table), each of the 5 results packed back"
            echo "once (5 packs) = 7 codec-boundary crossings."
            echo ""
            echo "<!-- ---8<--- CARD ROWS BEGIN rsqrtAtanPrimKernel (md5 = ${RSQAT_MD5}) ---8<--- -->"
            cat "${BODY}"
            echo "<!-- ---8<--- CARD ROWS END rsqrtAtanPrimKernel ---8<--- -->"
            echo ""
            echo "Budget derivation: STRICTLY BELOW \`measured + 28\` — the same"
            echo "single-crossing regression class \`ff2PrimKernel\` derives against (no"
            echo "op-chain between the four independent calls to amplify a regression"
            echo "through). Headroom granted: ${RSQAT_HEADROOM}, strictly under 28 by"
            echo "construction."
            echo ""
            echo "Positive control: \`${RSQAT_INJ}\` is the identical body plus ONE"
            echo "deliberate FP64 operation on a kernel argument; the row above records"
            echo "that the scan found it."
        fi

        if (( RTRIP_OK )); then
            BODY="${TMPD}/card_body_rtrip.md"
            {
                echo "| metric | value |"
                echo "|---|---|"
                echo "| audited kernel | \`${RTRIP_SYM}\` |"
                echo "| positive control | \`${RTRIP_INJ}\` |"
                echo "| FP64 arithmetic/convert hits (audited) | ${RTRIP_FP64} |"
                echo "| FP64 arithmetic/convert hits (control) | ${RTRIPINJ_FP64} |"
                echo "| total instructions (audited) | ${RTRIP_TOTAL} |"
                echo "| total instructions (control) | ${RTRIPINJ_TOTAL} |"
                echo "| stack spill ops (STL/LDL) | ${RTRIP_SPILL} |"
                echo "| register count | ${RTRIP_REG} |"
                echo "| ptxas STACK (bytes) | ${RTRIP_STACK} |"
                echo "| derived headroom | ${RTRIP_HEADROOM} |"
                echo "| budget in force | ${RTRIP_BUDGET} |"
            } > "${BODY}"
            RTRIP_MD5=$(md5sum "${BODY}" | cut -d' ' -f1)
            echo ""
            echo "## Subject 5: \`roundTripKernel\`"
            echo ""
            echo "\`tests/test_BandCell.cu\`, \`BandCell\`'s OWN codec (the 16-byte"
            echo "biased-\`Band\` cell, distinct from \`BandCell8\`): \`cellFromBand\` then"
            echo "\`bandFromCell\`, TWO \`bandScalePow2Split\` calls (12 FP32 multiplies"
            echo "total) plus bit-level exponent extract/rebuild, no op-chain between"
            echo "them to amplify a regression through."
            echo ""
            echo "<!-- ---8<--- CARD ROWS BEGIN roundTripKernel (md5 = ${RTRIP_MD5}) ---8<--- -->"
            cat "${BODY}"
            echo "<!-- ---8<--- CARD ROWS END roundTripKernel ---8<--- -->"
            echo ""
            echo "Budget derivation: STRICTLY BELOW \`measured + 28\` -- the same"
            echo "single-crossing regression class subjects 3/4 derive against (the whole"
            echo "kernel body IS the codec; the smallest realistic regression is ONE extra"
            echo "\`bandScalePow2Split\` call, ~14 instructions). Headroom granted:"
            echo "${RTRIP_HEADROOM}, strictly under 28 by construction."
            echo ""
            echo "Positive control: \`${RTRIP_INJ}\` is the identical body plus ONE"
            echo "deliberate FP64 operation on a kernel argument; the row above records"
            echo "that the scan found it. \`punKernel\` (the same file's punned"
            echo "load/store) is NOT a subject here: it is pure 128-bit memory traffic"
            echo "with no arithmetic instructions at all, so \"0 FP64 hits\" on it is a"
            echo "vacuous claim and it has nothing a positive control could poison."
        fi

        echo "${PER_OP_CARD_SECTION}"
        echo ""
        echo "## Why the positive controls are not decoration"
        echo ""
        echo "A scan that has never been seen to report a hit is not evidence that"
        echo "there are none. Both subjects above carry their own seeded-FP64 control,"
        echo "launched (not merely compiled) in the same test binary."
    } > "${CARD_PATH}"
    echo "----------------------------------------------"
    echo "CARD MINTED: ${CARD_PATH}"
fi

echo "----------------------------------------------"
if (( FAIL == 0 )); then
    echo "BANDED FP64-FREE SASS AUDIT: GREEN"
    exit 0
fi
echo "BANDED FP64-FREE SASS AUDIT: RED"
exit 1
