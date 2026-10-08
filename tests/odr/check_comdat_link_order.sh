#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# ===========================================================================
#  ODR / link-order invariance gate.
#
#  WHAT THIS CATCHES: `aether::detail::packetGet<...>`
#  (aether/backend/cpu/packet/LoadStore.h, the packet-lane materializer
#  `packetEval`/`packetFor` call into) is an `inline` template function
#  defined in a header. Every TU that instantiates one emits its OWN weak
#  (COMDAT) copy, and the linker keeps whichever copy appears FIRST on the
#  link line. The standard permits those copies to be different MACHINE
#  CODE (the inliner's decisions for a COMDAT body depend on the whole TU)
#  and requires only that they compute the same VALUES. A per-TU codegen
#  difference (FP-contraction is the confirmed case below, not a
#  hypothetical one) could make that false with no source change at all —
#  this gate exists so that class of regression cannot come back silently.
#
#  TWO STAGES, because aether's CURRENT test-suite TUs carry NO such
#  divergence (measured: every symbol shared between a normal and a fully-
#  reversed aether_tests relink is byte-identical, since
#  test_PacketExpr.cpp/test_Bundle.cpp — the only TUs presently
#  instantiating this shape — both already carry `-ffp-contract=off`):
#
#  STAGE A — INSTRUMENT CAPABILITY (must swap, hard-fail if not): two
#  throwaway TUs, `comdatProbeAxpy3` (the SAME axpy3 `a + k*b` shape those
#  two files use), ONE compiled with `-ffp-contract=off` and ONE without.
#  Confirms `comdat_copy_diff.py` — and this toolchain, on this machine —
#  CAN see a real divergence when one exists, independent of whether
#  aether_tests' own TUs currently contain one.
#
#  STAGE B — THE REAL GATE: re-link aether_tests with the FULL object list
#  REVERSED, report `comdat_copy_diff.py`'s swap count as INFORMATIONAL
#  (Stage A already proved the tool is not blind), and require the whole
#  suite to run the SAME test SET and PASS IDENTICALLY under both link
#  orders — that comparison is unconditional regardless of whether
#  anything swapped. Stage B reverses the WHOLE object list rather than
#  moving a single known-culprit TU, since aether has no such TU yet — a
#  general perturbation that does not depend on knowing which TU would
#  matter.
#
#  SCOPE: cpp-mode ONLY. Executing this gate means RUNNING the built
#  binary (twice, Stage B) — the house rule against GPU execution never
#  lets this companion run a CUDA-mode aether_tests, so this script only
#  ever targets a cpp-mode (AETHER_CPP_MODE) build tree.
#
#  Usage: check_comdat_link_order.sh <cpp_build_tree>
# ===========================================================================
set -u

BUILD_TREE="${1:-}"
if [ -z "${BUILD_TREE}" ] || [ ! -d "${BUILD_TREE}" ]; then
    echo "usage: $0 <cpp_build_tree>" >&2
    exit 2
fi
BUILD_TREE=$(cd "${BUILD_TREE}" && pwd)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${HERE}/../.." && pwd)

LINKDIR="${BUILD_TREE}/tests/CMakeFiles/aether_tests.dir"
LINKTXT="${LINKDIR}/link.txt"
NORMAL_BIN="${BUILD_TREE}/tests/aether_tests"

fail() { echo "$*" >&2; echo "COMDAT LINK-ORDER GATE: RED" >&2; exit 1; }

[ -x "${NORMAL_BIN}" ] || fail "FAIL: no test binary at ${NORMAL_BIN}"
[ -f "${LINKTXT}" ] || fail "FAIL: no link.txt at ${LINKTXT} — this gate needs the build tree's link command"

CXX="${CXX:-g++}"
command -v "${CXX}" > /dev/null 2>&1 || fail "FAIL: compiler '${CXX}' not found"

TMPD=$(mktemp -d)
trap 'rm -rf "${TMPD}"' EXIT

echo "======================================================================"
echo " COMDAT link-order invariance (ODR weak-symbol selection)"
echo " build tree : ${BUILD_TREE}"
echo "======================================================================"

# ---------------------------------------------------------------------------
# STAGE A — instrument capability probe.
# ---------------------------------------------------------------------------
echo "-- STAGE A: instrument capability (deliberate FP-contraction probe) --"
cat > "${TMPD}/probe.h" <<EOF
#pragma once
#include <aether/aether.h>
// The exact axpy3 shape tests/test_PacketExpr.cpp/test_Bundle.cpp already
// found FP-contraction-sensitive (their own docstrings, tests/CMakeLists.txt's
// -ffp-contract=off/-fmad=false for those two files ONLY).
inline void comdatProbeAxpy3(aether::Vec3dView out, aether::Vec3dView a, aether::Vec3dView b, double k)
{
    aether::packetEval(out, a + k * b);
}
EOF
cat > "${TMPD}/probe_fast.cpp" <<EOF
#include "probe.h"
void comdatProbeForceEmitFast()
{
    aether::Vec3dView out, a, b;
    comdatProbeAxpy3(out, a, b, 1.5);
}
EOF
cat > "${TMPD}/probe_off.cpp" <<EOF
#include "probe.h"
void comdatProbeForceEmitOff()
{
    aether::Vec3dView out, a, b;
    comdatProbeAxpy3(out, a, b, 1.5);
}
EOF

CXXFLAGS_COMMON=(-std=c++23 -DAETHER_CPP_MODE=1 -O3 -march=x86-64-v3 -fopenmp -I"${REPO_ROOT}")
( cd "${TMPD}" && "${CXX}" "${CXXFLAGS_COMMON[@]}" -c probe_fast.cpp -o probe_fast.o ) \
    > "${TMPD}/probe_fast.log" 2>&1
FAST_RC=$?
( cd "${TMPD}" && "${CXX}" "${CXXFLAGS_COMMON[@]}" -ffp-contract=off -c probe_off.cpp -o probe_off.o ) \
    > "${TMPD}/probe_off.log" 2>&1
OFF_RC=$?
if [ "${FAST_RC}" -ne 0 ] || [ "${OFF_RC}" -ne 0 ]; then
    cat "${TMPD}/probe_fast.log" "${TMPD}/probe_off.log" >&2
    fail "FAIL: the instrument-capability probe itself failed to compile (rc=${FAST_RC}/${OFF_RC})"
fi

python3 "${HERE}/comdat_copy_diff.py" "${TMPD}/probe_fast.o" "${TMPD}/probe_off.o" "packetGet" \
    > "${TMPD}/probe_diff.log" 2>&1
PROBE_DIFF_RC=$?
cat "${TMPD}/probe_diff.log"
if [ "${PROBE_DIFF_RC}" -ne 0 ]; then
    fail "FAIL: comdat_copy_diff.py failed on the probe objects (rc=${PROBE_DIFF_RC})"
fi
PROBE_SWAPPED=$(sed -n 's/^SWAPPED_COUNT=//p' "${TMPD}/probe_diff.log")
if [ -z "${PROBE_SWAPPED}" ] || [ "${PROBE_SWAPPED}" -eq 0 ]; then
    fail "FAIL: the DELIBERATE fast-vs-off probe pair produced IDENTICAL code for packetGet — the instrument (comdat_copy_diff.py, or this toolchain/flags) cannot detect a divergence it is KNOWN to contain. A gate that cannot see its own positive control cannot be trusted on the real object set either."
fi
echo "  instrument capability CONFIRMED: ${PROBE_SWAPPED} packetGet body(ies) genuinely differ under -ffp-contract=fast vs =off"

# ---------------------------------------------------------------------------
# STAGE B — the real gate: full object-list reversal of aether_tests.
# ---------------------------------------------------------------------------
echo "-- STAGE B: aether_tests full-reversal relink --"
LINE=$(cat "${LINKTXT}")
OBJS=$(tr ' ' '\n' <<<"${LINE}" | grep '\.o$')
N_OBJS=$(wc -l <<<"${OBJS}")
if [ "${N_OBJS}" -lt 2 ]; then
    fail "FAIL: only ${N_OBJS} object(s) on the link line — nothing to re-order, so a green would certify the empty set."
fi

NEWOBJS_FLAT=$(tac <<<"${OBJS}" | tr '\n' ' ')
PREFIX=$(sed 's| CMakeFiles/aether_tests.dir/[^ ]*\.o.*||' <<<"${LINE}")
SUFFIX=$(sed 's|.*-o aether_tests ||' <<<"${LINE}")
REV_BIN="${TMPD}/aether_tests_reversed"

( cd "${BUILD_TREE}/tests" && eval "${PREFIX} ${NEWOBJS_FLAT} -o ${REV_BIN} ${SUFFIX}" ) \
    > "${TMPD}/link.log" 2>&1
LINK_RC=$?
if [ "${LINK_RC}" -ne 0 ]; then
    sed -n '1,40p' "${TMPD}/link.log" >&2
    fail "FAIL: re-link with the object list reversed failed (rc=${LINK_RC})"
fi
echo "  re-linked with ${N_OBJS} objects in REVERSED order   ok"

echo "  swap count on the REAL objects (informational — Stage A already proved"
echo "  the tool is not blind; a 0 here means aether's own TUs presently agree,"
echo "  not that the check is vacuous):"
python3 "${HERE}/comdat_copy_diff.py" "${NORMAL_BIN}" "${REV_BIN}" "packetEval|packetFor|packetGet" \
    2>&1 | sed 's/^/  /'

echo "----------------------------------------------------------------------"
echo " running the suite against the reversed selection"
run_suite() {
    env -u GTEST_FILTER -u GTEST_ALSO_RUN_DISABLED_TESTS \
        -u GTEST_REPEAT -u GTEST_SHUFFLE -u GTEST_BREAK_ON_FAILURE \
        "$1" > "$2" 2>&1
    echo $?
}
NORM_RC=$(run_suite "${NORMAL_BIN}" "${TMPD}/normal.log")
REV_RC=$(run_suite "${REV_BIN}" "${TMPD}/reversed.log")

count_ran()    { sed -n 's/^\[==========\] \([0-9]*\) tests from .* ran.*/\1/p' "$1" | tail -1; }
count_passed() { sed -n 's/^\[  PASSED  \] \([0-9]*\) tests\?\..*/\1/p' "$1" | tail -1; }

N_RAN=$(count_ran "${TMPD}/normal.log");   N_PASS=$(count_passed "${TMPD}/normal.log")
R_RAN=$(count_ran "${TMPD}/reversed.log"); R_PASS=$(count_passed "${TMPD}/reversed.log")

printf "  %-30s rc=%s ran=%s passed=%s\n" "normal link order"     "${NORM_RC}" "${N_RAN:-?}" "${N_PASS:-?}"
printf "  %-30s rc=%s ran=%s passed=%s\n" "reversed link order"   "${REV_RC}"  "${R_RAN:-?}" "${R_PASS:-?}"

if [ -z "${N_RAN}" ] || [ -z "${R_RAN}" ]; then
    sed -n '1,40p' "${TMPD}/reversed.log" >&2
    fail "FAIL: could not read a ran-count from a suite run — the runner did not report, so no verdict is available."
fi
if [ "${N_RAN}" -eq 0 ] || [ "${R_RAN}" -eq 0 ]; then
    fail "FAIL: a run executed ZERO tests. That is the #102 vacuity, not a pass."
fi
if [ "${N_RAN}" != "${R_RAN}" ]; then
    fail "FAIL: the two links ran different test SETS (${N_RAN} vs ${R_RAN}). They are the same objects; a different set means the comparison below is between two different things."
fi
if [ "${NORM_RC}" -ne 0 ]; then
    grep -E '^\[  FAILED  \]' "${TMPD}/normal.log" | sort -u >&2
    fail "FAIL: the NORMAL link is already red (rc=${NORM_RC}). Fix that first — this gate compares two links and cannot speak while the baseline is broken."
fi
if [ "${REV_RC}" -ne 0 ] || [ "${R_PASS}" != "${N_PASS}" ]; then
    echo "" >&2
    echo "rows that fail ONLY when a different TU's copy is selected:" >&2
    grep -E '^\[  FAILED  \]' "${TMPD}/reversed.log" | sort -u >&2
    fail "FAIL: reversed link rc=${REV_RC}, passed ${R_PASS} vs ${N_PASS}"
fi

echo "----------------------------------------------------------------------"
echo "  every shared packetEval/packetFor/packetGet body computes the same values in every TU"
echo "COMDAT LINK-ORDER GATE: GREEN"
