#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile-fail gate for the banded chain states (`aether/banded/Chain.h`): an
# unrestored value (`BandRawT<k>`, or a cancel-capable `BandRawCancel`/`BandRawCancelN`) must
# not reach a consumer that reads the limbs in order without passing through
# `normalize`/`normalizeSafe`. The implicit normalising bridges to `Band` are
# positive rows in the gtest suite, not arms here. g++ only, CPP_MODE, no GPU.
#
# Every FAIL arm must be rejected AND name the right reason (RC alone is never
# the verdict); the POSITIVE arm proves the sanctioned spellings compile, so a
# broken header cannot satisfy the FAIL arms by failing everywhere.
#
#   ARM CHAIN-ABS              `abs(a * a)`  names the fix
#   ARM CHAIN-FMAX             `fmax(a + a, a)` names the fix
#   ARM CHAIN-CANCEL-TO-RAW    a cancel-capable sum is accepted by `mulRaw` NEVER
#   ARM CHAIN-FMA-RAW-ADDEND   `fma(a * a, a, a * a)` names the fix (the fused-hop bound needs a normalised addend)
#   ARM CHAIN-COMPARE          `a * a == a`
#   ARM CHAIN-COPYSIGN/FMIN/FACADE-ABS
#                              the same unrestored values rejected by each of those consumers
#   ARM CHAIN-POSITIVE         normalize / normalizeSafe / BandedReal compile
set -uo pipefail

REPO_ROOT="${1:-${AETHER_REPO_ROOT:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
CXX="${CXX:-g++}"
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
FAIL=0

if ! command -v "${CXX}" > /dev/null 2>&1; then
    echo "WARNING [fatal]: compiler '${CXX}' not found; this gate cannot run"
    exit 1
fi
BAND_H="${REPO_ROOT}/aether/banded/banded.h"
if [[ ! -f "${BAND_H}" ]]; then
    echo "WARNING [fatal]: '${BAND_H}' does not exist — REPO_ROOT is wrong" >&2
    exit 1
fi

arm() {
    local tag="$1" expect="$2" pattern="$3" body="$4"
    local src="${WORK}/${tag}.cpp" log="${WORK}/${tag}.log" rc=0
    {
        echo "#include \"${BAND_H}\""; echo "#include \"${REPO_ROOT}/aether/math/math.h\""
        echo "using namespace aether::banded; namespace bd = aether::banded::detail;"
        echo "int main() { const Band a{1.5f, 0.f, 0.f}; ${body} return 0; }"
    } > "${src}"
    "${CXX}" -std=c++23 -DAETHER_CPP_MODE=1 -I"${REPO_ROOT}" -c "${src}" -o "${WORK}/${tag}.o" > "${log}" 2>&1 || rc=$?
    if [[ "${expect}" == "fail" ]]; then
        if [[ ${rc} -eq 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — compiled, but must be rejected"
            FAIL=1
        elif ! grep -qF -- "${pattern}" "${log}"; then
            echo "ARM ${tag}: VERDICT RED — rejected but not for '${pattern}'"
            sed -n '1,15p' "${log}" >&2
            FAIL=1
        else
            echo "ARM ${tag}: PASS (rc=${rc}, matched '${pattern}')"
        fi
    elif [[ ${rc} -ne 0 ]]; then
        echo "ARM ${tag}: VERDICT RED — a SANCTIONED path stopped compiling"
        sed -n '1,30p' "${log}" >&2
        FAIL=1
    else
        echo "ARM ${tag}: PASS (rc=${rc})"
    fi
}

echo "=========================================="
echo "  banded chain typing gate (${CXX}, AETHER_CPP_MODE)"
echo "=========================================="
arm CHAIN-ABS fail "normalize(R) or normalizeSafe(C)" 'const Band b = bd::abs(a * a); (void)b;'
arm CHAIN-FMAX fail "fmax reads the limbs in order" 'const Band b = bd::fmax(a + a, a); (void)b;'
arm CHAIN-CANCEL-TO-RAW fail "could not convert" 'const auto r = bd::mulRaw(bd::BandRawCancel{}, bd::BandRaw(a)); (void)r;'
arm CHAIN-COMPARE fail "no match for" 'const bool e = (a * a == a); (void)e;'
arm CHAIN-COPYSIGN fail "copysign reads the limbs in order" 'const auto x = bd::copysign(a, a * a); (void)x;'
arm CHAIN-FMIN fail "fmin reads the limbs in order" 'const auto x = bd::fmin(a * a, a); (void)x;'
arm CHAIN-FACADE-ABS fail "aether::math::abs: T must be float or double" 'const auto x = aether::math::abs(a * a); (void)x;'
arm CHAIN-FMA-RAW-ADDEND fail "fma needs a normalised addend: normalize" 'const auto x = bd::fma(a * a, a, a * a); (void)x;'
arm CHAIN-POSITIVE pass - 'const Band n = bd::normalize(a * a); const Band m = bd::normalizeSafe(a + a); const BandedReal s(a * a); const BandedReal t(a + a); (void)n; (void)m; (void)s; (void)t;'

if [[ ${FAIL} -ne 0 ]]; then
    echo "BANDCHAIN TYPING GATE: RED"
    exit 1
fi
echo "BANDCHAIN TYPING GATE: GREEN"
