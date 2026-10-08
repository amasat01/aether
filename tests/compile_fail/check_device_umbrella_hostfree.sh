#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile-fail gate for the device-safe umbrella's HOST-FREE CONTRACT:
# `aether/device.h` alone must not expose any host-only symbol that reaches
# `<string>`/`aether/err/Error.h`/allocation — those live only in
# `aether/aether.h` and the split-out `view/Make*.h` factory headers.
#
# Ported in the house pattern from
# `tests/compile_fail/check_bandedreal_typing_rejected.sh` (same `arm()`
# helper, same structure): g++ only, AETHER_CPP_MODE, no CUDA toolchain, no
# GPU involved. The device-SIDE half of this claim (does `aether/device.h`
# parse+compile UNDER NVRTC) is a separate gate:
# `tools/nvrtc/audit_device_umbrella.sh`.
#
# ## What this instrument can see, and what it cannot
# Every negative claim below is that some expression MUST NOT COMPILE. None
# of them can live inside the aether_tests binary: a test that compiles is a
# test whose subject compiled. The POSITIVE arms are the complement:
# `tests/test_DeviceUmbrella.{cpp,cu}` carries the same `make_view()`
# reachability claims as live, running assertions (values checked, not just
# "it compiled") — this script's POSITIVE arms exist only for the
# non-vacuity role explained at CONTROL below, not to duplicate that
# coverage.
#
# ## Arms — every one mandatory, and RC alone is NEVER the verdict
#
#   ARM DEVICE-NO-MAKEVIEW    `aether::make_view<...>(...)` reached through
#                             ONLY `aether/device.h` MUST fail; stderr MUST
#                             name `is not a member of`. `make_view()` moved
#                             OUT to `aether/view/MakeView.h` — this is the
#                             exact claim this gate exists to check.
#   ARM DEVICE-NO-ERR         `aether::err::fail(...)` reached through ONLY
#                             `aether/device.h` MUST fail; stderr MUST name
#                             `has not been declared`. Stronger than the
#                             make_view arm above: proves the whole
#                             `aether/err/Error.h` HOST-ONLY error contract
#                             (not just one factory function) is unreachable
#                             — the root NVRTC finding this gate exists to catch.
#   ARM CONTROL               an undeclared symbol, reached through
#                             `aether/device.h`. MUST fail (any diagnostic).
#                             Proves the invocation genuinely compiles the
#                             file: a mis-ordered flag returns RC=0 having
#                             compiled nothing, which would make both
#                             rejections above vacuous.
#   ARM POSITIVE-AETHER-H     `aether::make_view<...>(...)` reached through
#                             `aether/aether.h` MUST compile. Without this,
#                             ARM DEVICE-NO-MAKEVIEW is also satisfied by a
#                             `make_view` that no longer exists ANYWHERE.
#   ARM POSITIVE-MAKEVIEW-H   the same call, reached through
#                             `aether/view/MakeView.h` DIRECTLY (no
#                             `aether/aether.h` in this TU at all) MUST
#                             compile — the split moved `make_view()` to a
#                             NAMED header, not into a black box only the
#                             full umbrella can reach.
#
# Usage: `tests/compile_fail/check_device_umbrella_hostfree.sh [<aether repo root>]`
# Exit code IS the verdict.

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

# ---------------------------------------------------------------------------
# ★ PRECONDITION — gate the SOURCE TREE, never an installed copy. A stale
# $CONDA_PREFIX/include/aether answers every question below about headers
# this checkout has never seen. Two locks: the entry headers must exist
# under REPO_ROOT, and every snippet includes them BY ABSOLUTE PATH.
# ---------------------------------------------------------------------------
DEVICE_H="${REPO_ROOT}/aether/device.h"
AETHER_H="${REPO_ROOT}/aether/aether.h"
MAKEVIEW_H="${REPO_ROOT}/aether/view/MakeView.h"
for f in "${DEVICE_H}" "${AETHER_H}" "${MAKEVIEW_H}"; do
    if [[ ! -f "${f}" ]]; then
        echo "WARNING [fatal]: '${f}' does not exist — REPO_ROOT is wrong, and" \
             "this gate would silently test an INSTALLED aether instead" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# arm <TAG> <fail|compile> <required-stderr-substring-or-'-'> <source file>
# ---------------------------------------------------------------------------
arm() {
    local tag="$1" expect="$2" pattern="$3" src="$4"
    local log="${WORK}/${tag}.log"
    local gen="${WORK}/${tag}.gen.cpp"
    local rc=0

    sed -e "s|@DEVICE_H@|${DEVICE_H}|" -e "s|@AETHER_H@|${AETHER_H}|" -e "s|@MAKEVIEW_H@|${MAKEVIEW_H}|" \
        "${src}" > "${gen}"

    "${CXX}" -std=c++23 -DAETHER_CPP_MODE=1 -I"${REPO_ROOT}" -c "${gen}" \
        -o "${WORK}/${tag}.o" > "${log}" 2>&1 || rc=$?

    if [[ "${expect}" == "fail" ]]; then
        if [[ ${rc} -eq 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — compiled, but must be rejected (rc=0)"
            FAIL=1
            return
        fi
        if [[ "${pattern}" != "-" ]] && ! grep -qF -- "${pattern}" "${log}"; then
            echo "ARM ${tag}: VERDICT RED — rejected (rc=${rc}) but for the" \
                 "WRONG reason: diagnostic does not contain '${pattern}'"
            sed -n '1,20p' "${log}" >&2
            FAIL=1
            return
        fi
        if [[ "${pattern}" == "-" ]]; then
            echo "ARM ${tag}: PASS (rc=${rc}, any diagnostic accepted)"
        else
            echo "ARM ${tag}: PASS (rc=${rc}, matched '${pattern}')"
        fi
    else
        if [[ ${rc} -ne 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — a SANCTIONED path stopped compiling (rc=${rc})"
            sed -n '1,40p' "${log}" >&2
            FAIL=1
            return
        fi
        echo "ARM ${tag}: PASS (rc=${rc})"
    fi
}

echo "=========================================="
echo "  device umbrella host-free gate (${CXX}, AETHER_CPP_MODE)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---- ARM DEVICE-NO-MAKEVIEW ------------------------------------------------
cat > "${WORK}/DEVICE-NO-MAKEVIEW.cpp" <<'EOF'
#include "@DEVICE_H@"
int main()
{
    double buf[4] = { 0.0, 0.0, 0.0, 0.0 };
    /* MUST NOT COMPILE -- make_view() lives in view/MakeView.h, not device.h */
    auto v = aether::make_view<double, aether::dyn>(buf, aether::Device(kDLCPU), 4);
    return static_cast<int>(v.samples());
}
EOF
arm DEVICE-NO-MAKEVIEW fail "is not a member of" "${WORK}/DEVICE-NO-MAKEVIEW.cpp"

# ---- ARM DEVICE-NO-ERR -----------------------------------------------------
cat > "${WORK}/DEVICE-NO-ERR.cpp" <<'EOF'
#include "@DEVICE_H@"
int main()
{
    /* MUST NOT COMPILE -- the whole err/Error.h HOST-ONLY contract is
     * unreachable from device.h alone. */
    aether::err::fail("op", "where", 0);
    return 0;
}
EOF
arm DEVICE-NO-ERR fail "has not been declared" "${WORK}/DEVICE-NO-ERR.cpp"

# ---- ARM CONTROL ------------------------------------------------------------
cat > "${WORK}/CONTROL.cpp" <<'EOF'
#include "@DEVICE_H@"
int main()
{
    return thisSymbolDoesNotExist();  /* MUST NOT COMPILE — proves the gate compiles */
}
EOF
arm CONTROL fail "-" "${WORK}/CONTROL.cpp"

# ---- ARM POSITIVE-AETHER-H --------------------------------------------------
cat > "${WORK}/POSITIVE-AETHER-H.cpp" <<'EOF'
#include "@AETHER_H@"
int main()
{
    double buf[4] = { 0.0, 0.0, 0.0, 0.0 };
    auto v = aether::make_view<double, aether::dyn>(buf, aether::Device(kDLCPU), 4);
    return v.samples() == 4 ? 0 : 1;
}
EOF
arm POSITIVE-AETHER-H compile "-" "${WORK}/POSITIVE-AETHER-H.cpp"

# ---- ARM POSITIVE-MAKEVIEW-H -------------------------------------------------
cat > "${WORK}/POSITIVE-MAKEVIEW-H.cpp" <<'EOF'
#include "@MAKEVIEW_H@"
int main()
{
    double buf[4] = { 0.0, 0.0, 0.0, 0.0 };
    auto v = aether::make_view<double, aether::dyn>(buf, aether::Device(kDLCPU), 4);
    return v.samples() == 4 ? 0 : 1;
}
EOF
arm POSITIVE-MAKEVIEW-H compile "-" "${WORK}/POSITIVE-MAKEVIEW-H.cpp"

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "DEVICE UMBRELLA HOST-FREE GATE: RED"
    exit 1
fi
echo "DEVICE UMBRELLA HOST-FREE GATE: GREEN"
