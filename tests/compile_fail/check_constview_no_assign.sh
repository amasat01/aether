#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile-fail gate for two locked "must not compile" claims:
#
#   (1) the CONST read path carries NO assignment operators — `cv[i] = expr`
#       must be rejected outright (`ConstSampleRef` declares no `operator=`),
#       with a CONTROL proving the mutable path still assigns fine.
#   (2) the scalar overloads require an EXACT type match — `v[i] = 2` (an
#       `int` literal) into a `double` view must be rejected by the
#       `static_assert` in `aether/expr/nodes/Constant.h`
#       (SampleRef::operator=(scalar) requires `S` to be EXACTLY
#       `element_type`, no implicit conversion), with a CONTROL proving
#       `v[i] = 2.0` (the exact scalar type) still assigns fine.
#
# ## What this instrument can see, and what it cannot
# Every claim below is that some expression MUST NOT COMPILE. None of them
# can live inside the gtest binary: a test that compiles is a test whose
# subject compiled — see `tests/test_ConstView.cpp`/`tests/
# test_ScalarBroadcast.cpp`'s own docstrings, which point back here for the
# COMPLEMENT. Mirrors `tests/compile_fail/check_bandedreal_typing_rejected.sh`'s
# own arm shape (`arm <TAG> <fail|compile> <required-stderr-substring-or-'-'>
# <source file>`).
#
# g++ only, CPP_MODE (`AETHER_CPP_MODE`), no CUDA toolchain, no GPU involved
# (both arms are DEVICEHOST-safe surface — see the test files' own note).

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
# ★ PRECONDITION — gate the SOURCE TREE, never an installed copy (same
# guard as check_bandedreal_typing_rejected.sh).
# ---------------------------------------------------------------------------
AETHER_H="${REPO_ROOT}/aether/aether.h"
if [[ ! -f "${AETHER_H}" ]]; then
    echo "WARNING [fatal]: '${AETHER_H}' does not exist — REPO_ROOT is wrong, and" \
         "this gate would silently test an INSTALLED aether instead" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# arm <TAG> <fail|compile> <required-stderr-substring-or-'-'> <source file>
# ---------------------------------------------------------------------------
arm() {
    local tag="$1" expect="$2" pattern="$3" src="$4"
    local log="${WORK}/${tag}.log"
    local gen="${WORK}/${tag}.gen.cpp"
    local rc=0

    sed -e "s|@AETHER_H@|${AETHER_H}|" "${src}" > "${gen}"

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
        echo "ARM ${tag}: PASS (rc=${rc}, RED-by-design, matched '${pattern}')"
    else
        if [[ ${rc} -ne 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — a CONTROL stopped compiling (rc=${rc})"
            sed -n '1,40p' "${log}" >&2
            FAIL=1
            return
        fi
        echo "ARM ${tag}: PASS (rc=${rc}, control GREEN)"
    fi
}

echo "=========================================="
echo "  const-view / scalar-broadcast typing gate (${CXX}, AETHER_CPP_MODE)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---- ARM CONST-NO-ASSIGN (RED-by-design) -----------------------------------
cat > "${WORK}/CONST-NO-ASSIGN.cpp" <<'EOF'
#include "@AETHER_H@"
#include <vector>
int main()
{
    using namespace aether;
    std::vector<double> buf(3);
    auto v = make_view<double, 3, dyn>(buf.data(), Device(kDLCPU), 1);
    const auto& cv = v;
    const SampleIndex i = SampleIndex::make(0);
    cv[i] = cv[i].get();  /* MUST NOT COMPILE — ConstSampleRef has no operator= */
    return 0;
}
EOF
arm CONST-NO-ASSIGN fail "no match for 'operator='" "${WORK}/CONST-NO-ASSIGN.cpp"

# ---- ARM CONST-NO-ASSIGN-CONTROL (mutable path still assigns) -------------
cat > "${WORK}/CONST-NO-ASSIGN-CONTROL.cpp" <<'EOF'
#include "@AETHER_H@"
#include <vector>
int main()
{
    using namespace aether;
    std::vector<double> buf(3);
    auto v = make_view<double, 3, dyn>(buf.data(), Device(kDLCPU), 1);
    const SampleIndex i = SampleIndex::make(0);
    v[i] = v[i].get();  /* MUST compile — the mutable overload is unaffected */
    return v(0, 0) == 0.0 ? 0 : 1;
}
EOF
arm CONST-NO-ASSIGN-CONTROL compile "-" "${WORK}/CONST-NO-ASSIGN-CONTROL.cpp"

# ---- ARM D10-SCALAR-INT (RED-by-design) ------------------------------------
cat > "${WORK}/D10-SCALAR-INT.cpp" <<'EOF'
#include "@AETHER_H@"
#include <vector>
int main()
{
    using namespace aether;
    std::vector<double> buf(3);
    auto v = make_view<double, 3, dyn>(buf.data(), Device(kDLCPU), 1);
    const SampleIndex i = SampleIndex::make(0);
    v[i] = 2;  /* MUST NOT COMPILE — int into a double view, no implicit conversion (D10) */
    return 0;
}
EOF
arm D10-SCALAR-INT fail "no implicit conversion (D10)" "${WORK}/D10-SCALAR-INT.cpp"

# ---- ARM D10-SCALAR-CONTROL (exact scalar type still assigns) -------------
cat > "${WORK}/D10-SCALAR-CONTROL.cpp" <<'EOF'
#include "@AETHER_H@"
#include <vector>
int main()
{
    using namespace aether;
    std::vector<double> buf(3);
    auto v = make_view<double, 3, dyn>(buf.data(), Device(kDLCPU), 1);
    const SampleIndex i = SampleIndex::make(0);
    v[i] = 2.0;  /* MUST compile — S deduces to EXACTLY element_type */
    return v(0, 0) == 2.0 ? 0 : 1;
}
EOF
arm D10-SCALAR-CONTROL compile "-" "${WORK}/D10-SCALAR-CONTROL.cpp"

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "CONST-VIEW / SCALAR-BROADCAST TYPING GATE: RED"
    exit 1
fi
echo "CONST-VIEW / SCALAR-BROADCAST TYPING GATE: GREEN"
