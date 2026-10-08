#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile-fail gate for `TableHandle<T, Texture>`'s NO-WRITE-PATH contract.
# Same instrument shape as
# tests/compile_fail/check_bandedreal_typing_rejected.sh: every claim below
# is that some expression MUST NOT COMPILE, so it cannot live inside the
# test binary — a test that compiles is a test whose subject compiled.
#
# #119 (guards key on the SELECTED FORM, never on a leftover boolean
# template parameter some form doesn't carry): `TableHandle<T, Texture>` is
# a full partial specialization with NO non-const `operator[]` overload at
# all, so there is no body guard to pin here — the type surface itself
# (`operator[]` returns `const T` by value) is the whole of the contract,
# and the ONLY way to observe "no write path exists" is to attempt one and
# watch it fail. The COMPLEMENT (a compile-time, in-binary re-observation
# of the same type-surface fact) is `TextureWriteInvariant.Anchor` in
# `tests/test_TextureBinding.{cpp,cu}`.
#
# g++ only, CPP_MODE (`AETHER_CPP_MODE`), no CUDA toolchain, no GPU
# involved — `TableHandle<T, Texture>` degrades to a plain-pointer read in
# CPP_MODE (see `aether/view/TableHandle.h`'s file docstring) but its
# read-only TYPE SURFACE (no non-const `operator[]`) is IDENTICAL in both
# build modes, so a CPP_MODE compile is a faithful probe of the contract —
# exactly the same reasoning `check_bandedreal_typing_rejected.sh` uses.
#
# ## Arms — every one mandatory, and RC alone is NEVER the verdict
#
#   ARM WRITE-REJECTED   `h[0] = 5.0;` through a `TableHandle<double,
#                         Texture>` MUST fail; stderr MUST name `lvalue
#                         required`. `operator[]` returns `const double` BY
#                         VALUE, and a built-in simple-assignment requires a
#                         modifiable LVALUE on its left — this is the
#                         diagnostic g++ actually gives (verified against
#                         this tree before writing this arm).
#   ARM CONTROL           an undeclared symbol. MUST fail (any diagnostic).
#                         Proves the invocation genuinely compiles the file:
#                         a mis-ordered flag returns RC=0 having compiled
#                         nothing, which would make WRITE-REJECTED vacuous.
#   ARM POSITIVE          a legitimate READ through BOTH carriers
#                         (`TableHandle<double, Texture>` and
#                         `TableHandle<double, Plain>`, default-constructing
#                         the latter to touch it too) MUST compile. Without
#                         it, WRITE-REJECTED could be satisfied by a type
#                         that fails to compile ANY expression at all.
#
# Usage: `tests/compile_fail/check_texture_write_rejected.sh [<aether repo root>]`
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
# ★ PRECONDITION — gate the SOURCE TREE, never an installed copy (this
# lock mirrors check_bandedreal_typing_rejected.sh's own precondition).
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
echo "  TableHandle<T,Texture> write-rejection gate (${CXX}, AETHER_CPP_MODE)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---- ARM WRITE-REJECTED ----------------------------------------------------
cat > "${WORK}/WRITE-REJECTED.cpp" <<'EOF'
#include "@AETHER_H@"
int main()
{
    double buf[4] = { 1.0, 2.0, 3.0, 4.0 };
    aether::TableHandle<double, aether::Texture> h(buf, 0ull, 0);
    h[0] = 5.0;  /* MUST NOT COMPILE — texture memory is read-only, and
                  * operator[] returns `const double` BY VALUE, so this is
                  * a built-in assignment with no modifiable lvalue on its
                  * left */
    return 0;
}
EOF
arm WRITE-REJECTED fail "lvalue required" "${WORK}/WRITE-REJECTED.cpp"

# ---- ARM CONTROL ------------------------------------------------------------
cat > "${WORK}/CONTROL.cpp" <<'EOF'
#include "@AETHER_H@"
int main()
{
    return thisSymbolDoesNotExist();  /* MUST NOT COMPILE — proves the gate compiles */
}
EOF
arm CONTROL fail "-" "${WORK}/CONTROL.cpp"

# ---- ARM POSITIVE -------------------------------------------------------
cat > "${WORK}/POSITIVE.cpp" <<'EOF'
#include "@AETHER_H@"
int main()
{
    double buf[4] = { 1.0, 2.0, 3.0, 4.0 };
    aether::TableHandle<double, aether::Texture> h(buf, 0ull, 1);
    const double v = h[1];  /* sanctioned read */

    aether::TableHandle<double, aether::Plain> hp;  /* touch the plain carrier too */

    return (v == 3.0 && hp.data() == nullptr) ? 0 : 1;
}
EOF
arm POSITIVE compile "-" "${WORK}/POSITIVE.cpp"

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "TEXTURE WRITE-REJECTION GATE: RED"
    exit 1
fi
echo "TEXTURE WRITE-REJECTION GATE: GREEN"
