#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/volatile_audit.sh — SASS-level CSE-blocking audit for
# `View::as_volatile()` (aether/view/View.h).
#
# Compiles `tools/volatile_probe.cu`'s twin kernels (plain vs `as_volatile()`
# view over the SAME __shared__ buffer, each doing a deliberate triple-read
# of one location) via `nvcc -c` (ptxas -v), then disassembles the resulting
# object with `cuobjdump -sass` — ptxas/cuobjdump only, no GPU execution,
# nothing here is ever launched.
#
# Assertions:
#   1. CSE is blocked: the volatile kernel's LDS (shared-memory load) count
#      is strictly greater than the plain kernel's.
#   2. No stack-backed mirror in either kernel: zero STL/LDL instructions.
#
# Usage: tools/volatile_audit.sh
#
set -u -o pipefail

PROG="tools/volatile_audit.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROBE_TU="$REPO_ROOT/tools/volatile_probe.cu"
if [ ! -f "$PROBE_TU" ]; then
    echo "$PROG: no probe TU at $PROBE_TU" >&2
    exit 1
fi

NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
    echo "$PROG: nvcc not found on PATH" >&2
    exit 1
fi
CUOBJDUMP="$(command -v cuobjdump || true)"
if [ -z "$CUOBJDUMP" ]; then
    echo "$PROG: cuobjdump not found on PATH" >&2
    exit 1
fi
DEMANGLE="$(command -v cu++filt || command -v c++filt || true)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "$PROG: compiling $PROBE_TU against $REPO_ROOT (sm_61, c++20, ptxas -v) — NO GPU execution" >&2
"$NVCC" -arch=sm_61 -std=c++20 -I"$REPO_ROOT" \
    --ptxas-options=-v \
    -c "$PROBE_TU" -o "$TMP/volatile_probe.o" > "$TMP/ptxas.log" 2>&1
NVCC_RC=$?
sed 's/^/  | /' "$TMP/ptxas.log" >&2

if [ "$NVCC_RC" -ne 0 ]; then
    echo "$PROG: nvcc compile FAILED (rc=$NVCC_RC)" >&2
    exit 1
fi

echo "$PROG: disassembling with cuobjdump -sass — NO GPU execution" >&2
"$CUOBJDUMP" -arch sm_61 -sass "$TMP/volatile_probe.o" > "$TMP/sass.txt" 2>"$TMP/cuobjdump.err"
CUOBJDUMP_RC=$?
if [ "$CUOBJDUMP_RC" -ne 0 ]; then
    echo "$PROG: cuobjdump FAILED (rc=$CUOBJDUMP_RC)" >&2
    sed 's/^/  | /' "$TMP/cuobjdump.err" >&2
    exit 1
fi

python3 - "$TMP/sass.txt" "$DEMANGLE" <<'PYEOF'
import re
import subprocess
import sys

sass_path, demangle_bin = sys.argv[1], sys.argv[2]
text = open(sass_path).read()


def demangle(name: str) -> str:
    if not demangle_bin:
        return name
    try:
        out = subprocess.run([demangle_bin, name], capture_output=True, text=True, check=True).stdout.strip()
        return out or name
    except Exception:
        return name


# cuobjdump -sass prints one "Function : <mangled>" header per kernel,
# followed by its instruction listing until the next "Function :" or EOF.
func_re = re.compile(r"^\s*Function : (\S+)\s*$", re.MULTILINE)
blocks = {}
matches = list(func_re.finditer(text))
for idx, m in enumerate(matches):
    mangled = m.group(1)
    start = m.end()
    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
    body = text[start:end]
    name = demangle(mangled)
    simple = name.split("(")[0]
    if "::" in simple:
        simple = simple.rsplit("::", 1)[-1]
    blocks[simple] = body

if not blocks:
    print("volatile_audit.sh: no 'Function :' entries found in cuobjdump output — nothing to report", file=sys.stderr)
    sys.exit(1)

print("volatile_audit.sh: kernels seen in cuobjdump output:", file=sys.stderr)
for name in blocks:
    print(f"  - {name}", file=sys.stderr)

TARGETS = {
    "plain": "volatileProbePlainKernel",
    "volatile": "volatileProbeVolatileKernel",
}

missing = [label for label, name in TARGETS.items() if name not in blocks]
if missing:
    print(f"volatile_audit.sh: MISSING kernel(s) in cuobjdump output: {missing}", file=sys.stderr)
    sys.exit(1)


def count(pattern, body):
    return len(re.findall(pattern, body))


lds_plain = count(r"\bLDS\b", blocks[TARGETS["plain"]])
lds_vol = count(r"\bLDS\b", blocks[TARGETS["volatile"]])
stack_plain = count(r"\bSTL\b|\bLDL\b", blocks[TARGETS["plain"]])
stack_vol = count(r"\bSTL\b|\bLDL\b", blocks[TARGETS["volatile"]])

print("==========================================")
print("  as_volatile() SASS audit  (sm_61)")
print("==========================================")
print(f"  LDS (shared-mem reads)             plain={lds_plain:<3}  volatile={lds_vol:<3}")
print(f"  STL|LDL (stack-mirror traffic)     plain={stack_plain:<3}  volatile={stack_vol:<3}")
print("------------------------------------------")

red = False

if lds_vol > lds_plain:
    print(f"PASS: CSE blocked (volatile LDS={lds_vol} > plain LDS={lds_plain})")
else:
    print(f"FAIL: volatile LDS ({lds_vol}) should be > plain LDS ({lds_plain}) — CSE not blocked?")
    red = True

if stack_plain == 0:
    print("PASS: plain kernel has no stack-mirror traffic (STL/LDL=0)")
else:
    print(f"FAIL: plain kernel has stack-mirror traffic (STL/LDL={stack_plain})")
    red = True

if stack_vol == 0:
    print("PASS: volatile kernel has no stack-mirror traffic (STL/LDL=0)")
else:
    print(f"FAIL: volatile kernel has stack-mirror traffic (STL/LDL={stack_vol})")
    red = True

print()
if red:
    print("volatile_audit.sh: VERDICT: RED", file=sys.stderr)
    sys.exit(1)
print("volatile_audit.sh: VERDICT: GREEN", file=sys.stderr)
sys.exit(0)
PYEOF
PY_RC=$?
exit "$PY_RC"
