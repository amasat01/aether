#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/bandwidth_audit.sh — SASS-level bandwidth audit for
# `View::as_readonly()` and `DeviceBundle`/`BundleIndex`, following the
# same pattern as `tools/volatile_audit.sh`.
#
# Compiles `tools/bandwidth_probe.cu`'s three kernels via `nvcc -c`
# (ptxas -v), then disassembles the resulting object with `cuobjdump -sass`
# — ptxas/cuobjdump only, no GPU execution, nothing here is ever launched.
#
# Assertions (see `bandwidth_probe.cu`'s file docstring for the full
# rationale, in particular why the fallback probe uses `layout_stride`
# rather than a genuinely misaligned pointer):
#   1. `bandwidthProbeBundleKernel`   must contain `LDG.E.128` and `STG.E.128`.
#   2. `bandwidthProbeReadOnlyKernel` must contain a `.CI` (read-only data
#      cache) load.
#   3. `bandwidthProbeFallbackKernel` must contain neither `LDG.E.128` nor
#      `STG.E.128` anywhere.
#
# `-DAETHER_HAS_CUDA=1` is passed explicitly: unlike `volatile_probe.cu`
# (whose `as_volatile()` path has no `AETHER_HAS_CUDA` dependency at all),
# `bundleGet`/`bundleStore`'s vectorized fast path (`backend/cuda/bundle/
# LoadStore.h`) is `#ifdef AETHER_HAS_CUDA`-gated (compiled in both modes,
# only the CUDA-only vectorized branch needs the macro) — the real CMake
# build defines this automatically for every CUDA-mode target, but a bare
# `nvcc -c` invocation like this one does not.
#
# Usage: tools/bandwidth_audit.sh
#
set -u -o pipefail

PROG="tools/bandwidth_audit.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROBE_TU="$REPO_ROOT/tools/bandwidth_probe.cu"
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
"$NVCC" -arch=sm_61 -std=c++20 -DAETHER_HAS_CUDA=1 -I"$REPO_ROOT" \
    --ptxas-options=-v \
    -c "$PROBE_TU" -o "$TMP/bandwidth_probe.o" > "$TMP/ptxas.log" 2>&1
NVCC_RC=$?
sed 's/^/  | /' "$TMP/ptxas.log" >&2

if [ "$NVCC_RC" -ne 0 ]; then
    echo "$PROG: nvcc compile FAILED (rc=$NVCC_RC)" >&2
    exit 1
fi

echo "$PROG: disassembling with cuobjdump -sass — NO GPU execution" >&2
"$CUOBJDUMP" -arch sm_61 -sass "$TMP/bandwidth_probe.o" > "$TMP/sass.txt" 2>"$TMP/cuobjdump.err"
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
    print("bandwidth_audit.sh: no 'Function :' entries found in cuobjdump output — nothing to report", file=sys.stderr)
    sys.exit(1)

print("bandwidth_audit.sh: kernels seen in cuobjdump output:", file=sys.stderr)
for name in blocks:
    print(f"  - {name}", file=sys.stderr)

TARGETS = {
    "bundle": "bandwidthProbeBundleKernel",
    "readonly": "bandwidthProbeReadOnlyKernel",
    "fallback": "bandwidthProbeFallbackKernel",
}

missing = [label for label, name in TARGETS.items() if name not in blocks]
if missing:
    print(f"bandwidth_audit.sh: MISSING kernel(s) in cuobjdump output: {missing}", file=sys.stderr)
    sys.exit(1)


def count(pattern, body):
    return len(re.findall(pattern, body))


ldg128_bundle = count(r"\bLDG\.E\.128\b", blocks[TARGETS["bundle"]])
stg128_bundle = count(r"\bSTG\.E\.128\b", blocks[TARGETS["bundle"]])
ci_readonly = count(r"\bLDG\.E\.CI\b", blocks[TARGETS["readonly"]])
ldg128_fallback = count(r"\bLDG\.E\.128\b", blocks[TARGETS["fallback"]])
stg128_fallback = count(r"\bSTG\.E\.128\b", blocks[TARGETS["fallback"]])

print("==========================================")
print("  bandwidth audit  (sm_61)")
print("==========================================")
print(f"  bundleKernel    LDG.E.128={ldg128_bundle:<3}  STG.E.128={stg128_bundle:<3}")
print(f"  readOnlyKernel  LDG.E.CI={ci_readonly:<3}")
print(f"  fallbackKernel  LDG.E.128={ldg128_fallback:<3}  STG.E.128={stg128_fallback:<3}")
print("------------------------------------------")

red = False

if ldg128_bundle > 0:
    print(f"PASS: bundle kernel contains LDG.E.128 ({ldg128_bundle})")
else:
    print("FAIL: bundle kernel MUST contain LDG.E.128 (vectorized load absent)")
    red = True

if stg128_bundle > 0:
    print(f"PASS: bundle kernel contains STG.E.128 ({stg128_bundle})")
else:
    print("FAIL: bundle kernel MUST contain STG.E.128 (vectorized store absent)")
    red = True

if ci_readonly > 0:
    print(f"PASS: readonly kernel contains LDG.E.CI ({ci_readonly})")
else:
    print("FAIL: readonly kernel MUST contain a .CI (read-only data cache) load")
    red = True

if ldg128_fallback == 0 and stg128_fallback == 0:
    print("PASS: fallback kernel contains NEITHER LDG.E.128 NOR STG.E.128")
else:
    print(f"FAIL: fallback kernel MUST NOT contain LDG.E.128/STG.E.128 "
          f"(found LDG.E.128={ldg128_fallback}, STG.E.128={stg128_fallback})")
    red = True

print()
if red:
    print("bandwidth_audit.sh: VERDICT: RED", file=sys.stderr)
    sys.exit(1)
print("bandwidth_audit.sh: VERDICT: GREEN", file=sys.stderr)
sys.exit(0)
PYEOF
PY_RC=$?
exit "$PY_RC"
