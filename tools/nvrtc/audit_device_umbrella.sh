#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/nvrtc/audit_device_umbrella.sh — gate: is `aether/device.h`
# actually NVRTC-parseable (not just documented as such), and does the
# audit still know how to fail?
#
# Two subjects, both compiled by tools/nvrtc/device_umbrella_probe.cpp with
# the SAME canonical option list tools/nvrtc/README.md documents (derived
# here at runtime from the active toolchain, mirroring
# tools/nvrtc_parse_audit.sh — never a hard-coded path):
#
#   1. DEVICE  (gating GREEN) — `#include <aether/device.h>` alone, then a
#      MATERIALIZING `Item c = a + b` (not `auto c = a + b` — the whole
#      point is Item's out-of-line expression ctor, which used to reach
#      host-only `err/Error.h` transitively through `view/View.h` before
#      the two headers were split). MUST PASS (rc=0).
#   2. CONTROL (gating RED, non-vacuity) — the literal
#      `#include <aether/aether.h>` full umbrella. MUST FAIL (rc!=0): an
#      audit that cannot fail is certifying nothing. Still RED today because
#      `aether.h` unconditionally pulls in host-only headers (e.g.
#      `accum/AccumPlane.h`'s `<algorithm>`) device.h deliberately excludes.
#
# This script's own exit code is 0 iff BOTH hold (device PASS, control
# RED); non-zero if either subject answers the wrong way. Both subjects'
# individual rc and first diagnostic line are always printed, regardless of
# the verdict, so a caller never has to re-run to see which one flipped.
#
# Usage: tools/nvrtc/audit_device_umbrella.sh [gpu-arch, default compute_61]

set -u -o pipefail

PROG="tools/nvrtc/audit_device_umbrella.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCH="${1:-compute_61}"

PROBE_SRC="$SCRIPT_DIR/device_umbrella_probe.cpp"
if [ ! -f "$PROBE_SRC" ]; then
    echo "$PROG: no probe source at $PROBE_SRC" >&2
    exit 1
fi

# PRECONDITION — gate the SOURCE TREE, never an installed copy (this
# lock mirrors tests/headers/check_header_diet.sh and
# tests/compile_fail/check_bandedreal_typing_rejected.sh's own preconditions).
for f in aether/device.h aether/aether.h; do
    if [ ! -f "$REPO_ROOT/$f" ]; then
        echo "$PROG: '$REPO_ROOT/$f' does not exist — REPO_ROOT is wrong" >&2
        exit 1
    fi
done

SHIM_DIR="$SCRIPT_DIR/shim_include"
if [ ! -f "$SHIM_DIR/cstddef" ]; then
    echo "$PROG: no NVRTC shim headers at $SHIM_DIR (expected cstddef, cstdint, ...)" >&2
    exit 1
fi

NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
    echo "$PROG: nvcc not found on PATH (needed to locate the NVRTC headers/library)" >&2
    exit 1
fi
CUDA_ROOT="$(cd "$(dirname "$NVCC")/.." && pwd)"

NVRTC_INCLUDE=""
for cand in "$CUDA_ROOT/targets/x86_64-linux/include" "$CUDA_ROOT/include"; do
    if [ -f "$cand/nvrtc.h" ]; then
        NVRTC_INCLUDE="$cand"
        break
    fi
done
if [ -z "$NVRTC_INCLUDE" ]; then
    echo "$PROG: could not find nvrtc.h under $CUDA_ROOT" >&2
    exit 1
fi

NVRTC_LIBDIR=""
for cand in "$CUDA_ROOT/targets/x86_64-linux/lib" "$CUDA_ROOT/lib64" "$CUDA_ROOT/lib"; do
    if [ -f "$cand/libnvrtc.so" ]; then
        NVRTC_LIBDIR="$cand"
        break
    fi
done
if [ -z "$NVRTC_LIBDIR" ]; then
    echo "$PROG: could not find libnvrtc.so under $CUDA_ROOT" >&2
    exit 1
fi

CUDA_STD_INCLUDE=""
for cand in "$CUDA_ROOT/targets/x86_64-linux/include" "$CUDA_ROOT/include"; do
    if [ -f "$cand/cuda/std/cstddef" ]; then
        CUDA_STD_INCLUDE="$cand"
        break
    fi
done
if [ -z "$CUDA_STD_INCLUDE" ]; then
    echo "$PROG: could not find libcu++ (cuda/std/cstddef) under $CUDA_ROOT — this toolkit" >&2
    echo "$PROG: is too old for the NVRTC shim; see tools/nvrtc/README.md 'manual steps'" >&2
    exit 1
fi

GXX="$(command -v g++ || true)"
if [ -z "$GXX" ]; then
    echo "$PROG: g++ not found on PATH" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PROBE_BIN="$TMP/device_umbrella_probe"

echo "$PROG: compiling probe driver against $NVRTC_INCLUDE / $NVRTC_LIBDIR" >&2
if ! "$GXX" -std=c++17 -I"$NVRTC_INCLUDE" -o "$PROBE_BIN" "$PROBE_SRC" \
    -L"$NVRTC_LIBDIR" -Wl,-rpath,"$NVRTC_LIBDIR" -lnvrtc; then
    echo "$PROG: RED — failed to compile/link the NVRTC probe driver itself" >&2
    exit 1
fi

# Canonical NVRTC option list — see tools/nvrtc/README.md.
ARCH_OPT="--gpu-architecture=$ARCH"
STD_OPT="--std=c++20"
REPO_OPT="--include-path=$REPO_ROOT"
SHIM_OPT="--include-path=$SHIM_DIR"
CUDASTD_OPT="--include-path=$CUDA_STD_INCLUDE"
PRE_OPT="--pre-include=$SHIM_DIR/__aether_nvrtc_prelude.h"
CANONICAL_OPTS=("$ARCH_OPT" "$STD_OPT" "$REPO_OPT" "$SHIM_OPT" "$CUDASTD_OPT" "$PRE_OPT")
echo "$PROG: canonical NVRTC option list: ${CANONICAL_OPTS[*]}" >&2

STATUS=0

# --- 1. DEVICE (gating GREEN) --------------------------------------------
echo "$PROG: [1/2] aether/device.h alone, materializing Item c = a + b (host-side compilation only, NO GPU execution, NO CUDA context created)" >&2
DEVICE_OUT="$("$PROBE_BIN" device "${CANONICAL_OPTS[@]}" 2>&1)"
DEVICE_RC=$?
if [ $DEVICE_RC -ne 0 ]; then
    echo "DEVICE: RED (rc=$DEVICE_RC) — aether/device.h FAILED to compile under NVRTC (should be GREEN):"
    echo "$DEVICE_OUT"
    STATUS=1
else
    echo "DEVICE: GREEN (rc=0) — aether/device.h parses+compiles and materializes Item c = a + b under NVRTC (--gpu-architecture=$ARCH)"
fi

# --- 2. CONTROL (gating RED, non-vacuity) --------------------------------
echo "$PROG: [2/2] full aether/aether.h umbrella (expect RED — host-only headers still unconditional there)" >&2
CONTROL_OUT="$("$PROBE_BIN" control "${CANONICAL_OPTS[@]}" 2>&1)"
CONTROL_RC=$?
if [ $CONTROL_RC -eq 0 ]; then
    echo "CONTROL: RED (rc=0) — aether/aether.h unexpectedly PASSED under NVRTC; this audit is VACUOUS (either aether.h became device-safe, update this script and tools/nvrtc/README.md, or the audit's own option derivation is broken)"
    STATUS=1
else
    CONTROL_LINE="$(printf '%s\n' "$CONTROL_OUT" | grep -m1 -E 'catastrophic error|error:')"
    echo "CONTROL: GREEN (rc=$CONTROL_RC, correctly RED) — $CONTROL_LINE"
fi

echo "=========================================="
echo "device.h rc=$DEVICE_RC (expect 0)   control (aether.h) rc=$CONTROL_RC (expect non-zero)"
if [ $STATUS -ne 0 ]; then
    echo "DEVICE UMBRELLA NVRTC AUDIT: RED"
else
    echo "DEVICE UMBRELLA NVRTC AUDIT: GREEN"
fi
exit $STATUS
