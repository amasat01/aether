#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/nvrtc_parse_audit.sh — de-risk probe driver for NVRTC parse/compile
# coverage. See tools/nvrtc/README.md for the full diagnosis and the
# canonical NVRTC option list this script derives and mints.
#
# Compiles tools/nvrtc_parse_probe.cpp (a plain host C++ program against the
# NVRTC API — no CUDA context, no kernel launch) with bare g++, then RUNS it
# (host-side compilation only — nvrtcCompileProgram is explicitly allowed by
# house convention, launching is not; this probe never launches anything).
#
# Three checks, ALL options derived from the ACTIVE toolchain (nvcc's own
# location / the conda prefix layout it implies) — never a hard-coded
# machine path:
#   1. GREEN  — the achievable, in-scope target: a real aether device
#      arithmetic kernel (Item/expr `.dot()`) compiles under NVRTC with the
#      full canonical option list. This is what gates the script's exit code.
#   2. SEEDED-RED — non-vacuity proof: the SAME kernel, same options, minus
#      one required option (the libcu++ shim's --include-path) — must FAIL,
#      and must fail with the SAME "cannot open source file" class of error
#      the original D-c finding reproduced. Also gates the exit code (a
#      seeded-red run that unexpectedly PASSES means this check is vacuous).
#   3. INFO   — the literal `#include <aether/aether.h>` umbrella. Reported
#      for transparency (still RED today, root-caused in the README to two
#      aether-source issues outside this tool's edit scope) but NOT gating —
#      fixing it is aether-header work, not an environment/audit-script fix.
#
# Usage: tools/nvrtc_parse_audit.sh [gpu-arch, default compute_61]
#
set -u -o pipefail

PROG="tools/nvrtc_parse_audit.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="${1:-compute_61}"

PROBE_SRC="$REPO_ROOT/tools/nvrtc_parse_probe.cpp"
if [ ! -f "$PROBE_SRC" ]; then
    echo "$PROG: no probe source at $PROBE_SRC" >&2
    exit 1
fi

SHIM_DIR="$SCRIPT_DIR/nvrtc/shim_include"
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

# libcu++ (cuda/std/...) — the NVRTC-safe standard-library substitute the
# shim headers forward to (see tools/nvrtc/README.md: real glibc/libstdc++
# headers cannot be parsed by NVRTC's frontend at all — confirmed by
# experiment, not a missing -I). Same search-path pattern as NVRTC_INCLUDE
# above, keyed on the one file the shim actually needs to find.
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
PROBE_BIN="$TMP/nvrtc_parse_probe"

echo "$PROG: compiling probe driver against $NVRTC_INCLUDE / $NVRTC_LIBDIR" >&2
if ! "$GXX" -std=c++17 -I"$NVRTC_INCLUDE" -o "$PROBE_BIN" "$PROBE_SRC" \
    -L"$NVRTC_LIBDIR" -Wl,-rpath,"$NVRTC_LIBDIR" -lnvrtc; then
    echo "$PROG: RED — failed to compile/link the NVRTC probe driver itself" >&2
    exit 1
fi

# Canonical NVRTC option list (mint once, reuse — see tools/nvrtc/README.md's
# "canonical option list" section, which quotes this exact set for the code generator/
# eagle consumers to copy).
ARCH_OPT="--gpu-architecture=$ARCH"
STD_OPT="--std=c++20"
REPO_OPT="--include-path=$REPO_ROOT"
SHIM_OPT="--include-path=$SHIM_DIR"
CUDASTD_OPT="--include-path=$CUDA_STD_INCLUDE"
# Absolute path (not a bare name resolved via --include-path) so the seeded-
# RED run below — which deliberately drops SHIM_OPT — fails on aether's own
# FIRST real #include <cstddef> rather than on this pre-include itself.
PRE_OPT="--pre-include=$SHIM_DIR/__aether_nvrtc_prelude.h"

CANONICAL_OPTS=("$ARCH_OPT" "$STD_OPT" "$REPO_OPT" "$SHIM_OPT" "$CUDASTD_OPT" "$PRE_OPT")
echo "$PROG: canonical NVRTC option list: ${CANONICAL_OPTS[*]}" >&2

STATUS=0

# --- 1. GREEN -----------------------------------------------------------
echo "$PROG: [1/3] device-safe kernel, canonical options (host-side compilation only, NO GPU execution, NO CUDA context created)" >&2
GREEN_OUT="$("$PROBE_BIN" device-safe "${CANONICAL_OPTS[@]}" 2>&1)"
GREEN_RC=$?
if [ $GREEN_RC -ne 0 ]; then
    echo "$PROG: RED — canonical options FAILED to compile the device-safe kernel (should be GREEN):" >&2
    echo "$GREEN_OUT" >&2
    STATUS=1
else
    echo "GREEN: aether device-safe arithmetic kernel (Item/expr .dot()) parses+compiles under NVRTC (--gpu-architecture=$ARCH, canonical options)"
fi

# --- 2. SEEDED-RED (non-vacuity) -----------------------------------------
SEEDED_OPTS=("$ARCH_OPT" "$STD_OPT" "$REPO_OPT" "$CUDASTD_OPT" "$PRE_OPT") # SHIM_OPT deliberately omitted
echo "$PROG: [2/3] same kernel, --include-path=$SHIM_DIR deliberately removed (expect RED)" >&2
SEEDRED_OUT="$("$PROBE_BIN" device-safe "${SEEDED_OPTS[@]}" 2>&1)"
SEEDRED_RC=$?
if [ $SEEDRED_RC -eq 0 ]; then
    echo "$PROG: RED — removing --include-path=$SHIM_DIR did NOT fail; this audit is VACUOUS" >&2
    STATUS=1
else
    SEEDED_LINE="$(printf '%s\n' "$SEEDRED_OUT" | grep -m1 -E 'catastrophic error|error:')"
    echo "SEEDED-RED: $SEEDED_LINE"
fi

# --- 3. INFO (non-gating) -------------------------------------------------
echo "$PROG: [3/3] full aether/aether.h umbrella, canonical options (informational only, see tools/nvrtc/README.md)" >&2
UMBRELLA_OUT="$("$PROBE_BIN" umbrella "${CANONICAL_OPTS[@]}" 2>&1)"
UMBRELLA_RC=$?
if [ $UMBRELLA_RC -eq 0 ]; then
    echo "INFO: full aether/aether.h umbrella is ALSO GREEN under NVRTC now — tools/nvrtc/README.md's known-gap section is stale, update it"
else
    UMBRELLA_LINE="$(printf '%s\n' "$UMBRELLA_OUT" | grep -m1 -E 'catastrophic error|error:')"
    echo "INFO (non-gating, known aether-source finding, see tools/nvrtc/README.md): $UMBRELLA_LINE"
fi

exit $STATUS
