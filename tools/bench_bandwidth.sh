#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/bench_bandwidth.sh — bandwidth microbench driver.
# Builds `tools/bench_bandwidth.cu` (release-shaped: -O3) and runs it —
# unlike `tools/bandwidth_audit.sh` (ptxas/cuobjdump only, never launched),
# this script does execute a GPU binary.
#
# The binary itself performs 3 warmup launches + 5 timed reps per arm and
# prints the median — this script adds no statistics of its own, it is a
# thin build+run wrapper. Prints an evidence card (GB/s + achieved fraction
# of the pinned 140 GB/s P2000 peak per arm) — no pass/fail, predeclared
# comparison (exploratory, not confirmatory).
#
# Usage: CUDA_VISIBLE_DEVICES=<n> tools/bench_bandwidth.sh
#
set -u -o pipefail

PROG="tools/bench_bandwidth.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_ROOT/tools/bench_bandwidth.cu"
if [ ! -f "$SRC" ]; then
    echo "$PROG: no source at $SRC" >&2
    exit 1
fi

NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
    echo "$PROG: nvcc not found on PATH" >&2
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$PROG: CUDA_VISIBLE_DEVICES is unset; GPU selection and serialisation are the" \
         "caller's responsibility and this run may collide with another concurrent GPU user" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/bench_bandwidth"

echo "$PROG: building $SRC (sm_61, -O3, release-shaped)" >&2
"$NVCC" -arch=sm_61 -std=c++20 -O3 -DAETHER_HAS_CUDA=1 -I"$REPO_ROOT" -Xcompiler=-fopenmp \
    "$SRC" -o "$BIN" > "$TMP/build.log" 2>&1
BUILD_RC=$?
sed 's/^/  | /' "$TMP/build.log" >&2
if [ "$BUILD_RC" -ne 0 ]; then
    echo "$PROG: build FAILED (rc=$BUILD_RC)" >&2
    exit 1
fi

echo "$PROG: running $BIN — GPU execution" >&2
"$BIN"
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
    echo "$PROG: bench_bandwidth exited rc=$RUN_RC" >&2
    exit 1
fi
exit 0
