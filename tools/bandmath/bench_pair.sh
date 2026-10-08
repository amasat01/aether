#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/bandmath/bench_pair.sh — bench driver, mirrors
# tools/bench_bandwidth.sh's own governance pattern: builds
# tools/bandmath/bench_pair.cu (release-shaped: -O3) and runs one arm.
#
# HOST ARM ("host", the default): runs directly — no GPU touched, safe to
# invoke directly.
# DEVICE ARM ("device"): GPU execution. Build-verified regardless of
# whether it is run (a full nvcc -DAETHER_HAS_CUDA=1 link, rc=0); running
# it needs a GPU, selected via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   tools/bandmath/bench_pair.sh host
#   CUDA_VISIBLE_DEVICES=<n> tools/bandmath/bench_pair.sh device
#
set -u -o pipefail

PROG="tools/bandmath/bench_pair.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$SCRIPT_DIR/bench_pair.cu"

ARM="${1:-host}"
if [ "$ARM" != "host" ] && [ "$ARM" != "device" ]; then
    echo "$PROG: usage: $PROG <host|device>" >&2
    exit 2
fi

NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
    echo "$PROG: nvcc not found on PATH" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/bench_pair_$ARM"

if [ "$ARM" = "host" ]; then
    echo "$PROG: building $SRC (host arm, sm_61, -O3)" >&2
    "$NVCC" -arch=sm_61 -std=c++20 -O3 -I"$REPO_ROOT" "$SRC" -o "$BIN" \
        > "$TMP/build.log" 2>&1
else
    echo "$PROG: building $SRC (device arm, sm_61, -O3, AETHER_HAS_CUDA)" >&2
    "$NVCC" -arch=sm_61 -std=c++20 -O3 -DAETHER_HAS_CUDA=1 -I"$REPO_ROOT" "$SRC" -o "$BIN" \
        > "$TMP/build.log" 2>&1
fi
BUILD_RC=$?
sed 's/^/  | /' "$TMP/build.log" >&2
if [ "$BUILD_RC" -ne 0 ]; then
    echo "$PROG: build FAILED (rc=$BUILD_RC)" >&2
    exit 1
fi

if [ "$ARM" = "host" ]; then
    echo "$PROG: running $BIN --host — no GPU touched" >&2
    "$BIN" --host
    exit $?
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$PROG: CUDA_VISIBLE_DEVICES is unset; GPU selection and serialisation are the" \
         "caller's responsibility and this run may collide with another concurrent GPU user" >&2
fi
echo "$PROG: running $BIN --device — GPU execution" >&2
"$BIN" --device
exit $?
