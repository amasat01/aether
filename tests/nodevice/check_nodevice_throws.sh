#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# aether's "no device" condition is structural, not runtime-masked: a
# CPP_MODE build has AETHER_HAS_CUDA undefined, so there is no CUDA backend
# AT ALL — Chunk::allocate(kDLCUDA/kDLCUDAHost, ...) throws aether::Error
# unconditionally (aether/chunk/Chunk.h's #else arm), no device probe, no
# skip branch. tests/test_NoDeviceCheck.cpp's two cases therefore RUN for
# real on every ordinary cpp-mode gate already; this script's OWN job is a
# non-vacuity discipline: pin the expected count FROM THE MANIFEST (never a
# literal), filter to exactly NoDeviceCheck.*, and assert ran==passed==pinned
# with ZERO skips — an ambient GTEST_FILTER or a silently-dropped TU (the
# configure-time GLOB) reds here instead of passing vacuously.
#
# Usage: check_nodevice_throws.sh <path/to/aether_tests>   (CPP_MODE binary)
#
set -u -o pipefail

BIN="${1:?usage: check_nodevice_throws.sh <path/to/aether_tests>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/../expected_tests_cpp.txt"

if [ ! -x "$BIN" ]; then
    echo "NODEVICE CHECK GATE: RED — '$BIN' is not an executable file" >&2
    exit 1
fi

PINNED=$(grep -c '^NoDeviceCheckTest\.' "$MANIFEST" || true)
if [ "$PINNED" -lt 1 ]; then
    echo "NODEVICE CHECK GATE: RED — the manifest pins ZERO NoDeviceCheck tests" \
         "(TU dropped by the configure-time GLOB, or manifest never re-minted)" >&2
    exit 1
fi

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

RC=0
env -u GTEST_FILTER -u TESTBRIDGE_TEST_ONLY \
    "$BIN" --gtest_filter='NoDeviceCheckTest.*' > "$LOG" 2>&1 || RC=$?

RAN=$(grep -oP '^\[==========\] \K[0-9]+(?= tests? from .* ran)' "$LOG" | tail -1)
PASSED=$(grep -oP '^\[  PASSED  \] \K[0-9]+(?= tests?)' "$LOG" | tail -1)
SKIPPED=$(grep -c '^\[  SKIPPED \]' "$LOG" || true)

if [ "$RC" -ne 0 ]; then
    echo "NODEVICE CHECK GATE: RED — binary rc=$RC (a crash here is the class" \
         "of defect this gate exists to catch: a discarded CUDA status" \
         "reaching an unwritten out-parameter)" >&2
    sed -n '1,40p' "$LOG" >&2
    exit 1
fi
if [ -z "${RAN:-}" ] || [ "$RAN" -ne "$PINNED" ] \
   || [ -z "${PASSED:-}" ] || [ "$PASSED" -ne "$PINNED" ] \
   || [ "$SKIPPED" -ne 0 ]; then
    echo "NODEVICE CHECK GATE: RED — ran=${RAN:-<none>} passed=${PASSED:-<none>}" \
         "skipped=$SKIPPED vs $PINNED pinned (a skip or an ambient filter is not a pass)" >&2
    sed -n '1,40p' "$LOG" >&2
    exit 1
fi

echo "NODEVICE CHECK GATE: GREEN ($PASSED/$PINNED NoDeviceCheck tests threw typed aether::Error, CPP_MODE has no CUDA backend to skip around)"
