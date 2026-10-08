#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tools/dlpack_preone_audit.sh — de-risk probe driver for eagle's pre-1.0
# DLPack adapter.
#
# Compiles+runs tools/dlpack_preone_probe.cpp: a pure compile-time ABI
# comparison (static_assert/offsetof/sizeof) between aether's own vendored
# DLPack v1.0 header and eagle's vendored PRE-1.0 subset
# (eagle/plugin/dlpack.h, READ-ONLY — never modified by this probe). Host
# C++, no GPU, no CUDA toolchain needed at all.
#
# Usage: tools/dlpack_preone_audit.sh [path-to-SWDevel-workspace-root]
#   defaults to two levels up from this script (aether/tools/.. .. == the
#   SWDevel workspace root that also holds eagle/).
#
set -u -o pipefail

PROG="tools/dlpack_preone_audit.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${1:-$(cd "$REPO_ROOT/.." && pwd)}"

PROBE_SRC="$REPO_ROOT/tools/dlpack_preone_probe.cpp"
EAGLE_HDR="$WORKSPACE_ROOT/eagle/plugin/dlpack.h"

if [ ! -f "$PROBE_SRC" ]; then
    echo "$PROG: no probe source at $PROBE_SRC" >&2
    exit 1
fi
if [ ! -f "$EAGLE_HDR" ]; then
    echo "$PROG: eagle/plugin/dlpack.h not found under $WORKSPACE_ROOT (pass the SWDevel workspace root explicitly)" >&2
    exit 1
fi

GXX="$(command -v g++ || true)"
if [ -z "$GXX" ]; then
    echo "$PROG: g++ not found on PATH" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PROBE_BIN="$TMP/dlpack_preone_probe"

echo "$PROG: compiling $PROBE_SRC (-I$REPO_ROOT -I$WORKSPACE_ROOT), READ-ONLY eagle header untouched" >&2
if ! "$GXX" -std=c++23 -Wall -Wextra -I"$REPO_ROOT" -I"$WORKSPACE_ROOT" -o "$PROBE_BIN" "$PROBE_SRC"; then
    echo "$PROG: RED — the ABI comparison static_asserts (or a compile error) failed; see the diagnostic above" >&2
    exit 1
fi

echo "$PROG: running probe (prints the findings already proven at compile time — no GPU, no CUDA context)" >&2
"$PROBE_BIN"
exit $?
