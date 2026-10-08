#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/langsplit/check_langsplit.sh — per-language standard split gate.
#
# aether's device code is capped at CUDA std 20 (nvcc's device-code standard
# ceiling) while host code targets C++23
# (CMakeLists.txt: target_compile_features(aether INTERFACE cxx_std_23)).
# A text-search "grep for c++23 constructs" gate cannot fail for the right
# reason — a compile PROBE with a KNOWN answer can.
# This script is that probe: three tiny TUs, each with a definite expected
# compile outcome, checked by actual compiler return code:
#
#   device_cxx20_pass.cu    -> MUST compile    at -std=c++20 (nvcc)   -> PASS
#   device_cxx23_canary.cu  -> MUST FAIL        at -std=c++20 (nvcc)  -> FAIL
#     (uses an explicit object parameter, `this Self&&` / P0847
#     "deducing this" — ill-formed before C++23; its rejection at std=20 is
#     the positive control proving the device std=20 cap is enforced, not
#     merely documented)
#   host_cxx23_pass.cpp     -> MUST compile    at -std=c++23 (g++)    -> PASS
#     (uses `if consteval`, P1938 — a C++23-only construct)
#
# Compile-only (`-c`); none of the three TUs has (or needs) a `main()`.
#
# Usage:
#   tests/langsplit/check_langsplit.sh [nvcc-path] [g++-path]
#
# Both arguments are optional; each defaults to the first match on PATH.
# Wired as a SCRIPT, not a ctest — invoke it directly, e.g. from a CI job
# step.
#
set -u -o pipefail

PROG="tests/langsplit/check_langsplit.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NVCC="${1:-}"
CXX="${2:-}"
[ -n "$NVCC" ] || NVCC="$(command -v nvcc || true)"
[ -n "$CXX" ]  || CXX="$(command -v g++ || true)"

if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
    echo "$PROG: nvcc not found (pass it as \$1, or put it on PATH)" >&2
    exit 2
fi
if [ -z "$CXX" ] || [ ! -x "$CXX" ]; then
    echo "$PROG: g++ not found (pass it as \$2, or put it on PATH)" >&2
    exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

RED=0

# ---------------------------------------------------------------------------
# 1. device_cxx20_pass.cu MUST compile at CUDA std 20.
# ---------------------------------------------------------------------------
if "$NVCC" -std=c++20 -I"$REPO_ROOT" -c "$SCRIPT_DIR/device_cxx20_pass.cu" \
    -o "$TMP/device_cxx20_pass.o" > "$TMP/r1.log" 2>&1; then
    R1="PASS"
else
    R1="FAIL"
    RED=1
    echo "GATE RED: device_cxx20_pass.cu did NOT compile at -std=c++20 (expected: compiles):" >&2
    sed 's/^/  | /' "$TMP/r1.log" >&2
fi

# ---------------------------------------------------------------------------
# 2. device_cxx23_canary.cu MUST FAIL at CUDA std 20 — the positive control.
# ---------------------------------------------------------------------------
if "$NVCC" -std=c++20 -I"$REPO_ROOT" -c "$SCRIPT_DIR/device_cxx23_canary.cu" \
    -o "$TMP/device_cxx23_canary.o" > "$TMP/r2.log" 2>&1; then
    R2="PASS"
    RED=1
    echo "GATE RED: device_cxx23_canary.cu COMPILED at -std=c++20 (expected: REJECTED)." >&2
    echo "          Its explicit object parameter is a C++23-only construct; if this" >&2
    echo "          compiles under std=20 the device-code language cap is not enforced." >&2
else
    R2="FAIL"
fi

# ---------------------------------------------------------------------------
# 3. host_cxx23_pass.cpp MUST compile at host C++23. AETHER_CPP_MODE is
#    defined explicitly here because this TU is compiled directly by g++,
#    outside the aether CMake target that would otherwise supply it — the
#    HOST()/DEVICEHOST() macros used inside need SOME mode to resolve to
#    (CPP_MODE => empty; CUDA mode => __host__/__device__, which plain g++
#    does not know).
# ---------------------------------------------------------------------------
if "$CXX" -std=c++23 -DAETHER_CPP_MODE -I"$REPO_ROOT" -c "$SCRIPT_DIR/host_cxx23_pass.cpp" \
    -o "$TMP/host_cxx23_pass.o" > "$TMP/r3.log" 2>&1; then
    R3="PASS"
else
    R3="FAIL"
    RED=1
    echo "GATE RED: host_cxx23_pass.cpp did NOT compile at -std=c++23 (expected: compiles):" >&2
    sed 's/^/  | /' "$TMP/r3.log" >&2
fi

echo "LANGSPLIT: device_cxx20_pass=$R1  device_cxx23_canary=$R2  host_cxx23_pass=$R3  (expected PASS/FAIL/PASS)"

if [ "$RED" -ne 0 ]; then
    echo "VERDICT: RED" >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
