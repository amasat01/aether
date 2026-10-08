#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/headers/check_device_header_standalone.sh — a "host compile-probe TU":
# a translation unit that #includes
# ONLY `aether/device.h` (no `aether/aether.h`, no `view/MakeView.h` or any
# other split-out `Make*.h`) and instantiates a materializing
# `Item c = a + b` over `View<double>` — in BOTH build modes.
#
# This is a HOST-side compile check (plain g++/nvcc, no NVRTC, no GPU) — it
# answers "does a real consumer TU that reaches for nothing but device.h
# compile, on this toolchain, right now", complementing
# `tools/nvrtc/audit_device_umbrella.sh` (which answers the NVRTC/JIT-mode
# question) and `tests/compile_fail/check_device_umbrella_hostfree.sh`
# (which answers "does device.h stay HOST-FREE"). None of the three
# subsumes another.
#
# Modes:
#   cpp   g++ -std=c++23 -DAETHER_CPP_MODE=1 -c   (no CUDA toolchain needed)
#   cuda  nvcc -std=c++20 -x cu -c                (host+device passes, no
#         GPU execution — nvcc compiling a .cu file to an object needs no
#         device present, only RUNNING one would)
#
# Usage: tests/headers/check_device_header_standalone.sh [<aether repo root>]
# Exit code IS the verdict (0 = both modes compiled).

set -uo pipefail

REPO_ROOT="${1:-${AETHER_REPO_ROOT:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DEVICE_H="${REPO_ROOT}/aether/device.h"
if [[ ! -f "${DEVICE_H}" ]]; then
    echo "WARNING [fatal]: '${DEVICE_H}' does not exist — REPO_ROOT is wrong" >&2
    exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
FAIL=0

# ---------------------------------------------------------------------------
# The probe body itself — ONLY device.h, a materializing Item c = a + b over
# View<double>. Written once, compiled
# under both toolchains below.
# ---------------------------------------------------------------------------
cat > "${WORK}/probe.src" <<EOF
#include "${DEVICE_H}"

int main()
{
    using Ext = aether::extents<3, aether::dyn>;
    Ext ext(1);
    aether::layout_right::mapping<Ext> map(ext);
    double abuf[3] = { 1.0, 2.0, 3.0 };
    double bbuf[3] = { 4.0, 5.0, 6.0 };
    aether::View<double, Ext> av(abuf, map, aether::Device(kDLCPU));
    aether::View<double, Ext> bv(bbuf, map, aether::Device(kDLCPU));
    const aether::SampleIndex i = aether::SampleIndex::make(0);

    aether::Item<double, 3> a = av[i].get();
    aether::Item<double, 3> b = bv[i].get();
    aether::Item<double, 3> c = a + b; /* materializing ET assignment */

    return (c(0) == 5.0 && c(1) == 7.0 && c(2) == 9.0) ? 0 : 1;
}
EOF

echo "=========================================="
echo "  device.h standalone compile-probe (both modes)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---- cpp mode ---------------------------------------------------------------
CXX="${CXX:-g++}"
if ! command -v "${CXX}" > /dev/null 2>&1; then
    echo "ARM CPP-MODE: VERDICT RED — compiler '${CXX}' not found"
    FAIL=1
else
    cp "${WORK}/probe.src" "${WORK}/probe_cpp.cpp"
    log="${WORK}/cpp.log"
    if "${CXX}" -std=c++23 -DAETHER_CPP_MODE=1 -I"${REPO_ROOT}" -c "${WORK}/probe_cpp.cpp" \
        -o "${WORK}/probe_cpp.o" > "${log}" 2>&1; then
        echo "ARM CPP-MODE: PASS (${CXX}, AETHER_CPP_MODE, rc=0)"
    else
        echo "ARM CPP-MODE: VERDICT RED — device.h-only TU failed to compile in AETHER_CPP_MODE"
        sed -n '1,40p' "${log}" >&2
        FAIL=1
    fi
fi

# ---- cuda mode --------------------------------------------------------------
NVCC="${NVCC:-nvcc}"
if ! command -v "${NVCC}" > /dev/null 2>&1; then
    echo "ARM CUDA-MODE: VERDICT RED — compiler '${NVCC}' not found"
    FAIL=1
else
    cp "${WORK}/probe.src" "${WORK}/probe_cuda.cu"
    log="${WORK}/cuda.log"
    # -c only: compiles host+device passes to an object file. Needs no GPU
    # present -- only RUNNING the result would (never done here, house
    # convention: NO GPU execution by companions).
    if "${NVCC}" -std=c++20 -arch=sm_61 -DAETHER_HAS_CUDA=1 -I"${REPO_ROOT}" -c "${WORK}/probe_cuda.cu" \
        -o "${WORK}/probe_cuda.o" > "${log}" 2>&1; then
        echo "ARM CUDA-MODE: PASS (${NVCC}, sm_61, rc=0, compile-only — no GPU execution)"
    else
        echo "ARM CUDA-MODE: VERDICT RED — device.h-only TU failed to compile under nvcc"
        sed -n '1,40p' "${log}" >&2
        FAIL=1
    fi
fi

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "DEVICE HEADER STANDALONE PROBE: RED"
    exit 1
fi
echo "DEVICE HEADER STANDALONE PROBE: GREEN"
