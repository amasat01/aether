#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# ===========================================================================
#  tests/sass/check_host_tu_includes.sh — HOST-TU INCLUDE GATE.
#
#  THE REQUIREMENT. A CUDA-mode aether build may contain translation
#  units compiled by the HOST compiler (`g++`) as well as by nvcc. eagle's
#  own test suite compiles six host-only `.cpp` tests INSIDE its CUDA build;
#  against an aether build without this fix they died at
#      aether/device/Device.h's default Device() constructor: '__host__' does not name a type
#  because `AETHER_HOST()` expanded to a raw `__host__` in a TU no CUDA
#  header had ever defined it for.
#
#  THE TWO AXES this gate enforces (macros.h, `AETHER_DEVICE_COMPILER`):
#    BUILD MODE  (`AETHER_HAS_CUDA` xor `AETHER_CPP_MODE`) -> the member set.
#    COMPILER    (`__CUDACC__`)                            -> the legal SYNTAX.
#  A host TU in a CUDA-mode build must therefore see the same types and be
#  able to allocate / upload / download / view / index — and must NOT be
#  able to define a kernel or write a `<<<>>>` launch.
#
#  KNOWN ANSWERS, NOT INSPECTION. Every probe below has a definite expected
#  outcome checked by an actual compiler return code, and each REFUSAL probe
#  is paired with the SAME source compiled by nvcc, where it MUST succeed.
#  That pairing is what stops a refusal probe from passing for the wrong
#  reason (a typo, a missing include, a stale path): if the refusal probe
#  also failed under nvcc, the gate reports RED rather than a false green.
#  (Same discipline as tests/langsplit/check_langsplit.sh's canary arm.)
#
#  PROBES
#    1 umbrella_host      g++,  CUDA mode   -> MUST COMPILE  (the host-TU claim)
#    2 real_host_tu       g++,  CUDA mode   -> MUST COMPILE  (tests/hosttu/
#                                              host_tu_probe.cpp, the very
#                                              TU the CUDA test build links)
#    3 umbrella_cppmode   g++,  CPP_MODE    -> MUST COMPILE  (regression
#                                              guard: the macros.h rework
#                                              must not disturb CPP_MODE)
#    4 kernel_def         g++,  CUDA mode   -> MUST FAIL     (AETHER_KERNEL
#                                              is deliberately NOT DEFINED
#                                              for a host compiler)
#      kernel_def         nvcc, CUDA mode   -> MUST COMPILE  (control)
#    5 launch_call        g++,  CUDA mode   -> MUST FAIL     (runtimeEval on
#                                              a CUDA device: static_assert,
#                                              never a silent host loop over
#                                              device pointers)
#      launch_call        nvcc, CUDA mode   -> MUST COMPILE  (control)
#
#  Compile-only (`-fsyntax-only` / `-c`); no probe has or needs a `main()`,
#  and NOTHING here executes on the GPU.
#
#  Usage:
#    tests/sass/check_host_tu_includes.sh [repo-root] [g++-path] [cuda-include-dir]
#
#  All three arguments are optional: repo-root defaults to the tree this
#  script lives in, g++ to /usr/bin/g++ (else PATH), and the CUDA include
#  dir is derived from `nvcc` when not given. Wired as a SCRIPT, not a
#  ctest — invoke it directly.
# ===========================================================================
set -u -o pipefail

PROG="tests/sass/check_host_tu_includes.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${1:-}"
CXX="${2:-}"
CUDA_INC="${3:-}"

[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

if [ -z "$CXX" ]; then
    if [ -x /usr/bin/g++ ]; then CXX=/usr/bin/g++; else CXX="$(command -v g++ || true)"; fi
fi
NVCC="$(command -v nvcc || true)"

if [ -z "$CXX" ] || [ ! -x "$CXX" ]; then
    echo "$PROG: g++ not found (pass it as \$2, or put it on PATH)" >&2
    exit 2
fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
    echo "$PROG: nvcc not found (needed for the CONTROL arms; put it on PATH)" >&2
    exit 2
fi
if [ -z "$CUDA_INC" ]; then
    # `dirname nvcc`/../include is NOT enough: a conda-hosted nvcc keeps the
    # runtime headers under targets/<triple>/include, and it is those (not
    # the bin-adjacent ones) that carry vector_types.h. Probe, in order, the
    # layouts this house actually has.
    for cand in \
        "$(dirname "$NVCC")/../targets/x86_64-linux/include" \
        "$(dirname "$NVCC")/../include" \
        /usr/local/cuda/include; do
        if [ -f "$cand/vector_types.h" ]; then CUDA_INC="$(cd "$cand" && pwd)"; break; fi
    done
fi
if [ -z "$CUDA_INC" ] || [ ! -f "$CUDA_INC/vector_types.h" ]; then
    echo "$PROG: no CUDA include dir with vector_types.h found (pass it as \$3)" >&2
    exit 2
fi
if [ ! -f "$REPO_ROOT/aether/aether.h" ]; then
    echo "$PROG: '$REPO_ROOT' does not look like an aether tree (no aether/aether.h)" >&2
    exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The CUDA-mode define set a real consumer sees: exactly what root
# CMakeLists.txt puts on the aether INTERFACE target when AETHER_CPP_MODE is
# OFF. AETHER_CPP_MODE is NOT among them, and must never be added per-source
# to make a host TU compile (that flips the member set for one TU only).
CUDA_MODE_DEFS=(-DAETHER_HAS_CUDA=1 -DAETHER_WARP_SIZE=32)
CPP_MODE_DEFS=(-DAETHER_CPP_MODE=1 -DAETHER_WARP_SIZE=32)
HOST_FLAGS=(-std=c++23 -I"$REPO_ROOT" -I"$CUDA_INC" -fsyntax-only)
# -arch=sm_61 is aether's LOWEST configured architecture, and the control
# arms need it: atomicAdd(double*, double) (aether/accum/atomic.h) exists
# only from sm_60, so nvcc's bare default would make the controls fail for a
# reason unrelated to the host-TU claim.
NVCC_FLAGS=(-std=c++20 -x cu -arch=sm_61 -Wno-deprecated-gpu-targets -I"$REPO_ROOT" -c)

RED=0
declare -a SUMMARY=()

# report <label> <expected: PASS|FAIL> <actual rc> <log>
judge() {
    local label="$1" expect="$2" rc="$3" log="$4"
    local actual="PASS"
    [ "$rc" -eq 0 ] || actual="FAIL"
    SUMMARY+=("$label=$actual(want $expect)")
    if [ "$actual" != "$expect" ]; then
        RED=1
        echo "GATE RED: $label -> $actual, expected $expect" >&2
        sed 's/^/  | /' "$log" | head -25 >&2
    fi
}

# --- probe sources ---------------------------------------------------------
cat > "$TMP/umbrella.cpp" <<'EOF'
// The host-TU claim in its smallest form: a plain C++ TU includes the CUDA-mode
// aether umbrella and uses a type from it.
#include <aether/aether.h>
aether::Device r8ProbeDevice() { return aether::Device(kDLCPU, 0); }
EOF

cat > "$TMP/kernel_def.cu" <<'EOF'
// REFUSAL PROBE: a kernel DEFINITION. Legal for nvcc, and a loud compile
// error for the host compiler, because AETHER_KERNEL is deliberately left
// UNDEFINED there rather than expanded to nothing (which would silently
// produce a host function of the same name).
#include <aether/aether.h>
AETHER_KERNEL() void r8ProbeKernel(float* p) { p[0] = 1.0F; }
EOF

cat > "$TMP/launch_call.cu" <<'EOF'
// REFUSAL PROBE: the one aether entry point that can reach a <<<>>> launch.
// Under nvcc it compiles; under the host compiler it must be REFUSED (a
// static_assert), never silently degrade to the serial host loop — that
// loop would dereference device pointers on the host.
#include <aether/aether.h>
void r8ProbeLaunch(const aether::RuntimeView& a, const aether::RuntimeView& out)
{
    aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, a, aether::RuntimeView{}, out);
}
EOF

# --- 1. umbrella, host compiler, CUDA mode: MUST COMPILE -------------------
"$CXX" "${HOST_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" "$TMP/umbrella.cpp" > "$TMP/p1.log" 2>&1
judge "umbrella_host" PASS "$?" "$TMP/p1.log"

# --- 2. the REAL committed host TU: MUST COMPILE ---------------------------
"$CXX" "${HOST_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" -Wall -Wextra \
    "$REPO_ROOT/tests/hosttu/host_tu_probe.cpp" > "$TMP/p2.log" 2>&1
judge "real_host_tu" PASS "$?" "$TMP/p2.log"

# --- 3. umbrella, host compiler, CPP_MODE: MUST COMPILE (regression) -------
"$CXX" -std=c++23 -I"$REPO_ROOT" -fsyntax-only "${CPP_MODE_DEFS[@]}" \
    "$TMP/umbrella.cpp" > "$TMP/p3.log" 2>&1
judge "umbrella_cppmode" PASS "$?" "$TMP/p3.log"

# --- 4. kernel definition: host MUST FAIL, nvcc MUST COMPILE --------------
"$CXX" "${HOST_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" -x c++ "$TMP/kernel_def.cu" > "$TMP/p4h.log" 2>&1
judge "kernel_def_host" FAIL "$?" "$TMP/p4h.log"
"$NVCC" "${NVCC_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" "$TMP/kernel_def.cu" -o "$TMP/p4.o" > "$TMP/p4n.log" 2>&1
judge "kernel_def_nvcc" PASS "$?" "$TMP/p4n.log"

# --- 5. launch path: host MUST FAIL, nvcc MUST COMPILE --------------------
"$CXX" "${HOST_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" -x c++ "$TMP/launch_call.cu" > "$TMP/p5h.log" 2>&1
judge "launch_call_host" FAIL "$?" "$TMP/p5h.log"
# ... and it must fail for the RIGHT REASON: the host-only static_assert,
# not a parse error and not an unrelated breakage.
if ! grep -q "kernel launches need nvcc" "$TMP/p5h.log"; then
    RED=1
    echo "GATE RED: launch_call_host failed, but NOT with the host-only static_assert" >&2
    echo "          (expected the message 'kernel launches need nvcc'):" >&2
    sed 's/^/  | /' "$TMP/p5h.log" | head -15 >&2
fi
if grep -qE "expected primary-expression before .<<|error: expected" "$TMP/p5h.log"; then
    RED=1
    echo "GATE RED: launch_call_host produced a PARSE error — this claim requires a" >&2
    echo "          clear diagnostic, not nvcc grammar leaking into g++:" >&2
    sed 's/^/  | /' "$TMP/p5h.log" | head -15 >&2
fi
"$NVCC" "${NVCC_FLAGS[@]}" "${CUDA_MODE_DEFS[@]}" "$TMP/launch_call.cu" -o "$TMP/p5.o" > "$TMP/p5n.log" 2>&1
judge "launch_call_nvcc" PASS "$?" "$TMP/p5n.log"

echo "HOST_TU_INCLUDES: ${SUMMARY[*]}"
if [ "$RED" -ne 0 ]; then
    echo "VERDICT: RED" >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
