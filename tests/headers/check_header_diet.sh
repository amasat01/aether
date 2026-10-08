#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/headers/check_header_diet.sh — include-graph gate for aether's
# umbrella header diet. Technique: nvcc --keep's dual preprocessed pass
# (`<tu>.cpp1.ii` is the DEVICE pass, `<tu>.cpp4.ii` is the HOST pass;
# provenance comes from the `# <line> "<file>"` linemarkers gcc's
# preprocessor emits) pointed at aether's OWN documented umbrella
# exclusions:
#
#   `aether/aether.h`'s own doc comment: "Deliberately does NOT include
#   aether/dtype/Format.h -- that header is host-only pretty-printing kept
#   separate precisely so the umbrella stays device-safe"; and "Deliberately
#   does NOT include interop/Mdspan.h -- same reasoning ... (host-only, and
#   on this workspace's pinned toolchain also environment-gated inert)".
#
# A third exclusion is a COST rather than a layering preference:
# `aether/banded/banded.h`. That umbrella reaches the whole emulated
# carrier and its codec, and putting it in the library umbrella would hand
# every TU that computes `sin(double)` about two thousand lines it has no
# use for. The `aether::math` facade routing for the banded family is
# reached instead through the DECLARATION-ONLY
# `aether/math/detail/BandedFwd.h` (which `math/detail/MathDispatch.h`
# includes and which names no definitions at all), so the routing exists
# in every math TU while the definitions do not. THAT header is expected
# in the umbrella's passes; only `banded/banded.h` must be absent.
#
# aether's invariant is UNCONDITIONAL: the umbrella never `#include`s
# either header on ANY path, so both passes must show absence.
#
# Arms (every one mandatory, RC alone is never the verdict):
#   UMBRELLA-DEVICE-NO-FORMAT   umbrella TU, device pass: no dtype/Format.h
#   UMBRELLA-HOST-NO-FORMAT     umbrella TU, host pass:   no dtype/Format.h
#   UMBRELLA-DEVICE-NO-MDSPAN   umbrella TU, device pass: no interop/Mdspan.h
#   UMBRELLA-HOST-NO-MDSPAN     umbrella TU, host pass:   no interop/Mdspan.h
#   UMBRELLA-DEVICE-NO-BANDED   umbrella TU, device pass: no banded/banded.h
#   UMBRELLA-HOST-NO-BANDED     umbrella TU, host pass:   no banded/banded.h
#   CONTROL-FORMAT-VISIBLE      a TU that DOES #include dtype/Format.h:
#                               present in BOTH passes (the absence arms
#                               above are observable, not a blind grep)
#   CONTROL-BANDED-VISIBLE      a TU that DOES #include banded/banded.h:
#                               present in BOTH passes (same non-vacuity role)
#   BANDEDFWD-PRESENT           the declaration-only aether/math/detail/
#                               BandedFwd.h IS present in the umbrella's device
#                               pass. Not decoration: it is what makes the
#                               absence arm above a statement about the COST
#                               rather than about the ROUTING -- if the fwd
#                               header vanished too, the banded facade entries
#                               would simply not exist and the diet would be
#                               "green" for the wrong reason.
#   CONTROL-MDSPAN-VISIBLE      a TU that DOES #include interop/Mdspan.h:
#                               present in BOTH passes (linemarker only —
#                               the header's BODY is __has_include(<mdspan>)-
#                               gated inert on this toolchain, Mdspan.h's
#                               own doc note; the FILE still gets pulled in
#                               and preprocessed, which is what the absence
#                               arms above are a claim about)
#
# Usage: tests/headers/check_header_diet.sh [<aether repo root>]
#
set -uo pipefail

REPO_ROOT="${1:-${AETHER_REPO_ROOT:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

NVCC="${NVCC:-nvcc}"
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

FAIL=0
red()  { echo "ARM $1: VERDICT RED — $2"; FAIL=1; }
pass() { echo "ARM $1: PASS — $2"; }

if ! command -v "${NVCC}" > /dev/null 2>&1; then
    echo "WARNING [fatal]: compiler '${NVCC}' not found; this gate cannot run"
    exit 1
fi

# PRECONDITION — gate the SOURCE TREE, never an installed copy: the
# files under test must exist under REPO_ROOT, and
# every probe includes its entry header by ABSOLUTE PATH.
for f in aether/aether.h aether/dtype/Format.h aether/interop/Mdspan.h \
         aether/banded/banded.h aether/math/detail/BandedFwd.h \
         aether/device.h aether/err/Error.h aether/chunk/Chunk.h; do
    if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
        echo "WARNING [fatal]: '${REPO_ROOT}/${f}' does not exist — REPO_ROOT is wrong" >&2
        exit 1
    fi
done

NVFLAGS=(-forward-unknown-to-host-compiler -std=c++20 -arch=sm_61
         -DAETHER_HAS_CUDA=1 -Wno-deprecated-gpu-targets -march=x86-64-v3 -fopenmp)

echo "=========================================="
echo "  header-diet gate (aether umbrella exclusions, ${NVCC}, sm_61)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---------------------------------------------------------------------------
# keep_and_split <name> <probe.cu>: nvcc --keep, then locate the device
# (.cpp1.ii) and host (.cpp4.ii) preprocessed passes it wrote.
# ---------------------------------------------------------------------------
keep_and_split() {
    local name="$1" src="$2"
    local dir="${WORK}/${name}"
    mkdir -p "${dir}"
    ( cd "${dir}" && "${NVCC}" "${NVFLAGS[@]}" -I"${REPO_ROOT}" --keep -c "${src}" -o "${name}.o" ) \
        > "${dir}/nvcc.log" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "WARNING [fatal]: nvcc --keep failed for ${name} (rc=${rc})" >&2
        sed -n '1,40p' "${dir}/nvcc.log" >&2
        exit 1
    fi
    DEVICE_II=$(ls "${dir}"/*.cpp1.ii 2>/dev/null | head -1)
    HOST_II=$(ls "${dir}"/*.cpp4.ii 2>/dev/null | head -1)
    if [[ -z "${DEVICE_II}" || -z "${HOST_II}" ]]; then
        echo "WARNING [fatal]: --keep did not produce both .cpp1.ii (device) and" \
             ".cpp4.ii (host) for ${name} — nvcc --keep's layout changed" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Probe 1: the umbrella itself, a trivial device kernel.
# ---------------------------------------------------------------------------
cat > "${WORK}/umbrella.cu" <<EOF
#include "${REPO_ROOT}/aether/aether.h"
__global__ void headerDietProbeKernel(double* p) { if (p) *p = 0.0; }
EOF
keep_and_split umbrella "${WORK}/umbrella.cu"
UMBRELLA_DEVICE_II="${DEVICE_II}"; UMBRELLA_HOST_II="${HOST_II}"

for tag in FORMAT:aether/dtype/Format.h MDSPAN:aether/interop/Mdspan.h \
           BANDED:aether/banded/banded.h; do
    label="${tag%%:*}"; header="${tag#*:}"
    if grep -qF "\"${REPO_ROOT}/${header}\"" "${UMBRELLA_DEVICE_II}"; then
        red "UMBRELLA-DEVICE-NO-${label}" "${header} IS present in the umbrella's DEVICE pass — the umbrella now includes it (accidentally or by design); update aether/aether.h's own doc note either way"
    else
        pass "UMBRELLA-DEVICE-NO-${label}" "${header} absent from the umbrella's device pass"
    fi
    if grep -qF "\"${REPO_ROOT}/${header}\"" "${UMBRELLA_HOST_II}"; then
        red "UMBRELLA-HOST-NO-${label}" "${header} IS present in the umbrella's HOST pass"
    else
        pass "UMBRELLA-HOST-NO-${label}" "${header} absent from the umbrella's host pass"
    fi
done

# ---------------------------------------------------------------------------
# Probe 2 (control, non-vacuity): a TU that explicitly includes BOTH
# excluded headers directly — proves the grep above is observable, not
# blind to everything.
# ---------------------------------------------------------------------------
cat > "${WORK}/control.cu" <<EOF
#include "${REPO_ROOT}/aether/aether.h"
#include "${REPO_ROOT}/aether/dtype/Format.h"
#include "${REPO_ROOT}/aether/interop/Mdspan.h"
#include "${REPO_ROOT}/aether/banded/banded.h"
__global__ void headerDietControlKernel(double* p) { if (p) *p = 0.0; }
EOF
keep_and_split control "${WORK}/control.cu"

for tag in FORMAT:aether/dtype/Format.h MDSPAN:aether/interop/Mdspan.h \
           BANDED:aether/banded/banded.h; do
    label="${tag%%:*}"; header="${tag#*:}"
    if grep -qF "\"${REPO_ROOT}/${header}\"" "${DEVICE_II}" && grep -qF "\"${REPO_ROOT}/${header}\"" "${HOST_II}"; then
        pass "CONTROL-${label}-VISIBLE" "${header} present in BOTH passes when explicitly included — the absence arms above are a real observation, not a blind grep"
    else
        red "CONTROL-${label}-VISIBLE" "${header} did NOT appear even when explicitly #included — the grep pattern (or --keep's linemarker format) is broken, and the absence arms above are certifying nothing"
    fi
done

# ---------------------------------------------------------------------------
# Probe 3: the DECLARATION-ONLY banded forward header MUST be present in
# the umbrella's device pass. Without this arm the BANDED absence arm above
# would go green on a build that had lost the banded facade routing entirely,
# which is the wrong kind of green: the diet is a claim about COST, not about
# the routing having been deleted.
# ---------------------------------------------------------------------------
BANDEDFWD="aether/math/detail/BandedFwd.h"
if grep -qF "\"${REPO_ROOT}/${BANDEDFWD}\"" "${UMBRELLA_DEVICE_II}"; then
    pass "BANDEDFWD-PRESENT" "${BANDEDFWD} present in the umbrella's device pass -- the banded facade routing exists in every math TU while its DEFINITIONS do not, which is what the BANDED absence arms are a claim about"
else
    red "BANDEDFWD-PRESENT" "${BANDEDFWD} is ABSENT from the umbrella's device pass -- the banded facade routing is gone, so UMBRELLA-*-NO-BANDED above is green for the wrong reason"
fi

# ---------------------------------------------------------------------------
# aether/device.h's own diet. UNCONDITIONAL absence, unlike the
# aether.h arms above (which distinguish device vs host pass) -- device.h
# is meant to carry NOTHING host-only in EITHER pass: it is the umbrella
# tools/nvrtc/audit_device_umbrella.sh compiles standalone under NVRTC,
# which has no host pass of its own at all. err/Error.h is the load-bearing
# one (the exact header whose transitive reachability through view/View.h
# was the root NVRTC finding); chunk/Chunk.h is the second, independent
# host-only header make_view()'s Chunk overload used to pull in.
#
#   DEVICE-DEVICE-NO-ERR/DEVICE-HOST-NO-ERR        err/Error.h absent, device.h TU, both passes
#   DEVICE-DEVICE-NO-CHUNK/DEVICE-HOST-NO-CHUNK    chunk/Chunk.h absent, device.h TU, both passes
#   CONTROL-DEVICE-ERR-VISIBLE                     a TU that DOES #include BOTH device.h and
#                                                   err/Error.h: err/Error.h present in BOTH passes
#                                                   (non-vacuity — the absence arms above are an
#                                                   observation, not a blind grep)
# ---------------------------------------------------------------------------
cat > "${WORK}/device.cu" <<EOF
#include "${REPO_ROOT}/aether/device.h"
__global__ void headerDietDeviceProbeKernel(double* p) { if (p) *p = 0.0; }
EOF
keep_and_split device "${WORK}/device.cu"
DEVICE_DEVICE_II="${DEVICE_II}"; DEVICE_HOST_II="${HOST_II}"

for tag in ERR:aether/err/Error.h CHUNK:aether/chunk/Chunk.h; do
    label="${tag%%:*}"; header="${tag#*:}"
    if grep -qF "\"${REPO_ROOT}/${header}\"" "${DEVICE_DEVICE_II}"; then
        red "DEVICE-DEVICE-NO-${label}" "${header} IS present in device.h's DEVICE pass -- device.h now pulls in a host-only header (accidentally or by design); update aether/device.h's own doc note either way"
    else
        pass "DEVICE-DEVICE-NO-${label}" "${header} absent from device.h's device pass"
    fi
    if grep -qF "\"${REPO_ROOT}/${header}\"" "${DEVICE_HOST_II}"; then
        red "DEVICE-HOST-NO-${label}" "${header} IS present in device.h's HOST pass -- device.h is meant to carry NOTHING host-only in EITHER pass (it is compiled standalone under NVRTC, which has no host pass at all)"
    else
        pass "DEVICE-HOST-NO-${label}" "${header} absent from device.h's host pass"
    fi
done

cat > "${WORK}/device_control.cu" <<EOF
#include "${REPO_ROOT}/aether/device.h"
#include "${REPO_ROOT}/aether/err/Error.h"
__global__ void headerDietDeviceControlKernel(double* p) { if (p) *p = 0.0; }
EOF
keep_and_split device_control "${WORK}/device_control.cu"
if grep -qF "\"${REPO_ROOT}/aether/err/Error.h\"" "${DEVICE_II}" && grep -qF "\"${REPO_ROOT}/aether/err/Error.h\"" "${HOST_II}"; then
    pass "CONTROL-DEVICE-ERR-VISIBLE" "aether/err/Error.h present in BOTH passes when explicitly included alongside device.h -- the absence arms above are a real observation, not a blind grep"
else
    red "CONTROL-DEVICE-ERR-VISIBLE" "aether/err/Error.h did NOT appear even when explicitly #included -- the grep pattern (or --keep's linemarker format) is broken, and the DEVICE-*-NO-ERR arms above are certifying nothing"
fi

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "HEADER DIET GATE: RED"
    exit 1
fi
echo "HEADER DIET GATE: GREEN"
