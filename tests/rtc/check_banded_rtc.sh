#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# tests/rtc/check_banded_rtc.sh — the banded family must compile under NVRTC with
# ONLY the std headers served by the repo's own rtc/ shims (the include budget: any
# new std include is "file not found" and fails the row). The shims re-export CCCL's
# cuda::std, found under $CUDA_PATH/include (or $CCCL_INCLUDE).
#   ARM-COMPILES   banded_ok.cu (banded.h + BandedRealOps.h + a device chain) compiles.
#   ARM-RED        banded_host_symbol.cu (one host-only call) must FAIL to compile,
#                  so a pass of the first arm cannot be a blind compiler.
#   ARM-BUDGET     the shim directory holds exactly the allowed set.
# NVRTC is found via $NVRTC_ROOT, else $CUDA_PATH, else the pip wheel next to $PYTHON.
# Not finding it is a FAILURE (rc 2), never a skip.
# Usage: tests/rtc/check_banded_rtc.sh [<scratch dir>]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
scratch="${1:-$(mktemp -d)}"
mkdir -p "$scratch"

allowed="atomic cmath concepts cstddef cstdint type_traits utility"
have="$(ls "$root/rtc" | LC_ALL=C sort | tr '\n' ' ')"
want="$(echo $allowed | tr ' ' '\n' | LC_ALL=C sort | tr '\n' ' ')"
if [ "$have" != "$want" ]; then
    echo "FAIL ARM-BUDGET: rtc/ [$have] != allowed [$want]"
    exit 1
fi

inc=""; lib=""
for r in "${NVRTC_ROOT:-}" "${CUDA_PATH:-}"; do
    [ -n "$r" ] || continue
    if [ -f "$r/include/nvrtc.h" ]; then inc="$r/include"; lib="$(dirname "$(ls "$r"/lib64/libnvrtc.so* "$r"/lib/libnvrtc.so* 2>/dev/null | head -1)")"; break; fi
done
if [ -z "$inc" ] && [ -n "${PYTHON:-}" ]; then
    wheel="$("$PYTHON" -I -c 'import nvidia.cuda_nvrtc as m,os;print(os.path.dirname(m.__path__[0]) + "/cuda_nvrtc")' 2>/dev/null)"
    [ -n "$wheel" ] && [ -f "$wheel/include/nvrtc.h" ] && { inc="$wheel/include"; lib="$wheel/lib"; }
fi
libfile="$(ls "$lib"/libnvrtc.so* 2>/dev/null | grep -v builtins | grep -v alt | head -1)"
if [ -z "$inc" ] || [ -z "$libfile" ]; then
    echo "FAIL: no NVRTC (set NVRTC_ROOT)"; exit 2
fi

"${CXX:-g++}" -std=c++17 -I"$inc" "$here/nvrtc_compile.cpp" -o "$scratch/nvrtc_compile" "$libfile" -Wl,-rpath,"$lib" \
    || { echo "FAIL: driver build"; exit 2; }

"$scratch/nvrtc_compile" "$here/banded_ok.cu" "$root" "$root/rtc" "${CCCL_INCLUDE:-${CUDA_PATH:-}/include}" 2> "$scratch/ok.log"
rc=$?
if [ $rc -ne 0 ]; then echo "FAIL ARM-COMPILES (rc $rc)"; head -40 "$scratch/ok.log"; exit 1; fi

"$scratch/nvrtc_compile" "$here/banded_host_symbol.cu" "$root" "$root/rtc" "${CCCL_INCLUDE:-${CUDA_PATH:-}/include}" 2> "$scratch/red.log"
rc=$?
if [ $rc -ne 1 ]; then echo "FAIL ARM-RED: host-only symbol was not rejected (rc $rc)"; exit 1; fi
if ! grep -q "fromDouble" "$scratch/red.log"; then echo "FAIL ARM-RED: rejected, but not for fromDouble"; head -20 "$scratch/red.log"; exit 1; fi
echo "PASS banded NVRTC row (budget: $allowed)"
