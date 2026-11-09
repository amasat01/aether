#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Checks of the aarch64 GCC route of aether::math::pow (aether/math/detail/HostVectorMath.h).
#   check_hvm_arm.sh <g++>
# On aarch64 with GCC, for AETHER_HVM_MASKED=0 and =1: the unconditional loop of
# hvm_arm_repro.cpp is vectorised, its results are right, the vector variant is defined
# (the masked one exactly under 1), and no libm pow is imported. Anywhere else it says
# SKIP and exits 0.
set -u
CXX=${1:?usage: check_hvm_arm.sh <g++>}
if [ "$(uname -m)" != aarch64 ] || "$CXX" -dM -E -x c++ /dev/null | grep -q __clang__; then
    echo "SKIP: needs GCC on aarch64"
    exit 0
fi
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC=$HERE/hvm_arm_repro.cpp
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FLAGS=(-std=c++23 -O3 -ffp-contract=fast -fno-trapping-math -fno-math-errno -mcpu=native -DAETHER_CPP_MODE -DAETHER_HOST_VECTOR_MATH "-I$ROOT")
fail() { echo "FAIL: $*"; exit 1; }
ln=$(grep -n "// LOOP$" "$SRC" | head -1 | cut -d: -f1)
for v in 0 1; do
    "$CXX" "${FLAGS[@]}" -DAETHER_HVM_MASKED=$v -fopt-info-vec-optimized -c "$SRC" -o "$TMP/$v.o" 2> "$TMP/$v.vec" \
        || { cat "$TMP/$v.vec"; fail "MASKED=$v: build failed"; }
    "$CXX" "$TMP/$v.o" -o "$TMP/$v.exe" || fail "MASKED=$v: link failed"
    grep -q "hvm_arm_repro.cpp:$ln:[0-9]*: optimized: loop vectorized" "$TMP/$v.vec" || fail "MASKED=$v: the loop is not vectorised"
    "$TMP/$v.exe" || fail "MASKED=$v: wrong results"
    nm "$TMP/$v.o" | grep -q ' [TtWw] _ZGVnN2vv_aether_hvm_pow$' || fail "MASKED=$v: _ZGVnN2vv_aether_hvm_pow is not defined"
    if nm "$TMP/$v.o" | grep -q '_ZGVnM2vv_aether_hvm_pow'; then masked=1; else masked=0; fi
    [ "$masked" = "$v" ] || fail "MASKED=$v: masked variant present=$masked"
    if nm -u "$TMP/$v.o" | grep -qE ' pow(@|$)'; then fail "MASKED=$v: libm pow is still called"; fi
    echo "MASKED=$v ok"
done
echo "PASS"
