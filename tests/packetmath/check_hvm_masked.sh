#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Checks of AETHER_HVM_MASKED (aether/math/detail/HostVectorMath.h) with GCC.
#   check_hvm_masked.sh <g++> [--necessity]
# Default: builds hvm_masked_repro.cpp under the default macro value, requires
# all three loops correct, then for AETHER_HVM_MASKED=0 and =1 requires loops
# b and c vectorised, loop a vectorised only under 1, correct results under 0,
# no `_ZGV.M` symbol under 0 and some under 1.
# --necessity: builds with AETHER_HVM_MASKED=1 and passes only if loop a gives
# wrong results (the compiler carries the bug the default rule works around).
set -u
CXX=${1:?usage: check_hvm_masked.sh <g++> [--necessity]}
MODE=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC=$HERE/hvm_masked_repro.cpp
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FLAGS=(-std=c++23 -O3 -march=native -DAETHER_HOST_VECTOR_MATH -I"$ROOT")

build() { # <tag> [extra flags...] ; writes $TMP/<tag>.{exe,vec,o}
    local tag=$1; shift
    "$CXX" "${FLAGS[@]}" "$@" -fopt-info-vec-optimized -c "$SRC" -o "$TMP/$tag.o" 2> "$TMP/$tag.vec" || { echo "build $tag failed"; cat "$TMP/$tag.vec"; exit 2; }
    "$CXX" "$TMP/$tag.o" -o "$TMP/$tag.exe" || exit 2
}
fail() { echo "FAIL: $*"; exit 1; }
vectorised() { # <tag> <marker> : is the marked loop reported vectorised?
    local ln; ln=$(grep -n "// $2$" "$SRC" | head -1 | cut -d: -f1)
    grep -q "hvm_masked_repro.cpp:$ln:[0-9]*: optimized: loop vectorized" "$TMP/$1.vec"
}

if [ "$MODE" = "--necessity" ]; then
    build m1 -DAETHER_HVM_MASKED=1
    out=$("$TMP/m1.exe"); echo "$out"
    echo "$out" | grep -q '^a: MISMATCH' || fail "loop a is correct with masked variants: this compiler does not carry the bug"
    echo "$out" | grep -q '^b: OK' && echo "$out" | grep -q '^c: OK' || fail "loops b/c must stay correct"
    echo "NECESSITY OK: masked variants miscompile loop a on this compiler"
    exit 0
fi

build def
out=$("$TMP/def.exe"); echo "default: $out" | tr '\n' ' '; echo
[ "$(echo "$out" | grep -c ': OK')" = 3 ] || fail "default macro: a loop gives wrong results"

for v in 0 1; do
    build m$v -DAETHER_HVM_MASKED=$v
    vectorised m$v LOOP_B || fail "MASKED=$v: loop b not vectorised"
    vectorised m$v LOOP_C || fail "MASKED=$v: loop c not vectorised"
    if [ $v = 0 ]; then
        vectorised m0 LOOP_A && fail "MASKED=0: loop a still vectorised"
        "$TMP/m0.exe" > /dev/null || fail "MASKED=0: wrong results"
        nm "$TMP/m0.o" | grep -q '_ZGV.M' && fail "MASKED=0: masked symbol present"
    else
        # GCC 12 and 13.4 decline to if-convert the conditional call even with
        # masked variants; a compiler that does (14, and the buggy 13.1-13.3)
        # must report the loop vectorised, so it is required from major 14 on.
        if vectorised m1 LOOP_A; then echo "MASKED=1: loop a vectorised"
        elif [ "$("$CXX" -dumpversion | cut -d. -f1)" -ge 14 ]; then fail "MASKED=1: loop a not vectorised"
        else echo "MASKED=1: loop a not vectorised (this compiler does not if-convert it)"; fi
        nm "$TMP/m1.o" | grep -q '_ZGV.M' || fail "MASKED=1: no masked symbol"
    fi
    echo "MASKED=$v ok"
done
echo "PASS"
