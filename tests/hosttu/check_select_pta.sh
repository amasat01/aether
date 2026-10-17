#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Nested `select` over value leaves (`Item`, `Constant`) must not read stale stack.
#   check_select_pta.sh <g++> [--necessity]
# Default: builds select_pta_repro.cpp at -O2 and -O3, with and without
# AETHER_HOST_VECTOR_MATH, for the nested, single and large select shapes; each run
# dirties the stack, calls the kernel twice and must print exact 0.
# --necessity: rebuilds against a temp copy of aether/ with the old by-value alias
# restored (select_pta_old_alias.patch) and passes only if some build MISMATCHES, i.e.
# the compiler carries the bug the by-reference rule works around.
# Measured against the old alias: g++ 11.5 is RED (nested shape, -O2 and -O3); g++ 14.3 is
# not RED on this reduction (the emitted kernel in hawk's tests/test_host_stale_stack.py is),
# so on a compiler where it is not RED this script is a regression test for the
# by-reference rule and non-vacuity rests on that hawk test.
set -u
CXX=${1:?usage: check_select_pta.sh <g++> [--necessity]}
MODE=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
INC=$ROOT
if [ "$MODE" = "--necessity" ]; then
    mkdir -p "$TMP/old" && cp -r "$ROOT/aether" "$TMP/old/aether"
    (cd "$TMP/old" && patch -s -p1 < "$HERE/select_pta_old_alias.patch") || { echo "patch failed"; exit 2; }
    INC=$TMP/old
fi
bad=0
for opt in -O2 -O3; do
  for hvm in "" "-DAETHER_HOST_VECTOR_MATH -ffp-contract=fast -fno-trapping-math"; do
    for shape in "" -DSHAPE_SINGLE -DSHAPE_LARGE; do
      # shellcheck disable=SC2086
      "$CXX" -std=c++23 $opt -DAETHER_CPP_MODE $hvm $shape -I"$INC" "$HERE/select_pta_repro.cpp" -o "$TMP/r.exe" 2> "$TMP/err" \
          || { echo "build failed: $opt $hvm $shape"; cat "$TMP/err"; exit 2; }
      out=$(LD_BIND_NOW=1 "$TMP/r.exe" 2>&1); rc=$?
      echo "$opt ${hvm:+hvm} ${shape:-nested}: $out"
      [ $rc = 0 ] || bad=$((bad+1))
    done
  done
done
if [ "$MODE" = "--necessity" ]; then
    [ $bad -gt 0 ] && { echo "NECESSITY OK: the by-value alias miscompiles on this compiler ($bad builds)"; exit 0; }
    echo "FAIL: old alias is correct on this compiler: it does not carry the bug"; exit 1
fi
[ $bad = 0 ] || { echo "FAIL: $bad builds read a stale value"; exit 1; }
echo "OK"
