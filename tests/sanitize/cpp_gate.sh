#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The host sanitizer gate (CI `sanitize-cpp`, `make gate-sanitize`) on a CPP_MODE build made with
# -DAETHER_SANITIZERS=address,undefined. Three checks keep it from going vacuously green:
#   (i) the ASan runtime is linked, (ii) every canary mode trips with its banner,
#   (iii) the suite runs through tests/check_gate.sh against the committed manifest.
# Usage: tests/sanitize/cpp_gate.sh BUILD_DIR
set -euo pipefail
build=${1:?build dir}
here=$(cd "$(dirname "$0")/../.." && pwd)

{ nm -D "$build/tests/aether_tests"; nm "$build/tests/aether_tests"; } 2>/dev/null | grep -q __asan_init \
  || { echo "RED: aether_tests carries no __asan_init: this job is not sanitizing"; exit 1; }

export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
out=$(mktemp)
trap 'rm -f "$out"' EXIT
fail=0
for pair in "heap-oob:AddressSanitizer: heap-buffer-overflow" \
            "leak:LeakSanitizer: detected memory leaks" \
            "uninit:runtime error: load of value" \
            "shift-ub:runtime error: shift exponent"; do
  mode="${pair%%:*}"; banner="${pair#*:}"
  set +e; "$build/tests/aether_canary" "$mode" > "$out" 2>&1; rc=$?; set -e
  if [ "$rc" -eq 0 ] || ! grep -q "$banner" "$out"; then
    echo "RED: canary '$mode' rc=$rc, banner '$banner' present: $(grep -c "$banner" "$out")"; cat "$out"; fail=1
  else
    echo "canary '$mode' tripped (rc=$rc)"
  fi
done
[ "$fail" -eq 0 ] || exit 1

LSAN_OPTIONS="suppressions=$here/tests/sanitize/lsan.supp" "$here/tests/check_gate.sh" cpp "$build/tests/aether_tests" --config debug
