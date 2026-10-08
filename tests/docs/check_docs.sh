#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/docs/check_docs.sh — aether docs gate. Runs with no GPU. Four checks,
# ALL must pass for RC=0:
#
#   1. doxygen warnings over aether/aether/** are a SUBSET of
#      docs/doxygen_allowlist.txt, exact match after path normalization
#      (any warning not on the list = RED, including a brand-new one).
#   2. every `aether::...` symbol / `aether/...h` path cited across
#      docs/**/*.rst exists in the tree (tests/docs/check_citations.py).
#   3. `sphinx-build -W` succeeds — warnings are errors.
#      Invoked as `python3 -m sphinx`, not the bare `sphinx-build` command:
#      see the Makefile's `docs:` target comment for why.
#   4. every `.. aether-example::` anchor names a real tests/test_*.cpp line
#      range whose embedded excerpt matches verbatim — no drift
#      (tests/docs/check_examples.py).
#
# Usage: tests/docs/check_docs.sh   (run from anywhere; cds to the repo root)
set -u
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
FAIL=0

echo "== [1/4] doxygen warnings (allowlist) =="
DOXY_LOG="$(mktemp)"
( cd docs && WARN_LOGFILE="$DOXY_LOG" QUIET=YES doxygen Doxyfile.in >/dev/null )
sed -i "s|^$ROOT/||" "$DOXY_LOG"
ALLOW="$(mktemp)"
grep -v '^[[:space:]]*#' docs/doxygen_allowlist.txt | grep -v '^[[:space:]]*$' > "$ALLOW"
NEW=0
while IFS= read -r line; do
    grep -qxF "$line" "$ALLOW" || { echo "NEW WARNING (not in allowlist): $line"; NEW=1; }
done < <(grep 'warning:' "$DOXY_LOG")
rm -f "$DOXY_LOG" "$ALLOW"
if [ "$NEW" -ne 0 ]; then
    echo "[1/4] FAIL"
    FAIL=1
else
    echo "[1/4] OK"
fi

echo "== [2/4] cited symbol/path existence =="
if python3 tests/docs/check_citations.py; then
    echo "[2/4] OK"
else
    echo "[2/4] FAIL"
    FAIL=1
fi

echo "== [3/4] sphinx-build -W =="
SPHINX_OUT="$(mktemp -d)"
SPHINX_LOG="$(mktemp)"
if python3 -m sphinx -W -b html docs "$SPHINX_OUT" >"$SPHINX_LOG" 2>&1; then
    echo "[3/4] OK"
else
    echo "[3/4] FAIL"
    cat "$SPHINX_LOG"
    FAIL=1
fi
rm -rf "$SPHINX_OUT" "$SPHINX_LOG"

echo "== [4/4] example-anchor drift =="
if python3 tests/docs/check_examples.py; then
    echo "[4/4] OK"
else
    echo "[4/4] FAIL"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "check_docs.sh: RED"
else
    echo "check_docs.sh: GREEN"
fi
exit "$FAIL"
