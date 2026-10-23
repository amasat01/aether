#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/c_core/check_c99.sh — compile the minted C reference corpus as strict C99.
#
# tests/c_core/aether_c_corpus.h (the umbrella) and the parts under
# tests/c_core/corpus/ are generated plain C99 (see tools/c_core/mint_golden.py).
# This gate proves they really are: the umbrella compiles, and so does every
# part ALONE, which proves each part carries the typedefs and structs it needs.
# Any diagnostic under the flags below, or a missing header, is a non-zero exit.
# Silent on success. Usable from any cwd. Override the compiler with CC=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HEADER="$REPO_ROOT/tests/c_core/aether_c_corpus.h"
PARTS_DIR="$REPO_ROOT/tests/c_core/corpus"

if [ ! -f "$HEADER" ]; then
    echo "check_c99.sh: no such file: $HEADER" >&2
    exit 1
fi
if [ ! -d "$PARTS_DIR" ]; then
    echo "check_c99.sh: no such directory: $PARTS_DIR" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

compile_one() {
    printf '#include <math.h>\n#include "%s"\n' "$1" > "$TMP/tu.c"
    "${CC:-cc}" -std=c99 -pedantic -Wall -Wextra -Wconversion -Wsign-conversion \
        -Wdouble-promotion -Wno-unused-function -Werror -fsyntax-only -x c "$TMP/tu.c"
}

compile_one "$HEADER"
parts=0
for part in "$PARTS_DIR"/*/*.h; do
    [ -f "$part" ] || continue
    compile_one "$part"
    parts=$((parts + 1))
done
if [ "$parts" -eq 0 ]; then
    echo "check_c99.sh: no corpus parts under $PARTS_DIR" >&2
    exit 1
fi
