#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/layers/check_layers.sh <layered tree> -- the mechanical checks of the header layering, for a tree that holds
# root/, components/ and variants/<name>/:
#   1. every AETHER_* macro the shared layers use is defined by root, components or each variant (macro_closure.py);
#   2. self-test: the same check FAILS on a scratch copy with a planted use of an undefined macro.
# Compile-only parses of the layers are the cut's own script; this one needs python3 only.
set -u
tree=${1:?usage: check_layers.sh <layered tree>}
here=$(cd "$(dirname "$0")" && pwd)
check="$here/../../tools/layers/macro_closure.py"
[ -d "$tree/root" ] && [ -d "$tree/components" ] && [ -d "$tree/variants" ] || { echo "not a layered tree: $tree" >&2; exit 2; }
python3 -I "$check" "$tree" || { echo "check_layers: macro closure FAILED on $tree" >&2; exit 1; }
scratch=$(mktemp -d)
cp -R "$tree/root" "$tree/components" "$tree/variants" "$scratch/"
f=$(find "$scratch/components" -name '*.h' | head -1)
printf '\n#ifdef AETHER_PLANTED_UNDEFINED_MACRO\n#endif\n' >> "$f"
python3 -I "$check" "$scratch" > /dev/null
rc=$?
rm -rf -- "$scratch"
[ "$rc" -ne 0 ] || { echo "check_layers: the planted violation was NOT detected" >&2; exit 1; }
echo "check_layers: OK (clean tree closed; planted violation rejected, rc=$rc)"
