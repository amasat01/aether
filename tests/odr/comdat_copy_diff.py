#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Compares the SELECTED machine code of a shared `inline` COMDAT body
between two linked binaries; the comparison logic is generic, parameterized
by a regex argument. DEFAULT_RE targets `packetEval`/`packetFor`
(`aether/backend/cpu/Tiled.h`, the CPU SIMD packet evaluation loop), which
survive as real, non-trivial, template-instantiated `inline` bodies shared
across every TU that drives a batched expression through the packet path
(test_PacketExpr.cpp, test_Bundle.cpp, ...) — genuine arithmetic, COMDAT-
linked, instantiated identically-shaped in multiple TUs. (`aether::math::
{sqrt,fma,...}` was tried first, but at -O3 those tiny `AETHER_FORCEINLINE`
wrappers are inlined away completely at every call site — measured via
`nm`: zero surviving symbols in `aether_tests` — so a target that leaves no
COMDAT body behind cannot be this gate's subject.)

Compare the SELECTED machine code of those shared inline bodies between two
linked binaries.

Such bodies are `inline` (often template) functions in a header, so EVERY
translation unit that instantiates one emits its own weak (COMDAT) copy and
the linker keeps whichever appears first on the link line. Those copies are
not required to be identical machine code -- the compiler's inlining
decisions for a COMDAT body depend on the whole translation unit -- so
re-ordering the objects can change which code actually runs.

This script answers one question for `check_comdat_link_order.sh`: DID the
re-order actually swap anything? If it did not, the link-order experiment is
blind and its green means nothing.

Comparison is on the MNEMONIC SEQUENCE, not on bytes: the two binaries have
different layouts, so identical code still carries different branch targets
and different rip-relative displacements. A mnemonic sequence is
layout-independent and still changes the moment a callee is inlined in one
copy and called in the other.

Usage:  comdat_copy_diff.py <binaryA> <binaryB> [name-regex]
Exit:   0 always (the caller reads the printed counts).
"""
import re
import subprocess
import sys

OBJDUMP = "objdump"
DEFAULT_RE = r"packetEval|packetFor|packetGet"


def defined_symbols(binary, pat):
    out = subprocess.run(["nm", "--defined-only", binary],
                         capture_output=True, text=True).stdout
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("W", "T", "t", "w"):
            if re.search(pat, parts[2]):
                syms.add(parts[2])
    return syms


def mnemonics(binary, sym):
    out = subprocess.run([OBJDUMP, "-d", "--disassemble=" + sym, binary],
                         capture_output=True, text=True).stdout
    seq = []
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3:
            tok = cols[2].strip().split()
            if tok:
                seq.append(tok[0])
    return seq


def main():
    a, b = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_RE
    sa, sb = defined_symbols(a, pat), defined_symbols(b, pat)
    shared = sorted(sa & sb)
    same = differ = empty = 0
    names = []
    for s in shared:
        ma, mb = mnemonics(a, s), mnemonics(b, s)
        if not ma or not mb:
            empty += 1
            continue
        if ma == mb:
            same += 1
        else:
            differ += 1
            dem = subprocess.run(["c++filt", s], capture_output=True,
                                 text=True).stdout.strip()
            names.append((dem, len(ma), len(mb)))
    print(f"  shared inline bodies in both binaries : {len(shared)}")
    print(f"  identical machine code                : {same}")
    print(f"  DIFFERENT machine code selected       : {differ}")
    print(f"  not disassemblable (skipped)          : {empty}")
    for dem, la, lb in names[:12]:
        print(f"    swapped: {dem}  ({la} vs {lb} instructions)")
    if len(names) > 12:
        print(f"    ... and {len(names) - 12} more")
    print(f"SWAPPED_COUNT={differ}")


if __name__ == "__main__":
    main()
