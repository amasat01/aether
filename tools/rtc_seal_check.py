#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""NVRTC-clean CLI: every payload header compiles under NVRTC with zero
diagnostics, in one translation unit that includes them all.

A thin wrapper: the manifest, the NVRTC options, the served-name
translation and the compile-and-report primitive itself all live in
:mod:`aether_dsc._rtc_check` (self-contained package code + package
data) — this script's own job is just to read this checkout's manifested
files off disk and hand them to that primitive. `aether_dsc` is the one
canonical home for that logic; this script only imports it.

    python tools/rtc_seal_check.py            # this repo's manifested payload
    python tools/rtc_seal_check.py --verbose
    python tools/rtc_seal_check.py --list-unmanifested

Exit code 0 = every header compiled clean; 1 = at least one diagnostic (a
compile error or a warning: zero diagnostics is required, not just zero
errors).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aether_dsc import _rtc_check as rc

REPO_ROOT = Path(__file__).resolve().parents[1]


def unmanifested_headers() -> list[str]:
    """Every `aether/*.h` file NOT in the manifest — the diagnostic behind
    the manifest's own "~28 other headers" comment. Not used by the
    GREEN/RED check itself; exists so growing the manifest starts from a
    live list rather than a stale comment."""
    aether_dir = REPO_ROOT / "aether"
    manifested = set(rc.manifest_paths())
    out = []
    for f in sorted(aether_dir.rglob("*")):
        if f.is_file():
            rel = f"aether/{f.relative_to(aether_dir).as_posix()}"
            if rel not in manifested:
                out.append(rel)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                     help="print every payload header name before compiling")
    ap.add_argument("--list-unmanifested", action="store_true",
                     help="print aether/*.h files NOT in the manifest, and exit "
                          "(diagnostic only — does not compile anything)")
    args = ap.parse_args(argv)

    if args.list_unmanifested:
        for rel in unmanifested_headers():
            print(rel)
        return 0

    headers = rc.repo_headers(REPO_ROOT, REPO_ROOT / "rtc")
    if args.verbose:
        for name in sorted(headers):
            print(f"  payload: {name}")
        print(f"{len(headers)} headers, NVRTC options: {rc.NVRTC_OPTIONS}")

    diagnostics = rc.check_headers_nvrtc_clean(headers)
    if diagnostics:
        print(f"RED: {len(diagnostics)} NVRTC diagnostic line(s) (zero required):",
              file=sys.stderr)
        for line in diagnostics:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"GREEN: {len(headers)} payload headers compile under NVRTC with zero "
          "diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
