#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tests/docs/check_examples.py — gate item.

Every runnable example a docs page embeds is a literal excerpt of an
EXISTING `tests/test_*.cpp` line range, cited with the pair of directive
markers this script recognizes:

    .. aether-example:: tests/test_Foo.cpp:START-END
    .. code-block:: cpp

       <copied excerpt, indented>

    .. aether-example-end::

The checker verifies (a) the cited file:START-END exists, and (b) the
indented excerpt between the two markers, dedented, matches those exact
source lines verbatim (trailing whitespace ignored per line) — a source
edit that outpaces the docs page is DRIFT, and DRIFT is RED. A tree with no
examples scans without erroring: an empty scan is a legitimate "nothing to
check yet", not a false pass.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_RE = re.compile(r"^\.\. aether-example:: (tests/test_[A-Za-z0-9_]+\.cpp):(\d+)-(\d+)\s*$")
END_RE = re.compile(r"^\.\. aether-example-end::\s*$")


def check_file(rst: Path, errors: list[str]) -> None:
    lines = rst.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = START_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        path, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        j = i + 1
        while j < len(lines) and not END_RE.match(lines[j].strip()):
            j += 1
        if j == len(lines):
            errors.append(f"{rst}:{i+1}: aether-example has no matching aether-example-end::")
            i = j
            continue
        body = [ln for ln in lines[i + 1:j] if ".. code-block::" not in ln]
        excerpt = [ln[3:] if ln.startswith("   ") else ln for ln in body if ln.strip()]
        src = ROOT / path
        if not src.is_file():
            errors.append(f"{rst}:{i+1}: cited file '{path}' does not exist")
        else:
            actual = src.read_text().splitlines()[start - 1:end]
            if [ln.rstrip() for ln in excerpt] != [ln.rstrip() for ln in actual]:
                errors.append(f"{rst}:{i+1}: excerpt of '{path}:{start}-{end}' has DRIFTED from source")
        i = j + 1


def main() -> int:
    errors: list[str] = []
    n = 0
    for rst in sorted((ROOT / "docs").rglob("*.rst")):
        if "_build" in rst.parts or "_doxybuild" in rst.parts:
            continue  # generated output, not an authored page
        check_file(rst, errors)
        n += 1
    if errors:
        print("\n".join(errors))
        print(f"check_examples: FAIL ({len(errors)} issue(s))")
        return 1
    print(f"check_examples: OK ({n} page(s) scanned, 0 issues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
