#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tests/docs/check_citations.py — gate item: every
``aether::...`` symbol / ``aether/...h`` path cited (RST double-backtick)
across docs/**/*.rst must exist under aether/ (whole-word grep for a
symbol's trailing identifier; plain existence check for a path)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SRC = ROOT / "aether"

SYMBOL_RE = re.compile(r"``(aether::[A-Za-z0-9_:<>,\s]*?[A-Za-z0-9_])``")
PATH_RE = re.compile(r"``(aether/[A-Za-z0-9_./]+\.h)``")

errors = []
headers = list(SRC.rglob("*.h"))
for rst in sorted(DOCS.rglob("*.rst")):
    if "_build" in rst.parts or "_doxybuild" in rst.parts:
        continue  # generated output, not an authored page
    text = rst.read_text()
    for m in SYMBOL_RE.finditer(text):
        sym = m.group(1)
        ident = sym.rsplit("::", 1)[-1].split("<")[0].strip()
        hit = ident and any(re.search(rf"\b{re.escape(ident)}\b", h.read_text()) for h in headers)
        if not hit:
            errors.append(f"{rst.relative_to(ROOT)}: cited symbol `{sym}` (identifier `{ident}`) not found under aether/")
    for m in PATH_RE.finditer(text):
        p = m.group(1)
        if not (ROOT / p).is_file():
            errors.append(f"{rst.relative_to(ROOT)}: cited path `{p}` does not exist")

if errors:
    print("\n".join(errors))
    print(f"check_citations: FAIL ({len(errors)} issue(s))")
    sys.exit(1)
print("check_citations: OK (0 issues)")
