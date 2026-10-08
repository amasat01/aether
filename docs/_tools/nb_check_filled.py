#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""docs/_tools/nb_check_filled.py — `make nbcheck`'s gate.

Every `.ipynb` under the directories named on the command line must have
been EXECUTED: every code cell needs a set `execution_count` and no
`error` output. A tree with no notebooks yet is not itself a failure (an
empty scan passes) — the point is to catch a notebook that was authored
but never run, not to demand a minimum count.

Also refuses a notebook whose OUTPUT (stream text, `text/plain`,
`text/html`, or an error traceback) contains an absolute local-machine
path — these repos go public, and a committed output with
`/local/...`, `/home/<user>/...` etc. leaks the machine that built
the docs. Source cells are not scanned (a path in a code comment is not a
leak); only what got rendered into the committed output is.
"""
import json
import re
import sys
from pathlib import Path

# Deliberately narrow: absolute local-filesystem roots a committed output
# should never contain, not a general "looks like a path" heuristic.
_LEAK_PATTERNS = [re.compile(p) for p in (r"/local/", r"/home/", r"/tmp/", r"/root/", r"/Users/", r"C:\\Users")]


def _text_of(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def _leaked_path(text: str) -> str | None:
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


def check_notebook(path: Path) -> list[str]:
    errors = []
    nb = json.loads(path.read_text())
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        errors.append(f"{path}: no code cells at all")
    for idx, cell in enumerate(code_cells):
        if cell.get("execution_count") is None:
            errors.append(f"{path}: code cell {idx} was never executed (execution_count is null)")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(f"{path}: code cell {idx} has an error output ({out.get('ename')}: {out.get('evalue')})")
            texts = []
            if out.get("output_type") == "stream":
                texts.append(_text_of(out.get("text", "")))
            elif out.get("output_type") in ("display_data", "execute_result"):
                data = out.get("data", {})
                texts.extend(_text_of(data[mime]) for mime in ("text/plain", "text/html") if mime in data)
            elif out.get("output_type") == "error":
                texts.append(_text_of(out.get("traceback", [])))
            for text in texts:
                pattern = _leaked_path(text)
                if pattern:
                    errors.append(f"{path}: code cell {idx} output leaks a local path (matched {pattern!r})")
    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: nb_check_filled.py <dir> [<dir> ...]")
        return 2
    errors: list[str] = []
    n = 0
    for d in argv:
        for nb_path in sorted(Path(d).rglob("*.ipynb")):
            if ".ipynb_checkpoints" in nb_path.parts:
                continue
            errors.extend(check_notebook(nb_path))
            n += 1
    if errors:
        print("\n".join(errors))
        print(f"nbcheck: FAIL ({len(errors)} issue(s) across {n} notebook(s))")
        return 1
    print(f"nbcheck: OK ({n} notebook(s), all cells executed cleanly)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
