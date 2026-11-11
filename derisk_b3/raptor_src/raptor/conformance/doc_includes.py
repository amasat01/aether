# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.conformance.doc_includes — reusable Sphinx `include`/`literalinclude`
target checker.

**Why this exists.** `eagle/docs/conf.py` sets `suppress_warnings = [...,
"docutils", ...]` -- the category Sphinx uses for "Include file ... not
found" -- so a broken `literalinclude` target prints one warning line and the
build still exits 0. Several of eagle's targets once pointed at a directory
later extracted to a downstream package, and this shipped silently for the
length of the pass. Blanket `-W` is not an option (most surfaced eagle
warnings are pre-existing doxygen duplicate-ID noise under the SAME
"docutils" category, so the suppression cannot be narrowed without also
un-suppressing that noise). This module is a targeted,
non-Sphinx-warning-dependent check instead: a plain filesystem scan that
reproduces Sphinx's own path-resolution rules and asserts every
`include`/`literalinclude` target exists.

**Why it lives here, not downstream:** the logic was always path-free (it
takes `docs_root` as an argument and knows nothing about which repos exist
on disk or where they live), but it once shipped from a downstream sibling's
own suite. That placement broke silently the day that sibling's own suite
shape changed (a sibling-free venv-gate leg could no longer resolve the tool
at all, landing the file blocked-at-collection instead of running). The
claim "eagle's includes resolve" is eagle's; a claim gated only in a
downstream sibling's suite stops running the day that suite changes shape.
Moving the LOGIC here (raptor ships to every consumer; `include =
["raptor*"]`) lets each owner (`eagle/python/tests/test_doc_includes.py`,
a downstream package's own test) certify its OWN docs tree directly, with
raptor keeping zero sibling/repo knowledge.

**Resolution rules (reproduce Sphinx's `include`/`literalinclude` directives
exactly -- verified against the live eagle tree, not assumed):**

1. a target starting with "/" is relative to the docs SOURCE directory
   (`docs_root`, the directory `conf.py` lives in) -- never the filesystem
   root (a target like `/schemas/manifest-v1.schema.json` resolves to
   `<docs_root>/schemas/manifest-v1.schema.json`, not `/schemas/...` on
   disk);
2. otherwise the target is relative to the directory containing the
   document that names it (the common case);
3. `_build/` is excluded -- it holds generated copies of the same source
   pages (`_build/html/_sources/**`), so scanning it double-counts every
   hit and would also flag the generated tree's own copies once a source
   fix has already landed but the stale `_build/` has not been rebuilt.

Handles both reStructuredText directives (``.. literalinclude:: target``,
eagle's dialect) and MyST fenced directives (a code fence whose info string is
``{literalinclude} target`` -- three backticks followed by that info string,
the rest of the family's dialect) with the same scan, so the same call is
reusable across the family regardless of which markup a repo's docs use.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: reST: ".. include:: target" / ".. literalinclude:: target" (optionally
#: indented, e.g. nested inside a ".. dropdown::").
_RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+(include|literalinclude)::\s+(\S+)\s*$")

#: MyST: "```{include} target" / "```{literalinclude} target" (the directive's
#: argument sits on the fence line itself).
_MYST_DIRECTIVE = re.compile(r"^\s*`{3,}\{(include|literalinclude)\}\s+(\S+)\s*$")

#: Source file suffixes scanned. `.rst` (docutils) and `.md` (MyST) are the
#: two dialects in use across the family's docs trees today.
SOURCE_SUFFIXES = (".rst", ".md")


@dataclass(frozen=True)
class BrokenInclude:
    """One `include`/`literalinclude` directive whose target does not exist."""

    doc: Path
    line: int
    directive: str
    target: str
    resolved: Path

    def __str__(self) -> str:
        return (
            f"{self.doc}:{self.line}: .. {self.directive}:: {self.target} "
            f"-> missing {self.resolved}"
        )


@dataclass(frozen=True)
class DocsScanReport:
    """One `scan_docs` result: the root scanned, the TOTAL number of
    `include`/`literalinclude` directives found (regardless of whether they
    resolved), and the broken subset.

    `n_directives` closes a vacuity trap: a scan pointed at an empty or
    mis-rooted `docs_root` reports `broken == []` vacuously -- nothing was
    found to break. A caller must assert `n_directives >= 1` alongside
    `broken == []`, or an empty/wrong root passes silently.
    """

    docs_root: Path
    n_directives: int
    broken: list[BrokenInclude]


def resolve_target(target: str, doc_path: Path, docs_root: Path) -> Path:
    """Resolve an `include`/`literalinclude` target using Sphinx's own rule.

    ``target`` starting with ``/`` is rooted at ``docs_root`` (the directory
    ``conf.py`` lives in), never the filesystem root. Otherwise it is
    resolved relative to ``doc_path``'s own directory (the document that
    names it), matching both docutils' `include` directive and Sphinx's
    `literalinclude` extension of it.
    """
    if target.startswith("/"):
        return (docs_root / target[1:]).resolve()
    return (doc_path.parent / target).resolve()


def _iter_directives(text: str):
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _RST_DIRECTIVE.match(line)
        if m is None:
            m = _MYST_DIRECTIVE.match(line)
        if m is not None:
            yield lineno, m.group(1), m.group(2)


def scan_docs(docs_root) -> DocsScanReport:
    """Scan every ``.rst``/``.md`` file under ``docs_root`` (excluding
    ``_build/``) for `include`/`literalinclude` directives, resolving each
    with :func:`resolve_target`. One filesystem walk produces both the total
    directive count and the broken subset (see :class:`DocsScanReport`).
    """
    docs_root = Path(docs_root).resolve()
    n_directives = 0
    broken: list[BrokenInclude] = []
    for suffix in SOURCE_SUFFIXES:
        for doc_path in sorted(docs_root.rglob(f"*{suffix}")):
            if "_build" in doc_path.relative_to(docs_root).parts:
                continue
            text = doc_path.read_text(encoding="utf-8")
            for lineno, directive, target in _iter_directives(text):
                n_directives += 1
                resolved = resolve_target(target, doc_path, docs_root)
                if not resolved.is_file():
                    broken.append(
                        BrokenInclude(doc_path, lineno, directive, target, resolved)
                    )
    return DocsScanReport(docs_root, n_directives, broken)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the directive count and every broken include
    found under each docs root given."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(
            "usage: python -m raptor.conformance.doc_includes <docs_root> "
            "[<docs_root> ...]",
            file=sys.stderr,
        )
        return 2
    red = 0
    for root in argv:
        report = scan_docs(Path(root))
        print(f"{root}: {report.n_directives} directive(s) scanned", file=sys.stderr)
        for broken in report.broken:
            red = 1
            print(broken)
    return red


if __name__ == "__main__":
    raise SystemExit(main())
