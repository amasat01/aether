# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`python -m aether_dsc.seal` — the packaging CLI.

Seals this repository's own checkout, runs the NVRTC-clean gate (every
payload header must compile under NVRTC with zero diagnostics), and only
then writes `aether_dsc/_payload/<digest>.bin` — a payload that fails the
gate is refused, never shipped silently broken. The actual sealing logic
lives in :mod:`aether_dsc._seal_impl` (see that module's docstring for
why: this file being importable as `aether_dsc.seal` — which `-m` does
internally — would otherwise shadow the `aether_dsc.seal(*roots)`
function exposed from `aether_dsc/__init__.py`).

    python -m aether_dsc.seal --facts-dir DIR [--layout-header PATH]
    python -m aether_dsc.seal --no-gate [--layout-header PATH]

`--facts-dir` names the folder holding a raptor wing's generated `raptor_facts.h` (CMake's
`AETHER_FACTS_DIR`): the gate includes it first, as every real compile does; it is never sealed.

`--no-gate` writes the blob without the NVRTC check, for a machine that has no NVRTC (CPU-only
macOS): the check ran on the Linux seal of the same headers.
"""
from __future__ import annotations

import sys

from ._seal_impl import seal, seal_and_write

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Seal this checkout, gate it on the NVRTC-clean check, write the
    blob. ``--layout-header PATH`` names eagle's ``plugin/gref_layout.h``
    when eagle is not the sibling checkout; ``--facts-dir DIR`` is required
    by the gate. Prints the outcome; exit 0 on a written blob, 1 on a gate
    failure or a missing input file."""
    args = list(sys.argv[1:] if argv is None else argv)
    gate = "--no-gate" not in args
    layout_header = facts_dir = None
    for i, a in enumerate(args):
        if a == "--layout-header" and i + 1 < len(args):
            layout_header = args[i + 1]
        elif a.startswith("--layout-header="):
            layout_header = a.split("=", 1)[1]
        elif a == "--facts-dir" and i + 1 < len(args):
            facts_dir = args[i + 1]
        elif a.startswith("--facts-dir="):
            facts_dir = a.split("=", 1)[1]
    try:
        p = seal(layout_header=layout_header)
    except FileNotFoundError as exc:
        print(f"RED: {exc}", file=sys.stderr)
        return 1
    try:
        blob_path = seal_and_write(p.headers, p.host_only_names, gate=gate, facts_dir=facts_dir)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"RED: {exc}", file=sys.stderr)
        return 1
    print(f"GREEN: sealed {len(p.headers)} headers "
          f"({len(p.host_only_names)} host-only), digest {p.digest[:16]}… -> {blob_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
