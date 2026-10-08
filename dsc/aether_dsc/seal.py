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

    python -m aether_dsc.seal
"""
from __future__ import annotations

import sys

from ._seal_impl import seal, seal_and_write

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Seal this checkout, gate it on the NVRTC-clean check, write the
    blob. Prints the outcome; exit 0 on a written blob, 1 on a gate
    failure or a missing input file."""
    del argv  # no options today — the CLI seals exactly this checkout
    try:
        p = seal()
    except FileNotFoundError as exc:
        print(f"RED: {exc}", file=sys.stderr)
        return 1
    try:
        blob_path = seal_and_write(p.headers, p.host_only_names)
    except RuntimeError as exc:
        print(f"RED: {exc}", file=sys.stderr)
        return 1
    print(f"GREEN: sealed {len(p.headers)} headers "
          f"({len(p.host_only_names)} host-only), digest {p.digest[:16]}… -> {blob_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
