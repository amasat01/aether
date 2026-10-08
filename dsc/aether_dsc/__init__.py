# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""aether-dsc — the sealed AETHER header payload.

aether ships to Python-only users as a SEALED PAYLOAD, never as readable
headers inside a wheel's include tree: one digest-named, zlib-compressed tar
blob (opacity, not secrecy — see :mod:`aether_dsc._core`) carrying the
`aether/`/`rtc/` headers a HAWK-emitted device kernel needs, plus eagle's
`plugin/gref_layout.h` (device-clean) and `plugin/gref_abi.h` (host-only —
NVRTC never sees its real bytes; :mod:`hawk.compile.nvrtc` serves an alias to
the layout header's instead). Two consumers:

* **device** — NVRTC compiles a kernel straight from :func:`payload`'s
  in-memory `headers` mapping; nothing touches disk.
* **host, or any file-only compiler** — :meth:`Payload.serve` materialises
  the payload into a private, owner-only RAM directory for the compile's
  lifetime and removes it afterwards, failure path included.

Typical use::

    import aether_dsc
    p = aether_dsc.payload()        # the blob shipped in this wheel
    # device: hand p.digest / p.headers to NVRTC (hawk.compile.nvrtc)
    with p.serve() as include_root:  # host / any file compiler
        ...                          # -I include_root

Building a payload from a live checkout (never the shipped-wheel path — a
developer's own aether tree, or the CLI that regenerates the shipped blob)
goes through :func:`seal`; see :mod:`aether_dsc.seal`'s module docstring.
"""
from __future__ import annotations

from ._core import Payload, PAYLOAD_DIR, _unpack, digest_of
# Imported from `_seal_impl`, deliberately NOT from `.seal`:
# `aether_dsc/seal.py` is `python -m aether_dsc.seal`'s CLI file, and
# Python binds `aether_dsc.seal` to that MODULE the instant anything imports
# it — which running the CLI does internally. Importing the function from
# its own home (`_seal_impl.py`) instead means `aether_dsc.seal` stays
# callable even in a process that has also run the CLI or imported
# `aether_dsc.seal` as a submodule (see `_seal_impl`'s docstring;
# `tests/test_aether_dsc.py` exercises both in one process).
from ._seal_impl import seal

__all__ = ["Payload", "seal", "payload", "version", "PAYLOAD_DIR"]

#: This distribution's version — tracks aether's own (`aether/version.h`:
#: `constexpr version()`), because a sealed payload speaks for exactly one
#: aether checkout and a version skew between the two would silently claim
#: otherwise.
version = "0.2.0"

_PAYLOAD_CACHE: Payload | None = None


def payload() -> Payload:
    """The sealed blob shipped in this distribution — decompressed
    once per process and cached; every call after the first returns the same
    object.

    Raises `FileNotFoundError` when this checkout carries no blob yet (a
    source tree that has never run ``python -m aether_dsc.seal``, or an
    editable install before the first seal). Nothing here falls back to
    sealing the live tree — that fallback belongs to
    ``hawk.compile.payload.current_payload()``, gated on
    `$HAWK_AETHER_INCLUDE`: a caller that wants "seal if there is no
    blob" asks for it explicitly, `payload()` never guesses.
    """
    global _PAYLOAD_CACHE
    if _PAYLOAD_CACHE is not None:
        return _PAYLOAD_CACHE
    blobs = sorted(PAYLOAD_DIR.glob("*.bin")) if PAYLOAD_DIR.is_dir() else []
    if not blobs:
        raise FileNotFoundError(
            f"no sealed payload under {PAYLOAD_DIR} — run `python -m "
            "aether_dsc.seal` from the aether/dsc checkout to build one")
    if len(blobs) > 1:
        raise RuntimeError(
            f"{PAYLOAD_DIR} carries {len(blobs)} blobs "
            f"({[b.name for b in blobs]}) — exactly one is expected; a stale "
            "blob from a previous seal was not cleaned up")
    blob_path = blobs[0]
    claimed_digest = blob_path.stem
    headers, host_only_names = _unpack(blob_path.read_bytes())
    actual_digest = digest_of(headers)
    if actual_digest != claimed_digest:
        raise ValueError(
            f"{blob_path} is named for digest {claimed_digest} but its content "
            f"digests to {actual_digest} — corrupt or hand-edited blob")
    _PAYLOAD_CACHE = Payload(actual_digest, headers, host_only_names)
    return _PAYLOAD_CACHE


def _reset_payload_cache() -> None:
    """Test/dev hook: forget the cached :func:`payload` so a re-seal (or a
    freshly written blob in a test's tmp `_payload/` dir) is picked up
    without restarting the process."""
    global _PAYLOAD_CACHE
    _PAYLOAD_CACHE = None
