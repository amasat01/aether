# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The real `seal()` implementation, kept out of `aether_dsc/seal.py` on
purpose: that file is `python -m aether_dsc.seal`'s CLI entry point, and
Python sets `aether_dsc.seal` to the MODULE object the moment anything
does `import aether_dsc.seal` (exactly what running it as `-m` does
internally) — which would silently shadow a `seal` FUNCTION of the same
name re-exported through `aether_dsc/__init__.py` (`aether_dsc.seal`
would stop being callable). Splitting the implementation into its own
module and having BOTH `aether_dsc/__init__.py` and `aether_dsc/seal.py`
import it from here (never from each other) means both `seal` spellings —
the function `aether_dsc.seal(*roots)` and the CLI
`python -m aether_dsc.seal` — can coexist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from . import _rtc_check as _tools
from ._core import Payload, PAYLOAD_DIR, _pack, digest_of

__all__ = ["seal", "seal_and_write"]

#: This checkout's own repo root (`aether_dsc/_seal_impl.py` -> `aether_dsc/`
#: -> `dsc/` -> the `aether` repo root) and its workspace parent — the same
#: "the four *-abi worktrees and aether hang off one directory" convention
#: `hawk._roots.WORKSPACE` documents. Used only to DEFAULT `seal()`'s own
#: `aether_root`/`layout_header` arguments to THIS checkout — never to reach
#: outside the package for logic, which now lives entirely in `_rtc_check`
#: (`aether_dsc` mutates no `sys.path` and needs nothing on disk outside
#: its own package directory to be imported, only to seal).
_AETHER_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _AETHER_REPO_ROOT.parent


#: Served, never force-#include'd by the NVRTC-clean check (see
#: `tools.check_headers_nvrtc_clean`'s `exclude_from_toplevel` docstring):
#: eagle's real `plugin/gref_abi.h` reaches `<stdexcept>`/`<string>`, so it is
#: sealed for the host `-I <served root>` path only. `hawk.compile.nvrtc`'s
#: own NVRTC serving layer answers a request for this exact name with
#: `plugin/gref_layout.h`'s bytes instead (`GREF_ABI_ALIAS_NAME`) — hawk's
#: emitted prelude text spells `#include "plugin/gref_abi.h"` on both
#: backends unconditionally, and that string is pinned byte-for-byte by
#: read-only corpora that must not change independently: hawk's own
#: `tests/test_index_width.py::test_the_mandated_include_order_is_aether_first`
#: and the golden kernel sources of its consumers. The
#: substitution is transparent to what a device kernel actually computes: it
#: never reaches `gref_abi.h`'s host-only validation functions, only the
#: mirror PODs/constants both files carry identically.
GREF_ABI_HOST_NAME = "plugin/gref_abi.h"


def _default_layout_header() -> Path:
    """`eagle/plugin/gref_layout.h` from the sibling eagle checkout under this
    workspace. `aether_dsc` depends on this ONE file from eagle (not on eagle
    as a whole): the sealed payload must carry it because a HAWK-emitted
    device kernel `#include`s it, and nothing else about eagle belongs in an
    AETHER payload.
    """
    for name in ("eagle", "eagle-abi"):
        candidate = _WORKSPACE / name / "plugin" / "gref_layout.h"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "seal(): no eagle checkout with plugin/gref_layout.h found under "
        f"{_WORKSPACE} — pass layout_header= explicitly, or check out eagle as "
        "a sibling of this aether checkout")


def seal(aether_root: str | Path | None = None,
         rtc_dir: str | Path | None = None,
         layout_header: str | Path | None = None) -> Payload:
    """Build a fresh :class:`~aether_dsc._core.Payload` from headers on disk.
    Every argument defaults to THIS checkout's own paths,
    so a bare `seal()` reproduces exactly what `python -m aether_dsc.seal`
    ships; a caller with a different aether install (`hawk.compile.toolchain
    .aether_include()`'s resolved root, when it differs from this checkout)
    passes its own roots instead.

    * `aether_root` — an aether INCLUDE ROOT: a directory such that
      `aether_root/aether/<name>` are the library's own headers, exactly
      what a compiler resolves for `#include <aether/...>` against
      `-I aether_root`. Only the paths listed in this PACKAGE's own
      `aether_dsc/payload_headers.txt` (the NVRTC-clean manifest, shipped
      as package data — see that file's header comment for why a manifest,
      not a directory walk: aether/ holds headers not yet claimed to be
      NVRTC-clean) are sealed from this root — never a blind walk of
      everything under it.
    * `rtc_dir` — the directory of the seven NVRTC std shims; every one
      becomes a payload header named `"rtc/<filename>"`.
    * `layout_header` — the path to eagle's `plugin/gref_layout.h`;
      sealed as exactly `"plugin/gref_layout.h"`. Its own directory is also
      where eagle's REAL host-only `gref_abi.h` is found, sealed as
      :data:`GREF_ABI_HOST_NAME` for the host `-I` path (see that constant's
      docstring — NVRTC never receives it under this content, only under an
      alias to `gref_layout.h`'s bytes).

    A second, HOST-ONLY set of headers — this package's own
    `payload_headers_host.txt` manifest — is sealed into the SAME payload
    alongside the above: never claimed NVRTC-clean (excluded from
    :func:`seal_and_write`'s gate entirely, not merely from its
    force-`#include` set) and never served to an NVRTC compile
    (:attr:`~aether_dsc._core.Payload.device_headers` excludes them), but
    served to a host `-I` root exactly like every other entry
    (:meth:`~aether_dsc._core.Payload.serve`). Resolved from the SAME
    `aether_root`, since both manifests name files under one aether
    checkout.

    Raises `FileNotFoundError` naming the missing file when a manifested
    header, a shim, or a layout/abi header is not where expected — never
    silently seals a smaller payload than either manifest promises.
    """
    root = Path(aether_root) if aether_root is not None else _AETHER_REPO_ROOT
    rtc = Path(rtc_dir) if rtc_dir is not None else (_AETHER_REPO_ROOT / "rtc")
    layout = Path(layout_header) if layout_header is not None else _default_layout_header()
    abi = layout.parent / "gref_abi.h"

    headers: dict[str, bytes] = {}
    for rel in _tools.manifest_paths():
        f = root / rel
        if not f.is_file():
            raise FileNotFoundError(f"seal(): manifested header missing: {f}")
        headers[rel] = f.read_bytes()
    for name in _tools.RTC_SHIM_NAMES:
        f = rtc / name
        if not f.is_file():
            raise FileNotFoundError(f"seal(): NVRTC shim missing: {f}")
        headers[f"rtc/{name}"] = f.read_bytes()
    if not layout.is_file():
        raise FileNotFoundError(f"seal(): eagle layout header missing: {layout}")
    if not abi.is_file():
        raise FileNotFoundError(f"seal(): eagle host-abi header missing: {abi}")
    headers["plugin/gref_layout.h"] = layout.read_bytes()
    headers[GREF_ABI_HOST_NAME] = abi.read_bytes()

    host_only_names = frozenset(_tools.host_only_manifest_paths())
    for rel in host_only_names:
        f = root / rel
        if not f.is_file():
            raise FileNotFoundError(f"seal(): host-only manifested header missing: {f}")
        headers[rel] = f.read_bytes()

    return Payload(digest_of(headers), headers, host_only_names)


def seal_and_write(headers: Mapping[str, bytes],
                   host_only_names: frozenset[str] = frozenset()) -> Path:
    """Runs the NVRTC-clean gate over the NVRTC-clean section ONLY, then
    writes `_payload/<digest>.bin` — refuses to write a payload that fails
    the gate. Removes any stale blob first: `aether_dsc.payload()` globs
    `_payload/*.bin` and would otherwise have two candidates after a
    re-seal that changed the digest.

    `host_only_names` — `headers` entries never passed to the gate at all
    (not merely excluded from its force-#include set, the way
    `GREF_ABI_HOST_NAME` is — see its own docstring): the host-only section
    is never claimed NVRTC-clean and the gate never even sees its content.
    """
    nvrtc_headers = ({k: v for k, v in headers.items() if k not in host_only_names}
                     if host_only_names else headers)
    diagnostics = _tools.check_headers_nvrtc_clean(
        nvrtc_headers, exclude_from_toplevel=frozenset({GREF_ABI_HOST_NAME}))
    if diagnostics:
        raise RuntimeError(
            "NVRTC-clean check FAILED: the sealed payload does not compile clean under NVRTC "
            f"({len(diagnostics)} diagnostic line(s)):\n" + "\n".join(diagnostics))
    digest = digest_of(headers)
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for stale in PAYLOAD_DIR.glob("*.bin"):
        stale.unlink()
    blob_path = PAYLOAD_DIR / f"{digest}.bin"
    blob_path.write_bytes(_pack(headers, host_only_names))
    return blob_path
