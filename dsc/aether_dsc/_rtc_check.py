# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The NVRTC-clean-compile check: the manifest of aether headers claimed
to be NVRTC-clean, the seven ``rtc/`` shim names, the NVRTC options
measured to compile cleanly, and the one compile-and-report primitive
both :func:`aether_dsc._seal_impl.seal_and_write`'s gate and the
standalone ``aether/tools/rtc_seal_check.py`` CLI share.

Self-contained by design: this module reaches nowhere outside
``aether_dsc`` — no ``sys.path`` mutation, no dependency on
``aether/tools/`` being on disk. The manifest ships as package data
(``aether_dsc/payload_headers.txt``, declared in ``pyproject.toml`` so a
wheel carries it), and the one third-party import this module needs
(``cuda-bindings``, for :func:`check_headers_nvrtc_clean`'s NVRTC calls)
happens inside that function only — ``import aether_dsc`` never pays for
a CUDA binding just because this module exists.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from pathlib import Path
from typing import Mapping

__all__ = ["RTC_SHIM_NAMES", "NVRTC_OPTIONS", "PAYLOAD_MANIFEST",
           "HOST_ONLY_MANIFEST", "manifest_paths", "host_only_manifest_paths",
           "repo_headers", "cccl_include_dir", "served_name",
           "check_headers_nvrtc_clean"]

#: The seven NVRTC compatibility shims — see ``aether/rtc/``'s own files for the served text.
RTC_SHIM_NAMES = ("cstddef", "cstdint", "type_traits", "utility", "concepts",
                   "cmath", "atomic")

#: NVRTC options measured to compile cleanly: c++20 device standard, this
#: box's default arch (P2000 = sm_61), ``-default-device`` so a bare
#: ``AETHER_DEVICE()``-qualified free function needs no
#: ``extern "C" __global__`` wrapper to be reachable from the probe TU.
NVRTC_OPTIONS = ["-std=c++20", "-arch=compute_61", "-default-device"]

#: The committed manifest, shipped as package data — the ``aether/...``
#: paths claimed to be NVRTC-clean. A directory walk is deliberately
#: NOT the source of truth (see ``payload_headers.txt``'s own header
#: comment): growing the payload to cover more headers is a deliberate,
#: evidenced addition made later, not a default.
PAYLOAD_MANIFEST = Path(__file__).resolve().parent / "payload_headers.txt"

#: The host-only manifest (``payload_headers_host.txt``) — headers sealed
#: into the payload's host-only section (see that file's own header
#: comment): never claimed NVRTC-clean, never part of the NVRTC-clean gate
#: this module runs, never handed to an NVRTC compile at all
#: (:data:`aether_dsc._core.Payload.device_headers` excludes every name
#: this manifest lists).
HOST_ONLY_MANIFEST = Path(__file__).resolve().parent / "payload_headers_host.txt"


def manifest_paths() -> list[str]:
    """The manifest's ``aether/...`` entries — comments (``#``) and blank
    lines skipped, order preserved as committed (already sorted)."""
    lines = PAYLOAD_MANIFEST.read_text().splitlines()
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def host_only_manifest_paths() -> list[str]:
    """:data:`HOST_ONLY_MANIFEST`'s entries, same shape as
    :func:`manifest_paths` — never fed to :func:`check_headers_nvrtc_clean`
    by anything in this package; that gate reads :func:`manifest_paths`
    alone."""
    lines = HOST_ONLY_MANIFEST.read_text().splitlines()
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def repo_headers(aether_root: Path, rtc_dir: Path) -> dict[str, bytes]:
    """Every manifested ``aether/...`` file under ``aether_root``, plus the
    seven ``rtc/<name>`` shims under ``rtc_dir`` — named exactly as a sealed
    payload would name them. Raises ``FileNotFoundError`` naming the missing
    file; never silently returns fewer headers than the manifest promises."""
    headers: dict[str, bytes] = {}
    for rel in manifest_paths():
        f = aether_root / rel
        if not f.is_file():
            raise FileNotFoundError(
                f"{PAYLOAD_MANIFEST} lists {rel!r}, which does not exist under {aether_root}")
        headers[rel] = f.read_bytes()
    for name in RTC_SHIM_NAMES:
        f = rtc_dir / name
        if not f.is_file():
            raise FileNotFoundError(f"NVRTC shim missing: {f}")
        headers[f"rtc/{name}"] = f.read_bytes()
    return headers


def _default_options(nvrtc) -> list[str]:
    """:data:`NVRTC_OPTIONS` with ``-arch`` set to the oldest architecture the
    loaded NVRTC supports (``nvrtcGetSupportedArchs``): ``compute_61`` on a
    CUDA 12.x NVRTC, ``compute_75`` on CUDA 13, whose NVRTC dropped the older
    targets. Falls back to the constant when the query is unavailable."""
    try:
        err, archs = nvrtc.nvrtcGetSupportedArchs()
        oldest = min(int(a) for a in archs)
    except Exception:
        return list(NVRTC_OPTIONS)
    return [f"-arch=compute_{oldest}" if o.startswith("-arch=") else o
            for o in NVRTC_OPTIONS]


def _loaded_nvrtc_root() -> str | None:
    """The toolkit root (the directory above ``lib``/``lib64``) of the
    ``libnvrtc`` this process has loaded, or ``None`` when none is mapped."""
    try:
        from cuda.bindings import nvrtc  # noqa: F401  (loads the library)
    except Exception:
        pass
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split(None, 5)[-1].strip() if line.count(" ") >= 5 else ""
                if "/libnvrtc.so" in path or "/libnvrtc-" in path:
                    lib = Path(path).resolve().parent
                    while lib.name in ("lib", "lib64", "x86_64-linux", "sbsa-linux",
                                       "aarch64-linux") or lib.parent.name == "targets":
                        lib = lib.parent
                    return str(lib.parent if lib.name == "targets" else lib)
    except OSError:
        pass
    return None


def cccl_include_dir() -> str:
    """Where ``<cuda/std/...>`` lives: the ``nvidia-cuda-cccl-cu12`` wheel's
    ``include/`` directory when it is installed, else a CUDA toolkit's own
    headers. Every CUDA 12.x/13 toolkit ships CCCL, either in a flat
    ``include/`` (CUDA 13: ``include/cccl/``) or in NVIDIA's
    ``targets/<arch>-linux/include/`` layout (a
    conda toolkit has only the latter). Toolkit roots are tried in order:
    the root of the loaded NVRTC, ``$CUDA_PATH``, ``$CONDA_PREFIX``, the toolkit holding the ``nvcc`` on
    ``PATH``, then ``/usr/local/cuda``.

    ``importlib.util.find_spec`` on a DOTTED name imports the PARENT first
    (``nvidia``) to look inside it, and raises ``ModuleNotFoundError`` rather
    than returning ``None`` when that parent does not exist at all — the
    exact shape of a box with no ``nvidia-cuda-*`` wheel installed to create
    the ``nvidia`` namespace package at all. That is a "not installed"
    answer, the same as the wheel existing but lacking ``cuda_cccl``, so it
    is caught here rather than left to crash a caller that only wanted to
    know whether the wheel is there."""
    try:
        spec = importlib.util.find_spec("nvidia.cuda_cccl")
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            candidate = Path(loc) / "include"
            if (candidate / "cuda" / "std").is_dir():
                return str(candidate)
    for candidate in _toolkit_include_dirs():
        if (candidate / "cuda" / "std").is_dir():
            return str(candidate)
    raise RuntimeError(
        "no CCCL include directory found: install nvidia-cuda-cccl-cu12 or "
        "set $CUDA_PATH to a toolkit whose include/cuda/std exists")


def _toolkit_include_dirs() -> list[Path]:
    """Candidate toolkit include directories for :func:`cccl_include_dir`,
    most specific first: for each toolkit root, its flat ``include/``, then
    ``targets/<arch>-linux/include/`` (this machine's arch first, then any
    other ``targets/*-linux`` the toolkit carries, e.g. ``sbsa-linux``)."""
    roots = [_loaded_nvrtc_root(), os.environ.get("CUDA_PATH"),
             os.environ.get("CONDA_PREFIX")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(str(Path(nvcc).resolve().parent.parent))
    roots.append("/usr/local/cuda")
    dirs: list[Path] = []
    for root in dict.fromkeys(r for r in roots if r):
        base = Path(root)
        targets = base / "targets"
        own = targets / f"{platform.machine()}-linux" / "include"
        others = sorted(targets.glob("*-linux/include")) if targets.is_dir() else []
        flat = [base / "include", own, *others]
        # CUDA 13 moved CCCL into an ``include/cccl`` subdirectory.
        for d in [p for x in flat for p in (x / "cccl", x)]:
            if d not in dirs:
                dirs.append(d)
    return dirs


def served_name(payload_name: str) -> str:
    """The name NVRTC is asked to resolve for a payload entry — what a REAL
    ``#include`` directive spells. ``aether/...``/``plugin/...`` entries are
    ``#include``d by that exact prefixed name, so they serve unchanged; an
    ``rtc/<std name>`` shim is never included by that stored path itself — it
    satisfies aether's own plain ``#include <cstdint>`` et al., so it must be
    offered under the bare std name with the ``rtc/`` storage prefix
    stripped (NVRTC's header lookup is an exact string match)."""
    return payload_name.split("/", 1)[1] if payload_name.startswith("rtc/") else payload_name


def check_headers_nvrtc_clean(headers: Mapping[str, bytes], *,
                               cccl_include: str | None = None,
                               options: list[str] | None = None,
                               exclude_from_toplevel: frozenset[str] = frozenset(),
                               ) -> list[str]:
    """Compile ONE synthetic TU ``#include``-ing every TOP-LEVEL name in
    ``headers`` (sorted, for a reproducible diagnostic order) through NVRTC,
    serving every name from ``headers`` itself — never from disk — under its
    :func:`served_name`. Only ``aether/...``/``plugin/...`` entries not in
    ``exclude_from_toplevel`` are ``#include``d directly; ``rtc/...`` shims
    are reachable only indirectly, exactly as real aether source reaches
    them.

    ``exclude_from_toplevel`` is for an entry that is served (resolvable if
    something reaches it) but never itself the subject of the NVRTC-clean
    claim — see :mod:`aether_dsc._seal_impl`'s ``GREF_ABI_HOST_NAME`` for
    the one case this applies to.

    Returns the NVRTC log's diagnostic lines (empty means the claim holds);
    raises only on an environment failure (no NVRTC binding, no CCCL) that
    is not itself the thing under test.
    """
    from cuda.bindings import nvrtc  # function-local: only a real check pays for this

    cccl_include = cccl_include or cccl_include_dir()
    opts = list(options or _default_options(nvrtc)) + [f"-I{cccl_include}"]
    names = sorted(headers)
    top_level = [n for n in names
                 if not n.startswith("rtc/") and n not in exclude_from_toplevel]
    src = "".join(f'#include "{n}"\n' for n in top_level)
    src += "\n// reachability only — the TU need not use anything.\n"

    _, prog = nvrtc.nvrtcCreateProgram(
        src.encode(), b"rtc_seal_check.cu",
        len(names), [headers[n] for n in names],
        [served_name(n).encode() for n in names])
    try:
        nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log_buf = bytearray(log_size)
        nvrtc.nvrtcGetProgramLog(prog, log_buf)
        log = bytes(log_buf).decode(errors="replace").rstrip("\x00")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    return [line for line in log.splitlines()
            if line.strip() and not _is_target_deprecation_notice(line)]


def _is_target_deprecation_notice(line: str) -> bool:
    """NVRTC 12.x warns that pre-sm_75 targets are deprecated. That notice is about
    the toolkit's support window for the chosen ``-arch``, not about the headers
    under test, so it never counts against the NVRTC-clean claim."""
    return "Architectures prior to" in line and "are deprecated" in line
