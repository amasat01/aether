# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The `Payload` type and its (de)serialisation — everything `aether_dsc`
needs at IMPORT time, with zero third-party dependencies. Kept apart
from :mod:`aether_dsc.seal`, which needs `cuda-bindings` for its NVRTC-clean
gate: an ordinary consumer who only ever calls `payload()`/`.serve()` must
never pay for (or need installed) a CUDA binding just to unpack a blob.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import zlib
from pathlib import Path
from typing import Iterator, Mapping

__all__ = ["Payload", "PAYLOAD_DIR", "digest_of"]

#: Where a sealed blob lives once `python -m aether_dsc.seal` has written one
#: — `aether_dsc/_payload/<digest>.bin`, shipped as package data.
PAYLOAD_DIR = Path(__file__).resolve().parent / "_payload"


def digest_of(headers: Mapping[str, bytes]) -> str:
    """The payload's content digest: sha256 over the
    SORTED `(name, bytes)` pairs. Sorting is what makes it order-independent
    — two payloads built by walking the same files in a different order (a
    different filesystem, a different `os.walk` order) still digest to the
    same value, and a single renamed or altered file changes it.
    """
    h = hashlib.sha256()
    for name in sorted(headers):
        encoded = name.encode()
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
        data = headers[name]
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


#: The tar member :func:`_pack`/:func:`_unpack` use to carry the host-only
#: name set through the blob. Never a real header's own name — every real
#: payload entry spells ``aether/...``, ``rtc/...`` or ``plugin/...``, so a
#: bare dunder name with no such prefix cannot collide with one, now or as
#: the manifests grow.
_HOST_ONLY_MANIFEST_MEMBER = "__host_only_names__"


def _pack(headers: Mapping[str, bytes],
          host_only_names: frozenset[str] = frozenset()) -> bytes:
    """Serialise `headers` (both sections — NVRTC-clean and host-only —
    already merged into one mapping by the caller) to the on-disk blob
    format: a zlib-compressed tar, one member per header plus one more
    (:data:`_HOST_ONLY_MANIFEST_MEMBER`) naming which of those members are
    host-only, newline-joined and sorted. Opacity, not secrecy, so a viewer
    cannot casually `strings` or `cat` the payload."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in sorted(headers):
            data = headers[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if host_only_names:
            data = "\n".join(sorted(host_only_names)).encode()
            info = tarfile.TarInfo(name=_HOST_ONLY_MANIFEST_MEMBER)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return zlib.compress(buf.getvalue(), level=9)


def _unpack(blob: bytes) -> tuple[dict[str, bytes], frozenset[str]]:
    """The inverse of :func:`_pack`: ``(headers, host_only_names)`` — the
    latter empty when the blob predates the host-only section (an older
    payload sealed only the NVRTC-clean set)."""
    raw = zlib.decompress(blob)
    headers: dict[str, bytes] = {}
    host_only_names: frozenset[str] = frozenset()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            data = extracted.read() if extracted else b""
            if member.name == _HOST_ONLY_MANIFEST_MEMBER:
                host_only_names = frozenset(
                    ln for ln in data.decode().splitlines() if ln)
            else:
                headers[member.name] = data
    return headers, host_only_names


def _is_tmpfs(path: Path) -> bool:
    """Whether `path` is itself a tmpfs mount (prefer `/dev/shm` when it is
    one). Read from `/proc/mounts` rather than assumed from the name —
    `/dev/shm` is conventional, not guaranteed, and measurement found a box
    where a private tmpfs mount is refused entirely (SELinux enforcing), so
    this function only ever REPORTS what is already mounted, never mounts
    anything itself."""
    try:
        target = str(path.resolve())
    except OSError:
        return False
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == target and parts[2] == "tmpfs":
                    return True
    except OSError:
        return False
    return False


def _ram_dir() -> Path:
    """Where :meth:`Payload.serve` stages files: `/dev/shm` when it is a
    tmpfs, else `$TMPDIR` (the measured fallback when a private mount
    namespace is refused)."""
    shm = Path("/dev/shm")
    if shm.is_dir() and _is_tmpfs(shm):
        return shm
    return Path(os.environ.get("TMPDIR", "/tmp"))


class Payload:
    """A sealed set of headers plus its content digest.

    `headers` maps include-relative names (`aether/...`, `rtc/<std name>`,
    `plugin/gref_layout.h`) to their bytes; `digest` is :func:`digest_of`
    applied to that mapping — two `Payload` instances with the same file set and
    content always carry the same digest, independent of how or in what
    order their files were discovered (:mod:`aether_dsc.seal`'s own
    order-independence guarantee, exercised by
    `tests/test_aether_dsc.py::test_seal_digest_order_independent`).

    `host_only_names` names the subset of `headers` that is the HOST-ONLY
    section: sealed (so :meth:`serve` writes them out for a host `-I` root
    exactly like every other entry) but never claimed NVRTC-clean and never
    handed to an NVRTC compile — see :attr:`device_headers`. `digest` covers
    `headers` whole, host-only entries included, so a changed host-only file
    changes the digest too (the cache-validity term every compile keys on).
    """

    __slots__ = ("digest", "headers", "host_only_names")

    def __init__(self, digest: str, headers: Mapping[str, bytes],
                host_only_names: frozenset[str] = frozenset()):
        self.digest = digest
        self.headers = dict(headers)
        self.host_only_names = frozenset(host_only_names)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Payload(digest={self.digest[:12]}…, {len(self.headers)} headers, "
                f"{len(self.host_only_names)} host-only)")

    @property
    def device_headers(self) -> dict[str, bytes]:
        """`headers` minus the host-only section — what an NVRTC compile is
        actually served (:mod:`hawk.compile.drivers`'s device path,
        :mod:`hawk.compile.nvrtc`'s public ``device``/``cubin``): the
        host-only headers are never NVRTC-clean-checked and must never
        reach NVRTC even as an unreferenced, served-but-unused file."""
        if not self.host_only_names:
            return self.headers
        return {k: v for k, v in self.headers.items() if k not in self.host_only_names}

    @contextlib.contextmanager
    def serve(self) -> Iterator[Path]:
        """Materialise every header into a PRIVATE, owner-only directory for
        the compile's lifetime, then remove it — on the normal exit path and
        when the `with` block raises.

        Prefers a tmpfs (`/dev/shm`, checked via `/proc/mounts` — never
        assumed from the path alone) so nothing touches spinning or
        networked storage; falls back to `$TMPDIR` otherwise. A private
        mount namespace with its own tmpfs is an opportunistic upgrade this
        function does not attempt: measurement found it refused outright on
        an SELinux-enforcing box, so the plain RAM directory is the one
        path every caller can rely on.
        """
        base = _ram_dir()
        base.mkdir(parents=True, exist_ok=True)
        served = Path(tempfile.mkdtemp(prefix="aether_dsc_", dir=str(base)))
        os.chmod(served, 0o700)
        try:
            for name, data in self.headers.items():
                dest = served / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                os.chmod(dest, 0o600)
            yield served
        finally:
            shutil.rmtree(served, ignore_errors=True)
