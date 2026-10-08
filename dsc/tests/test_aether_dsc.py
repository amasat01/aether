# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""aether_dsc acceptance: the seal digest, the payload's shape, `serve()`'s
two leak halves, and the blob's opacity. Also covers the package's own
self-containment: no `sys.path` mutation anywhere under `aether_dsc/`,
and `import aether_dsc` succeeds from a copy holding only the package
directory — nothing on `sys.path` beside it.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aether_dsc
from aether_dsc import _seal_impl
from aether_dsc._core import digest_of

_tools = _seal_impl._tools  # tools/rtc_seal_check.py, imported by _seal_impl.py

# NOTE ON `import aether_dsc.seal`: deliberately never done in this file.
# Python sets `aether_dsc.seal` to the SUBMODULE the instant anything imports
# it that way (`aether_dsc/seal.py`'s own docstring explains why that file
# exists apart from `_seal_impl.py`) — which would shadow the
# `aether_dsc.seal(*roots)` FUNCTION this test file calls throughout. The
# two spellings (`import aether_dsc.seal` for the CLI machinery,
# `aether_dsc.seal(...)` for the function) are used from DIFFERENT
# processes in real operation (`python -m aether_dsc.seal` vs. a normal
# `import aether_dsc`) and are not expected to coexist in one.


def test_seal_function_is_reachable_without_importing_the_cli_submodule():
    assert callable(aether_dsc.seal)


# ---------------------------------------------------------------------------
# seal digest: content-based, order-independent
# ---------------------------------------------------------------------------

def test_digest_is_content_based():
    a = digest_of({"x": b"hello", "y": b"world"})
    b = digest_of({"x": b"hello", "y": b"world"})
    assert a == b


def test_digest_changes_with_content():
    a = digest_of({"x": b"hello"})
    b = digest_of({"x": b"hellO"})
    assert a != b


def test_digest_changes_with_a_renamed_file():
    a = digest_of({"x": b"hello"})
    b = digest_of({"y": b"hello"})
    assert a != b


def test_digest_is_order_independent():
    """Two dicts built by inserting the SAME (name, bytes) pairs in a
    different order digest identically — the walk order a real filesystem
    hands back is not guaranteed, so the digest must not depend on it."""
    forward = {}
    for name, data in [("a", b"1"), ("b", b"2"), ("c", b"3")]:
        forward[name] = data
    backward = {}
    for name, data in [("c", b"3"), ("a", b"1"), ("b", b"2")]:
        backward[name] = data
    assert list(forward) != list(backward)  # the insertion orders really differ
    assert digest_of(forward) == digest_of(backward)


def test_seal_builds_an_order_independent_digest(tmp_path):
    """The same guarantee through :func:`aether_dsc.seal`'s own real
    file-reading path: two manifests naming the SAME files in reverse line
    order seal to the same digest — the payload is a set, not a sequence."""
    aether_dir = tmp_path / "root" / "aether"
    aether_dir.mkdir(parents=True)
    (aether_dir / "one.h").write_text("// one\n")
    (aether_dir / "two.h").write_text("// two\n")
    rtc_dir = tmp_path / "rtc"
    rtc_dir.mkdir()
    for name in _tools.RTC_SHIM_NAMES:
        (rtc_dir / name).write_text(f"// shim {name}\n")
    layout = tmp_path / "gref_layout.h"
    layout.write_text("// layout\n")
    (tmp_path / "gref_abi.h").write_text("// abi\n")  # seal()'s sibling lookup

    # A MINIMAL two-file manifest, in each order, so this test does not
    # depend on the real aether tree's current shape.
    manifest_forward = tmp_path / "manifest_forward.txt"
    manifest_forward.write_text("aether/one.h\naether/two.h\n")
    manifest_backward = tmp_path / "manifest_backward.txt"
    manifest_backward.write_text("aether/two.h\naether/one.h\n")
    # Empty host-only manifest: this test's fake tree carries none of the
    # real host-only files, and the host-only section is not what this test
    # is about.
    host_only_manifest = tmp_path / "manifest_host_only_empty.txt"
    host_only_manifest.write_text("")

    orig_manifest = _tools.PAYLOAD_MANIFEST
    orig_host_only_manifest = _tools.HOST_ONLY_MANIFEST
    try:
        _tools.HOST_ONLY_MANIFEST = host_only_manifest
        _tools.PAYLOAD_MANIFEST = manifest_forward
        p1 = aether_dsc.seal(aether_root=tmp_path / "root", rtc_dir=rtc_dir,
                              layout_header=layout)
        _tools.PAYLOAD_MANIFEST = manifest_backward
        p2 = aether_dsc.seal(aether_root=tmp_path / "root", rtc_dir=rtc_dir,
                              layout_header=layout)
    finally:
        _tools.PAYLOAD_MANIFEST = orig_manifest
        _tools.HOST_ONLY_MANIFEST = orig_host_only_manifest

    assert p1.digest == p2.digest
    assert p1.digest == digest_of(p1.headers)


# ---------------------------------------------------------------------------
# payload().headers shape
# ---------------------------------------------------------------------------

def test_payload_headers_are_relative_with_no_absolute_path():
    p = aether_dsc.payload()
    assert p.headers, "payload() returned an empty header set"
    for name in p.headers:
        assert not name.startswith("/"), f"{name!r} looks like an absolute path"
        assert ":" not in name, f"{name!r} looks like a Windows absolute path"
        assert ".." not in name.split("/"), f"{name!r} escapes its root"


def test_payload_includes_the_seven_shims_and_the_layout_and_abi_headers():
    p = aether_dsc.payload()
    for shim in _tools.RTC_SHIM_NAMES:
        assert f"rtc/{shim}" in p.headers, f"missing shim rtc/{shim}"
    assert "plugin/gref_layout.h" in p.headers
    assert "plugin/gref_abi.h" in p.headers, (
        "the REAL host-only gref_abi.h must be sealed too, for the host -I "
        "path — NVRTC never sees it under this content (nvrtc.py aliases it "
        "to gref_layout.h's bytes)")


def test_payload_digest_matches_its_own_content():
    p = aether_dsc.payload()
    assert p.digest == digest_of(p.headers)


# ---------------------------------------------------------------------------
# the host-only section: sealed, served to a host root, withheld from NVRTC
# ---------------------------------------------------------------------------

def test_payload_carries_the_host_only_manifest_and_nothing_else_as_host_only():
    p = aether_dsc.payload()
    assert p.host_only_names == frozenset(_tools.host_only_manifest_paths())
    assert p.host_only_names, "no host-only headers sealed"
    for name in p.host_only_names:
        assert name in p.headers, f"host-only name {name!r} missing from headers"


def test_device_headers_excludes_every_host_only_name():
    p = aether_dsc.payload()
    dh = p.device_headers
    assert set(dh) == set(p.headers) - p.host_only_names
    for name in p.host_only_names:
        assert name not in dh, f"{name!r} must never reach an NVRTC compile"


def test_serve_materialises_the_host_only_section_too():
    p = aether_dsc.payload()
    with p.serve() as root:
        for name in p.host_only_names:
            assert (root / name).is_file(), f"host-only {name!r} not served"


# ---------------------------------------------------------------------------
# serve(): 0700, owner-only files, absent after normal exit AND after a raise
# ---------------------------------------------------------------------------

def test_serve_directory_is_0700_and_files_are_owner_only():
    p = aether_dsc.payload()
    with p.serve() as root:
        mode = stat.S_IMODE(os.stat(root).st_mode)
        assert mode == 0o700, f"served dir mode {oct(mode)}, want 0700"
        checked = 0
        for name in list(p.headers)[:5]:
            f = root / name
            assert f.is_file()
            fmode = stat.S_IMODE(os.stat(f).st_mode)
            assert fmode & 0o077 == 0, f"{f} is group/other-accessible: {oct(fmode)}"
            checked += 1
        assert checked > 0


def test_serve_directory_absent_after_normal_exit():
    p = aether_dsc.payload()
    with p.serve() as root:
        served_path = root
        assert served_path.is_dir()
    assert not served_path.exists(), "serve()'s directory survived a normal exit"


def test_serve_directory_absent_after_an_exception_inside_the_block():
    p = aether_dsc.payload()
    served_path = None
    with pytest.raises(ValueError):
        with p.serve() as root:
            served_path = root
            assert served_path.is_dir()
            raise ValueError("simulated compile failure inside serve()")
    assert served_path is not None
    assert not served_path.exists(), "serve()'s directory survived a raised exception"


def test_serve_uses_a_ram_directory():
    """The RAM-directory fallback: `/dev/shm` when it is a tmpfs, else
    `$TMPDIR` — never a bare `/tmp` guess and never the CWD."""
    from pathlib import Path

    from aether_dsc._core import _is_tmpfs, _ram_dir

    base = _ram_dir()
    if base == Path("/dev/shm"):
        assert _is_tmpfs(base), "_ram_dir() picked /dev/shm but it is not a tmpfs"
    else:
        assert str(base) == os.environ.get("TMPDIR", "/tmp")


# ---------------------------------------------------------------------------
# cccl_include_dir(): the nvidia.cuda_cccl lookup survives a box with no
# top-level `nvidia` package at all, and falls back through a toolkit's
# targets/<arch>-linux layout.
# ---------------------------------------------------------------------------

def _hide_cccl_wheel_entirely(monkeypatch):
    """`importlib.util.find_spec("nvidia.cuda_cccl")` as it behaves on a box
    with NO `nvidia-cuda-*` wheel installed at all: the dotted lookup tries
    to import the PARENT (`nvidia`) first and raises `ModuleNotFoundError`
    rather than returning `None` — this is the exact bug
    `cccl_include_dir()` used to crash on (hawk's `hawk.compile.nvrtc`
    carries the identical fix)."""
    real = _tools.importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "nvidia.cuda_cccl":
            raise ModuleNotFoundError("simulated: no module named 'nvidia'")
        return real(name, *a, **k)

    monkeypatch.setattr(_tools.importlib.util, "find_spec", fake)


def _hide_cccl_wheel_missing_submodule(monkeypatch):
    """The DIFFERENT "not installed" shape: the `nvidia` namespace package
    exists (some other `nvidia-*` wheel created it) but `cuda_cccl` is not
    one of its submodules — `find_spec` returns `None` here, no raise."""
    real = _tools.importlib.util.find_spec
    monkeypatch.setattr(
        _tools.importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "nvidia.cuda_cccl" else real(name, *a, **k))


def _toolkit(root, include: str):
    d = root / include
    (d / "cuda" / "std").mkdir(parents=True)
    return d


def test_cccl_include_dir_does_not_raise_when_no_nvidia_namespace_package_exists(
        monkeypatch, tmp_path):
    """The actual bug: `find_spec` raising `ModuleNotFoundError` (not
    returning `None`) used to propagate straight out of `cccl_include_dir()`
    instead of being treated as "the wheel is not installed" and falling
    through to the toolkit search below."""
    _hide_cccl_wheel_entirely(monkeypatch)
    want = _toolkit(tmp_path, "include")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert _tools.cccl_include_dir() == str(want)


def test_cccl_found_in_a_conda_toolkit_targets_layout(monkeypatch, tmp_path):
    """A conda CUDA toolkit has no flat include/cuda/std; its CCCL lives
    under targets/<arch>-linux/include."""
    _hide_cccl_wheel_missing_submodule(monkeypatch)
    want = _toolkit(tmp_path, f"targets/{_tools.platform.machine()}-linux/include")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert _tools.cccl_include_dir() == str(want)


def test_cccl_found_through_conda_prefix_when_cuda_path_is_unset(monkeypatch, tmp_path):
    _hide_cccl_wheel_missing_submodule(monkeypatch)
    want = _toolkit(tmp_path, f"targets/{_tools.platform.machine()}-linux/include")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert _tools.cccl_include_dir() == str(want)


def test_cccl_found_in_a_cuda13_include_cccl_layout(monkeypatch, tmp_path):
    """CUDA 13 moved CCCL under include/cccl."""
    _hide_cccl_wheel_missing_submodule(monkeypatch)
    monkeypatch.setattr(_tools, "_loaded_nvrtc_root", lambda: None)
    want = _toolkit(tmp_path, "include/cccl")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert _tools.cccl_include_dir() == str(want)


def test_cccl_prefers_a_flat_toolkit_include_over_the_targets_layout(monkeypatch, tmp_path):
    _hide_cccl_wheel_missing_submodule(monkeypatch)
    want = _toolkit(tmp_path, "include")
    _toolkit(tmp_path, f"targets/{_tools.platform.machine()}-linux/include")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert _tools.cccl_include_dir() == str(want)


# ---------------------------------------------------------------------------
# opacity: the blob is not readable source
# ---------------------------------------------------------------------------

def test_blob_file_is_not_utf8_text_and_hides_source_strings():
    blobs = sorted(aether_dsc.PAYLOAD_DIR.glob("*.bin"))
    assert blobs, "no sealed blob to check — run `python -m aether_dsc.seal` first"
    raw = blobs[0].read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert b"namespace aether" not in raw
    assert b"#include" not in raw


# ---------------------------------------------------------------------------
# self-containment: no sys.path mutation anywhere under the
# package, and importable from a copy holding nothing but the package itself.
# ---------------------------------------------------------------------------

#: A real mutation call/assignment, not an ENGLISH mention of the phrase in a
#: comment or docstring (this file's own docstrings say "no ... mutation" —
#: a bare substring match on the two words would flag its own explanation).
_SYS_PATH_MUTATION = re.compile(r"sys\s*\.\s*path\s*(?:\.\s*\w+\(|=(?!=))")


def test_no_module_under_aether_dsc_mutates_sys_path():
    """The package used to reach OUTSIDE itself (`sys.path.insert(0, ...
    "tools")`, borrowing `aether/tools/rtc_seal_check.py`'s logic) — that
    logic now lives INSIDE the package (`_rtc_check.py`), so nothing under
    `aether_dsc/` may touch `sys.path` at all. A grep-style structural check,
    not an import-time behavioural probe: a mutation is a property of the
    SOURCE, and a single run exercising the import-succeeds path would not
    by itself prove none exists."""
    pkg_dir = Path(aether_dsc.__file__).resolve().parent
    offenders = [str(p.relative_to(pkg_dir.parent))
                 for p in sorted(pkg_dir.rglob("*.py"))
                 if _SYS_PATH_MUTATION.search(p.read_text())]
    assert offenders == [], f"sys.path mutation found under aether_dsc/: {offenders}"


def test_import_succeeds_from_a_copy_holding_only_the_package(tmp_path):
    """The real test of self-containment: copy JUST `aether_dsc/` (no
    `aether/tools/`, no aether repo at all — the exact shape a wheel
    install is) onto an otherwise-empty `PYTHONPATH`, run a FRESH
    interpreter from `cwd=/` (so no ambient relative path can help), and
    `import aether_dsc; aether_dsc.payload()` must still work: the manifest
    and the shipped blob are both PACKAGE DATA now, never reached through a
    sibling checkout the copy does not carry."""
    import shutil

    pkg_dir = Path(aether_dsc.__file__).resolve().parent
    copy_root = tmp_path / "isolated"
    copy_root.mkdir()
    shutil.copytree(pkg_dir, copy_root / "aether_dsc",
                     ignore=shutil.ignore_patterns("__pycache__"))

    done = subprocess.run(
        [sys.executable, "-c",
         "import aether_dsc; print(aether_dsc.payload().digest)"],
        capture_output=True, text=True, cwd="/",
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(copy_root),
             "HOME": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == aether_dsc.payload().digest, (
        "the isolated copy sealed a DIFFERENT digest than this process's own "
        f"payload(): {done.stdout!r} vs {aether_dsc.payload().digest!r}"
    )


def test_the_target_deprecation_notice_does_not_count_as_a_diagnostic():
    from aether_dsc._rtc_check import _is_target_deprecation_notice

    assert _is_target_deprecation_notice(
        "nvrtc: warning: Architectures prior to '<compute/sm>_75' are deprecated and may be "
        "removed in a future release")
    assert not _is_target_deprecation_notice("error: identifier \"foo\" is undefined")
