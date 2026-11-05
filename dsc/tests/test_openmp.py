# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`aether_dsc.openmp()`: where the one shipped OpenMP runtime is, for the packages that link it."""
import ctypes
import os
import re
import sys

import pytest

import aether_dsc


def test_a_package_without_the_runtime_reports_none(tmp_path, monkeypatch):
    (tmp_path / "aether").mkdir()
    (tmp_path / "aether" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "aether", raising=False)
    assert aether_dsc.openmp() is None


def test_a_package_with_the_runtime_reports_its_file_and_flags(tmp_path, monkeypatch):
    name = "libomp.dylib" if sys.platform == "darwin" else "libgomp.so.1"
    omp = tmp_path / "aether" / "_omp"
    omp.mkdir(parents=True)
    (tmp_path / "aether" / "__init__.py").write_text("")
    (omp / name).write_bytes(b"")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "aether", raising=False)
    found = aether_dsc.openmp()
    assert found is not None and found.library == omp / name and found.directory == omp
    assert found.rpath() == str(omp)
    assert f"-Wl,-rpath,{omp}" in found.link_flags and str(omp / name) in found.link_flags
    assert "-fopenmp" in found.compile_flags


_RUNTIME = re.compile(r"lib(gomp|omp|iomp5)[^/]*\.(so(\.\d+)*|dylib)$")


def _loaded_runtimes():
    """The distinct OpenMP runtime files mapped into this process."""
    if sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        names = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    else:
        names = [line.split()[-1] for line in open("/proc/self/maps") if "/" in line.split()[-1]]
    return {os.path.realpath(n) for n in names if _RUNTIME.search(n)}


def test_one_openmp_runtime_is_loaded_and_it_is_the_shipped_one():
    shipped = aether_dsc.openmp()
    if shipped is None:
        pytest.skip("this aether install ships no OpenMP runtime (a source checkout)")
    import aether
    from aether import fft

    fft.fft(aether.zeros((64, 4096, 2), "float64"))   # the first OpenMP region runs here
    runtimes = _loaded_runtimes()
    assert runtimes == {os.path.realpath(str(shipped.library))}, runtimes
    if (os.cpu_count() or 1) > 1 and "OMP_NUM_THREADS" not in os.environ:
        assert ctypes.CDLL(str(shipped.library)).omp_get_max_threads() > 1
