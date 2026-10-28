# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`aether_dsc.openmp()`: where the one shipped OpenMP runtime is, for the packages that link it."""
import sys

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
