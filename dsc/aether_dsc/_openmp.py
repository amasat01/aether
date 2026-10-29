# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The OpenMP runtime aether-dsc ships, for the packages that build against it.

One OpenMP runtime may live in a process: a second copy aborts on macOS and doubles the thread pool on Linux.
aether-dsc owns it (``aether/_omp/`` inside the installed package) and `aether._core` finds it by a relative rpath.
A dependent (eagle's extension, a host kernel) links that same file instead of vendoring its own, and its wheel
repair excludes the runtime (``auditwheel repair --exclude libgomp.so.1``, ``delocate-wheel --exclude libomp.dylib``).
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenMP:
    """Where the shipped runtime is and how to build against it."""

    library: Path               #: the runtime file, ``.../aether/_omp/libgomp.so.1`` or ``libomp.dylib``
    directory: Path             #: its directory: the rpath a dependent adds
    include: Path | None        #: a directory holding ``omp.h`` (macOS only; Linux compilers bring their own)
    compile_flags: tuple[str, ...]   #: flags that turn OpenMP on in a compile
    link_flags: tuple[str, ...]      #: link line: the runtime by full path and the rpath to it

    def rpath(self) -> str:
        """The directory to put on a dependent's rpath."""
        return str(self.directory)


def openmp() -> OpenMP | None:
    """The OpenMP runtime shipped in this aether install, or ``None`` when the install carries none.

    Link against ``result.library`` with ``result.link_flags`` and compile with ``result.compile_flags``; add nothing
    else for OpenMP, so the whole process shares this one runtime.
    """
    spec = importlib.util.find_spec("aether")
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        directory = Path(root) / "_omp"
        if sys.platform == "darwin":
            library = directory / "libomp.dylib"
        else:   # libgomp.so.1 from a GCC build, libomp.so (LLVM) from a clang build: whichever the wheel carries
            found = sorted(directory.glob("lib*omp*.so*")) if directory.is_dir() else []
            library = found[0] if found else directory / "libgomp.so.1"
        if library.is_file():
            include = directory / "include"
            has_include = (include / "omp.h").is_file()
            if sys.platform == "darwin":
                compile_flags = ("-Xpreprocessor", "-fopenmp") + ((f"-I{include}",) if has_include else ())
            else:
                compile_flags = ("-fopenmp",)
            return OpenMP(library, directory, include if has_include else None, compile_flags,
                          (f"-Wl,-rpath,{directory}", str(library)))
    return None
