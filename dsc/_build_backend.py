# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build backend: scikit-build-core, plus the headers a source distribution needs.

``aether._core`` compiles against aether's own C++ headers, which live one directory above this
package in a checkout. A source distribution carries nothing from outside itself, so the sdist hook
copies the headers ``_core`` and the plugins include (``aether/interop/DLPack.h``, ``DLPackLegacy.h``,
``aether/chunk/Chunk.h``, ``Fill.h``, ``aether/residency/Transport.h``) and every header they include into
``_sdist_include/`` (git-ignored, listed in ``sdist.include``) and removes the copy again afterwards;
CMakeLists.txt finds it as its last resort. Every other hook runs scikit-build-core unchanged.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from scikit_build_core.build import *  # noqa: F403  (every other hook, unchanged)
from scikit_build_core.build import __all__  # noqa: F401
from scikit_build_core.build import build_sdist as _build_sdist

_HERE = Path(__file__).resolve().parent
_INCLUDE = _HERE / "_sdist_include"
_INCLUDE_LINE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def _copy_closure(root: Path, start: list) -> None:
    pending = [(rel, root) for rel in start]
    seen = set()
    while pending:
        rel, base = pending.pop()
        source = next((c / rel for c in (base, root) if (c / rel).is_file()), None)
        if source is None:
            continue  # a system or standard header
        source = source.resolve()
        if source in seen:
            continue
        seen.add(source)
        target = _INCLUDE / source.relative_to(root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for name in _INCLUDE_LINE.findall(source.read_text()):
            pending.append((name, source.parent))


def build_sdist(sdist_directory, config_settings=None):
    """Write the sdist with the aether headers ``_core`` needs."""
    root = _HERE.parent
    shutil.rmtree(_INCLUDE, ignore_errors=True)
    _copy_closure(root, ["aether/interop/DLPack.h", "aether/interop/DLPackLegacy.h", "aether/chunk/Chunk.h",
                         "aether/chunk/Fill.h", "aether/residency/Transport.h"])
    try:
        return _build_sdist(sdist_directory, config_settings)
    finally:
        shutil.rmtree(_INCLUDE, ignore_errors=True)
