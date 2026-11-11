# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.ir is a closed definition: the standard library and raptor, nothing else, statically and at run time."""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

import raptor.ir

_ROOT = pathlib.Path(raptor.ir.__file__).parent
_STDLIB = set(sys.stdlib_module_names)


def test_static_imports_are_stdlib_or_raptor():
    stray = []
    for path in sorted(_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                tops = ["raptor"] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            stray += [f"{path.name}:{node.lineno} imports {t}" for t in tops if t not in _STDLIB and t != "raptor"]
    assert not stray, stray


def test_loading_every_submodule_loads_no_outside_package():
    import pkgutil

    names = sorted(m.name for m in pkgutil.walk_packages(raptor.ir.__path__, "raptor.ir."))
    probe = ("import importlib, json, sys\n"
             f"sys.path[:0] = [{str(_ROOT.parent.parent)!r}]\n"
             f"for name in ['raptor.ir'] + {names!r}:\n    importlib.import_module(name)\n"
             "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))")
    done = subprocess.run([sys.executable, "-S", "-c", probe], capture_output=True, text=True, check=True)
    tops = json.loads(done.stdout.strip().splitlines()[-1])
    assert [t for t in tops if t not in _STDLIB and t not in ("raptor", "__main__") and not t.startswith("_")] == []
