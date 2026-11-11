# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``raptor.schema.blocks`` resolves and its entry point runs on a
numpy-less, eagle-less, cupy-less interpreter.

Measured that every symbol the moved ``graph_block`` clause needs already
exists in raptor and imports on a numpy-blocked interpreter; this test
productizes that measurement as raptor's own gate on the zero-dep LOCK
("raptor stays zero-hard-dep and numpy-less-importable").

``python -I`` (isolated mode: no PYTHONPATH, no user site-packages) is run as
a SUBPROCESS that inserts raptor's own repo root onto ``sys.path`` itself
(the one thing ``-I`` does not prevent — it blocks the automatic additions,
not an explicit ``sys.path.insert`` the program performs), then installs a
``sys.meta_path`` blocker for ``numpy``/``eagle``/``cupy`` BEFORE importing
``raptor.schema.blocks`` — the stronger claim (mirrors a downstream
package's own equivalent floor test): it holds even against a lazy or
transitive import a static scan cannot see, not merely those packages'
absence from the environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

#: raptor's own repo root (holds the importable ``raptor`` package) — this
#: file is raptor/tests/test_blocks_zero_dep_floor.py, so .parents[1] is it.
RAPTOR_ROOT = Path(__file__).resolve().parents[1]

_BLOCKED_PREAMBLE = textwrap.dedent(
    """
    import sys

    class _Blocker:
        \"\"\"Makes numpy + eagle + cupy unimportable even if reachable on
        sys.path -- the stronger claim: holds against a lazy/dynamic import a
        static scan cannot see, not merely their absence.\"\"\"

        _BLOCKED = ("numpy", "eagle", "cupy")

        def find_spec(self, fullname, path=None, target=None):
            if fullname in self._BLOCKED or any(
                fullname.startswith(name + ".") for name in self._BLOCKED
            ):
                raise ImportError(f"{fullname} blocked by the zero-dep gate")
            return None

    sys.meta_path.insert(0, _Blocker())
    """
)


def test_blocks_module_resolves_and_entry_point_runs_numpy_eagle_cupy_blocked():
    """``python -I``: numpy/eagle/cupy meta_path-blocked; ``import
    raptor.schema.blocks`` and CALL ``validate_block_descriptor`` on a
    well-formed descriptor -- it must resolve, run, and return without ever
    touching numpy/eagle/cupy."""
    program = _BLOCKED_PREAMBLE + textwrap.dedent(
        f"""
        sys.path.insert(0, {str(RAPTOR_ROOT)!r})

        import raptor.schema.blocks as blocks

        meta = {{
            "schema_version": 1,
            "pattern": blocks.GRAPH_BLOCK_PATTERN,
            "scalar_type": "float64",
            "forward_exec": {{"kind": "kernel", "kernel": "floor_probe_fwd"}},
            "kernel": "floor_probe",
            "in_degree": 1,
            "out_degree": 1,
            "input_width": 1,
            "output_width": 1,
            "state_width": 1,
            "param_width": 1,
            "scatter_policy": "unique_write",
            "arg_spec": [],
        }}
        result = blocks.validate_block_descriptor(meta, name="floor_probe")
        assert result is meta, result

        assert "numpy" not in sys.modules, sorted(sys.modules)
        assert "eagle" not in sys.modules, sorted(sys.modules)
        assert "cupy" not in sys.modules, sorted(sys.modules)
        print("A5-OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0 and "A5-OK" in result.stdout, (
        "raptor.schema.blocks must import and validate_block_descriptor "
        "must resolve+run with numpy/eagle/cupy blocked under `python -I`.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_old_pattern_name_is_refused_by_name():
    import pytest

    from raptor.schema import validate_manifest

    with pytest.raises(ValueError, match="'neural_block' was renamed to 'graph_block' in 0.7.3"):
        validate_manifest({"schema_version": 1, "pattern": "neural_block"})
