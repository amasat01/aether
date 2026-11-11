# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""schema/manifest.py + schema/dtypes.py contain ZERO of hawk's banned
substrings (identifiers AND string literals, case-insensitive) — this is
what lets hawk's writer bind against these modules and keep its own
core-purity gate green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: A guardrail: every test in this file reads raptor's own schema/*.py
#: source AS TEXT from a repo-relative path -- never shipped as package data
#: in the wheel. Deselected only in the isolated-venv leg, always run
#: repo-rooted (CI, dev runs). See pyproject.toml's marker docstring.
pytestmark = pytest.mark.repo_local

BANNED = (
    "active_indices",
    "scatter",
    "gather",
    "graph_block",
    "block_type",
    "source_indices",
    "target_indices",
)

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "raptor" / "schema"


def _hits(text: str) -> dict:
    lower = text.lower()
    return {tok: lower.count(tok) for tok in BANNED if tok in lower}


def test_manifest_py_is_purity_clean():
    text = (SCHEMA_DIR / "manifest.py").read_text()
    hits = _hits(text)
    assert not hits, f"banned substrings found in manifest.py: {hits}"


def test_dtypes_py_is_purity_clean():
    text = (SCHEMA_DIR / "dtypes.py").read_text()
    hits = _hits(text)
    assert not hits, f"banned substrings found in dtypes.py: {hits}"


def test_blocks_py_is_the_extension_layer_and_carries_graph_block_vocabulary():
    """Sanity twin: blocks.py is NOT purity-gated and legitimately carries the
    graph_block vocabulary — confirms the split is real, not accidentally neutered."""
    text = (SCHEMA_DIR / "blocks.py").read_text()
    assert "graph_block" in text.lower()
