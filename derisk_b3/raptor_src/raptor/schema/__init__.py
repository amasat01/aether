# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.schema — the two-layer kernel-manifest schema.

:mod:`raptor.schema.manifest` is the bird-neutral layer (hawk's writer binds
here only); :mod:`raptor.schema.blocks` is the ``graph_block`` extension
(hawk never imports it); :mod:`raptor.schema.dtypes` is the scalar-type tag
vocabulary. This top-level module is NOT part of the purity gate (only
``manifest.py`` and ``dtypes.py`` are) — it is the combined public surface, so
it is the one place allowed to reference both layers.

:func:`validate_manifest` is a MINIMAL shape validator (key order, top keys,
schema_version, pattern vocabulary) — it is NOT a port of eagle's deep
``validate_sidecar`` (eagle stays the deep validator; raptor validates SHAPE).
"""

from __future__ import annotations

from .. import _core
from . import blocks, dtypes, manifest

#: the full recognized pattern vocabulary (base + the graph_block extension).
ALL_PATTERNS = manifest.BASE_PATTERNS | {blocks.GRAPH_BLOCK_PATTERN}

__all__ = [
    "ALL_PATTERNS",
    "blocks",
    "dtypes",
    "manifest",
    "validate_manifest",
]


def validate_manifest(doc: dict) -> None:
    """Validate the SHAPE of a manifest or sidecar document; raise on failure.

    Checks, in order:

    1. ``doc`` is a ``dict``.
    2. ``schema_version`` (or the legacy ``version`` key) is present-or-absent
       -leniently valid and not greater than
       :data:`raptor.schema.manifest.MAX_SCHEMA_VERSION` (forward-strict; a
       transition bridge — v1 and v2 both load today).
    3. The v2 execution axis (:func:`raptor.schema.manifest.check_execution_axis`):
       v1 must not carry any ``exec_*`` key; v2 requires ``exec_targets`` +
       ``exec_access`` and validates their vocabulary + ``exec_op``'s
       mapreduce-conditional requiredness. Either version's ``aether_abi``, when
       present, must match its version's tag.
    4. ``launch.lane_space``, when present, has the documented shape
       (:func:`raptor.schema.manifest.check_lane_space`); the optional ``launch.index_set`` /
       ``launch.local_*`` fields and the team-stage fields (``stages[k].team``, ``team_planes``,
       ``keyed``) are shape-checked the same way
       (:func:`raptor.schema.manifest.check_launch_fields`,
       :func:`raptor.schema.manifest.check_team_fields`).
    5. ``pattern``, when present, is one of :data:`ALL_PATTERNS`.
    6. For a MANIFEST-shaped document (one carrying a ``plugins`` key), the
       top-level keys that are present appear in the contractual relative order: the
       graph_block order (:data:`raptor.schema.blocks.BLOCK_TOP_LEVEL_KEY_ORDER`, with
       ``blocks``) when the document itself carries a ``blocks`` key, else the
       version-appropriate base order
       (:data:`raptor.schema.manifest.TOP_LEVEL_KEY_ORDER` for v1,
       :data:`raptor.schema.manifest.TOP_LEVEL_KEY_ORDER_V2` for v2). A bare sidecar (no
       ``plugins``) is not order-checked here — that is eagle's deep validator's job,
       not this shape check's.
    """
    _core.schema_check("document", doc)
