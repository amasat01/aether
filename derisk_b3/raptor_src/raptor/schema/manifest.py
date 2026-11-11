# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The bird-neutral (neutral across the family's own eagle/hawk/raptor
packages) kernel-manifest schema — the layer hawk touches.

This module MUST stay clean of the extension family's vocabulary (declared in
:mod:`raptor.schema.blocks`, the sibling module this file never imports — the
purity direction is one-way: ``blocks`` may reference ``manifest``, never the
reverse). This is what lets hawk's writer bind against this module alone and
keep its own core-purity gate green.

Mirrors ``eagle/python/eagle/roles.py`` (the Python half of the schema-v1
protocol) for the bird-neutral subset of its vocabulary, and the C++ twin
``eagle/plugin/roles.h`` / ``eagle/plugin/plugin_registry/manifest.h``.

NOTE on the top-level key order: the full order for the EXTENDED manifest
kind (see :mod:`raptor.schema.blocks`) is ``schema_version, pattern, aether_abi,
blocks, plugins`` — spelling the literal ``blocks`` here would tie this
neutral module to the extension's carriage key, so the FULL order is housed
in :mod:`raptor.schema.blocks` instead; :data:`TOP_LEVEL_KEY_ORDER` below is
the neutral 4-key order for an ordinary (unextended) manifest.

A toy manifest document, just enough to pass :func:`check_schema_version`
and :func:`check_execution_axis`::

    doc = {
        "schema_version": 1,
        "pattern": "pure",
        "aether_abi": "aether-abi/1",
        "plugins": [],
    }
    check_schema_version(doc)        # -> 1
    check_execution_axis(doc, 1)     # -> None (no exec keys allowed at v1)
"""

from __future__ import annotations

from .. import _core

#: raptor's own plugin-schema version — the schema triad's number of record.
#: ``eagle.roles.SCHEMA_VERSION`` derives from this value directly (an import,
#: not a copy); the C++ half (``plugin/roles.h::kPluginSchemaVersion``) stays
#: a literal, cross-checked against the derived Python value by
#: ``test_cpp_and_python_schema_version_identical`` in eagle's
#: ``python/tests/test_roles_vocab.py``.
#: hawk's own contracts module carries a second, SANCTIONED literal copy (a
#: severing law forbids it importing raptor) — guarded by eagle's own
#: structural AST scan and hawk's own import check.
SCHEMA_VERSION = 1

#: the highest ``schema_version`` this loader ACCEPTS today — a TRANSITION
#: BRIDGE: raptor accepts SCHEMA_VERSION 1, 2 AND 3 on the aether branches.
#: v2 added the execution axis; v3 is v2 with ``cells`` replacing
#: ``sample_sums`` in the sidecar (the record of reductions over samples,
#: ``{name: {op, columns}}``); the manifest shape and ABI tag are v2's.
#: Deliberately a SEPARATE constant from :data:`SCHEMA_VERSION`, not a bump of
#: it: :data:`SCHEMA_VERSION` is the cross-repo FOUR-WAY PIN against eagle's
#: C++ ``kPluginSchemaVersion`` literal and hawk's contracts-module literal
#: copy — both still ``1`` today (the twin-site bump across all four repos is
#: a separate, later work item, not this raptor-only change). Bumping
#: :data:`SCHEMA_VERSION` here would break that pin against repos this change
#: does not touch, for no benefit: :func:`check_schema_version`'s
#: forward-strict ceiling is the only thing that needs to move to let v2 load.
MAX_SCHEMA_VERSION = 3

#: entry-point symbol of every kernel artifact, generated or hand-written
#: (verbatim value of ``eagle.launch.KERNEL_NAME``).
KERNEL_NAME = "raptor_kernel"

#: the FIELD name carrying the AETHER ABI tag on a manifest/sidecar. The VALUE
#: (e.g. ``"aether-abi/1"``) stays eagle-owned; raptor only names the field.
AETHER_ABI_FIELD = "aether_abi"

#: the ``aether_abi`` WIRE VALUE a schema-v1 document must carry, when the field is
#: present (schema v1 does not require the field at all — see
#: :func:`check_execution_axis`). Named here, newly, only so the shape validator can
#: catch a version/tag mismatch; this is still not the deep ABI-tag-vs-symbol
#: agreement check, which stays eagle's loader (``eagle_abi_tag`` dlsym read).
AETHER_ABI_V1 = "aether-abi/1"

#: the ``aether_abi`` WIRE VALUE a schema-v2 document must carry, when the field is
#: present. See :data:`AETHER_ABI_V1`.
AETHER_ABI_V2 = "aether-abi/2"

#: the ``exec_targets`` vocabulary a schema-v2 manifest may name (the two
#: structural entry points device kernels and host teams launch through). MPI
#: ranks and NCCL device groups are not distinct target VALUES — they are
#: execution STRUCTURES eagle's planner may choose for an existing target, so
#: they never appear here.
EXEC_TARGETS = frozenset({"device", "host"})

#: the ``exec_access`` vocabulary a schema-v2 manifest must declare: how the
#: plugin body touches sample-local vs. cross-sample data, as classified by the
#: emitter from the trace, never guessed.
EXEC_ACCESS_CLASSES = frozenset(
    {"sample_local", "cross_sample_read", "cross_sample_write", "mapreduce"}
)

#: the reduction-operator vocabulary a schema-v2 manifest's ``exec_op`` may declare —
#: REQUIRED iff ``exec_access == "mapreduce"`` and FORBIDDEN otherwise (the
#: ``mapreduce(op)`` declared aether functor).
EXEC_OPS = frozenset({"sum", "times", "max", "land"})

#: the v2-only FLAT top-level keys naming the execution axis. Flat, not a
#: single nested ``execution`` block: the manifest is a strict-key,
#: order-contractual document a C++ text scanner reads (module NOTE above) —
#: flat keys are the format this schema already uses for every other axis
#: (``pattern``, ``aether_abi``), so the axis is flattened to match rather
#: than introducing the schema's first nested object.
EXEC_KEYS = ("exec_targets", "exec_access", "exec_op")

#: top-level key order of an ORDINARY (unextended) manifest — contractual for
#: the C++ text scanner. See the module NOTE above for why the extended
#: variant (with ``blocks``) is not spelled here.
TOP_LEVEL_KEY_ORDER = ("schema_version", "pattern", "aether_abi", "plugins")

#: top-level key order of an ORDINARY schema-v2 manifest: :data:`TOP_LEVEL_KEY_ORDER`
#: with :data:`EXEC_KEYS` inserted between ``aether_abi`` and ``plugins`` (the
#: execution axis is a new, versioned block of top-level keys). The ``blocks``-carrying
#: extended variant (see :mod:`raptor.schema.blocks`) is unaffected — v2 exec keys do
#: not currently combine with that extension.
TOP_LEVEL_KEY_ORDER_V2 = (
    "schema_version",
    "pattern",
    "aether_abi",
    *EXEC_KEYS,
    "plugins",
)

#: the base plugin-family vocabulary a bird-neutral manifest/sidecar may name
#: (schema v1). The extension family (:mod:`raptor.schema.blocks`) adds
#: exactly one more value.
BASE_PATTERNS = frozenset({"vector", "pure"})

#: the ``kind`` discriminant of an exec reference
#: (``{"kind": ..., "kernel": ...}``). One value in v1: the referenced
#: artifact is a plain plugin kernel.
EXEC_REF_KINDS = frozenset({"kernel"})

#: the manifest-entry ``format`` vocabulary (schema v1) — the artifact
#: container a plugin manifest entry names. ``host`` is a host shared library
#: (a CPU plugin); the others are device artifacts.
MANIFEST_FORMATS = frozenset({"ptx", "cubin", "fatbin", "host"})


def check_schema_version(
    meta: dict, *, name: str = "<document>", allow_legacy_version_key: bool = False
) -> int:
    """Return the artifact's plugin-schema version, rejecting one too new to load.

    ``schema_version`` absent means v1 (backward-lenient: pre-freeze artifacts
    stay loadable). A version greater than :data:`MAX_SCHEMA_VERSION` is rejected
    (forward-strict) — the transition bridge: v1, v2 AND v3 load today, v4+
    does not. Mirrors ``eagle.roles.check_schema_version``
    (``eagle/python/eagle/roles.py::check_schema_version``) and the C++
    ``kPluginSchemaVersion`` gate. ``allow_legacy_version_key`` scopes the
    legacy ``version``-key fallback to MANIFESTS only, matching the
    eagle-side asymmetry.

    This is the ONE implementation of the version compare in Python,
    family-wide: ``eagle.roles.check_schema_version`` is a re-export of this
    function, not a twin. The rejection message is BYTE-IDENTICAL to the one
    eagle emitted before that unification — including the ``upgrade eagle``
    suffix, which is corpus-pinned in two languages (rows 04/04b/04c) and
    whose re-wording is therefore a corpus + C++ event, refused here. A floor
    road emitting different words would break road-message uniformity, the
    property that makes every floor message corpus-pinned for free. (The
    rejection THRESHOLD moved from :data:`SCHEMA_VERSION` to
    :data:`MAX_SCHEMA_VERSION` — see that constant's docstring for why they
    are two constants, not one bumped in place; the message WORDING is
    unchanged, it just now names the new ceiling.)
    """

    return _core.schema_check("schema_version", meta, name, 1, allow_legacy_version_key)


def check_execution_axis(meta: dict, version: int, *, name: str = "<document>") -> None:
    """Validate the schema-v2 execution axis, given the ALREADY-resolved
    ``version`` (:func:`check_schema_version`'s return).

    schema v1 (``version == 1``): none of :data:`EXEC_KEYS` may be PRESENT — the
    legacy bridge is whole-view/single-device by the AXIS'S ABSENCE, so a v1
    document naming one is a strict-key violation, never a silent upgrade. If
    :data:`AETHER_ABI_FIELD` is present it must read :data:`AETHER_ABI_V1`.

    schema v2 and v3 (``version >= 2``; :func:`check_schema_version` has already
    rejected anything newer): ``exec_targets`` and ``exec_access`` are REQUIRED ("absence
    = load refused"); ``exec_targets`` must be a non-empty list drawn from
    :data:`EXEC_TARGETS` with no duplicates; ``exec_access`` must be one of
    :data:`EXEC_ACCESS_CLASSES`; ``exec_op`` is REQUIRED iff
    ``exec_access == "mapreduce"`` and FORBIDDEN otherwise, and when present must be
    one of :data:`EXEC_OPS`. If :data:`AETHER_ABI_FIELD` is present it must read
    :data:`AETHER_ABI_V2`.

    Only a SHAPE check, matching :func:`raptor.schema.validate_manifest`'s own
    scope note: the deep tag-vs-exported-symbol agreement (the
    ``eagle_abi_tag``/``eagle_layout_sizes`` dlsym read) stays eagle's loader, not
    this module's.
    """

    _core.schema_check("execution_axis", meta, name, version)


def check_lane_space(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``launch.lane_space`` block (shape only; absent = valid).

    ``axes``: non-empty list of ``{label: str | None, extent: int >= 1 | None}``,
    row-major, last axis fastest. Non-null labels are unique; only the FIRST
    extent may be ``None`` (known at run time). ``group``: int >= 1, the
    compaction unit. Unknown keys are refused. Mirrors ``lane_space`` in
    ``docs/schemas/sidecar-v1.schema.json``.
    """

    _core.schema_check("lane_space", meta, name)




def check_launch_fields(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``launch.box``, ``launch.index_set`` and ``launch.local_*`` fields (shape only; absent = valid).

    ``box``: exactly ``{"axis": int >= 0, "T": int >= 2, "stride": int >= 1}``.

    ``index_set``: ``{"width": 32 | 64}`` (a map of lane numbers), or the prefix form
    ``{"width": 32, "prefix": true, "count": <plane name>}`` (no map; the named plane holds how
    many leading lanes run). ``local_bytes`` (int >= 0), ``local_source`` (``"ptxas"`` | ``"ptx"``)
    and ``local_limit`` (int >= 1) come together. Unknown ``index_set`` keys are refused. Mirrors
    ``launch`` in ``docs/schemas/sidecar-v1.schema.json``.
    """

    _core.schema_check("launch_fields", meta, name)


TEAM_GRIDS = frozenset({"lanes", "keys", "chunks", "targets"})
TEAM_PLANE_DTYPES = frozenset({"real", "int32"})





def check_index_planes(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``index_planes`` list (shape only; absent = valid): the names of the input planes
    declared ``Index``, which bind as ``int32`` (a runtime converts an ``int64`` array whose values fit). A
    non-empty list of distinct plane names, each one a name in ``arg_spec``. Mirrors
    ``docs/schemas/sidecar-v1.schema.json``."""

    _core.schema_check("index_planes", meta, name)

def check_team_fields(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional team-stage fields of a split kernel (shape only; absent = valid).

    ``stages[k].team``: ``{"threads": int >= 1, "grid"?: lanes | keys | chunks | targets, ...}``;
    grid ``keys`` carries ``keyed`` (a keyed record), ``chunks`` and ``targets`` carry ``of`` (the
    ordered-add target plane) and ``chunks`` also ``status`` (the out-of-range flag plane).
    ``team_planes``: ``{plane: {"dtype": real | int32, "per_lane": int >= 1, ...}}`` with at most
    one of ``keys`` (keyed record), ``per_target`` (plane name) and ``once`` (true), and an optional
    integer ``fill``. ``keyed``: ``{plane: keyed record}``. Unknown keys of these records are
    refused. Mirrors ``docs/schemas/sidecar-v1.schema.json``.
    """

    _core.schema_check("team_fields", meta, name)
