# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``graph_block`` extension layer of the schema.

hawk never imports this module — only :mod:`raptor.schema.manifest` (the
family-neutral layer) is part of hawk's core-purity surface. This module is
free to reference ``manifest``, never the reverse.

Values below are copied VERBATIM from ``eagle/python/eagle/roles.py``, cited
by symbol below (symbol citations do not rot the way line citations do);
eagle's roles.py is not modified by this move — a later change may re-point
it at this module instead.

**The descriptor VALIDATOR lives here too.** The clause that enforces this
family's wire contract — :func:`validate_block_descriptor` and the
private ``_validate_graph_block`` / ``_validate_exec_ref`` it is built from —
moved VERBATIM out of ``eagle/python/eagle/sidecar.py`` into this module,
because enforcement lives with the contract's OWNER at (never above) its
lowest-floor consumer: the descriptor wire contract is declared here and
consumed at the raptor+numpy floor, so its validator is spine content. It is
ONE implementation with two importers — a downstream package's two roads and
``eagle.sidecar.validate_sidecar``'s pattern-conditional dispatch — never a
copy. What did NOT move is everything serving the KERNEL families (``ROLES``
/ ``validate_roles``, ``parse_derivative``, the recognized-pattern and
buffer-kind loops, ``MANIFEST_FORMATS``): that vocabulary is eagle-owned and
consumed only by eagle loaders, so its validator stays in eagle for the
``graph_block`` family only — eagle stays the deep validator everywhere
else. This module imports nothing but :mod:`raptor.schema.manifest`, so the
clause runs on a numpy-less, eagle-less interpreter.
"""

from __future__ import annotations

from .. import _core
from . import manifest

#: the graph-block family name (verbatim value from
#: ``eagle/python/eagle/roles.py::RECOGNIZED_PATTERNS``, one element of that
#: set).
GRAPH_BLOCK_PATTERN = "graph_block"

#: the full top-level key order for a graph_block manifest — houses the literal
#: "blocks" carriage key (kept out of the neutral module, per
#: :mod:`raptor.schema.manifest`'s note).
BLOCK_TOP_LEVEL_KEY_ORDER = (
    "schema_version",
    "pattern",
    "aether_abi",
    "blocks",
    "plugins",
)

#: the declared TERMINAL-WRITE CONTRACT of a block's scatter (verbatim value
#: from ``eagle/python/eagle/roles.py::SCATTER_POLICIES``).
SCATTER_POLICIES = frozenset({"unique_write", "accumulate"})

#: the fields a ``graph_block`` descriptor MUST carry, beyond the general
#: required set (verbatim value from
#: ``eagle/python/eagle/roles.py::BLOCK_REQUIRED_FIELDS``, minus a schema-key
#: rename: ``fanin``/``fanout`` -> ``in_degree``/``out_degree``, both ends
#: renamed together, SCHEMA_VERSION stays 1).
BLOCK_REQUIRED_FIELDS = frozenset(
    {
        "in_degree",
        "out_degree",
        "input_width",
        "output_width",
        "state_width",
        "param_width",
        "scatter_policy",
        "forward_exec",
    }
)

#: the keys whose values are exec references (verbatim value from
#: ``eagle/python/eagle/roles.py::BLOCK_EXEC_REF_FIELDS``).
BLOCK_EXEC_REF_FIELDS = frozenset({"forward_exec", "vjp_exec", "jvp_exec"})

#: kernel-machinery fields a descriptor must NOT carry (verbatim value from
#: ``eagle/python/eagle/roles.py::BLOCK_FORBIDDEN_FIELDS``).
BLOCK_FORBIDDEN_FIELDS = frozenset(
    {"aether_abi", "derivative", "buffers", "mutables", "mat_shapes", "host_entry"}
)

#: the declared-buffer ``kind`` vocabulary, schema v1 (verbatim value from
#: ``eagle/python/eagle/roles.py::BUFFER_KINDS``).
BUFFER_KINDS = frozenset({"lookup"})

#: the full pattern vocabulary this extension recognizes: the neutral base
#: (:data:`raptor.schema.manifest.BASE_PATTERNS`) plus ``graph_block``.
ALL_PATTERNS = manifest.BASE_PATTERNS | {GRAPH_BLOCK_PATTERN}

# --------------------------------------------------------------------------- #
# The descriptor-family VALIDATOR: runs in Rust (rcore::manifest). Its checks moved VERBATIM from
# ``eagle/python/eagle/sidecar.py`` — only the vocabulary SPELLING changed
# (``EXEC_REF_KINDS`` -> ``manifest.EXEC_REF_KINDS``); the checks, their order
# and every message byte are the ones the shared conformance corpus and the C++
# twin (``eagle/plugin/sidecar.h``) already pin.
# --------------------------------------------------------------------------- #









def validate_block_descriptor(meta: dict, *, name: str) -> dict:
    """Validate a ``graph_block`` descriptor document; raise on any breach.

    THE public entry point of the descriptor family's wire contract, and the ONE
    implementation of it: a downstream package's graph-block spec type calls it
    directly on construction and when loading from a sidecar (the raptor+numpy
    floor road), and :func:`eagle.sidecar.validate_sidecar` calls it as its
    pattern-conditional ``graph_block`` clause (the eagle road). Two importers,
    no copy — which is what makes the two roads' verdicts and messages identical by
    construction rather than by parallel maintenance.

    The order is the CORPUS order and is contractual: the schema-version gate
    (:func:`raptor.schema.manifest.check_schema_version`) runs BEFORE the clause, so a
    descriptor that is both too new and malformed is refused for being too new — what
    corpus row ``row04c`` pins. ``meta`` is the descriptor as loaded from JSON (or as
    serialized in process); ``name`` names the artifact in every message. Returns
    ``meta`` unchanged, for call-site chaining.
    """
    _core.schema_check("block_descriptor", meta, name)
    return meta
