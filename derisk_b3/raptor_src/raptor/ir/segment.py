# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The names and records of the ``segmented`` dispatch policy: the planes a segmented unit adds, and the unit records.
The split itself is a rewrite and lives in the lowering passes."""

from __future__ import annotations

from . import _gen
from .. import _core

# pure-data records of the manifest (raptor/ir/core.json, arm `records`, home 'segment')
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__, 'segment'))

#: The offsets plane rides the SAME role every other run-time-addressed
#: plane does (``lookup``), read like any other lookup table.
OFFSETS_ROLE = "lookup"

#: K+1 exclusive-prefix entries, the language index type's width (HAWK types an
#: integer leaf ``i64`` regardless of the emitted wire width).
OFFSETS_DTYPE = "i64"

def offsets_name(dispatch_name: str) -> str:
    """The offsets plane's slot name for dispatch group ``dispatch_name``."""
    return f"{dispatch_name}_offsets"


def mix_name(dispatch_name: str) -> str:
    """The K-counter ``lookup`` plane an ``auto`` dispatch counts the classes
    it picks into (one uint32 per branch, integer-atomic): the branch mix a
    launching runtime's chooser reads."""
    return f"{dispatch_name}_mix"


def class_name(dispatch_name: str) -> str:
    """The per-sample class plane a segmented unit writes (the clamped
    selector: the next sort's key; its bincount is the branch mix)."""
    return f"{dispatch_name}_class"


def stale_name(dispatch_name: str) -> str:
    """The per-sample mask a segmented unit sets where the selector no
    longer picks the unit's own branch (the sample sits in a stale run)."""
    return f"{dispatch_name}_stale"


def classify_name(kernel_name: str) -> str:
    """The classify kernel's name for segmented kernel ``kernel_name``."""
    return f"{kernel_name}__cls"
