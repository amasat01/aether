# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lowered unit: the one object a backend is handed after lowering.

Its node graph is the live IR (``sinks``, ``walk``); the rest is an envelope of plain data. The frame is typed here; the keys
inside ``mode``, ``kind``, ``form``, ``policy`` and ``cost`` belong to whoever lowers, and the same ``ir_version`` covers them.
``cost`` is read on first use: the filler hands in a function and the value is kept after the first read."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import TensorType


from . import _gen
from .. import _core

# pure-data records of the manifest (raptor/ir/core.json, arm `records`, home 'unit')
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__, 'unit'))
