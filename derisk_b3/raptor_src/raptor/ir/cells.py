# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""IR-pure helpers a lowering of any form shares: the parameter-cell read, the sink rebuilt over a new value, the row planes of a
cell, and the launch constant of an ordered scatter."""

from __future__ import annotations

from typing import Any

from .kinds import is_a
from .nodes import PARAM_CELL, AccumWrite, Assign, At, Finish, HawkError, Leaf, Node, RoleConst, SampleSum
from .types import TensorType
from .walk import substitute

#: The per-sample ``i32`` plane that holds each sample's parameter cell.
CELL_PLANE = "param_cell"

#: The most lane chunks an ordered scatter adds in (a chunk is one thread's run of lanes).
SCATTER_CHUNKS = 256

_INDEX = TensorType((), "i32")


def planes(cell: str, columns: int) -> tuple:
    """The row planes of a cell: ``name``, or ``name_0 .. name_{columns-1}``."""
    return (cell,) if columns == 1 else tuple(f"{cell}_{k}" for k in range(columns))


def param_cell(name: str, ttype: TensorType, per_cell: bool = False) -> At:
    """The read of parameter ``name``: cell 0 of a one-cell ``lookup`` plane,
    or, with ``per_cell``, the cell the sample's ``param_cell`` plane names."""
    plane = Leaf("table_read", "lookup", name, ttype)
    if per_cell:
        return At(plane, Leaf("vocab_read", "per_sample", CELL_PLANE, _INDEX), ttype)
    return At(plane, RoleConst(0, PARAM_CELL, _INDEX), ttype)


def with_value(sink: Node, value: Node) -> Node:
    """``sink`` committing ``value`` (a Mutable commit, a sample sum, a keyed accumulate, a finish or a team store)."""
    if isinstance(sink, (Finish, SampleSum, AccumWrite, Assign)) or is_a(sink, "team_store"):
        return sink.rebuilt((value, *sink.operands[1:]))
    raise HawkError(
        f"a promotion rebuilds Mutable commits, sample sums, keyed accumulates and a finish; "
        f"{sink.kind} {getattr(sink, 'name', '?')!r} is none of them")


def swap(sinks: Any, mapping: dict) -> tuple:
    """``sinks`` with the nodes in ``mapping`` (by ``id``) replaced."""
    if mapping and any(not hasattr(s, "value") for s in sinks):
        raise HawkError(
            "a kernel that reads a parameter cell cannot also be differentiated "
            "through a `for` that commits a gradient per iteration (a plane "
            "read at a runtime index inside the loop): the per-sample "
            "accumulate that feeds the per-32-sample sum of the parameter is "
            "not built. Differentiate the kernel with the parameter passed as a "
            "plain `Param` (not through param_planes), or read the plane outside "
            "the loop")
    values = substitute(tuple(s.value for s in sinks), mapping)
    return tuple(with_value(s, v) for s, v in zip(sinks, values))
