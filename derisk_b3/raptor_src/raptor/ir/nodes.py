# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tensor-typed IR nodes: leaf / op / sink / select.

ONE typed node family, not a class hierarchy per value shape: every node
carries a :class:`~hawk.types.TensorType` ``(shape, dtype, tag)`` as FIELDS.
Nodes are identity objects — structural equality is a KEY the canonical
walk computes (:mod:`hawk.ir.walk`), so a DAG may share a subexpression
freely. Leaves and sinks BIND a slot, keyed on ``(role, name)`` and merged
totally by the walk (:attr:`Node.binding`). Access is structural:
``own(i)`` (default), ``at(expr)`` (:class:`At`), ``scatter(expr)`` (a
wide/accum sink with an index operand).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from .types import Slot, TensorType

#: What a literal of a node is: plain data.
Plain = Union[int, float, bool, str, tuple, None]

from .. import _core
from . import _gen
from .._core import HawkError

_TABLE = _gen.parse(_core.manifest())
globals().update(_TABLE["const"])
globals().update(_TABLE["str"])


def refuse_factor_arms(what: str, ttype: TensorType) -> None:
    """Refuse a ``select`` or ``dispatch`` whose arms are factorisations (the check lives in the core)."""
    _core.run_check("refuse_factor_arms", what, ttype)


def plain_out(x: Any) -> Any:
    """A literal as JSON-safe data: scalars as they are, tuples tagged; anything else refused."""
    if x is None or type(x) in (int, float, str, bool):
        return x
    if type(x) is tuple:
        return {"tuple": [plain_out(i) for i in x]}
    raise HawkError(f"literal of type {type(x).__name__} not covered")


def plain_in(x: Any) -> Any:
    return tuple(plain_in(i) for i in x["tuple"]) if isinstance(x, dict) else x


_CLASSES = _gen.classes(_TABLE, globals())
globals().update(_CLASSES)
_core.setup(TensorType, Node, Leaf, _CLASSES)


#: HAWK's OWN canonical role order for ``Walk.arg_spec``, chosen to match
#: the previous generator's so artifacts stay comparable. The ONE
#: contractual property: ``arg_spec`` order == entry-signature order.
#: The op KIND an ``@raw_device`` block splices under: opaque author text,
#: not a HAWK op, so it has no type or derivative rule (differentiating it
#: refuses, naming the kind) and its access class must be DECLARED.
RAW_KIND = "raw_device"

#: The language a raw block is written in unless it says otherwise; a writer
#: refuses a block in a language it cannot write.
RAW_LANGUAGE = "aether"

#: The leaf ROLES an ``at(expr)`` read may target: ``lookup`` (table plane)
#: and ``wide_in`` (pinned to the SAME 32-byte ``ScalarHandle``, same access),
#: and the per-sample input planes, read at another lane through ``u.at(i)``.
#: ``GATHER_ROLES`` are the ones read ONLY through ``at()``.
GATHER_ROLES = ("lookup", "wide_in")

#: The width of the language's row reduction: a fixed pairwise tree over this
#: many consecutive samples, the same on every target.
ROW_WIDTH = 32


def extent_of(node: Any) -> str:
    """The extent of ``node``: ``all_samples`` for a reduction over samples,
    ``shared`` when nothing under it depends on the sample (constants, uniforms,
    the sample count), else ``per_sample``."""
    if isinstance(node, ReduceSamples):
        return "all_samples"
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, SampleIndex) or (
                isinstance(n, Leaf) and n.role not in ("uniform", "nsamples", "lookup",
                                                       "wide_in")):
            return "per_sample"
        if isinstance(n, At) and not is_param_cell(n):
            return "per_sample"
        stack.extend(n.operands)
    return "shared"


def peel_clamp(node: Node) -> Node:
    """``node`` without the ``clamp_index`` a table read holds its index in, else ``node`` itself."""
    return node.operands[0] if isinstance(node, Op) and node.kind == "clamp_index" else node


#: The label of the index a PARAMETER CELL carries: a ``uniform`` promoted to
#: a one-cell ``lookup`` plane (:func:`hawk.diff.param_planes`) is read as
#: ``At(plane, RoleConst(0, PARAM_CELL))``. The label tells the access
#: classifier it is one broadcast value, not a cross-sample read.


def is_param_cell(node: Any) -> bool:
    """Whether ``node`` is the read of a parameter cell."""
    return (isinstance(node, At) and isinstance(node.index, RoleConst)
            and node.index.label == PARAM_CELL)


def param_cell_names(nodes: Any) -> tuple:
    """The names of the parameter cells read in ``nodes``, in first-seen order."""
    out: dict = {}
    for node in nodes:
        if is_param_cell(node):
            out.setdefault(node.plane.name, None)
    return tuple(out)

#: The policies the renderer lowers as a ``switch`` statement.
SWITCH_RENDERED = ("switch", "auto")


def compensates(policy: str) -> bool:
    """Whether an accumulate under ``policy`` commits through a companion plane."""
    return policy in ("compensated", KEYED_AUTO)

#: The suffix of a compensated accumulate plane's companion.
COMPANION_SUFFIX = "_c"

#: The prefix of a vjp's adjoint plane (``bar_<plane>``).
ADJOINT_PREFIX = "bar"



class OrderedAccumWrite(AccumWrite):
    """The accumulate a DERIVATIVE of an ordered keyed sum states: add ``value`` into ``name`` at ``index``, the
    contributions of one cell applied in lane order, so the result is the same bits on every launch. operand 2 is the lane's key
    (the keyed sum's) and ``kept`` the kept lane axes ``((label, extent|None), ...)``. It states the math only; how a backend
    keeps the order (a log of the contributions and a fold in lane order) is that backend's lowering of it."""

    __slots__ = ("kept",)

    def state(self) -> dict:
        return {**AccumWrite.state(self), "kept": plain_out(self.kept)}

    @property
    def key(self) -> Node:
        return self.operands[2]

    @classmethod
    def from_state(cls, ttype, operands, state, edges=None):
        return cls(state["name"], operands[0], operands[1], operands[2], plain_in(state["kept"]), ttype)

    def __init__(self, name: str, value: Node, index: Node, key: Node, kept: tuple, ttype: TensorType | None = None) -> None:
        super().__init__(name, value, index, ttype, "ordered")
        self.operands = (*self.operands, key)
        self.kept = tuple(kept)


#: The suffix of a ``sample_sum`` plane's compensation companion.
SAMPLE_SUM_COMP = "_comp"



def sample_sums(sinks: Sequence[Node]) -> tuple:
    """The :class:`SampleSum` sinks of ``sinks``, in order: column ``k`` of
    the kernel's row of terms is the ``k``-th."""
    return tuple(s for s in sinks if isinstance(s, SampleSum))


def element_count(ttype: TensorType) -> int:
    """How many scalars one value of ``ttype`` holds."""
    out = 1
    for extent in ttype.shape:
        out *= int(extent)
    return out


#: The reserved ``lookup`` plane a finishing kernel counts its newly finished
#: samples into: ONE ``uint32`` cell. eagle's ``eagle.FINISHED_PLANE``
#: names the same plane.
FINISHED_TTYPE = TensorType((), "uint32")

#: The reserved ``lookup`` plane a ``steps="auto"`` kernel reads its trip
#: count from (:func:`hawk.steps`): set on the device by the runner, never
#: by the host; eagle's ``eagle.FUSED_STEPS_PLANE`` names the same plane.


def _through(node: Node) -> Node:
    """The node a reference to ``node`` is a reference TO: a Primitive
    stands for its forward and is never rendered itself."""
    while isinstance(node, Primitive):
        node = node.forward
    return node


def same_index(a: Node, b: Node, seen: dict | None = None) -> bool:
    """Whether two index expressions are the same value: one node, or equal
    kind, literal and operands (a key written twice is traced twice). ``seen``
    keeps a shared subexpression from being compared once per path."""
    if a is b:
        return True
    seen = {} if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen:
        return seen[pair]
    out = (type(a) is type(b) and a.ttype == b.ttype
           and len(a.operands) == len(b.operands))
    if out and isinstance(a, Const):
        out = a.literal == b.literal
    elif out:
        out = (isinstance(a, (Op, At)) and getattr(a, "kind", None) == getattr(b, "kind", None)
               and getattr(a, "literal", None) == getattr(b, "literal", None)
               and all(same_index(_through(x), _through(y), seen)
                       for x, y in zip(a.operands, b.operands)))
    seen[pair] = out
    return out


def replace(node: Node, **changes: Any) -> Node:
    """A copy of ``node`` with some slots replaced: a node is not changed after it is built, a pass that needs another
    operand tuple (or any other field) makes a new node with this and hands it on."""
    new = object.__new__(type(node))
    for cls in type(node).__mro__:
        for slot in cls.__dict__.get("__slots__", ()):
            try:
                object.__setattr__(new, slot, object.__getattribute__(node, slot))
            except AttributeError:
                pass
    for key, value in changes.items():
        object.__setattr__(new, key, tuple(value) if key == "operands" else value)
    return new


# ---- IR-building methods that stay Python (listed gap G3) ----

def _At_raw_index(self) -> Node:
    """The index before its clamp (:func:`peel_clamp`): what a pass reads for the index's SHAPE. A pass that
    rewrites the read keeps :attr:`index`, so the read stays clamped."""
    return peel_clamp(self.operands[1])
At.raw_index = property(_At_raw_index)

def _ReduceSamples_recount(self, count: Node) -> ReduceSamples:
    """The same keyed reduction over ``count`` kept coordinates."""
    return self.rebuilt(self.operands, {**self.edges(), "count": count})
ReduceSamples.recount = _ReduceSamples_recount

def _ReduceSamples_stored(self, name: str, ttype: TensorType | None = None) -> tuple:
    """The cell sinks that store this reduction to the plane ``name``: one
    for a scalar, one per component (planes ``name_0``, ``name_1``, ...)
    for a vector or matrix."""
    value = self.operands[0]
    shape = value.ttype.shape
    if self.key is not None:
        return self._keyed(name)
    if not shape:
        return (SampleSum(name, value, ttype, self.op),)
    scalar = TensorType((), value.ttype.dtype)
    index = [(r,) if len(shape) == 1 else (r, c)
             for r in range(shape[0]) for c in range(shape[-1] if len(shape) == 2 else 1)]
    width = len(index)
    return tuple(SampleSum(f"{name}_{k}", Op("component", (value,), scalar, ix if len(shape) == 2 else ix[0]),
                           scalar, self.op, cell=name, columns=width)
                 for k, ix in enumerate(index))
ReduceSamples.stored = _ReduceSamples_stored

def _ReduceSamples_keyed(self, name: str) -> tuple:
    """The accumulates of a reduction over some lane axes: entry ``k`` of
    the payload (row-major) adds into ``name`` at ``key + k * count``, so the
    plane is ``(*payload, *kept)``, payload first."""
    value = self.operands[0]
    shape = value.ttype.shape
    scalar = TensorType((), value.ttype.dtype)
    if not shape:
        return (AccumWrite(name, value, self.key, scalar, self.policy, self),)
    index = [(r,) if len(shape) == 1 else (r, c)
             for r in range(shape[0]) for c in range(shape[-1] if len(shape) == 2 else 1)]
    from .ops import make

    out = []
    for k, ix in enumerate(index):
        entry = Op("component", (value,), scalar, ix if len(shape) == 2 else ix[0])
        at = self.key if k == 0 else make("add", (self.key, make(
            "mul", (Const(k, self.count.ttype), self.count))))
        out.append(AccumWrite(name, entry, at, scalar, self.policy))
    return tuple(out)
ReduceSamples._keyed = _ReduceSamples_keyed

def _SampleSum_companion(self) -> Slot:
    """The compensation plane's slot (a sum only)."""
    return _core.companion(self, Slot, SAMPLE_SUM_COMP)
SampleSum.companion = property(_SampleSum_companion)
