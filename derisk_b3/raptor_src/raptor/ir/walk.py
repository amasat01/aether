# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""THE canonical walk.

``canonical(sinks)`` is the ONLY ordered traversal of a HAWK DAG:
depth-first post-order from the sink set, each node emitted once at its
first completion. Interior nodes dedup structurally on
``(kind, dtype, shape, tag, child_ids, literal)``; LEAVES and sinks dedup
on ``(role, name)`` with a TOTAL merge — a collision is refused, never
silently collapsed. Every consumer derives from :attr:`Walk.arg_spec`;
nothing re-walks. A second walk of the same DAG returns a byte-identical
:attr:`Walk.digest`, since every key is structural.
"""

from __future__ import annotations

from .. import _core
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .types import Slot, TensorType
from . import access as _access
from .access import Access
from .nodes import ADJOINT_PREFIX, COMPANION_SUFFIX, NAMESPACE_SEP, SAMPLE_SUM_COMP, AccumWrite, compensates, param_cell_names
from .compound import Quantity, QuantitySpan
from .loop_nodes import Loop, LoopCarry, LoopCount, LoopValue, TapeRead
from .nodes import (
    WideWrite,
    Assign,
    FINISHED_TTYPE,
    NAMESPACE_SEP,
    ROLE_ORDER,
    At,
    Dispatch,
    Finish,
    HawkError,
    LanePosition,
    PlaneExtent,
    Leaf,
    Node,
    SampleSum,
    Op,
    Primitive,
    ReduceSamples,
    Select,
    Sink,
)


from . import _gen
from .. import _core

# pure-data records of the manifest (raptor/ir/core.json, arm `records`, home 'walk')
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__, 'walk'))


@dataclass(frozen=True)
class Walk:
    """The frozen product of :func:`canonical`, computed once.

    ``order`` is the emission-order node list; ``leaves`` the ordered
    ``(role, name, ttype)`` records; ``arg_spec`` their canonical-order
    ``(role, name)`` projection; ``slot_of``/``slot_types`` its inverse and
    type projection; ``access`` the inferred class + op; ``quantities`` the
    reached compound quantities' contiguous wire spans; ``digest`` a
    content hash of all of the above (sidecar only); ``dispatches`` the
    per-kernel dispatch listing, positionally named; ``prior_reads`` the
    ``mutable``-role names read before their first store; ``finish`` is
    ``(mask, steps)`` for a self-finishing kernel, else ``None``."""

    order: tuple[Node, ...]
    leaves: tuple[Slot, ...]
    arg_spec: tuple[tuple[str, str], ...]
    slot_of: Mapping[tuple[str, str], int]
    slot_types: Mapping[tuple[str, str], TensorType]
    access: Access
    quantities: Mapping[str, QuantitySpan]
    digest: str
    dispatches: tuple[DispatchInfo, ...] = ()
    prior_reads: frozenset[str] = frozenset()
    finish: tuple[str, int | str] | None = None

    @property
    def sample_sums(self) -> tuple[str, ...]:
        """The ``accum_out`` planes written one row per 32 samples
        (:class:`~hawk.ir.nodes.SampleSum`), in sink order."""
        return tuple(n.name for n in self.order if isinstance(n, SampleSum))

    @property
    def kinds(self) -> frozenset:
        """The ``kind`` of every node the body reaches (a build mode refuses an op by it)."""
        return frozenset(getattr(n, "kind", None) for n in self.order)

    @property
    def reads_lane_position(self) -> bool:
        """``True`` when the body reads ``hawk.lane_position()`` (:class:`~hawk.ir.nodes.LanePosition`)."""
        return any(isinstance(n, LanePosition) for n in self.order)

    @property
    def reduces(self) -> bool:
        """``True`` when the kernel reduces over samples
        (:class:`~hawk.ir.nodes.ReduceSamples`) — its access class stays
        ``sample_local``, so the reduction is only visible here."""
        return any(isinstance(n, ReduceSamples) for n in self.order)

    @property
    def cells(self) -> dict:
        """The reductions over samples stored by this kernel: plane name to
        ``{"op", "columns"}`` (``columns`` is the width of one cell, 1 for a
        rank-0 value; a wider cell is the planes ``name_0 .. name_{columns-1}``)."""
        return {n.cell: {"op": n.op, "columns": n.columns}
                for n in self.order if isinstance(n, SampleSum)}

    @property
    def companions(self) -> dict:
        """The companion planes the runtime owns: ``<plane>_c`` to ``<plane>``
        for each ``policy="compensated"`` keyed accumulate. The caller never
        binds them."""
        return {n.name + COMPANION_SUFFIX: n.name for n in self.order
                if isinstance(n, AccumWrite) and compensates(n.policy)}

    @property
    def adjoint_like(self) -> dict:
        """``bar_<plane>`` to ``<plane>`` for each accumulate plane that is the
        adjoint of a lookup plane: it is that plane's size, whatever the lane
        count (a Table's length is the caller's)."""
        reads = {n for r, n in self.arg_spec if r in ("lookup", "wide_in")}
        prefix = ADJOINT_PREFIX + "_"
        return {n: n[len(prefix):] for r, n in self.arg_spec
                if r == "accum_out" and n.startswith(prefix) and n[len(prefix):] in reads}

    @property
    def gather_mats(self) -> tuple:
        """The ``mat_in`` planes read ONLY through ``at()``: their column count
        is the caller's (a broadcast Matrix binds compact), not the lane count."""
        from .nodes import At

        direct = {o.name for n in self.order
                  if not isinstance(n, At) for o in n.operands
                  if isinstance(o, Leaf) and o.role == "mat_in"}
        crossed = {n.operands[0].name for n in self.order
                   if isinstance(n, At) and n.operands[0].role == "mat_in"}
        return tuple(sorted(crossed - direct))

    @property
    def param_planes(self) -> tuple[str, ...]:
        """The ``lookup`` planes that are promoted uniforms (one-cell
        parameter planes, see :func:`hawk.diff.param_planes`)."""
        return param_cell_names(self.order)


_ENV = (Slot, QuantitySpan, TensorType, FINISHED_TTYPE, NAMESPACE_SEP, SAMPLE_SUM_COMP)


def canonical(sinks: Sequence[Sink], declared: Sequence[Slot] = (), *,
             force_access: str | None = None) -> Walk:
    """The ONE traversal. Refuses a mixed sink set and a total-merge collision before it returns anything.

    ``declared`` are slots a DECLARATION binds whether or not the body reaches them (the guard's ``terminated`` mask, like a compound
    quantity's unread wires); ``force_access`` FLOORS the inferred access class before the digest is computed (a ``segmented`` UNIT is
    ``cross_sample_write``). Everything happens in the core (``_core.canonical``: traversal, dedup, total merge, declared slots, compensation
    planes, role order and quantity spans, access, dispatch listing, digest); this builds the frozen :class:`Walk` record from its fields."""
    (order, leaves, arg_spec, slot_of, slot_types, cls, quantities, digest, dispatches, prior, finish) = _core.canonical(
        list(sinks), list(declared), force_access, _ENV)
    return Walk(order, leaves, arg_spec, MappingProxyType(slot_of), MappingProxyType(slot_types), Access(cls, None), MappingProxyType(quantities),
                digest, tuple(DispatchInfo(n, K, p, selector_reads=tuple(r), static_selector=s) for n, K, p, r, s in dispatches), frozenset(prior), finish)


def canonical_nodes(sinks: Sequence[Sink]) -> tuple[tuple[Node, ...],
                                                    Mapping[int, int]]:
    """THE traversal's node sequence and its ``id(node) -> position`` map.

    Same traversal as :func:`canonical`, handed to an IR->IR transform
    (:mod:`hawk.diff`) so a derivative is a function of the primal's
    STRUCTURE: structurally equal nodes share ONE position, so a duplicated
    subexpression accumulates one adjoint like a shared one. Derives no
    slots; ``arg_spec``/``slot_of`` stay :func:`canonical`'s alone."""
    nodes, position = _core.walk(list(sinks))[:2]
    return nodes, MappingProxyType(position)


# --------------------------------------------------------------------------
# the per-kernel dispatch listing


def _dispatches(order: tuple[Node, ...]) -> tuple[DispatchInfo, ...]:
    """The ``{name, K, policy}`` record per Dispatch node reached, POSITIONALLY named in the canonical order the digest hashes over; the
    listing is the core's (``_core.dispatch_listing``), DispatchInfo is its pure-data record."""
    return tuple(DispatchInfo(name, K, policy, selector_reads=reads, static_selector=static)
                 for name, K, policy, reads, static in _core.dispatch_listing(list(order)))


def _dispatch_nodes(order: Sequence[Node]) -> list:
    """The Dispatch nodes a group is named for: the top-level ones in canonical order, then those inside lowered loop bodies."""
    return _core.dispatch_nodes(list(order))


def written_names(order: Sequence[Node]) -> frozenset[str]:
    """Every plane name a node in ``order`` commits: sinks and the per-iteration commits a lowered loop carries."""
    return frozenset(_core.written(list(order)))


def dispatch_groups(walk: Walk) -> dict[str, Dispatch]:
    """``{group name: Dispatch node}`` for ``walk``'s dispatches (top level,
    then inside loop bodies), under the names :attr:`Walk.dispatches`
    publishes."""
    return dict(zip((info.name for info in walk.dispatches),
                    _dispatch_nodes(walk.order)))


def loop_body_groups(walk: Walk) -> frozenset[str]:
    """The names of ``walk``'s dispatch groups that sit inside a loop body."""
    top = sum(isinstance(n, Dispatch) for n in walk.order)
    return frozenset(info.name for info in walk.dispatches[top:])


def static_selectors(kernels: Sequence[Any]) -> dict[str, dict[str, bool]]:
    """The step-scoped selector classification: ``{kernel: {group: static}}``
    over ``kernels``, every kernel ONE step runs (a simulation step, a Dag
    stage). A group is static iff its selector reads no plane ANY kernel of
    the step writes — a plane another kernel rewrites changes the class
    between launches exactly as the dispatching kernel's own commit would."""
    written: set = set()
    for k in kernels:
        written |= written_names(k.walk.order)
    return {k.name: {d.name: not any(name in written for _, name in d.selector_reads)
                     for d in k.walk.dispatches}
            for k in kernels}


def _rebuild_loop(loop: Loop, operands: Sequence[Node]) -> Loop:
    """``loop`` with new boundary operands: its body is rewritten so every
    boundary value that changed is replaced by its new one, then the loop is
    minted again by :func:`~hawk.ir.loops.build_loop` (its boundary is
    recomputed from the new body). The header, the placeholders and the
    ``break`` structure are kept. A loop whose body commits sinks of its own
    (a derived loop) is not rebuilt this way."""
    from .loops import build_loop

    if loop.body_sinks:
        raise HawkError(
            "a loop that commits once per iteration (a derived loop) cannot be "
            "rewritten by substitution")
    swapped = {id(old): new for old, new in zip(loop.operands, operands)
               if old is not new}
    n_exit = 0 if loop.exit_cond is None else 1 + len(loop.exit_values)
    roots = loop.body_roots()
    done = list(substitute(roots, swapped)) if swapped else list(roots)
    nexts = done[:loop.carry_count]
    exit_cond, exit_values = None, ()
    if n_exit:
        exit_cond, exit_values = done[loop.carry_count], tuple(done[loop.carry_count + 1:])
    count = operands[-1] if loop.count is not None else None
    return build_loop(loop.index, loop.carries, operands[:loop.carry_count], nexts,
                      start=loop.start, stop=loop.stop, step=loop.step,
                      exit_cond=exit_cond, exit_values=exit_values, count=count,
                      count_tail=loop.count_tail, dynamic=loop.dynamic)


def rebuild(node: Node, operands: Sequence[Node]) -> Node:
    """``node`` with new operands, same kind, type and literal.

    Enumerated rather than generic because a node's constructor carries its
    own invariants (``At`` checks its plane's role, ``Select`` its branches
    agree); a family with no arm here refuses by name rather than silently
    passing through."""
    if isinstance(node, Loop):
        return _rebuild_loop(node, operands)
    return node.rebuilt(operands)


def substitute(roots: Sequence[Node], mapping: dict) -> tuple:
    """``roots`` with every node in ``mapping`` (by ``id``) replaced.

    Bottom-up with a memo; an unchanged subtree is REUSED, not copied — it
    must stay structurally shared with the primal wherever it didn't change,
    or the emitter's fan-out rule would see two copies of one expression."""
    memo: dict = dict(mapping)
    order: list = []
    seen: set = set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, expanded = stack.pop()
        if id(node) in seen or id(node) in mapping:
            continue
        if not expanded:
            stack.append((node, True))
            for child in reversed(node.operands):
                if id(child) not in seen and id(child) not in mapping:
                    stack.append((child, False))
            continue
        seen.add(id(node))
        order.append(node)
    for node in order:
        new = [memo.get(id(c), c) for c in node.operands]
        memo[id(node)] = (node if all(a is b for a, b in zip(new, node.operands))
                          else rebuild(node, new))
    return tuple(memo.get(id(r), r) for r in roots)


def resink(sink: Node, memo: dict) -> Node:
    """``sink`` over the rewritten values (``memo``, by ``id``) of its operands."""
    new = tuple(memo[id(c)] for c in sink.operands)
    if all(a is b for a, b in zip(new, sink.operands)):
        return sink
    if isinstance(sink, (SampleSum, Assign, WideWrite, AccumWrite)):
        return sink.rebuilt(new, {})
    return sink                                     # TapeWrite/Finish: untouched


def post_order(roots, stop=()) -> list:
    """Every node under ``roots``, children before parents, each once; a
    node whose id is in ``stop`` is kept but not entered."""
    out, seen = [], set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, done = stack.pop()
        if id(node) in seen:
            continue
        if done:
            seen.add(id(node))
            out.append(node)
            continue
        stack.append((node, True))
        if id(node) not in stop:
            stack.extend((c, False) for c in reversed(node.operands) if id(c) not in seen)
    return out


def structural_keys(order, memo: dict) -> None:
    """Fill ``memo`` with a structural key per node of ``order`` (children
    first), so a value traced twice counts as one value."""
    for node in order:
        if id(node) not in memo:
            memo[id(node)] = (type(node).__name__, node.kind, node.ttype,
                              getattr(node, "literal", None), node.dedup_extra,
                              getattr(node, "name", None),
                              tuple(memo.get(id(c)) for c in node.operands))
