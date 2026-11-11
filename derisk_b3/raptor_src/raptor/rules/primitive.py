# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A user primitive and its derivative rule, as DATA: a rule is a TEMPLATE (ordinary IR over placeholder leaves), registered
per input signature into a :class:`~raptor.ir.kinds.Registry` the caller passes in. Nothing callable is stored: instantiation
substitutes a node's operands for the placeholders, structurally."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.kinds import CORE, Registry
from ..ir.loop_nodes import TAPE_READ_KIND, Loop, LoopCarry, LoopIndex
from ..ir.nodes import PRIMITIVE_KIND, HawkError, Leaf, Node, Primitive
from ..ir.types import TensorType
from .core import zero_like

PREFIX = "$"                 # the name of a placeholder leaf starts with it; no plane or uniform may
DIRECTIONS = ("vjp", "jvp")
_WORDS = {"vjp": "reverse (vjp)", "jvp": "forward (jvp)"}


@dataclass(frozen=True)
class PrimitiveDecl:
    """What registration declares of one primitive: its name, its extension id and the directions it supplies a rule for."""

    name: str
    ext: str
    directions: tuple


def placeholder(tag: str, ttype: TensorType) -> Leaf:
    """The free leaf a template reads: ``in<i>``, ``out``, ``cot`` (the incoming adjoint), ``dot<i>`` (an input's tangent)."""
    return Leaf("uniform", "uniform", PREFIX + tag, ttype)


def signature_of(node: Primitive) -> tuple:
    """The input types and the output type of one primitive node: the key of its rule instance."""
    return tuple(x.ttype for x in node.inputs), node.ttype


def declare_primitive(reg: Registry, name: str, *, ext: str, directions: tuple) -> PrimitiveDecl:
    """Register a primitive's name; an equal re-registration returns the held one, a different one refuses."""
    if not isinstance(name, str) or not name:
        raise HawkError("a primitive is registered by NAME: the name is the boundary node's identity in the walk's digest")
    if name == PRIMITIVE_KIND or name in CORE or name in reg.kinds:
        raise HawkError(f"primitive {name!r}: the name is reserved (a core or declared kind)")
    decl = PrimitiveDecl(name, ext, tuple(directions))
    held = reg.primitives.get(name)
    if held is not None:
        if held[0] == decl:
            return decl
        raise HawkError(f"primitive {name!r} is already registered (by extension {held[0].ext!r}, rules {held[0].directions}); a "
                        "second registration under the same name is refused because the name is what identifies the boundary "
                        "node in Walk.digest")
    reg.primitives[name] = (decl, {})
    return decl


def _edge_nodes(n: Node) -> list:
    """The nodes a loop holds outside its operands: the body's results and the break (its index and carries are its own)."""
    if not isinstance(n, Loop):
        return []
    return [*n.nexts, *n.exit_values, *([] if n.exit_cond is None else [n.exit_cond])]


def _key(node: Node, memo: dict) -> tuple:
    got = memo.get(id(node))
    if got is None:
        got = memo[id(node)] = (node.kind, node.ttype, node.dedup_extra, getattr(node, "name", None) if isinstance(node, Leaf) else None,
                                repr(sorted(node.state().items())) if isinstance(node, Loop) else None,
                                tuple(_key(c, memo) for c in node.operands), tuple(_key(c, memo) for c in _edge_nodes(node)))
    return got


def _check_template(name: str, node: Node, allowed: set) -> None:
    """A template is CLOSED: its free leaves are the allowed placeholders (inside a loop body too), a loop's index and carries
    are read only inside that loop, and nothing in it writes a plane or reads a tape of a loop outside it."""
    owned: dict = {}                                   # id of a loop -> ids of its own index and carries
    stack, seen = [(node, ())], set()
    while stack:
        n, loops = stack.pop()
        if (id(n), loops) in seen:
            continue
        seen.add((id(n), loops))
        if n.edges() and not isinstance(n, Loop):
            raise HawkError(f"primitive {name!r}: a rule template may not hold a {n.kind} node (it holds edges outside a loop)")
        if isinstance(n, Leaf) and n.name not in allowed:
            raise HawkError(f"primitive {name!r}: a rule template reads {n.role} {n.name!r}"
                            f"{' inside a loop body' if loops else ''}: only its inputs, the incoming adjoint and the input "
                            f"tangents may be free in a rule ({sorted(allowed)})")
        if isinstance(n, (LoopIndex, LoopCarry)) and not any(id(n) in owned[lp] for lp in loops):
            raise HawkError(f"primitive {name!r}: a rule template reads the index or a carry of a loop outside that loop")
        if n.kind == TAPE_READ_KIND:
            raise HawkError(f"primitive {name!r}: a rule template may not read a tape (it belongs to a loop outside the template)")
        stack.extend((c, loops) for c in n.operands)
        if isinstance(n, Loop):
            if n.body_sinks:
                raise HawkError(f"primitive {name!r}: a loop in a rule template may not write a plane (it has body sinks)")
            owned[id(n)] = {id(n.index), *(id(c) for c in n.carries)}
            stack.extend((c, (*loops, id(n))) for c in _edge_nodes(n))


def register_rule(reg: Registry, name: str, signature: tuple, direction: str, templates) -> None:
    """Register a rule instance for ``signature``: vjp = a tuple of templates, one per input; jvp = one template. Validates
    arity, free leaves and the type each template owes; an equal re-registration is a no-op, a different one refuses."""
    held = reg.primitives.get(name)
    if held is None:
        raise HawkError(f"primitive {name!r} is not registered")
    ins, out = signature
    n = len(ins)
    if direction == "vjp":
        templates = tuple(templates)
        if len(templates) != n:
            raise HawkError(
                f"primitive {name!r}: its vjp returned {len(templates)} gradient(s) for {n} input(s). A reverse rule returns "
                "ONE adjoint per declared input, in call order — a short tuple would silently leave an input's gradient at zero")
        owed, allowed = ins, {PREFIX + t for t in ("out", "cot", *(f"in{i}" for i in range(n)))}
    elif direction == "jvp":
        templates, owed = (templates,), (out,)
        allowed = {PREFIX + t for t in ("out", *(f"in{i}" for i in range(n)), *(f"dot{i}" for i in range(n)))}
    else:
        raise HawkError(f"primitive {name!r}: unknown direction {direction!r}")
    for tpl, want in zip(templates, owed):
        if tpl.ttype != want:
            raise HawkError(
                f"primitive {name!r}: its {direction} produced a value typed {tpl.ttype!r} where {want!r} was owed. A supplied rule "
                "is checked against the type it stands in for, because a rank or dtype mismatch here reaches the emitter as a "
                "shape error far from the declaration that caused it")
        _check_template(name, tpl, allowed)
    instance = held[1].setdefault(signature, {})
    key = tuple(_key(t, {}) for t in templates)
    old = instance.get(direction)
    if old is not None and old[0] != key:
        raise HawkError(f"primitive {name!r} is already registered with a different {direction} rule for this signature")
    instance[direction] = (key, templates)


def has_rule(reg: Registry, node: Primitive, direction: str) -> bool:
    held = reg.primitives.get(node.name)
    return held is not None and direction in held[1].get(signature_of(node), {})


def substitute(template: Node, mapping: dict) -> Node:
    """``template`` with each placeholder leaf replaced by ``mapping[name]``, through ordinary operands and through loop bodies
    alike, in ONE traversal. A subgraph that reads no placeholder is kept as it is; a loop is rebuilt with a fresh index and
    fresh carries (so two instantiations never share loop-local nodes) and its body substituted once."""
    memo: dict = {}
    _substitute([template], mapping, memo)
    return memo[id(template)]


def _substitute(roots: list, mapping: dict, memo: dict) -> None:
    stack = list(roots)
    while stack:
        n = stack[-1]
        if id(n) in memo:
            stack.pop()
            continue
        if isinstance(n, Leaf):
            memo[id(n)] = mapping.get(n.name, n)
            stack.pop()
            continue
        todo = [c for c in n.operands if id(c) not in memo]
        if todo:
            stack.extend(todo)
            continue
        stack.pop()
        new = tuple(memo[id(c)] for c in n.operands)
        if isinstance(n, Loop):
            memo[id(n)] = _loop(n, new, mapping, memo)
        else:
            memo[id(n)] = n if all(a is b for a, b in zip(new, n.operands)) else n.rebuilt(new)


def _loop(n: Loop, operands: tuple, mapping: dict, memo: dict) -> Node:
    index = LoopIndex(n.index.depth, n.index.name, n.index.ttype, n.index.bounds)
    carries = tuple(LoopCarry(c.depth, c.slot, c.name, c.ttype) for c in n.carries)
    memo[id(n.index)] = index
    memo.update({id(o): c for o, c in zip(n.carries, carries)})
    _substitute(_edge_nodes(n), mapping, memo)
    edges = {"index": index, "carries": carries, "nexts": tuple(memo[id(x)] for x in n.nexts), "body_sinks": (),
             "exit_cond": None if n.exit_cond is None else memo[id(n.exit_cond)],
             "exit_values": tuple(memo[id(x)] for x in n.exit_values)}
    return n.rebuilt(operands, edges)


def _supplied(reg: Registry, node: Primitive, direction: str):
    held = reg.primitives.get(node.name)
    if held is None or direction not in held[0].directions:
        raise HawkError(
            f"primitive {node.name!r} supplies no {_WORDS[direction]} rule, so this direction cannot be taken. HAWK does not fall "
            "back to differentiating THROUGH a primitive's forward: the forward is inlined and the table would happily produce a "
            f"number, which is exactly the number {node.name!r} was declared to replace. Register a {direction} rule for it")
    got = held[1].get(signature_of(node), {}).get(direction)
    if got is None:
        raise HawkError(f"primitive {node.name!r}: no {direction} rule instance for this signature is registered")
    return got[1]


def vjp_contributions(reg: Registry, node: Primitive, g: Node) -> tuple:
    """``((input, adjoint), ...)`` for one boundary node, from the registered rule instance."""
    templates = _supplied(reg, node, "vjp")
    m = {PREFIX + f"in{i}": x for i, x in enumerate(node.inputs)}
    m[PREFIX + "cot"], m[PREFIX + "out"] = g, node
    return tuple((child, substitute(t, m)) for child, t in zip(node.inputs, templates))


def jvp_tangent(reg: Registry, node: Primitive, supplied) -> Node:
    """The output's tangent for one boundary node; an input with no incoming tangent is a structural zero."""
    (template,) = _supplied(reg, node, "jvp")
    m = {PREFIX + f"in{i}": x for i, x in enumerate(node.inputs)}
    m.update({PREFIX + f"dot{i}": (t if t is not None else zero_like(x.ttype)) for i, (t, x) in enumerate(zip(supplied, node.inputs))})
    m[PREFIX + "out"] = node
    return substitute(template, m)
