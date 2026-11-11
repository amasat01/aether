# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""No node state, registry entry or rule holds a callable: there is no list to add an exception to. The state of a node class is its
``__slots__``; a slot's type is the annotation of the ``__init__`` parameter that fills it. A slot typed ``Callable`` fails; a
slot typed ``Any`` or left unannotated fails (plain data is named by ``raptor.ir.nodes.Plain``)."""

import functools
import importlib
import inspect
import pkgutil
import types

import raptor.ir
from dataclasses import fields, is_dataclass

from raptor.ir.kinds import Registry
from raptor.ir.nodes import Node, Primitive
from raptor.rules.primitive import declare_primitive, placeholder, register_rule, signature_of

#: ``Any``-typed slots that hold plain data (a number, a string, a tuple, a quantity record).


def _classes():
    for m in pkgutil.walk_packages(raptor.ir.__path__, "raptor.ir."):
        importlib.import_module(m.name)
    todo, seen = [Node], []
    while todo:
        c = todo.pop()
        if c not in seen:
            seen.append(c)
            todo += c.__subclasses__()
    return seen


def _annotation(cls, slot):
    for c in cls.__mro__:
        init = c.__dict__.get("__init__")
        if init is not None and slot in (params := inspect.signature(init).parameters):
            note = params[slot].annotation
            return None if note is inspect.Parameter.empty else str(note)
    return "derived"                                   # not an ``__init__`` argument: filled by the class itself


def test_no_node_slot_is_a_callable_or_an_unclassified_any():
    found, loose = set(), set()
    for cls in _classes():
        for slot in cls.__dict__.get("__slots__", ()):
            note = _annotation(cls, slot)
            if note is not None and "Callable" in note:
                found.add((cls.__name__, slot))
            elif (note is None or note in ("Any", "typing.Any")):
                loose.add((cls.__name__, slot))
    assert not loose, f"slots typed Any or unannotated, not classified: {sorted(loose)}"
    assert not found, f"callable-typed node state: {sorted(found)}"


def _callables(root):
    seen, found, stack = set(), [], [("root", root)]
    while stack:
        path, o = stack.pop()
        if id(o) in seen or isinstance(o, (type(None), bool, int, float, str, bytes, type)):
            continue
        seen.add(id(o))
        if isinstance(o, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, functools.partial)):
            found.append(path)
        elif isinstance(o, dict):
            stack += [(f"{path}[{k!r}]", v) for k, v in o.items()]
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack += [(f"{path}[{i}]", v) for i, v in enumerate(o)]
        else:
            names = [f.name for f in fields(o)] if is_dataclass(o) else []
            names += [x for c in type(o).__mro__ for x in getattr(c, "__slots__", ())]
            stack += [(f"{path}.{n}", getattr(o, n)) for n in dict.fromkeys(names) if hasattr(o, n)]
    return found


def test_a_primitive_node_a_registry_and_a_rule_hold_no_callable():
    t = raptor.ir.TensorType((), "f64")
    reg = Registry()
    declare_primitive(reg, "p", ext="user", directions=("vjp",))
    node = Primitive("p", raptor.ir.nodes.Const(1.0, t), (raptor.ir.nodes.Const(2.0, t),))
    register_rule(reg, "p", signature_of(node), "vjp", (placeholder("cot", t),))
    assert node.state() == {"name": "p"}
    assert _callables(node) == [] and _callables(reg.primitives) == [] and _callables(reg.kinds) == []
