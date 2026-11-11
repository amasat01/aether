# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dialect extension protocol: a node kind is DECLARED BY DATA (a :class:`KindDecl`), never by subclassing.

Everything here is a function over a :class:`Registry` that the caller holds and passes in (no module-level table of kinds). A
declaration is plain data (JSON-shaped); :func:`declare` validates it and has the registry build ONE class for the kind, the
Python presentation of the declaration (a node holds its declaration; a pass asks :func:`is_a` and :func:`trait`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .nodes import Assign, At, HawkError, Leaf, Node, Sink, WideWrite, plain_in, plain_out
from .types import TensorType

#: The core kinds a declared kind may refine: name -> (family, class, role of a sink, wire kind).
def _core_kinds() -> MappingProxyType:
    """The core kinds a declared kind may refine, read from the manifest of the core (its rows marked ``refinable``)."""
    from . import nodes
    return MappingProxyType({row["name"]: (row["family"], getattr(nodes, row["class"]), row.get("role"), row.get("wire"))
                             for row in nodes._TABLE["man"]["kinds"] if row.get("refinable")})


CORE = _core_kinds()
FAMILIES = ("expr", "sink", "terminal")
FIELD_TYPES = ("int", "float", "bool", "str", "none", "tuple")
TRAITS = ("elementwise_sink", "reads_by_element", "fresh_read")
SPELL_ARGS = re.compile(r"^(node|target|ref\d+)$")
_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_KEYS = frozenset({"name", "dialect", "ext", "family", "refines", "wire", "operands", "fields", "result", "identity", "traits",
         "spelling", "diff", "no_writer"})


class Registry:
    """The declared kinds of one process or one test, as a value: ``kinds`` maps a name to its :class:`KindDecl`, ``classes``
    to the class built for it, ``primitives`` to the registered :class:`~raptor.rules.primitive.PrimitiveDecl`."""

    __slots__ = ("kinds", "classes", "primitives")

    def __init__(self) -> None:
        self.kinds: dict = {}
        self.classes: dict = {}
        self.primitives: dict = {}


@dataclass(frozen=True)
class KindDecl:
    name: str
    dialect: str
    ext: str
    family: str
    refines: str | None
    wire: str
    operands: tuple          # (min, max or None, families)
    fields: tuple            # ((name, type, default, required), ...)
    result: tuple
    identity: tuple | None
    traits: tuple            # ((name, bool), ...)
    spelling: tuple | None   # (dialect, name, args)
    diff: str
    no_writer: bool = False


def load_manifest(reg: Registry, manifest: dict, *, first_party: tuple = ()) -> tuple:
    """THE way the ``kinds`` arm of a manifest is loaded, raptor's own and an extension's alike: a row that names a ``class`` is a core
    kind (its class is generated from this very row at import: it is only checked to exist), any other row is declared into ``reg``.
    Returns the declarations of the rows it declared."""
    from . import nodes
    out = []
    for row in manifest["kinds"]:
        if "class" in row:
            if not hasattr(nodes, row["class"]):
                raise HawkError(f"manifest {manifest.get('name')!r}: core kind {row.get('name')!r} names class {row['class']!r}, which is not generated")
            continue
        out.append(declare(reg, {"ext": manifest["name"], "dialect": manifest.get("dialect", ""), **row}, first_party=first_party))
    return tuple(out)


def _refuse(name: str, why: str) -> HawkError:
    return HawkError(f"kind {name!r}: {why}")


def _root(reg: Registry, name: str) -> str:
    """The core kind a declared kind finally refines."""
    while name not in CORE:
        name = reg.kinds[name].refines or ("sink" if reg.kinds[name].family == "sink" else "node")
    return name


def declare(reg: Registry, data: dict, *, first_party: tuple = ()) -> KindDecl:
    """Validate ``data`` (a KindDecl as plain data) and add it to ``reg``; an equal re-declaration returns the held one."""
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise HawkError("kind declaration: `name` is required")
    unknown = set(data) - _KEYS
    if unknown:
        raise _refuse(name, f"unknown key(s) {sorted(unknown)}")
    ext = data.get("ext")
    if not isinstance(ext, str) or not _ID.match(ext):
        raise _refuse(name, f"extension id {ext!r} is not [a-z][a-z0-9_-]*")
    if ext not in first_party and not name.startswith(ext.replace("-", "_") + "."):
        raise _refuse(name, f"a third-party kind is spelled '{ext.replace('-', '_')}.<name>': unqualified names are reserved to the core and first-party extensions")
    if name in CORE:
        raise _refuse(name, "a core kind name is reserved")
    family = data.get("family")
    if family not in FAMILIES:
        raise _refuse(name, f"family {family!r} is not one of {FAMILIES}")
    refines = data.get("refines")
    parent = None
    if refines is not None:
        if refines in CORE:
            if CORE[refines][0] != ("sink" if family == "sink" else "expr"):
                raise _refuse(name, f"family {family} does not agree with the family of {refines!r}")
        elif refines in reg.kinds:
            parent = reg.kinds[refines]
            if (parent.family == "sink") != (family == "sink"):
                raise _refuse(name, f"family {family} does not agree with the family of {refines!r}")
        else:
            raise _refuse(name, f"refines {refines!r}, which is not declared")
    chain, cur = 1, parent
    while cur is not None:
        chain, cur = chain + 1, reg.kinds.get(cur.refines)
    if chain > 3:
        raise _refuse(name, "refinement deeper than 3")
    rootname = _root(reg, refines) if refines else ("sink" if family == "sink" else "node")
    root = "at" if rootname == "at" else ("sink" if CORE[rootname][0] == "sink" else "node")
    ops = data.get("operands")
    if ops is None and parent is not None:
        operands = parent.operands
    else:
        ops = ops or {"min": 0, "max": 0}
        mx = ops.get("max")
        operands = (int(ops["min"]), None if mx in (None, "many") else int(mx), tuple(ops.get("families", ())))
    if family == "terminal" and operands[1] != 0:
        raise _refuse(name, "a terminal takes no operands")
    own = tuple((f["name"], f["type"], f.get("default"), "default" not in f) for f in data.get("fields", ()))
    inherited = _all_fields(reg, parent)
    for fname, ftype, default, _req in own:
        if ftype not in FIELD_TYPES:
            raise _refuse(name, f"field {fname!r} has type {ftype!r}, not one of {FIELD_TYPES}")
        if fname in ("name", "role", "kind", "ttype", "operands", "decl") or fname in inherited or hasattr(Node, fname):
            raise _refuse(name, f"field {fname!r} collides with a builtin or inherited name")
        if default is not None and not _has_type(default, ftype):
            raise _refuse(name, f"field {fname!r}: default {default!r} is not a {ftype}")
    result = tuple(data["result"]) if data.get("result") is not None else (parent.result if parent else ("carried",))
    if result[0] not in ("same_as", "scalar_of", "fixed", "carried"):
        raise _refuse(name, f"result rule {result[0]!r} is unknown")
    if result[0] in ("same_as", "scalar_of") and not 0 <= result[1] < operands[0]:
        raise _refuse(name, f"result operand index {result[1]} is outside the {operands[0]} required operand(s)")
    names = {f[0] for f in own} | set(inherited) | set(("name", "role") if family == "sink" else ())
    identity = tuple(data["identity"]) if data.get("identity") is not None else (parent.identity if parent else None)
    if identity is not None and not set(identity) <= names:
        raise _refuse(name, f"identity names unknown field(s) {sorted(set(identity) - names)}")
    traits = dict(parent.traits) if parent else {}
    for t, v in (data.get("traits") or {}).items():
        if t not in TRAITS:
            raise _refuse(name, f"unknown trait {t!r}")
        traits[t] = bool(v)
    spell = data.get("spelling")
    spelling = (parent.spelling if parent else None) if spell is None else (
        spell["dialect"], spell["name"], tuple(spell.get("args", ())))
    if spelling is not None and not all(SPELL_ARGS.match(a) and (not a.startswith("ref") or int(a[3:]) < max(operands[0], 1)
                                                                   or int(a[3:]) < (operands[1] or 0)) for a in spelling[2]):
        raise _refuse(name, f"spelling args {spelling[2]} are not node|target|ref<i> within the operands")
    wire = data.get("wire") or (parent.wire if parent else (CORE[refines][3] if refines in CORE and CORE[refines][3] else name))
    decl = KindDecl(name, data.get("dialect", ""), ext, family, refines, wire, operands, own, result, identity,
                    tuple(sorted(traits.items())), spelling, data.get("diff", "none"), bool(data.get("no_writer", False)))
    held = reg.kinds.get(name)
    if held is not None:
        if held == decl:
            return held
        raise _refuse(name, f"already declared by extension {held.ext!r} with a different declaration")
    reg.kinds[name] = decl
    reg.classes[name] = _build_class(reg, decl, root)
    return decl


def _all_fields(reg: Registry, decl: KindDecl | None) -> dict:
    out: dict = {}
    while decl is not None:
        for f in decl.fields:
            out.setdefault(f[0], f)
        decl = reg.kinds.get(decl.refines)
    return out


def _has_type(v: Any, t: str) -> bool:
    if t == "int":
        return type(v) is int
    if t == "float":
        return type(v) in (int, float)
    if t == "bool":
        return type(v) is bool
    if t == "str":
        return type(v) is str
    if t == "none":
        return v is None
    try:
        plain_out(v)
    except HawkError:
        return False
    return isinstance(v, tuple)


def _build_class(reg: Registry, decl: KindDecl, root: str):
    base = reg.classes[decl.refines] if decl.refines in reg.kinds else CORE[decl.refines or ("sink" if decl.family == "sink" else "node")][1]
    own = tuple(f[0] for f in decl.fields)
    ns: dict = {"__slots__": own, "decl": decl, "decl_root": root, "__module__": __name__,
                "decl_chain": (decl.name, *getattr(base, "decl_chain", ())),
                **{t: v for t, v in decl.traits}}
    if root == "node":
        ns["kind"] = decl.wire
    if decl.identity is not None:
        ns["dedup_extra"] = property(lambda self: tuple(getattr(self, n) for n in self.decl.identity))
    names = [f[0] for f in decl.fields]

    def state(self) -> dict:
        out = {"name": self.name, "role": self.role} if self.decl_root == "sink" else {}
        for f in _all_fields_of(self):
            out[f] = plain_out(getattr(self, f))
        return out

    def from_state(cls, ttype, operands, state):
        return _make(cls, ttype, operands, **{k: plain_in(v) for k, v in state.items()})

    ns.update(state=state, from_state=classmethod(from_state))
    return type("".join(p.capitalize() for p in re.split(r"[._]", decl.name)), (base,), ns)


def _all_fields_of(node) -> list:
    return [f[0] for c in type(node).__mro__ if "decl" in c.__dict__ for f in c.__dict__["decl"].fields]


def _check_result(decl: KindDecl, ttype, operands) -> TensorType:
    rule = decl.result
    if rule[0] == "same_as":
        want = operands[rule[1]].ttype
    elif rule[0] == "scalar_of":
        want = TensorType((), operands[rule[1]].ttype.dtype)
    elif rule[0] == "fixed":
        want = TensorType(tuple(rule[2]), rule[1])
    else:
        if ttype is None and decl.family == "sink" and operands:
            ttype = operands[0].ttype
        if not isinstance(ttype, TensorType):
            raise _refuse(decl.name, f"carries its own type, got {ttype!r}")
        return ttype
    if ttype is not None and ttype != want:
        raise _refuse(decl.name, f"type {ttype!r} is not what its result rule derives, {want!r}")
    return want


def _make(cls, ttype, operands, **fields):
    """The ONE builder of a declared kind's node (verifies arity, families, result type and field types)."""
    d = cls.decl
    operands = tuple(operands)
    lo, hi, fams = d.operands[0], d.operands[1], d.operands[2]
    if len(operands) < lo or (hi is not None and len(operands) > hi):
        raise _refuse(d.name, f"takes {lo}..{'many' if hi is None else hi} operand(s), got {len(operands)}")
    for i, fam in enumerate(fams[:len(operands)]):
        if fam == "plane-leaf" and not isinstance(operands[i], Leaf):
            raise _refuse(d.name, f"operand {i} must be a plane leaf, got {operands[i]!r}")
    ttype = _check_result(d, ttype, operands)
    allf: dict = {}
    for c in cls.__mro__:
        if "decl" in c.__dict__:
            for f in c.__dict__["decl"].fields:
                allf.setdefault(f[0], f)
    builtin = ("name", "role") if cls.decl_root == "sink" else ()
    extra = set(fields) - set(allf) - set(builtin)
    if extra:
        raise _refuse(d.name, f"unknown field(s) {sorted(extra)}")
    for fname, (_n, ftype, default, required) in allf.items():
        if fname not in fields:
            if required:
                raise _refuse(d.name, f"field {fname!r} is required")
            fields[fname] = default
        elif not _has_type(fields[fname], ftype):
            raise _refuse(d.name, f"field {fname!r} must be a {ftype}, got {fields[fname]!r}")
    obj = object.__new__(cls)
    if cls.decl_root == "sink":
        core_role = CORE[_core_of(cls)][2]
        role = fields.get("role", core_role)
        if role is None or not isinstance(fields.get("name"), str):
            raise _refuse(d.name, "a sink needs `name` (and `role` unless the refined core kind fixes it)")
        Sink.__init__(obj, d.wire, role, fields["name"], operands[0], operands[1] if len(operands) > 1 else None, ttype)
    elif cls.decl_root == "at":
        At.__init__(obj, operands[0], operands[1], ttype)
        object.__setattr__(obj, "kind", d.wire)   # still inside the builder: the declared wire replaces At's, before the node escapes
    else:
        Node.__init__(obj, ttype, operands)
    for fname in allf:
        setattr(obj, fname, fields[fname])
    return obj


def _core_of(cls) -> str:
    for c in cls.__mro__:
        for k, v in CORE.items():
            if c is v[1] and k != "node":
                return k
    return "node"


def make_kind(reg: Registry, kind_: str, ttype, operands=(), /, **fields) -> Node:
    """A node of the declared kind ``kind_``; refuses by name an undeclared kind."""
    cls = reg.classes.get(kind_)
    if cls is None:
        raise HawkError(f"kind {kind_!r} is not declared")
    return _make(cls, ttype, operands, **fields)


def kind_name(node: Any) -> str | None:
    """The declared kind of ``node`` (its own, not a refined one's), or ``None`` for a core node."""
    chain = getattr(node, "decl_chain", None)
    return chain[0] if chain else None


def is_a(node: Any, name: str) -> bool:
    """Whether ``node`` is of kind ``name``: declared (itself or a refinement of it) or core."""
    if name in getattr(node, "decl_chain", ()):
        return True
    return name in CORE and isinstance(node, CORE[name][1])


def trait(node: Any, name: str) -> bool:
    if name not in TRAITS:
        raise HawkError(f"unknown trait {name!r}")
    return bool(getattr(node, name, False))
