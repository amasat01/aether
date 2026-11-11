# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The IR classes, generated at import from ONE declaration table (``rs/core/decl.tbl``, compiled into the core module).

Table lines (a class block is indented):  ``const NAME v...`` a tuple of strings; ``str NAME v`` a string; ``record Name f[=default]...`` a
frozen dataclass; ``class Name [: Base]`` then ``slots``, ``kind`` (class attribute), ``params`` (the signature), ``new`` (the core
function that checks the arguments and returns ``(ttype, operands, *vals)``), ``vals`` (the slots it sets, in order), ``binding``
(``slot`` or ``none``), ``dedup`` (read by the core), ``ctor`` (the declared steps and outputs the core runs), ``prop NAME FORM`` (``op N``, ``ops N:``, ``opq N``, ``opattr N A``, ``nops-1``,
``none``), ``state`` (``+Base``, ``f`` or ``f:plain``), ``edges`` (names omitted when None), ``from_edges``, ``from`` (the arguments of
``from_state``: ``ttype``, ``operands``, ``*operands``, ``op:N``, ``opq:N``, ``ops:N:``, ``@k``, ``@k:plain``, ``edge:k``, ``name=tok``).
Nothing here decides what a node IS: the checks, the dedup identity and the walk live in the core."""

from __future__ import annotations

import ast
import json
from collections import namedtuple
from dataclasses import field, make_dataclass

from .. import _core


def parse(text: str) -> dict:
    """The manifest text (raptor/ir/core.json) as the generator's working form."""
    man = json.loads(text)
    out = {"const": {}, "str": {}, "record": [(r["name"], r["fields"], r.get("kind", "dataclass"), r.get("eq", True), r.get("home")) for r in man.get("records", ())], "class": [], "checks": man.get("checks", {}),
           "man": man}
    for k, v in man.get("consts", {}).items():
        (out["const"] if isinstance(v, list) else out["str"])[k] = tuple(v) if isinstance(v, list) else v
    cls_of = {row["name"]: row["class"] for row in man["kinds"] if "class" in row}
    for row in man["kinds"]:
        if "class" not in row:
            continue
        L = {}
        for key, dst in (("slots", "slots"), ("wire_attr", "kind"), ("params", "params"), ("assigns", "vals"), ("binding", "binding"), ("dedup", "dedup"),
                         ("state", "state"), ("methods", "methods"), ("edges", "edges"), ("from", "from"), ("watch", "watch")):
            if key in row:
                L[dst] = row[key]
        if "ctor" in row:
            L["new"] = row["class"]
        if row.get("from_edges"):
            L["from_edges"] = True
        out["class"].append({"name": row["class"], "base": cls_of.get(row.get("refines")), "props": row.get("props", []), "lines": L})
    return out


def _tok(t: str) -> str:
    if t == "ttype":
        return "ttype"
    if t == "operands":
        return "operands"
    if t == "*operands":
        return "*operands"
    if t == "carry_inits":
        return 'operands[:len((edges or {})["carries"])]'
    if t == "free_ops":
        return 'operands[len((edges or {})["carries"]):len((edges or {})["carries"]) + state["n_free"]]'
    if t == "count_op":
        return '(operands[-1] if state["has_count"] else None)'
    if t.startswith("op:"):
        return f"operands[{t[3:]}]"
    if t.startswith("opq:"):
        n = t[4:]
        return f"(operands[{n}] if len(operands) > {n} else None)"
    if t.startswith("ops:"):
        return f"operands[{t[4:]}]"
    if t.startswith("edge:"):
        return f'(edges or {{}}).get("{t[5:]}")'
    if t.startswith("@"):
        k, _, f = t[1:].partition(":")
        return f'plain_in(state["{k}"])' if f == "plain" else f'state["{k}"]'
    raise ValueError(t)


def _prop(name: str, form: list) -> str:
    kind = form[0]
    form = list(form) + [""] * 3
    body = {"op": f"self.operands[{form[1] if len(form) > 1 else 0}]", "ops": f"self.operands[{form[1] if len(form) > 1 else 0}]",
            "opq": f"self.operands[{form[1] if len(form) > 1 else 0}] if len(self.operands) > {form[1] if len(form) > 1 else 0} else None",
            "opattr": f"self.operands[{form[1] if len(form) > 1 else 0}].{form[2] if len(form) > 2 else ''}",
            "nops-1": "len(self.operands) - 1", "none": "None",
            "slice": f"self.operands[{form[1]}:{form[2]}]", "core": f"_core.{form[1]}(self)", "intlit": "int(self.literal)"}[kind]
    return f"    @property\n    def {name}(self):\n        return {body}\n"


def class_source(c: dict) -> str:
    name, base, L = c["name"], c["base"], c["lines"]
    src = [f"class {name}({base or 'object'}):\n"]
    src.append(f"    __slots__ = {tuple(L.get('slots', ()))!r}\n")
    if "kind" in L:
        src.append(f"    kind = {L['kind']!r}\n")
    if "new" in L:
        params = L["params"]
        names = [p.strip().split("=")[0].split(":")[0].strip() for p in params.split(",") if p.strip() not in ("*", "")]
        vals = L["vals"] if "vals" in L else []
        src.append(f"    def __init__(self, {params}) -> None:\n"
                   f"        _construct({_core.ctor_index(name)}, self, {', '.join(names)})\n")
    if L.get("binding") == "slot":
        src.append("    @property\n    def binding(self):\n        return Slot(self.role, self.name, self.ttype)\n")
    elif L.get("binding") == "none":
        src.append("    @property\n    def binding(self):\n        return None\n")
    d = L.get("dedup")
    if d == ["py"]:
        src.append("    @property\n    def dedup_extra(self):\n        state = self.state()\n"
                   "        return _core.keytext(sorted(state.items())) if state else None\n")
    elif d:
        src.append("    @property\n    def dedup_extra(self):\n        return _core.dedup_extra(self)\n")
    for pname, *form in c["props"]:
        src.append(_prop(pname, form))
    st = L.get("state")
    if st is not None or name == "Node":
        lines = ["        out = {}\n"]
        for t in st or ():
            if t.startswith("+"):
                lines.append(f"        out.update({t[1:]}.state(self))\n")
            elif "=" in t:
                k, _, e = t.partition("=")
                lines.append(f'        out["{k}"] = {e}\n')
            else:
                k, _, f = t.partition(":")
                lines.append(f'        out["{k}"] = ' + (f"plain_out(self.{k})" if f else f"self.{k}") + "\n")
        src.append("    def state(self) -> dict:\n" + "".join(lines) + "        return out\n")
    for mname, expr in L.get("methods", ()):
        src.append(f"    def {mname}(self):\n        return {expr}\n")
    ed = L.get("edges")
    if ed is not None or name == "Node":
        if ed is None:
            src.append("    def edges(self) -> dict:\n        return {}\n")
        else:
            pairs = ", ".join(f'("{e.rstrip("?")}", self.{e.rstrip("?")})' for e in ed)
            src.append(f"    def edges(self) -> dict:\n        return {{k: v for k, v in ({pairs},) if v is not None}}\n")
    if "from" in L:
        args = []
        for t in L["from"]:
            if "=" in t and not t.startswith(("@", "edge", "op")):
                k, _, v = t.partition("=")
                args.append(f"{k}={_tok(v)}")
            else:
                args.append(_tok(t))
        sig = "cls, ttype, operands, state, edges=None" if "from_edges" in L else "cls, ttype, operands, state"
        src.append(f"    @classmethod\n    def from_state({sig}):\n        return cls({', '.join(args)})\n")
    if name == "Node":
        src.append('''    @classmethod
    def build(cls, ttype, operands, state, edges):
        """``from_state``, passing the edges only to a class that declares some."""
        return cls.from_state(ttype, operands, state, edges) if edges else cls.from_state(ttype, operands, state)

    def rebuilt(self, operands, edges=None):
        """This node over other operands (and edges), whatever its class (sinks included)."""
        return self.build(self.ttype, operands, self.state(), self.edges() if edges is None else edges)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} {self.kind} {self.ttype}>"

    def __setattr__(self, name, value):
        # a node is not changed after it is built: a slot that is set refuses (a private cache `_x` may be refreshed)
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
        else:
            if name[:1] != "_":
                raise HawkError(f"{type(self).__name__}.{name}: a node is not changed after it is built; make another with replace()")
            object.__setattr__(self, name, value)
''')
    return "".join(src)


def records(table: dict, module: str, home: str | None = None) -> dict:
    """The pure-data records of the manifest: frozen dataclasses (``eq: false`` compares by identity), or named tuples (``kind``); a default
    `{}` is a fresh dict per record, a field named `_x` is left out of the repr."""
    out = {}
    for name, fields, kind, eq, where in table["record"]:
        if where != home:
            continue
        names = [f.partition("=")[0] for f in fields]
        defaults = {f.partition("=")[0]: ast.literal_eval(f.partition("=")[2]) for f in fields if "=" in f}
        if kind == "namedtuple":
            cls = namedtuple(name, names, defaults=tuple(defaults[n] for n in names if n in defaults), module=module)
        else:
            spec = []
            for n in names:
                if n not in defaults:
                    spec.append((n, object))
                elif isinstance(defaults[n], dict):
                    spec.append((n, object, field(default_factory=dict, repr=not n.startswith("_"))))
                else:
                    spec.append((n, object, field(default=defaults[n])))
            cls = make_dataclass(name, spec, frozen=True, eq=eq, module=module)
        out[name] = cls
    return out


def classes(table: dict, ns: dict) -> dict:
    """Execute the generated sources into ``ns`` (which holds Slot, plain_in, plain_out) and return {name: class}."""
    _core.load()
    ns["_construct"] = _core.construct
    ns["_core"] = _core
    out = {}
    for c in table["class"]:
        exec(compile(class_source(c), f"<decl:{c['name']}>", "exec"), ns)
        ns[c["name"]].__module__ = ns["__name__"]
        out[c["name"]] = ns[c["name"]]
    return out
