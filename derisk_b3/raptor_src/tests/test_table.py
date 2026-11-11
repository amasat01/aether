"""Drift test of the ONE declaration manifest (raptor/ir/core.json): the generated Python classes, the specs compiled into the core and the
table text must agree. Plain asserts; run with run_test.sh (exit code = result). A planted extra field must FAIL the check."""
import inspect
from raptor import _core
from raptor.ir import _gen, nodes, types


def check_table(text: str) -> list:
    """Every disagreement between a table text, the specs compiled into the core and the classes generated at import."""
    import json
    bad, t = [], _gen.parse(text)
    bad += _core.check_manifest(text)
    if text != _core.manifest():
        bad.append('the manifest text differs from the one compiled into the core')
    rows = {r['class']: r for r in json.loads(text)['kinds'] if 'class' in r}
    rust = {n: (b, tuple(s), bind) for n, b, s, bind in _core.spec_dump()}
    by = {c["name"]: c for c in t["class"]}
    for c in t["class"]:
        L, n = c["lines"], c["name"]
        chain, k = [], c
        while k:
            chain.append(k); k = by.get(k["base"])
        slots = tuple(L.get("slots", ()))
        if n not in rust:
            bad.append(f"{n}: not in the compiled core"); continue
        rb, rs, rbind = rust[n]
        if (rb or None) != c["base"]: bad.append(f"{n}: base {c['base']} vs core {rb}")
        if rs != slots: bad.append(f"{n}: slots {slots} vs core {rs}")
        gen = getattr(nodes, n, None)
        if gen is None or tuple(gen.__dict__.get("__slots__", ())) != slots: bad.append(f"{n}: slots {slots} vs generated class")
        have = {v for k in chain for v in k["lines"].get("vals", ())} | ({"ttype", "operands"} if n == "Node" else set())
        for s in slots:
            if s not in have: bad.append(f"{n}: slot {s!r} is set by no `vals` line")
        known = {s for k in chain for s in k["lines"].get("slots", ())}
        for tok in list(L.get("dedup", ())) + list(L.get("state", ())):
            if tok.startswith(("names:", "skeys:")): f = tok.split(":")[1]
            elif tok.startswith("litif:"): f = tok.split(":")[2]
            elif tok.startswith("fieldif:"): f = tok.split(":")[1]
            elif tok == "body" or "=" in tok: continue
            else: f = tok.split(":")[1] if tok[:2] in ("f:", "r:") else tok.split(":")[1] if tok.startswith("fx:") else tok.split(":")[0]
            if tok.startswith(("+", "s:")) or tok in ("k", "none", "py", "one", "tup"): continue
            if f not in known and f not in ("literal",): bad.append(f"{n}: {tok!r} names no slot")
        if n in rows and 'ctor' in rows[n] and len(rows[n]['ctor']['out']) != len(L.get('vals', ())):
            bad.append(f"{n}: the ctor yields {len(rows[n]['ctor']['out'])} values for {len(L.get('vals', ()))} assigned slots")
        if "new" in L:
            if _core.ctor_index(n) is None: bad.append(f"{n}: the core compiled no constructor program")
            params = [p.strip().split("=")[0].split(":")[0].strip() for p in L["params"].split(",") if p.strip() not in ("", "*")]
            got = [p for p in inspect.signature(gen.__init__).parameters][1:]
            if params != got: bad.append(f"{n}: params {params} vs generated {got}")
    for n in rust:
        if n not in by: bad.append(f"{n}: in the core, not in the table")
    for name, fields, kind, _eq, home in t["record"]:
        cls = getattr(__import__("importlib").import_module("raptor.ir." + (home or "types")), name, None)
        have = list(cls._fields) if kind == "namedtuple" and cls is not None else list(getattr(cls, "__dataclass_fields__", ()))
        if cls is None or [f.split("=")[0] for f in fields] != have: bad.append(f"record {name}: differs from {cls}")
    return bad

real = _core.manifest()
assert real == open(__import__("raptor").__path__[0] + "/ir/core.json").read(), "the manifest file differs from the one compiled into the core (rebuild)"
assert check_table(real) == [], check_table(real)
import json
def planted(mutate, what):
    m = json.loads(real)
    mutate(m)
    t = json.dumps(m, indent=1, ensure_ascii=False) + "\n"
    assert t != real, what
    found = check_table(t)
    assert found, f"planted mismatch not caught: {what}"
    return len(found)
row = lambda m, c: next(r for r in m["kinds"] if r.get("class") == c)
def drop_row(m): m["kinds"].remove(row(m, "Finish"))
def rename_param(m): row(m, "Leaf")["ctor"]["steps"][1][1] = "knd not in LEAF_KINDS"
def bad_call(m): row(m, "Select")["ctor"]["steps"][0][1] = "refuse_factor"
def short_out(m): row(m, "Leaf")["ctor"]["out"].pop()
def extra_slot(m): row(m, "Leaf")["slots"].append("extra_field")
caught = {
 "table ahead: a slot on Leaf the core does not know": planted(extra_slot, "leaf slot"),
 "core ahead: a row dropped from the table": planted(drop_row, "finish row"),
 "a check names a parameter the row no longer has": planted(rename_param, "leaf check"),
 "a check calls an undeclared check": planted(bad_call, "select call"),
 "the ctor yields fewer values than slots assigned": planted(short_out, "leaf out"),
}
print("manifest drift test: real manifest clean (0 findings); planted mismatches caught (findings each):", caught)
