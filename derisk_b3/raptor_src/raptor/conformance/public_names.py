# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Public-name coverage, the family's one check: every public name of the given modules is referenced by a test, or is
on the allowlist.

    python -m raptor.conformance.public_names --src DIR --tests DIR --allow FILE (--module a.b | --walk pkg)... [--base REF]

Public = the names in a module's ``__all__``, else its top-level functions, classes and assignments without a leading
underscore; for a package, also the submodules its ``__init__`` imports with ``from . import name`` (its doors).
Referenced = the bare name appears as a word in a ``.py`` file under ``--tests``. The allowlist holds one
``module.name  # reason`` per line; it may only shrink: a listed name that a test now references, or that is no longer
public, fails the check until its line is deleted, and with ``--base REF`` a line added since that git ref fails too.
Static, no imports. Exit 0 clean, 1 on any finding.
"""
import argparse
import ast
import pathlib
import re
import subprocess
import sys


def _public_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doors = set()
    if path.name == "__init__.py":
        doors = {a.name for node in tree.body if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
                 for a in node.names if not a.name.startswith("_")}
    literals = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign) for t in node.targets
                if isinstance(t, ast.Name)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return sorted({n for n in _strings(node.value, literals) if not n.startswith("_")} | doors)
    names = set(doors)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return sorted(n for n in names if not n.startswith("_"))


def _strings(node, literals):
    """The names a literal ``__all__`` lists; ``*NAME`` takes the items (a dict: the keys) of that module-level literal."""
    if isinstance(node, ast.Dict):
        return [ast.literal_eval(k) for k in node.keys]
    out = []
    for item in node.elts:
        if isinstance(item, ast.Starred) and isinstance(item.value, ast.Name) and item.value.id in literals:
            out += _strings(literals[item.value.id], literals)
        else:
            out.append(ast.literal_eval(item))
    return out


def _module_file(src, dotted):
    base = src.joinpath(*dotted.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def _walk(src, package):
    root = src.joinpath(*package.split("."))
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(src).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        if not any(p.startswith("_") for p in parts):
            yield ".".join(parts)


def _entries(text):
    """``{module.name: reason}`` of an allowlist's text (blank lines and whole-line comments skipped)."""
    out = {}
    for line in text.splitlines():
        entry, _, reason = line.partition("#")
        if entry.strip():
            out[entry.strip()] = reason.strip()
    return out


def _toplevel(path):
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.resolve().parent, capture_output=True, text=True,
                         check=True)
    return pathlib.Path(top.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--tests", type=pathlib.Path, required=True)
    ap.add_argument("--allow", type=pathlib.Path, required=True)
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--walk", action="append", default=[])
    ap.add_argument("--base", help="a git ref: an allowlist line added since it is a finding")
    args = ap.parse_args(argv)

    modules = list(args.module)
    for package in args.walk:
        modules += _walk(args.src, package)
    public = {f"{m}.{n}" for m in modules for n in _public_names(_module_file(args.src, m))}
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(args.tests.rglob("*.py")))
    referenced = {q for q in public if re.search(rf"\b{re.escape(q.rsplit('.', 1)[1])}\b", corpus)}

    allowed = _entries(args.allow.read_text(encoding="utf-8"))
    problems = [f"no test references {q}: add one, or list it in {args.allow} with a reason" for q in sorted(public - referenced - set(allowed))]
    problems += [f"{q} is referenced by a test now: delete its line from {args.allow}" for q in sorted(set(allowed) & referenced)]
    problems += [f"{q} is not a public name any more: delete its line from {args.allow}" for q in sorted(set(allowed) - public)]
    problems += [f"{q} in {args.allow} has no reason after '#'" for q in sorted(q for q, r in allowed.items() if not r)]
    if args.base:
        old = subprocess.run(["git", "show", f"{args.base}:{args.allow.resolve().relative_to(_toplevel(args.allow))}"],
                             cwd=args.allow.resolve().parent, capture_output=True, text=True)
        if old.returncode == 0:
            problems += [f"{q} was added to {args.allow} since {args.base}: the list may only shrink"
                         for q in sorted(set(allowed) - set(_entries(old.stdout)))]
    print(f"public names: {len(public)}  referenced: {len(referenced)}  allowlisted (untested): {len(allowed)}")
    for p in problems:
        print("RED:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
