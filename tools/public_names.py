#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Public-name coverage: every public name of the given modules is referenced by a test, or is on the allowlist.

    public_names.py --src DIR --tests DIR --allow FILE (--module a.b | --walk pkg)...

Public = the names in a module's ``__all__``, else its top-level functions, classes and assignments without a leading
underscore. Referenced = the bare name appears as a word in a ``.py`` file under ``--tests``. The allowlist holds one
``module.name  # reason`` per line; it may only shrink: a listed name that a test now references, or that is no longer
public, fails the check until its line is deleted. Static, no imports. Exit 0 clean, 1 on any finding.
"""
import argparse
import ast
import pathlib
import re
import sys


def public_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return sorted(ast.literal_eval(node.value))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return sorted(n for n in names if not n.startswith("_"))


def module_file(src, dotted):
    base = src.joinpath(*dotted.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def walk(src, package):
    root = src.joinpath(*package.split("."))
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(src).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        if not any(p.startswith("_") for p in parts):
            yield ".".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--tests", type=pathlib.Path, required=True)
    ap.add_argument("--allow", type=pathlib.Path, required=True)
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--walk", action="append", default=[])
    args = ap.parse_args()

    modules = list(args.module)
    for package in args.walk:
        modules += walk(args.src, package)
    public = {f"{m}.{n}" for m in modules for n in public_names(module_file(args.src, m))}
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(args.tests.rglob("*.py")))
    referenced = {q for q in public if re.search(rf"\b{re.escape(q.rsplit('.', 1)[1])}\b", corpus)}

    allowed = {}
    for line in args.allow.read_text(encoding="utf-8").splitlines():
        entry, _, reason = line.partition("#")
        if entry.strip():
            allowed[entry.strip()] = reason.strip()
    problems = [f"no test references {q}: add one, or list it in {args.allow} with a reason" for q in sorted(public - referenced - set(allowed))]
    problems += [f"{q} is referenced by a test now: delete its line from {args.allow}" for q in sorted(set(allowed) & referenced)]
    problems += [f"{q} is not a public name any more: delete its line from {args.allow}" for q in sorted(set(allowed) - public)]
    problems += [f"{q} in {args.allow} has no reason after '#'" for q in sorted(q for q, r in allowed.items() if not r)]
    print(f"public names: {len(public)}  referenced: {len(referenced)}  allowlisted (untested): {len(allowed)}")
    for p in problems:
        print("RED:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
