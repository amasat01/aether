#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""docs/_tools/nb_verify.py -- `make nbverify`: prove every C++ snippet in
every tutorial/example notebook compiles and still prints what the page shows.

Two legs, one PASS/FAIL/SKIP line each:

 cpu   Execute the notebook (jupyter kernel, CUDA hidden so the CPU arm runs)
       in a scratch copy of the docs tree, and compare every code cell's
       fresh output with the committed one. Compiles use the shared harness
       (content/_shared/cpp_harness.py) with `-Wall -Wextra -Werror`, like
       aether's own build.
 cuda  Every snippet the notebook passes to `run_cpp` / `run_or_show` is
       compiled COMPILE-ONLY (`nvcc -c`, -Werror=all-warnings, no GPU
       needed) from the SAME source the harness would assemble. SKIP when
       nvcc is absent, unless --require-cuda (CI) turns that into FAIL.

Output normalisation (only what genuinely varies between runs/machines):
  * absolute temp/home paths           -> <PATH>
  * 0x... pointer addresses            -> <ADDR>
  * timings such as "12.3 ms", "4 us"  -> <TIME>
  * image payloads: only the presence of an image/png output is compared
    (pixels differ across matplotlib/font versions); text is compared exactly.

usage: nb_verify.py [--require-cuda] [--cpu-only|--cuda-only] [dir ...]
Writes a table to $GITHUB_STEP_SUMMARY when set. Exit 1 on any FAIL.
"""
import argparse
import contextlib
import io
import warnings
import difflib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

DOCS = pathlib.Path(__file__).resolve().parents[1]
REPO = DOCS.parent
# Lesson 02 deliberately builds named temporaries it never reads (to contrast
# with expression templates), so that one warning is switched off.
WARN_CPU = "-Wall -Wextra -Werror -Wno-unused-but-set-variable"
WARN_CUDA = ["-Werror=all-warnings"]

_NORMALISE = [
    (re.compile(r"(?:/tmp|/var/folders|/home|/local|/Users|/root)/\S+"), "<PATH>"),
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "<ADDR>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|us|µs|ns)\b"), "<TIME>"),
]


def norm(text: str) -> str:
    for pat, rep in _NORMALISE:
        text = pat.sub(rep, text)
    return text


def text_of(v) -> str:
    return "".join(v) if isinstance(v, list) else str(v)


def cell_signature(cell) -> str:
    """Comparable rendering of a code cell's outputs."""
    parts = []
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        if t == "stream":
            parts.append(f"[{o.get('name')}]\n" + norm(text_of(o.get("text", ""))))
        elif t in ("display_data", "execute_result"):
            data = o.get("data", {})
            if "text/plain" in data and "image/png" not in data:
                parts.append("[text/plain]\n" + norm(text_of(data["text/plain"])))
            if "image/png" in data:
                parts.append("[image/png present]")
        elif t == "error":
            parts.append(f"[error] {o.get('ename')}: {o.get('evalue')}")
    return "\n".join(parts)


def notebooks(dirs):
    out = []
    for d in dirs:
        d = pathlib.Path(d)
        out += [p for p in ([d] if d.is_file() else sorted(d.rglob("*.ipynb"))) if ".ipynb_checkpoints" not in p.parts]
    return out


# ---------------------------------------------------------------- CPU leg
def cpu_leg(nbs, scratch):
    import nbformat
    from nbclient import NotebookClient

    tree = scratch / "docs"
    shutil.copytree(DOCS / "content", tree / "content", ignore=shutil.ignore_patterns("_build", ".ipynb_checkpoints"))
    os.symlink(REPO / "aether", scratch / "aether")  # harness asserts aether/aether.h at the root
    results = {}
    os.environ["AETHER_NB_EXTRA_FLAGS"] = WARN_CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU arm everywhere, like the committed outputs
    for nb_path in nbs:
        rel = nb_path.resolve().relative_to(DOCS)
        work = tree / rel
        committed = json.loads(nb_path.read_text())
        try:
            nb = nbformat.read(str(work), as_version=4)
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(work.parent)}}).execute()
        except Exception as exc:  # CellExecutionError carries the compiler output
            msg = str(exc).strip().splitlines()
            keep = [l for l in msg if "error" in l.lower() or "Error" in l] or msg[-3:]
            results[nb_path] = ("FAIL", "execution failed: " + " | ".join(l.strip() for l in keep[:4]))
            continue
        fresh = [c for c in nb.cells if c.cell_type == "code"]
        old = [c for c in committed["cells"] if c["cell_type"] == "code"]
        diffs = []
        if len(fresh) != len(old):
            diffs.append(f"cell count {len(old)} committed vs {len(fresh)} fresh")
        for i, (f, o) in enumerate(zip(fresh, old)):
            a, b = cell_signature(o), cell_signature(f)
            if a != b:
                d = difflib.unified_diff(a.splitlines(), b.splitlines(), "committed", "fresh", lineterm="", n=0)
                diffs.append(f"cell {i} output differs:\n" + "\n".join(d))
        results[nb_path] = ("FAIL", "\n".join(diffs)) if diffs else ("PASS", f"{len(fresh)} code cells match")
    return results


# ---------------------------------------------------------------- CUDA leg
def cuda_leg(nbs, require):
    results = {}  # (nb_path, label) -> (status, detail)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        st = "FAIL" if require else "SKIP"
        for nb_path in nbs:
            results[(nb_path, "cuda")] = (st, "nvcc not found")
        return results

    shared = DOCS / "content" / "_shared"
    sys.path.insert(0, str(shared))
    import cpp_harness as H
    import matplotlib

    matplotlib.use("Agg")
    for nb_path in nbs:
        nb = json.loads(nb_path.read_text())
        count = [0]
        nbname = nb_path.name

        def compile_only(main_body, top="", includes=()):
            count[0] += 1
            label = f"cuda {count[0]}"
            src = H._assemble(main_body, top, includes)
            with tempfile.TemporaryDirectory() as tmp:
                f = pathlib.Path(tmp) / "snippet.cu"
                f.write_text(src)
                # -arch=sm_70: compile-only needs a fixed arch, not "native" (no GPU).
                cmd = [nvcc, f"-std={H._GPU_STD}", f"-I{H.ROOT}", "-DAETHER_HAS_CUDA=1", "-arch=sm_70", "-O3",
                       *WARN_CUDA, "-c", str(f), "-o", str(pathlib.Path(tmp) / "snippet.o")]
                p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                results[(nb_path, label)] = ("PASS", "compiles")
            else:
                errs = [l for l in p.stderr.splitlines() if "error" in l.lower()] or p.stderr.splitlines()[-3:]
                results[(nb_path, label)] = ("FAIL", " | ".join(l.strip() for l in errs[:4]))

        def fake_run_cpp(main_body, *, top="", includes=(), device=None):
            compile_only(main_body, top, includes)
            return ""

        def fake_run_or_show(main_body, *, top=""):
            compile_only(main_body, top)
            return ""

        ns = {"DEVICE": "gpu", "run_cpp": fake_run_cpp, "run_or_show": fake_run_or_show,
              "split_data": H.split_data, "plot_style": H.plot_style}
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = text_of(cell["source"])
            if src.lstrip().startswith("%run"):
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(src, f"{nbname}", "exec"), ns)
            except Exception:
                pass  # plotting/post-processing of the empty stub output; only the compiles matter
        if count[0] == 0:
            results[(nb_path, "cuda")] = ("PASS", "no C++ snippets")
    return results


# ---------------------------------------------------------------- main
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["content/tutorials", "content/examples"])
    ap.add_argument("--require-cuda", action="store_true", help="a missing nvcc is a FAIL, not a SKIP")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cpu-only", action="store_true")
    g.add_argument("--cuda-only", action="store_true")
    a = ap.parse_args(argv)
    nbs = notebooks(a.dirs)
    if not nbs:
        print("nbverify: FAIL (no notebooks found)")
        return 1

    rows = []  # (notebook, leg, status, detail)
    with tempfile.TemporaryDirectory(prefix="nbverify_") as tmp:
        if not a.cuda_only:
            for nb, (st, d) in cpu_leg(nbs, pathlib.Path(tmp)).items():
                rows.append((str(nb), "cpu", st, d))
    if not a.cpu_only:
        for (nb, label), (st, d) in cuda_leg(nbs, a.require_cuda).items():
            rows.append((str(nb), label, st, d))

    for nb, leg, st, d in rows:
        first = d.splitlines()[0] if d else ""
        print(f"{st:4}  {nb}  [{leg}]  {first}")
        for extra in d.splitlines()[1:]:
            print(f"        {extra}")
    fails = [r for r in rows if r[2] == "FAIL"]
    skips = [r for r in rows if r[2] == "SKIP"]
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("### Notebook verification\n\n| Notebook | Leg | Result | Detail |\n|---|---|---|---|\n")
            for nb, leg, st, d in rows:
                fh.write(f"| {nb} | {leg} | {st} | {d.splitlines()[0].replace('|', '/') if d else ''} |\n")
    print(f"nbverify: {'FAIL' if fails else 'OK'} ({len(rows)} checks, {len(fails)} failed, {len(skips)} skipped)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
