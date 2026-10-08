# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Shared C++ harness for aether's tutorial notebooks. Every notebook's
# first cell (hidden) is just:
#
#     %run ../_shared/setup.py
#
# which pulls in `DEVICE`, `run_cpp()`, `split_data()` and `plot_style()`.
# Device detection, the g++-vs-nvcc compile-and-run mechanics, the
# `#include`/`main()`/`return 0;` boilerplate, the CSV-row split that
# feeds a plot, and the shared chart style all live here, one copy, so
# no lesson cell has to carry them.
#
# This is aether's OWN copy: the family's other sites keep an independent
# helper with the same shape for their own door; no cross-repo doc
# dependency is introduced here.
import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap

# docs/content/_shared/cpp_harness.py -> repo root is three levels up.
ROOT = pathlib.Path(__file__).resolve().parents[3]
assert (ROOT / "aether" / "aether.h").exists(), f"not the aether repo root: {ROOT}"


def gpu_visible() -> bool:
    """nvcc on PATH *and* nvidia-smi sees a device. aether's C++ door has no
    cupy to ask, so this is the same "a GPU is visible" check the family's
    Python notebooks make, translated to the nvcc toolchain. An explicit
    empty CUDA_VISIBLE_DEVICES always means "no", exactly like cupy."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    if shutil.which("nvcc") is None:
        return False
    try:
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return probe.returncode == 0 and bool(probe.stdout.strip())
    except Exception:
        return False


#: Computed once, at import time, so a notebook never branches on device
#: itself -- it just imports the answer, and `run_cpp` defaults to it, so
#: the GPU arm actually runs whenever one is visible.
DEVICE = "gpu" if gpu_visible() else "cpu"

# nvcc 12.6 (this family's pinned CUDA toolchain) tops out at -std=c++20;
# it does not accept -std=c++23 at all ("fatal: Value 'c++23' is not
# defined for option 'std'"). The host (g++) arm uses c++23 -- aether's
# own standard -- since nothing stops it there. "The SAME source" claim
# on a GPU-switch page is about the aether CODE, not the compiler flag:
# no lesson body in this family uses a C++21-23-only feature, so one
# source compiles clean under both.
_CPU_STD = "c++23"
_GPU_STD = "c++20"


def _assemble(main_body: str, top: str, includes: tuple) -> str:
    extra_includes = "\n".join(f"#include {inc}" for inc in includes)
    return (
        "#include <aether/aether.h>\n#include <cstdio>\n"
        f"{extra_includes}\n\n"
        f"{textwrap.dedent(top).strip()}\n\n"
        "int main() {\n"
        f"{textwrap.dedent(main_body).strip()}\n"
        "return 0;\n}\n"
    )


def run_cpp(main_body: str, *, top: str = "", includes: tuple = (), device: str | None = None) -> str:
    """Compile and run a tiny aether program, returning stdout.

    `main_body` becomes the inside of `int main() { ...; return 0; }` --
    write ordinary statements, not a whole program. `top`, if given, is
    inserted before `main()`, for free functions, macros or `using`
    declarations `main()` needs. `#include <aether/aether.h>` and
    `<cstdio>` are always added; `includes` names more, e.g.
    `("<cmath>",)`.

    device=None (the default) runs on `DEVICE` -- the SAME call this
    notebook already decided "gpu" or "cpu" for, so a lesson cell never
    has to branch on it itself. Pass "cpu"/"gpu" explicitly only to force
    one side regardless of what is visible (used by the few "going
    deeper" cells that show the other side on purpose).

    Always compiled at this repo's normal optimization level (`-O3`) --
    performance in this family is always measured at `-O3`, and a claim
    this family makes (e.g. "zero temporaries") is never shown by
    turning optimization down until a gap appears; see
    `02_expressions_without_temporaries.ipynb` for how that claim is
    demonstrated instead (a compile-time type check, not a timing).
    """
    if device is None:
        device = DEVICE
    source = _assemble(main_body, top, includes)
    with tempfile.TemporaryDirectory() as tmp:
        if device == "gpu":
            src = pathlib.Path(tmp) / "example.cu"
            compiler = "nvcc"
            flags = [f"-std={_GPU_STD}", f"-I{ROOT}", "-DAETHER_HAS_CUDA=1", "-arch=native", "-O3"]
        else:
            src = pathlib.Path(tmp) / "example.cpp"
            compiler = "g++"
            flags = [f"-std={_CPU_STD}", f"-I{ROOT}", "-DAETHER_CPP_MODE=1", "-fopenmp", "-O3"]
        src.write_text(source)
        binary = pathlib.Path(tmp) / "example"
        subprocess.run([compiler, *flags, str(src), "-o", str(binary)], check=True)
        proc = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        return proc.stdout


def run_or_show(main_body: str, *, top: str = "") -> str:
    """For the one or two cells per site that are genuinely GPU-only (no
    host build can run them at all): run on the GPU if `DEVICE == "gpu"`;
    otherwise return a one-line placeholder -- the lesson already showed
    the kernel source in its own cell, so this does not re-echo it --
    without the calling cell itself branching on the device."""
    if DEVICE == "gpu":
        return run_cpp(main_body, top=top, device="gpu")
    return "(shown, not run: no GPU visible)"


def split_data(output: str):
    """Split `run_cpp`'s stdout into (text the lesson prints, data rows
    for a hidden plot). A line starting "D," feeds the plot and is
    dropped from the printed text; its comma-separated fields (the "D,"
    prefix removed) become one row, still plain strings -- a plotting
    cell converts what it needs with `float()`."""
    text_lines, rows = [], []
    for line in output.splitlines():
        if line.startswith("D,"):
            rows.append(line[2:].split(","))
        else:
            text_lines.append(line)
    return "\n".join(text_lines), rows


def plot_style() -> None:
    """Apply this site's shared, light+dark-legible matplotlib style.
    Call once per notebook, right before the first figure -- the one line
    every plotting cell needs; the rcParams dict itself never has to
    appear in a lesson cell again."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "none", "savefig.facecolor": "none", "axes.facecolor": "none",
        "text.color": "#898781", "axes.labelcolor": "#898781", "axes.edgecolor": "#898781",
        "xtick.color": "#898781", "ytick.color": "#898781", "grid.color": "#c3c2b7",
    })
