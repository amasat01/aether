"""All eight probe kernels through hawk with NO hawk twin: the plane declarations are built from
what the raise stub reports of the C++ signature (role, name, scalar type)."""
import pathlib

import numpy as np

from hawk import _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

out = pathlib.Path(__file__).parent / "out"
SCALAR = {"f64": decl.Scalar, "i32": decl.Int32}


def declaration(role: str, dtype: str):
    """The hawk declaration of one parameter of a raised kernel."""
    if role == "per_sample":
        return decl.Plane("in", SCALAR[dtype])
    if role == "uniform":
        return decl.Param
    return decl.Mutable[SCALAR[dtype]]


def raised(name: str) -> Kernel:
    """The kernel hawk makes of a raised IR document and its signature report."""
    roots = _ir.new_context().load((out / f"cpp_{name}.ir").read_text())
    planes = {}
    for line in (out / f"cpp_{name}.planes").read_text().splitlines():
        role, plane, dtype = line.split()
        planes[plane] = declaration(role, dtype)
    return Kernel(f"{name}_raised", tuple(roots), planes)


first = np.loadtxt(out / "values.txt")       # v, drag, clip, newton
second = np.loadtxt(out / "values2.txt")     # v, band, sides, horner, counted x, counted iters, whileRoot
positive = second[second[:, 0] > 0]
cases = [("drag", first, (0.47,), {"out": 1}), ("clip", first, (1.5,), {"y": 2}), ("newton", first, (), {"x": 3}),
         ("band", second, (0.5,), {"y": 1}), ("sides", positive, (), {"y": 2}), ("horner", second, (), {"y": 3}),
         ("counted", positive, (), {"x": 4, "iters": 5}), ("whileRoot", positive, (), {"x": 6})]
for name, rows, params, columns in cases:
    kernel = raised(name)
    v = np.ascontiguousarray(np.tile(rows[:, 0], 1000))
    got = kernel(v, *params)
    got = got if isinstance(got, (tuple, list)) else (got,)
    got = dict(zip(columns, got)) if not isinstance(got[0], dict) else got[0]
    worst = max(float(np.max(np.abs(np.asarray(got[plane], dtype=float) - np.tile(rows[:, column], 1000))))
                for plane, column in columns.items())
    print(f"{name}: built from its signature report, run on {v.size} samples; worst absolute difference to the normal C++ arm {worst:.3g}")
