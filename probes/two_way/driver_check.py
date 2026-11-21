"""The kernels the driver raised from the example project, built and run by hawk, against the normal arm."""
import pathlib

import numpy as np

from hawk import _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

import ulp

here = pathlib.Path(__file__).parent / "out" / "driver"


def raised(name):
    roots = _ir.new_context().load((here / f"physics_{name}.ir").read_text())
    planes = {}
    for line in (here / f"physics_{name}.planes").read_text().splitlines():
        role, plane, dtype = line.split()
        scalar = decl.Int32 if dtype == "i32" else decl.Scalar
        planes[plane] = {"per_sample": decl.Plane("in", scalar), "uniform": decl.Param}.get(role, decl.Mutable[scalar])
    return Kernel(f"{name}_driver", tuple(roots), planes)


rows = np.loadtxt(here / "values.txt")
gears = np.ascontiguousarray(np.tile(rows[:, 4].astype(np.int32), 500))
speeds = np.ascontiguousarray(np.tile(rows[:, 0], 500))
cases = (("drag", 1, (rows[:, 0], 0.47)), ("relax", 2, (np.abs(rows[:, 0]),)), ("band", 3, (rows[:, 0], 0.5)),
         ("gear", 5, (gears, speeds)))
for name, column, args in cases:
    first = args[0] if name == "gear" else np.ascontiguousarray(np.tile(args[0], 500))
    got = raised(name)(first, *args[1:])
    got = got[0] if isinstance(got, (tuple, list)) else got
    text = (here / f"physics_{name}.ir").read_text()
    limit = ulp.bound(text, trips=8 if "|Loop|" in text else 1)   # every loop of the example runs at most 8 times
    print(ulp.verdict(name, got, np.tile(rows[:, column], 500), limit))
