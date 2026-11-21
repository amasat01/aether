"""The kernels the driver raised from the example project, built and run by hawk, against the normal arm."""
import pathlib

import numpy as np

from hawk import _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

here = pathlib.Path(__file__).parent / "out" / "driver"


def raised(name):
    roots = _ir.new_context().load((here / f"physics_{name}.ir").read_text())
    planes = {}
    for line in (here / f"physics_{name}.planes").read_text().splitlines():
        role, plane, _ = line.split()
        planes[plane] = {"per_sample": decl.Plane("in", decl.Scalar), "uniform": decl.Param}.get(role, decl.Mutable[decl.Scalar])
    return Kernel(f"{name}_driver", tuple(roots), planes)


rows = np.loadtxt(here / "values.txt")
for name, column, args in (("drag", 1, (rows[:, 0], 0.47)), ("relax", 2, (np.abs(rows[:, 0]),)), ("band", 3, (rows[:, 0], 0.5))):
    got = raised(name)(np.ascontiguousarray(np.tile(args[0], 500)), *args[1:])
    got = got[0] if isinstance(got, (tuple, list)) else got
    print(name, "worst absolute difference", float(np.max(np.abs(np.asarray(got) - np.tile(rows[:, column], 500)))))
