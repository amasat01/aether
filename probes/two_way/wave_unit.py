"""The math kernel as a BUILT unit (the library's own math) against the normal C++ arm."""
import pathlib

import numpy as np

from hawk import _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

import ulp

out = pathlib.Path(__file__).parent / "out"
roots = _ir.new_context().load((out / "cpp_wave.ir").read_text())
planes = {"v": decl.Plane("in", decl.Scalar), "k": decl.Param, "y": decl.Mutable[decl.Scalar]}
rows = np.loadtxt(out / "values3.txt")
got = Kernel("wave_unit", tuple(roots), planes)(np.ascontiguousarray(np.tile(rows[:, 0], 100)), 0.8)
got = np.asarray(got[0] if isinstance(got, (tuple, list)) else got)
want = np.tile(rows[:, 1], 100)
print(ulp.verdict("wave as a built unit", got, want, ulp.bound((out / "cpp_wave.ir").read_text())))
