"""The integer kernel raised from native C++, built and run by hawk, against the normal arm."""
import pathlib

import numpy as np

from hawk import _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

out = pathlib.Path(__file__).parent / "out"
roots = _ir.new_context().load((out / "cpp_bucket.ir").read_text())
planes = {"key": decl.Plane("in", decl.Int32), "y": decl.Mutable[decl.Int32]}
rows = np.loadtxt(out / "values_bucket.txt", dtype=np.int64)
got = Kernel("bucket_raised", tuple(roots), planes)(np.ascontiguousarray(np.tile(rows[:, 0], 300).astype(np.int32)))
got = got[0] if isinstance(got, (tuple, list)) else got
print("bucket: mismatches against the normal C++ arm:", int(np.count_nonzero(np.asarray(got) != np.tile(rows[:, 1], 300))), "of", got.size)
