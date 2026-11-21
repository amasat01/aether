"""The kernel on aether's real math headers: its raised IR, evaluated, against the normal arm."""
import pathlib
import sys

import numpy as np

import hawk
from hawk import _ir

sys.path.insert(0, str(pathlib.Path(hawk.__file__).parent.parent / "tests"))
from _eval import evaluate  # noqa: E402

out = pathlib.Path(__file__).parent / "out"
roots = _ir.new_context().load((out / "cpp_wave.ir").read_text())
rows = np.loadtxt(out / "values3.txt")
worst = max(abs(float(np.asarray(evaluate(roots, {"v": v, "k": 0.8, "terminated": False})["y"]).reshape(-1)[0]) - want)
            / np.spacing(abs(want)) for v, want in rows)
print(f"wave (sinh, atan2, hypot, fmod, cbrt through aether's own headers): worst difference {worst:.3g} ulp over {len(rows)} inputs")
