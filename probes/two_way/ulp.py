"""The ulp bound of a value comparison between two builds of one kernel.

Two builds of the same kernel need not agree to the last bit: a compiler may fuse a multiply with
the addition that follows it (one rounding instead of two), and a math function may come from two
libraries. Each such site moves a result by at most one unit in the last place, so the bound is
their count, times the trip count of a loop that repeats them. It holds where no subtraction
cancels leading digits; a kernel that cancels needs a bound of its own.
"""
import re

import numpy as np

#: Operations every build rounds the same way, unless an addition is fused with a multiply.
_SAME_IN_EVERY_BUILD = {
    "add", "sub", "mul", "truediv", "div", "neg", "abs", "sqrt", "min", "max", "select",
    "lt", "le", "gt", "ge", "eq", "ne", "lnot", "land", "lor",
    # Element access moves values without computing on them.
    "component", "component_at", "set_component_at", "vec", "cast", "clamp_index",
}


def bound(ir_text: str, trips: int = 1) -> int:
    """The ulp bound of a kernel, from its IR text; `trips` is the trip count of its longest loop."""
    kinds = re.findall(r"^n\|Op\|.*\('kind', '([a-z0-9_]+)'\)", ir_text, flags=re.MULTILINE)
    fusable = sum(kind in ("add", "sub") for kind in kinds)
    library = sum(kind not in _SAME_IN_EVERY_BUILD for kind in kinds)
    return (fusable + library) * trips


def worst(got, want) -> float:
    """The largest difference between two arrays, in units in the last place of the wanted value."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    return float(np.max(np.abs(got - want) / np.spacing(np.abs(want))))


def verdict(name: str, got, want, limit: int) -> str:
    """One report line: the worst difference, the bound, and whether the difference is within it."""
    difference = worst(got, want)
    state = "within the bound" if difference <= limit else "OUTSIDE the bound"
    return f"{name}: worst difference {difference:.3g} ulp, bound {limit} ulp: {state}"
