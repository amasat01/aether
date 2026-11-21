"""The kernels that take views, raised by the driver, against hawk. For each kernel: a hawk-authored
twin, the declarations of the raised kernel built from the plane report (no hand twin of the
declarations), the digest of the canonical form of both, and the built unit on random inputs
against the normal C++ arm (built on the library's own view headers) within the derived ulp bound."""
import pathlib
import sys

import numpy as np

import hawk
from hawk import Index, Mutable, Param, Scalar, Table, Vector, WideOut, _ir
from hawk.trace import decl
from hawk.trace.kernel import Kernel

import ulp

here = pathlib.Path(__file__).parent
out = here / "out" / "views_driver"


@hawk.kernel
def vscale(vec: Vector[3], gain: Param, scaled: Mutable[Vector[3]]):
    scaled[0] = vec[0] * gain
    scaled[1] = vec[1] * gain
    scaled[2] = vec[2] * gain


@hawk.kernel
def gather(row: Index, table: Table[Scalar], out: Mutable[Scalar]):
    out = table.at(row) + table.at(row + 1)


@hawk.kernel
def scatter(slot: Index, value: Scalar, out: WideOut[Scalar]):
    hawk.scatter(out)[slot] = 2.0 * value


@hawk.kernel
def tally(key: Index, value: Scalar, bins: Mutable["bin"]):
    hawk.scatter(bins, contract="plain")[key] += value


@hawk.kernel
def vgate(vec: Vector[3], gate: Scalar, out: Mutable[Vector[3]]):
    if gate > 0.5:
        out[1] = vec[1] * 2.0


@hawk.kernel
def vsum(vec: Vector[3], out: Mutable[Scalar]):
    total = 0.0
    for k in range(3):
        total = total + vec[k]
    out = total


TWINS = {"vscale": vscale, "gather": gather, "scatter": scatter, "tally": tally, "vgate": vgate, "vsum": vsum}


def declaration(role: str, name: str, dtype: str, shape: str):
    """The hawk declaration of one plane of a raised view kernel, from its report line."""
    payload = Scalar if shape == "-" else Vector[int(shape)]
    if role == "uniform":
        return Param
    if role in ("per_sample", "vec_in"):
        return decl.Plane("in", Index if dtype == "index" else payload)
    if role == "lookup":
        return Table[Scalar]
    if role == "mutable":
        return Mutable[payload]
    if role == "wide_out":
        return WideOut[Scalar]
    if role == "accum_out":
        return Mutable["bin"]
    raise SystemExit(f"{name}: no declaration for role {role}")


def raised(name: str) -> Kernel:
    """The kernel hawk makes of a raised IR document and its plane report."""
    roots = _ir.new_context().load((out / f"physics_{name}.ir").read_text())
    planes = {}
    for line in (out / f"physics_{name}.planes").read_text().splitlines():
        role, plane, dtype, *rest = line.split()
        planes[plane] = declaration(role, plane, dtype, rest[0] if rest else "-")
    return Kernel(f"{name}_raised", tuple(roots), planes)


def arrays() -> dict:
    """Every array the normal arm printed, by name, in memory order."""
    table = {}
    for line in (here / "out" / "views_values.txt").read_text().splitlines():
        name, count, *values = line.split()
        assert int(count) == len(values), name
        table[name] = np.array([float(v) for v in values])
    return table


def run(name: str, **given):
    """Run the raised kernel through hawk; the arrays it writes are passed in, so they hold the result."""
    return raised(name)(**given)


def values_checks() -> list[str]:
    """The built units on the normal arm's inputs, against the normal arm's outputs."""
    a = arrays()
    n = len(a["gate"])
    vec = np.ascontiguousarray(a["vec"].reshape(3, n))                      # component-major, as the C++ view lays it out
    row, slot, key = (np.ascontiguousarray(a[k].astype(np.int32)) for k in ("row", "slot", "key"))
    cases = []

    scaled = np.zeros((3, n))
    run("vscale", vec=vec, gain=float(a["gain"][0]), scaled=scaled)
    cases.append(("vscale", scaled.ravel(), a["scaled"]))

    cases.append(("gather", np.asarray(run("gather", row=row, table=a["table"])), a["gathered"]))

    cases.append(("scatter", np.asarray(run("scatter", slot=slot, value=a["value"], out=np.zeros(n))), a["scattered"]))

    bins = np.zeros(len(a["bins"]))
    run("tally", key=key, value=a["value"], bins=bins)
    cases.append(("tally", bins, a["bins"]))

    gated = np.zeros((3, n))
    run("vgate", vec=vec, gate=a["gate"], out=gated)
    cases.append(("vgate", gated.ravel(), a["gated"]))

    cases.append(("vsum", np.asarray(run("vsum", vec=vec)), a["sums"]))

    lines = []
    for name, got, want in cases:
        text = (out / f"physics_{name}.ir").read_text()
        limit = ulp.bound(text, trips=3 if "|Loop|" in text else 1)
        if name == "tally":
            limit = max(limit, n)           # a sum of up to n values in an order the unit may change
        lines.append(ulp.verdict(name, got, want, limit))
    return lines


if __name__ == "__main__":
    for name, twin in TWINS.items():
        same = _ir.digest(twin.walk) == _ir.digest(raised(name).walk)
        print(f"digest {name} equal {same}")
    print("\n".join(values_checks()))
