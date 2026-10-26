# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Dtypes by name, and ``Device``: plain Python, no numpy."""

_NAMES = {"float64": (2, 64), "float32": (2, 32), "float16": (2, 16), "bfloat16": (4, 16),
          "int64": (0, 64), "int32": (0, 32), "int16": (0, 16), "int8": (0, 8),
          "uint64": (1, 64), "uint32": (1, 32), "uint16": (1, 16), "uint8": (1, 8), "bool": (6, 8)}
_PY = {float: "float64", int: "int64", bool: "bool"}


class DType:
    """A dtype: ``.name .itemsize .code .bits``. Equal to its name string."""
    __slots__ = ("name", "code", "bits")

    def __init__(self, name):
        self.name = name
        self.code, self.bits = _NAMES[name]

    itemsize = property(lambda self: self.bits // 8)

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        try:
            return self.name == parse(other).name
        except (TypeError, ValueError):
            return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"aether.DType({self.name!r})"


def parse(dtype):
    """A name string, a Python ``float``/``int``/``bool``, or any object whose ``str()`` or
    ``__name__`` is a dtype name (a numpy dtype or scalar type passes without importing numpy)."""
    from ._core import Error
    if isinstance(dtype, DType):
        return dtype
    candidates = []
    if isinstance(dtype, str):
        candidates.append(dtype)
    elif dtype in _PY:
        candidates.append(_PY[dtype])
    else:
        candidates += [getattr(dtype, "__name__", None), str(dtype)]
    for c in candidates:
        if c == "bool_":
            c = "bool"
        if c in _NAMES:
            return DType(c)
    names = " ".join(_NAMES)
    raise Error(f"dtype {dtype!r} is not supported ({names})")


class Device:
    """``Device(kind, index)``: kind is "cpu", "cuda" or "pinned"."""
    __slots__ = ("kind", "index")

    def __init__(self, kind, index=0):
        self.kind, self.index = kind, int(index)

    def __eq__(self, other):
        return isinstance(other, Device) and (self.kind, self.index) == (other.kind, other.index)

    def __hash__(self):
        return hash((self.kind, self.index))

    def __repr__(self):
        return f"Device({self.kind!r}, {self.index})"

    def __str__(self):
        return self.kind if self.kind != "cuda" else f"cuda:{self.index}"
