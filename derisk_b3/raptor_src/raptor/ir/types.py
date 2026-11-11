# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dependency-free type tier.

Small, frozen dataclasses describing HAWK's tensor-typed vocabulary —
the records a policy core (a downstream routing/dispatch layer) needs
to describe its own kernel shapes without paying for HAWK's
kernel-authoring machinery. This module is importable at near-zero
cost: importing ``hawk.types`` pulls in no other ``hawk`` module, so a
downstream consumer avoids paying for
``hawk.ir``/``hawk.emit``/``hawk.compile``.

Nothing here performs validation, tracing or codegen — types only.
``TensorType``/``Wire`` are the static slice of the vocabulary read off
a traced/emitted IR; ``Slot`` is one entry of ``Walk.leaves``.
:class:`LaneMeta` lives here rather than in the emitter so a policy core
composing lanes pays the cheap import, never the codegen one.
"""

from __future__ import annotations


#: IR value dtypes. The compiled scalar mode (float64/float32/banded) is
#: a separate axis carried by the artifact, never by a node. ``i32``/``i64``
#: are INDEX integers (addresses, counts; stored 64-bit as ``Int``);
#: ``int32``/``uint32`` are the 32-bit integer PLANES (:data:`INT_PLANE_DTYPES`).
DTYPES = ("f64", "f32", "i32", "i64", "bool", "int32", "uint32")

#: The 32-bit integer plane dtypes: stored at their own width, wrapping
#: arithmetic, never promoted to or from a float without an explicit cast,
#: and never differentiated.
INT_PLANE_DTYPES = ("int32", "uint32")

#: The roles whose plane is addressed at the current sample index by
#: construction: every leaf role here is read ``plane[i]``, so these are
#: the only roles whose plane's trailing extent is n_samples.
PER_SAMPLE_ROLES = ("per_sample", "vec_in", "mat_in", "terminated", "mutable", "out")

from .. import _core
from . import _gen

# TensorType, Wire, Slot, LaneMeta: frozen dataclasses generated from the one declaration table.
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__))

_INDEX = TensorType((), "i32")


def _call(self, *args: object, **kwargs: object) -> object:
    """Never called: a declaration is only annotated. Defined so that ``typing.Annotated``
    accepts it as its first argument before Python 3.11, which requires a callable."""
    raise TypeError(f"{self!r} is a type declaration: annotate with it, do not call it")


def _getitem(self, width: object) -> TensorType:
    """``Index[int64]``: the 64-bit index plane (``Index`` alone is the 32-bit one).
    ``int32`` spells the default; any other width is refused by name."""
    if self != _INDEX:
        raise TypeError(f"{self!r} is not subscriptable; only Index takes a width, Index[int64]")
    name = getattr(width, "name", None) or getattr(width, "__name__", None) or str(width)
    if name == "int64":
        return TensorType((), "i64")
    if name == "int32":
        return self
    raise TypeError(f"Index[...] takes int32 (the default) or int64, got {width!r}")


TensorType.__call__ = _call
TensorType.__getitem__ = _getitem
