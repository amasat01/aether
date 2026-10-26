# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""aether: CPU and GPU arrays that every other array library can read without a copy.

One ``Array`` type for every place memory can live: ``device="cpu"``, ``"pinned"`` (page-locked host),
``"cuda"`` or ``"cuda:N"``. Host and pinned arrays speak the buffer protocol, so ``numpy.asarray`` and
``memoryview`` see them without a copy; every array speaks DLPack, and device arrays also
``__cuda_array_interface__``. No maths lives here. Nothing here imports numpy.

``import aether`` needs no CUDA. The CUDA runtime comes from a plugin shipped in the wheel
(``libaether_cuda13.so`` then ``libaether_cuda12.so``, the first the driver accepts;
``AETHER_CUDA_MAJOR=12|13`` forces one) and loads on the first device call; without one a device call
raises ``CudaUnavailable``.

Python ``aether.Array`` mirrors the C++ ``Chunk`` + ``RuntimeView`` (one buffer on one device), not the
C++ ``aether::Array`` (a host-mirrored structure of arrays)."""

try:
    from . import _core
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("aether: _core is not built for this interpreter (no wheel for this platform or "
                      "Python, and no source build); aether_dsc, the sealed payload, still works") from exc

from ._core import Array, CudaUnavailable, Error, cuda_plugin, cuda_runtime_version
from ._dtype import Device, DType, parse as _dtype

_core._register(Device, DType)

__all__ = ["Array", "Device", "DType", "Error", "CudaUnavailable", "asarray", "copy", "cuda_plugin",
           "cuda_runtime_version", "empty", "empty_like", "from_dlpack", "full", "pinned", "zeros", "zeros_like"]

_KIND = {"cpu": 0, "cuda": 1, "pinned": 2}


def _device(device):
    """``(kind, index)`` of ``"cpu"``, ``"pinned"``, ``"cuda"`` (the current device), ``"cuda:N"`` or a Device."""
    if isinstance(device, Device):
        return _KIND[device.kind], device.index
    if device in ("cpu", "pinned"):
        return _KIND[device], 0
    if device == "cuda":
        return 1, _core._current_device()
    if isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit():
        return 1, int(device[5:])
    raise Error(f"device= wants 'cpu', 'pinned', 'cuda' or 'cuda:N', not {device!r}")


def _shape(shape):
    return [int(shape)] if isinstance(shape, int) else [int(s) for s in shape]


def _make(shape, dtype, device, stream, zero):
    kind, index = _device(device)
    return _core._new(kind, index, _shape(shape), _dtype(dtype).name, stream, zero)


def empty(shape, dtype="float64", device="cpu", stream=None):
    """An uninitialised array."""
    return _make(shape, dtype, device, stream, False)


def zeros(shape, dtype="float64", device="cpu", stream=None):
    return _make(shape, dtype, device, stream, True)


def full(shape, value, dtype="float64", device="cpu", stream=None):
    return _make(shape, dtype, device, stream, False).fill(value)


def _like(x, dtype, device):
    if device is None:
        device = x.device if isinstance(x, Array) else "cpu"
    return x.shape, (x.dtype if dtype is None else dtype), device


def empty_like(x, dtype=None, device=None, stream=None):
    shape, dt, dev = _like(x, dtype, device)
    return empty(shape, dt, dev, stream)


def zeros_like(x, dtype=None, device=None, stream=None):
    shape, dt, dev = _like(x, dtype, device)
    return zeros(shape, dt, dev, stream)


def pinned(shape, dtype="float64"):
    """Page-locked host memory, uninitialised: DMA copies to and from a device are direct."""
    return empty(shape, dtype, "pinned")


def from_dlpack(x, stream=None):
    """View a foreign array without copying; the producer's deleter runs when the view dies."""
    return _core._from_dlpack(x, stream)


def _view(x, stream):
    """``x`` as an Array without copying, or None when it is not something aether can view."""
    if isinstance(x, Array):
        return x
    if hasattr(x, "__cuda_array_interface__"):               # no stream ordering asked: legal inside a capture
        return _core._from_cai(x)
    if hasattr(x, "__dlpack_device__"):
        return from_dlpack(x, stream)
    try:
        memoryview(x)
    except TypeError:
        return None
    return _core._from_buffer(x)


def asarray(x, dtype=None, device=None, stream=None):
    """``x`` on ``device``: zero-copy when it is already there, a copy otherwise (``device=None`` keeps
    it where it is). Takes an Array, anything with a buffer, DLPack or ``__cuda_array_interface__``,
    or a (nested) list of numbers."""
    src = _view(x, stream)
    if src is None:
        src = _from_list(x, dtype)
    if dtype is not None and src.dtype != _dtype(dtype):
        raise Error(f"asarray: the data is {src.dtype}; aether does no casting (asked for {_dtype(dtype)})")
    if device is None:
        return src
    kind, index = _device(device)
    here = src.device
    if (here.kind, here.index) == (("cpu", "cuda", "pinned")[kind], index if kind == 1 else 0):
        return src
    out = _make(src.shape, src.dtype, device, stream, False)
    return copy(src, into=out, stream=stream)


def _from_list(x, dtype):
    flat, shape = [], []

    def walk(v, depth):
        if isinstance(v, (list, tuple)):
            if len(shape) <= depth:
                shape.append(len(v))
            for i in v:
                walk(i, depth + 1)
        else:
            flat.append(v)
    walk(x, 0)
    if dtype is None:
        dtype = "float64" if any(isinstance(v, float) for v in flat) else "bool" if flat and all(isinstance(v, bool) for v in flat) else "int64"
    out = empty(shape, dtype, "cpu")
    for i, v in enumerate(flat):
        out._set_flat(i, v)
    return out


def copy(src, into=None, stream=None):
    """Copy ``src`` into ``into`` (a new array on ``src``'s device when omitted). The direction is inferred.
    With ``stream=None`` it is synchronous; with a stream it is queued there (a memcpy node in a capture)."""
    s = _view(src, stream) if not isinstance(src, Array) else src
    if s is None:
        raise Error("copy: src is not an array aether can view")
    if into is None:
        into = empty_like(s, stream=stream)
    d = _view(into, None)
    _core._copy(s, d, stream)
    return into
