# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Batched fast Fourier transforms on aether arrays, host and GPU.

Complex data is a real ``float32`` / ``float64`` array whose LAST axis has extent 2 (real part, imaginary
part): there is no complex dtype. Convert without a copy with ``torch.view_as_complex`` / ``numpy``'s
``.view(complex64)`` on a host array. Every ``axis`` below counts the data axes only (the trailing 2 is not
one), so ``x`` of shape ``(batch, n, 2)`` is transformed along ``axis=-1`` or ``axis=1``.

Normalisation is numpy's ``norm``: ``"backward"`` (default: forward unscaled, inverse scaled by ``1/n``),
``"forward"`` or ``"ortho"``.

Host arrays use pocketfft (built in). CUDA arrays use cuFFT, which is an optional install, opened the first
time a plan is built: ``pip install 'aether-dsc[cufft12]'`` (CUDA 12) or ``'aether-dsc[cufft13]'`` (CUDA 13);
without it a device transform raises ``aether.FftUnavailable`` naming the extra.

Plans are built eagerly, outside a stream capture: ``plan(...)`` (and the first call of a transform on a new
shape) builds one, later calls reuse it. A plan's ``execute(input, output, stream=None)`` allocates nothing
and may run inside a capture, so build it first, capture after. ``stream=None`` runs on the input's
allocation stream."""

from . import _core
from ._core import Error

__all__ = ["fft", "ifft", "rfft", "irfft", "fftn", "ifftn", "rfftn", "irfftn", "plan"]

_C2C, _R2C, _C2R = 0, 1, 2
_NORMS = {"backward": 0, "forward": 1, "ortho": 2}
_cache = {}


def _norm(norm):
    if norm not in _NORMS:
        raise Error(f"norm= wants 'backward', 'forward' or 'ortho', not {norm!r}")
    return _NORMS[norm]


def plan(kind, dtype, shape, axes, forward=True, norm="backward", device="cpu", stream=None):
    """A transform plan. ``kind``: ``"c2c"``, ``"r2c"`` or ``"c2r"``. ``shape`` is the data shape (without
    the trailing 2): the complex shape for ``c2c``, the REAL side's shape for ``r2c`` / ``c2r`` (whose
    complex side has ``shape[axes[-1]] // 2 + 1`` along the last axis in ``axes``). ``forward`` applies to
    ``c2c`` only (``r2c`` is forward, ``c2r`` inverse). Plans are cached per (kind, dtype, shape, axes,
    direction, norm, device)."""
    from . import _device, _dtype
    kinds = {"c2c": _C2C, "r2c": _R2C, "c2r": _C2R}
    if kind not in kinds:
        raise Error(f"kind= wants 'c2c', 'r2c' or 'c2r', not {kind!r}")
    shape = tuple(int(s) for s in shape)
    axes = tuple(int(a) % len(shape) for a in axes) if shape else tuple(axes)
    dk, di = _device(device)
    if dk == 2:
        dk = 0
    name = _dtype(dtype).name
    key = (kind, name, shape, axes, bool(forward), norm, dk, di)
    p = _cache.get(key)
    if p is None:
        p = _core._fft_plan(kinds[kind], name, dk, di, list(shape), list(axes), bool(forward), _norm(norm), stream)
        _cache[key] = p
    return p


def _axes(axis, rank):
    if axis is None:
        axis = tuple(range(rank))
    elif isinstance(axis, int):
        axis = (axis,)
    out = []
    for a in axis:
        if not -rank <= a < rank:
            raise Error(f"axis {a} is out of range for {rank} data axes")
        out.append(a % rank)
    return tuple(out)


def _device_of(a):
    d = a.device
    return f"cuda:{d.index}" if d.kind == "cuda" else "cpu"


def _complex_shape(x):
    if x.ndim < 2 or x.shape[-1] != 2:
        raise Error("a complex array is a float array with a trailing axis of 2 (re, im); this one has shape "
                    f"{tuple(x.shape)}")
    return tuple(x.shape[:-1])


def _run(kind, x, out_shape, shape, axes, forward, norm, stream):
    from . import empty
    p = plan(kind, x.dtype, shape, axes, forward, norm, _device_of(x), stream)
    out = empty(out_shape, x.dtype, _device_of(x), stream if stream is not None else None)
    p.execute(x, out, stream)
    return out


def _c2c(x, axes, forward, norm, stream):
    shape = _complex_shape(x)
    return _run("c2c", x, shape + (2,), shape, _axes(axes, len(shape)), forward, norm, stream)


def fft(x, axis=-1, norm="backward", stream=None):
    """1-D transform of a complex array along ``axis``."""
    return _c2c(x, (axis,), True, norm, stream)


def ifft(x, axis=-1, norm="backward", stream=None):
    return _c2c(x, (axis,), False, norm, stream)


def fftn(x, axes=None, norm="backward", stream=None):
    """N-D transform over ``axes`` (default: every data axis)."""
    return _c2c(x, axes, True, norm, stream)


def ifftn(x, axes=None, norm="backward", stream=None):
    return _c2c(x, axes, False, norm, stream)


def rfftn(x, axes=None, norm="backward", stream=None):
    """N-D transform of REAL data: the output keeps ``n // 2 + 1`` points along the last of ``axes``."""
    shape = tuple(x.shape)
    ax = _axes(axes, len(shape))
    half = list(shape)
    half[ax[-1]] = shape[ax[-1]] // 2 + 1
    return _run("r2c", x, tuple(half) + (2,), shape, ax, True, norm, stream)


def rfft(x, axis=-1, norm="backward", stream=None):
    return rfftn(x, (axis,), norm, stream)


def irfftn(x, s=None, axes=None, norm="backward", stream=None):
    """Inverse of ``rfftn``. ``s`` is the REAL output's shape along ``axes``; the default recovers an even
    length on the last axis (``2 * (m - 1)``), as numpy does."""
    shape = _complex_shape(x)
    ax = _axes(axes, len(shape))
    out = list(shape)
    if s is not None:
        for a, n in zip(ax, s):
            out[a] = int(n)
    else:
        out[ax[-1]] = 2 * (shape[ax[-1]] - 1)
    if out[ax[-1]] // 2 + 1 != shape[ax[-1]]:
        raise Error(f"irfft: a real length of {out[ax[-1]]} has {out[ax[-1]] // 2 + 1} points, not {shape[ax[-1]]}")
    return _run("c2r", x, tuple(out), tuple(out), ax, False, norm, stream)


def irfft(x, n=None, axis=-1, norm="backward", stream=None):
    return irfftn(x, None if n is None else (n,), (axis,), norm, stream)
