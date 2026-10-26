# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The CUDA runtime, imported on the first device call (never at ``import aether``)."""

import threading

_lock = threading.Lock()
_mods = None


class CudaUnavailable(RuntimeError):
    """A device array was asked for and the CUDA runtime is not there."""


class CudaError(RuntimeError):
    """A CUDA runtime call failed."""


def mods():
    """``(cuda.bindings.runtime, cuda.bindings.driver)``, or ``CudaUnavailable``."""
    global _mods
    if _mods is None:
        with _lock:
            if _mods is None:
                try:
                    from cuda.bindings import driver, runtime
                except ImportError as exc:
                    raise CudaUnavailable(
                        "device arrays need cuda-bindings: pip install 'aether-dsc[cuda12]' "
                        "(or [cuda13] for a CUDA 13 toolkit)") from exc
                _mods = (runtime, driver)
    return _mods


def rt():
    return mods()[0]


def ck(res):
    """Unpack a cuda-bindings result tuple, raising ``CudaError`` on a failure."""
    err, *rest = res
    if int(err) != 0:
        raise CudaError(f"CUDA error {int(err)}: {err}")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


def current_device():
    return int(ck(rt().cudaGetDevice()))


class on_device:
    """``with on_device(n):`` makes ``n`` the calling thread's device, then restores it."""

    def __init__(self, index):
        self.index = index

    def __enter__(self):
        self.prev = current_device()
        if self.prev != self.index:
            ck(rt().cudaSetDevice(self.index))
        return self

    def __exit__(self, *exc):
        if self.prev != self.index:
            ck(rt().cudaSetDevice(self.prev))
