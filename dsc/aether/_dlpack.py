# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""DLPack capsules in pure Python (ctypes): export of an ``Array`` and import of a foreign one.

Both struct generations are written: the versioned one (``dltensor_versioned``) when the
consumer says ``max_version >= (1, 0)``, the legacy one otherwise. A producer stays alive in
a lock-guarded registry until the consumer calls the deleter (once); a capsule nobody ever
consumed releases it from the capsule's own destructor."""

import ctypes
import threading

kDLCPU, kDLCUDA, kDLCUDAHost = 1, 2, 3
_CODE_OF_KIND = {"i": 0, "u": 1, "f": 2, "b": 6}
_KIND_OF_CODE = {0: "i", 1: "u", 2: "f", 6: "b"}


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [("version", DLPackVersion), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER),
                ("flags", ctypes.c_uint64), ("dl_tensor", DLTensor)]


_api = ctypes.pythonapi
_new = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)(
    ("PyCapsule_New", _api))
_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(("PyCapsule_IsValid", _api))
_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(("PyCapsule_GetPointer", _api))
_rename = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(("PyCapsule_SetName", _api))
_get = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(("PyCapsule_GetPointer", _api))
_is = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object, ctypes.c_char_p)(("PyCapsule_IsValid", _api))

LEGACY, VERSIONED, USED = b"dltensor", b"dltensor_versioned", b"used_dltensor"
USED_VERSIONED = b"used_dltensor_versioned"

_lock = threading.Lock()
_live = {}               # address of the exported struct -> (struct, shape array, producer)


def _release(address):
    with _lock:
        _live.pop(address, None)


@DELETER
def _deleter(address):
    _release(address)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _capsule_destructor(capsule):
    # a capsule that was never consumed still carries its original name: release here
    for name in (LEGACY, VERSIONED):
        if _valid(capsule, name):
            _release(_pointer(capsule, name))
            return


def live_exports():
    """How many exported tensors have not had their deleter called yet."""
    with _lock:
        return len(_live)


def export(array, versioned):
    """A DLPack capsule over ``array`` (an ``aether.Array``)."""
    kind = array.dtype.kind
    if kind not in _CODE_OF_KIND:
        raise TypeError(f"dtype {array.dtype} has no DLPack code here")
    shape = (ctypes.c_int64 * max(array.ndim, 1))(*array.shape)
    strides = None
    if not array.is_contiguous():
        item = array.dtype.itemsize
        strides = (ctypes.c_int64 * array.ndim)(*[s // item for s in array.strides])
    tensor_type = DLManagedTensorVersioned if versioned else DLManagedTensor
    m = tensor_type()
    dl = m.dl_tensor
    dl.data = array.ptr
    dl.device = DLDevice(kDLCUDA, array.device.index)
    dl.ndim = array.ndim
    dl.dtype = DLDataType(_CODE_OF_KIND[kind], array.dtype.itemsize * 8, 1)
    dl.shape = shape
    dl.strides = strides
    dl.byte_offset = 0
    m.manager_ctx = None
    m.deleter = _deleter
    if versioned:
        m.version = DLPackVersion(1, 0)
        m.flags = 0
    address = ctypes.addressof(m)
    with _lock:
        _live[address] = (m, shape, strides, array)
    return _new(address, VERSIONED if versioned else LEGACY, ctypes.cast(_capsule_destructor, ctypes.c_void_p))


class Imported:
    """Holds a consumed foreign capsule; calls its deleter when the last view of it dies."""

    def __init__(self, address, deleter):
        self._address, self._deleter = address, deleter

    def release(self):
        address, self._address = self._address, 0
        if address:
            self._deleter(address)

    def __del__(self):
        try:
            self.release()
        except Exception:            # interpreter shutdown
            pass


def take(capsule):
    """Consume a capsule: returns ``(DLTensor, Imported)``; the capsule is renamed ``used_*``."""
    for name, used, tensor_type in ((VERSIONED, USED_VERSIONED, DLManagedTensorVersioned),
                                    (LEGACY, USED, DLManagedTensor)):
        if _is(capsule, name):
            address = _get(capsule, name)
            m = tensor_type.from_address(address)
            _rename(capsule, used)
            return m.dl_tensor, Imported(address, m.deleter)
    raise ValueError("not a DLPack capsule (or one already consumed)")


def dtype_of(dt):
    from numpy import dtype
    kind = _KIND_OF_CODE.get(dt.code)
    if kind is None or dt.lanes != 1:
        raise TypeError(f"DLPack dtype code {dt.code} bits {dt.bits} lanes {dt.lanes} is not supported "
                        "(bool, int, uint, float only)")
    return dtype("?") if kind == "b" else dtype(f"{kind}{dt.bits // 8}")
