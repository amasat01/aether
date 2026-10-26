# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Array``: one typed buffer on one CUDA device, and the verbs that make and move it."""

import atexit
import ctypes
import math
import threading
import weakref

import numpy as np

from . import _dlpack, _stream
from ._rt import ck, current_device, on_device, rt


class Device:
    """``Device(kind, index)``: ``kind`` is "cpu" or "cuda" (the DLPack device)."""
    __slots__ = ("kind", "index")

    def __init__(self, kind, index=0):
        self.kind, self.index = kind, int(index)

    def __eq__(self, other):
        return isinstance(other, Device) and (self.kind, self.index) == (other.kind, other.index)

    def __hash__(self):
        return hash((self.kind, self.index))

    def __repr__(self):
        return f"Device({self.kind!r}, {self.index})"


def _parse_device(device):
    """``"cpu"`` -> None; ``"cuda"`` / ``"cuda:N"`` -> the index (current device for bare "cuda")."""
    if isinstance(device, Device):
        device = "cpu" if device.kind == "cpu" else f"cuda:{device.index}"
    if device == "cpu":
        return None
    if device == "cuda":
        return current_device()
    if isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit():
        return int(device[5:])
    raise ValueError(f"device= wants 'cpu', 'cuda' or 'cuda:N', not {device!r}")


def _dtype(dtype):
    try:
        dt = np.dtype(dtype)
    except TypeError:
        raise TypeError(f"dtype {dtype!r} is not supported (bool, int, uint, float only)") from None
    if dt.kind not in "biuf" or dt.name == "bfloat16":
        raise TypeError(f"dtype {dt} is not supported (bool, int, uint, float only; no complex, no bfloat16)")
    return dt


def _shape(shape):
    return (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)


# ---------------------------------------------------------------- owned device memory

_registry_lock = threading.Lock()
_outstanding = weakref.WeakSet()


class _relaxed:
    """Relaxed capture mode for the calling thread only: allocation, initialisation and frees are not
    capturable work, and under another stream's global-mode capture they would invalidate it."""

    def __enter__(self):
        r = rt()
        self.prev = ck(r.cudaThreadExchangeStreamCaptureMode(r.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed))
        return self

    def __exit__(self, *exc):
        rt().cudaThreadExchangeStreamCaptureMode(self.prev)


class _Block:
    """One allocation and what freeing it needs. The free is queued on the allocation's own stream, so
    it runs after the work already queued there; while a capture blocks that stream it is deferred and
    done by the next free or allocation outside the capture. ``atexit`` frees what is left."""
    __slots__ = ("ptr", "host", "device", "stream", "handle", "__weakref__")

    def __init__(self, ptr, host, device, stream, handle):
        self.ptr, self.host, self.device, self.stream, self.handle = ptr, host, device, stream, handle
        with _registry_lock:
            _outstanding.add(self)

    def release(self, drain=False):
        with _registry_lock:
            ptr, self.ptr = self.ptr, 0
        if not ptr:
            return
        if drain:                                    # interpreter exit: nothing is capturing, wait for it all
            _free(ptr, self.host, self.device, self.handle, sync=True)
            return
        _flush_pending()
        entry = (ptr, self.host, self.device, self.handle)
        if _blocked(self.handle):
            with _registry_lock:
                _pending.append(entry)
        else:
            _free(*entry)

    def __del__(self):
        try:
            self.release()
        except Exception:            # interpreter shutdown: the atexit drain has run
            pass


_pending = []            # frees that arrived while a capture was in progress: (ptr, host, device, handle)


def _blocked(handle):
    """True when a free on ``handle`` now would join or break a capture: the stream is capturing, or
    (the legacy stream under a blocking capture) the runtime refuses to say. Asking is harmless."""
    r = rt()
    with _relaxed():
        res = r.cudaStreamIsCapturing(handle)
    return int(res[0]) != 0 or res[1] != r.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone


def _free(ptr, host, device, handle, sync=False):
    """Free in stream order after everything already queued on ``handle`` (never sooner), without
    touching a capture on another stream: the calling thread runs in relaxed capture mode."""
    r = rt()
    with _relaxed():
        if host:
            ck(r.cudaFreeHost(ptr))
        else:
            with on_device(device):
                ck(r.cudaFree(ptr) if sync else r.cudaFreeAsync(ptr, handle))


def _flush_pending():
    """Free what was deferred, for each entry once its stream is no longer blocked by a capture."""
    if not _pending:
        return
    with _registry_lock:
        todo, _pending[:] = list(_pending), []
    keep = []
    for entry in todo:
        (keep.append(entry) if _blocked(entry[3]) else _free(*entry))
    if keep:
        with _registry_lock:
            _pending.extend(keep)


@atexit.register
def _drain():
    with _registry_lock:
        todo, _pending[:] = list(_pending), []
        blocks = list(_outstanding)
    for entry in todo:
        try:
            _free(*entry, sync=True)
        except Exception:
            pass
    for block in blocks:
        try:
            block.release(drain=True)
        except Exception:
            pass


def _alloc(nbytes, index, stream):
    _flush_pending()
    h = _stream.handle(stream)
    if nbytes == 0:
        return _Block(0, False, index, stream, h)
    with on_device(index):
        ptr = int(ck(rt().cudaMallocAsync(nbytes, h)))
    return _Block(ptr, False, index, stream, h)


class Array:
    """A typed buffer on one CUDA device. C-contiguous when made here; an imported one may be
    strided (then it can be read and passed on, not filled or copied into)."""

    def __init__(self, ptr, shape, dtype, device, strides=None, stream=None, owner=None):
        self._ptr, self._shape, self._dtype = int(ptr), tuple(shape), np.dtype(dtype)
        self._device, self._owner = device, owner       # owner: a _Block or dlpack.Imported
        self._stream = stream
        item = self._dtype.itemsize
        if strides is None:
            strides, acc = [], item
            for extent in reversed(self._shape):
                strides.append(acc)
                acc *= max(extent, 1)
            strides = tuple(reversed(strides))
        self._strides = tuple(int(s) for s in strides)

    # -- the plain attributes
    shape = property(lambda self: self._shape)
    dtype = property(lambda self: self._dtype)
    strides = property(lambda self: self._strides)
    ndim = property(lambda self: len(self._shape))
    ptr = property(lambda self: self._ptr)
    device = property(lambda self: self._device)
    size = property(lambda self: math.prod(self._shape))
    nbytes = property(lambda self: self.size * self._dtype.itemsize)

    def is_contiguous(self):
        acc = self._dtype.itemsize
        for extent, stride in zip(reversed(self._shape), reversed(self._strides)):
            if extent != 1 and stride != acc:
                return False
            acc *= extent
        return True

    def _need_contiguous(self, what):
        if not self.is_contiguous():
            raise ValueError(f"{what} needs a contiguous array; this one is strided")

    def _handle(self, stream):
        return _stream.handle(self._stream if stream is None else stream)

    def __repr__(self):
        return f"aether.Array(shape={self._shape}, dtype={self._dtype}, device='cuda:{self._device.index}')"

    # -- filling (memset/memcpy nodes: legal inside a stream capture)
    def zero(self, stream=None):
        """Zero the array on ``stream`` (default: the stream it was made on)."""
        self._need_contiguous("zero")
        if self.nbytes:
            with on_device(self._device.index):
                ck(rt().cudaMemsetAsync(self._ptr, 0, self.nbytes, self._handle(stream)))
        return self

    def fill(self, value, stream=None):
        """Set every element to ``value`` on ``stream``."""
        self._need_contiguous("fill")
        raw = np.array(value, dtype=self._dtype).tobytes()
        n, item = self.size, self._dtype.itemsize
        if not n:
            return self
        h = self._handle(stream)
        r = rt()
        with on_device(self._device.index):
            if raw.count(raw[0:1]) == item:                       # one repeated byte: a plain memset
                ck(r.cudaMemsetAsync(self._ptr, raw[0], self.nbytes, h))
            elif item in (2, 4) or (item == 8 and raw[:4] == raw[4:]):
                width = 2 if item == 2 else 4
                pattern = int.from_bytes(raw[:width], "little")
                driver = _driver()
                fn = driver.cuMemsetD16Async if width == 2 else driver.cuMemsetD32Async
                count = n * (item // width)
                ck(fn(driver.CUdeviceptr(self._ptr), pattern, count, driver.CUstream(h)))
            else:
                self._fill_halves(raw, h)
        return self

    def _fill_halves(self, raw, h):
        # an 8-byte value with two different halves: two strided 32-bit memsets, one per half
        driver = _driver()
        lo, hi = (int.from_bytes(raw[i:i + 4], "little") for i in (0, 4))
        for offset, word in ((0, lo), (4, hi)):
            ck(driver.cuMemsetD2D32Async(driver.CUdeviceptr(self._ptr + offset), 8, word, 1, self.size,
                                         driver.CUstream(h)))

    def synchronize(self):
        """Block until the work queued on the array's stream is done."""
        with on_device(self._device.index):
            ck(rt().cudaStreamSynchronize(self._handle(None)))

    # -- interop
    @property
    def __cuda_array_interface__(self):
        h = self._handle(None)
        return {"shape": self._shape, "typestr": self._dtype.str, "data": (self._ptr, False), "version": 3,
                "strides": None if self.is_contiguous() else self._strides,
                "stream": h if h else None}

    def __dlpack_device__(self):
        return (_dlpack.kDLCUDA, self._device.index)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("aether.Array does not copy on export (copy=True refused)")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError(f"cannot export on {tuple(dl_device)}: the array is on {self.__dlpack_device__()}")
        if stream is not None and stream != -1:
            if stream == 0:
                raise ValueError("__dlpack__(stream=0) is not a valid CUDA stream code")
            self._order_before(int(stream))
        versioned = max_version is not None and tuple(max_version)[0] >= 1
        return _dlpack.export(self, versioned)

    def _order_before(self, consumer):
        """Make ``consumer`` wait for the work queued on the array's stream; the host never blocks."""
        r = rt()
        producer = self._handle(None)
        if producer == consumer or (producer == 0 and consumer == 1):
            return
        with on_device(self._device.index):
            ev = ck(r.cudaEventCreateWithFlags(r.cudaEventDisableTiming))
            try:
                ck(r.cudaEventRecord(ev, producer))
                ck(r.cudaStreamWaitEvent(consumer, ev, 0))
            finally:
                ck(r.cudaEventDestroy(ev))


def _driver():
    from ._rt import mods
    return mods()[1]


# ---------------------------------------------------------------- construction

def _new_device(shape, dtype, index, stream):
    shape, dt = _shape(shape), _dtype(dtype)
    block = _alloc(math.prod(shape) * dt.itemsize, index, stream)
    return Array(block.ptr, shape, dt, Device("cuda", index), stream=stream, owner=block)


def empty(shape, dtype=np.float64, device="cuda", stream=None):
    """An uninitialised array. ``device="cpu"`` gives a numpy array; "cuda"/"cuda:N" an ``Array``."""
    index = _parse_device(device)
    if index is None:
        return np.empty(_shape(shape), _dtype(dtype))
    with _relaxed():
        return _new_device(shape, dtype, index, stream)


def zeros(shape, dtype=np.float64, device="cuda", stream=None):
    index = _parse_device(device)
    if index is None:
        return np.zeros(_shape(shape), _dtype(dtype))
    with _relaxed():
        return _new_device(shape, dtype, index, stream).zero()


def full(shape, value, dtype=np.float64, device="cuda", stream=None):
    index = _parse_device(device)
    if index is None:
        return np.full(_shape(shape), value, _dtype(dtype))
    with _relaxed():
        return _new_device(shape, dtype, index, stream).fill(value)


def _device_name(x):
    if isinstance(x, Array):
        return f"cuda:{x.device.index}"
    if hasattr(x, "__dlpack_device__") and x.__dlpack_device__()[0] == _dlpack.kDLCUDA:
        return f"cuda:{x.__dlpack_device__()[1]}"
    return "cpu"


def empty_like(x, dtype=None, device=None, stream=None):
    return empty(x.shape, x.dtype if dtype is None else dtype, _device_name(x) if device is None else device, stream)


def zeros_like(x, dtype=None, device=None, stream=None):
    return zeros(x.shape, x.dtype if dtype is None else dtype, _device_name(x) if device is None else device, stream)


def pinned(shape, dtype=np.float64):
    """A numpy array over page-locked host memory, uninitialised; freed with the array."""
    shape, dt = _shape(shape), _dtype(dtype)
    nbytes = math.prod(shape) * dt.itemsize
    if nbytes == 0:
        return np.empty(shape, dt)
    r = rt()
    ptr = int(ck(r.cudaHostAlloc(nbytes, r.cudaHostAllocDefault)))
    block = (ctypes.c_char * nbytes).from_address(ptr)
    block._owner = _Block(ptr, True, 0, None, 0)
    return np.frombuffer(block, dtype=dt).reshape(shape)


# ---------------------------------------------------------------- import

def from_dlpack(x, stream=None):
    """View a foreign array without copying. CUDA producers give an ``Array``; host producers
    give a numpy array. The producer's deleter runs when the view dies."""
    kind, index = x.__dlpack_device__()
    if kind == _dlpack.kDLCPU:
        return np.from_dlpack(x)
    if kind != _dlpack.kDLCUDA:
        raise ValueError(f"from_dlpack: DLPack device type {kind} is not supported (cpu and cuda only)")
    ask = {}
    if stream is not None:
        h = _stream.handle(stream)
        ask["stream"] = h if h else 1                       # the DLPack code for the legacy stream
    try:
        capsule = x.__dlpack__(max_version=(1, 0), **ask)
    except TypeError:
        capsule = x.__dlpack__(**ask)
    t, held = _dlpack.take(capsule)
    try:
        dt = _dlpack.dtype_of(t.dtype)
        shape = tuple(t.shape[i] for i in range(t.ndim))
        strides = (None if not t.strides else tuple(t.strides[i] * dt.itemsize for i in range(t.ndim)))
        return Array(t.data + t.byte_offset, shape, dt, Device("cuda", index), strides=strides, owner=held)
    except BaseException:
        held.release()
        raise


def _from_cai(x):
    cai = x.__cuda_array_interface__
    ptr = cai["data"][0]
    dt = _dtype(np.dtype(cai["typestr"]))
    index = ck(rt().cudaPointerGetAttributes(ptr)).device if ptr else current_device()
    return Array(ptr, tuple(cai["shape"]), dt, Device("cuda", int(index)), strides=cai.get("strides"), owner=x)


def _host_ptr(a):
    return a.ctypes.data


def asarray(x, dtype=None, device=None, stream=None):
    """``x`` on ``device``: zero-copy when it is already resident there, a copy otherwise.
    ``device=None`` keeps it where it is (host data stays numpy)."""
    want = None if device is None else ("cpu" if _parse_device(device) is None else f"cuda:{_parse_device(device)}")
    if dtype is not None:
        dtype = _dtype(dtype)
    if isinstance(x, Array):
        src = x
    elif hasattr(x, "__cuda_array_interface__"):          # no stream ordering asked: legal inside a capture
        src = _from_cai(x)
    elif hasattr(x, "__dlpack_device__") and x.__dlpack_device__()[0] == _dlpack.kDLCUDA:
        src = from_dlpack(x, stream)
    else:
        host = np.asarray(x, dtype)
        if dtype is not None and host.dtype != dtype:
            host = host.astype(dtype)
        _dtype(host.dtype)
        if want in (None, "cpu"):
            return host
        out = empty(host.shape, host.dtype, want, stream)
        copy(np.ascontiguousarray(host), into=out, stream=stream)
        return out
    if dtype is not None and src.dtype != dtype:
        raise TypeError(f"asarray: the array is {src.dtype}; aether does no casting (asked for {dtype})")
    if want is None or want == f"cuda:{src.device.index}":
        return src
    out = empty(src.shape, src.dtype, want, stream)
    copy(src, into=out, stream=stream)
    return out


def _span(x):
    """``(pointer, nbytes, shape, dtype)`` of a host or device array."""
    if isinstance(x, np.ndarray):
        if not x.flags.c_contiguous:
            raise ValueError("copy needs C-contiguous arrays")
        return _host_ptr(x), x.nbytes, x.shape, x.dtype
    a = asarray(x)
    if isinstance(a, np.ndarray):
        return _span(a)
    a._need_contiguous("copy")
    return a.ptr, a.nbytes, a.shape, a.dtype


def copy(src, into=None, stream=None):
    """Copy ``src`` into ``into`` (a new array on ``src``'s device when omitted). The direction is
    inferred from the two pointers. With ``stream=None`` it is synchronous; with a stream it is
    queued there (a memcpy node when that stream is capturing)."""
    if into is None:
        into = empty_like(src)
    sp, sn, sshape, sdt = _span(src)
    dp, dn, dshape, ddt = _span(into)
    if (sshape, sdt) != (dshape, ddt):
        raise ValueError(f"copy: {sshape}/{sdt} into {dshape}/{ddt}")
    if sn:
        r = rt()
        h = _stream.handle(stream)
        ck(r.cudaMemcpyAsync(dp, sp, sn, r.cudaMemcpyKind.cudaMemcpyDefault, h))
        if stream is None:
            ck(r.cudaStreamSynchronize(0))
    return into
