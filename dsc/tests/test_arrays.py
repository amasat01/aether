# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""aether arrays: the acceptance rows. GPU rows skip when no CUDA plugin loads or no device is present;
numpy, cupy and torch rows skip when that library is absent (aether itself needs none of them)."""

import ctypes
import gc
import importlib.util
import pathlib
import struct
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import aether
from aether import _core

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _gpu():
    try:
        aether.cuda_plugin()
        return True
    except aether.CudaUnavailable:
        return False


gpu = pytest.mark.skipif(not _gpu(), reason="needs a CUDA plugin that loads and a device")
needs_cupy = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="needs cupy (the subprocess rows import it)")


def _run(code, **env):
    import os
    full = dict(os.environ, **env)
    return subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=180, env=full)


# ---- row 1: no numpy/cupy/cuda-bindings, no device
def test_import_without_numpy_or_a_device():
    out = _run("""
        import sys
        for m in ('numpy', 'cupy', 'cuda', 'cuda.bindings', 'torch'):
            sys.modules[m] = None
        import aether
        a = aether.zeros(4)
        assert aether.asarray([1.0, 2.0]).shape == (2,)
        try:
            aether.empty(8, device='cuda')
        except aether.CudaUnavailable:
            print('ok')
    """, CUDA_VISIBLE_DEVICES="")
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_no_numpy_import_in_the_package_sources():
    for path in list((ROOT / "aether").glob("*.py")) + list((ROOT / "src").glob("*")):
        text = path.read_text()
        assert "import numpy" not in text and "from numpy" not in text, path


def test_dtype_spellings_and_refusals():
    assert aether.zeros(2, "float32").dtype == "float32"
    assert aether.zeros(2, float).dtype == "float64" and aether.zeros(2, int).dtype.name == "int64"
    assert aether.zeros(2, bool).dtype == "bool"
    for bad in ("float128", "int4", "complex64", "complex128", "str", object):
        with pytest.raises(aether.Error):
            aether.empty(4, bad)


_NEW_DTYPES = [
    ("int8", -5, b"\xfb"), ("int16", -300, (-300).to_bytes(2, "little", signed=True)),
    ("uint16", 60000, (60000).to_bytes(2, "little")), ("float16", 1.5, struct.pack("<e", 1.5)),
    ("bfloat16", 1.0, b"\x80\x3f"),
]


def _bytes_of(a):
    host = a if a.device.kind != "cuda" else aether.copy(a, into=aether.empty(a.shape, a.dtype))
    return ctypes.string_at(host.ptr, host.nbytes)


@pytest.mark.parametrize("where", ["cpu", pytest.param("cuda", marks=gpu), pytest.param("pinned", marks=gpu)])
@pytest.mark.parametrize("name,value,raw", _NEW_DTYPES, ids=[d[0] for d in _NEW_DTYPES])
def test_added_dtypes_allocate_fill_and_roundtrip_through_dlpack(name, value, raw, where):
    a = aether.full((3, 2), value, name, where)
    assert a.dtype == name and a.nbytes == 6 * len(raw) and a.dtype.itemsize == len(raw)
    assert _bytes_of(a) == raw * 6
    assert _bytes_of(aether.zeros(4, name, where)) == bytes(4 * len(raw))
    b = aether.from_dlpack(a)                   # versioned capsule, and the legacy one below
    assert (b.ptr, b.shape, b.dtype) == (a.ptr, a.shape, a.dtype) and _bytes_of(b) == raw * 6
    legacy = aether.from_dlpack(type("L", (), {"__dlpack_device__": a.__dlpack_device__,
                                               "__dlpack__": lambda self, **kw: a.__dlpack__(**{k: v for k, v in kw.items() if k != "max_version"})})())
    assert legacy.ptr == a.ptr and legacy.dtype == name


def test_added_dtype_buffer_protocol():
    half = memoryview(aether.full(3, 1.5, "float16"))
    assert (half.format, half.itemsize, half.nbytes) == ("e", 2, 6)
    if sys.version_info >= (3, 12):             # memoryview reads format "e" from 3.12 on
        assert half.tolist() == [1.5] * 3
    assert memoryview(aether.full(3, -4, "int8")).tolist() == [-4] * 3
    with pytest.raises(BufferError):
        memoryview(aether.zeros(2, "bfloat16"))


def test_host_surface_and_values():
    a = aether.zeros((3, 5), "float32")
    assert (a.shape, a.strides, a.nbytes, a.ndim, a.is_contiguous) == ((3, 5), (20, 4), 60, 2, True)
    assert a.device == aether.Device("cpu", 0) and a.dtype.itemsize == 4
    for dt, val in (("float64", 0.1), ("float32", 2.5), ("int64", -7), ("int32", 7), ("uint64", 9),
                    ("uint32", 8), ("uint8", 200), ("bool", True)):
        f = aether.full(33, val, dt)
        assert all(x == (float(val) if dt.startswith("float") else val) or abs(x - val) < 1e-6
                   for x in memoryview(f).tolist()), dt
    assert aether.empty(0).ptr == 0
    assert aether.asarray([[1, 2], [3, 4]]).dtype == "int64"


def test_copy_and_asarray_on_the_host():
    src = aether.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dst = aether.copy(src)
    assert memoryview(dst).tolist() == memoryview(src).tolist() and dst.ptr != src.ptr
    assert aether.asarray(src) is src
    with pytest.raises(aether.Error):
        aether.copy(src, into=aether.zeros(5))


# ---- row 3: zero-copy views over cpu AND pinned arrays, writes visible both ways
@pytest.mark.parametrize("where", ["cpu", pytest.param("pinned", marks=gpu)])
def test_numpy_torch_memoryview_zero_copy(where):
    a = aether.zeros((3, 7), "float64", where)
    m = memoryview(a)
    m[1, 2] = 4.0
    assert memoryview(a)[1, 2] == 4.0
    np = pytest.importorskip("numpy")
    v = np.asarray(a)
    assert v.ctypes.data == a.ptr and v[1, 2] == 4.0
    v[0, 0] = 9.0
    assert memoryview(a)[0, 0] == 9.0
    if where == "cpu":
        assert np.from_dlpack(a).ctypes.data == a.ptr
        torch = pytest.importorskip("torch")
        t = torch.from_dlpack(a)
        assert t.data_ptr() == a.ptr and float(t[0, 0]) == 9.0
        back = aether.asarray(np.arange(6.0).reshape(2, 3))
        assert back.shape == (2, 3) and memoryview(back).tolist()[1] == [3.0, 4.0, 5.0]


def test_device_memory_has_no_buffer():
    if not _gpu():
        pytest.skip("needs a device")
    with pytest.raises(BufferError):
        memoryview(aether.zeros(4, device="cuda"))


# ---- GPU rows
@gpu
def test_gpu_surface_copy_and_fill():
    d = aether.zeros((3, 5), "float32", "cuda")
    assert d.device == aether.Device("cuda", 0) and d.nbytes == 60 and d.strides == (20, 4)
    for dt, val in (("float64", 0.1), ("float32", 2.5), ("int64", -3), ("int32", 7), ("uint64", 5),
                    ("uint32", 6), ("uint8", 200), ("bool", True)):
        f = aether.full(33, val, dt, "cuda")
        host = aether.asarray(f, device="cpu")
        assert all(abs(x - val) < 1e-6 for x in memoryview(host).tolist()), (dt, val)
    h = aether.pinned((3, 5), "float32")
    aether.copy(aether.full((3, 5), 2.5, "float32", "cuda"), into=h)
    assert set(memoryview(h).tolist()[0]) == {2.5}
    up = aether.asarray(aether.asarray([[1.0, 2.0], [3.0, 4.0]]), device="cuda")
    assert aether.copy(up, into=aether.empty((2, 2), device="cpu")).shape == (2, 2)
    assert aether.asarray(up) is up and aether.zeros_like(up).device == up.device
    assert aether.cuda_plugin().startswith("cuda") and aether.cuda_runtime_version() > 12000
    with pytest.raises(aether.Error):
        aether.copy(up, into=aether.zeros(5))


# ---- row 4: cupy / torch round trips, deleters exactly once, stream codes
@gpu
def test_cupy_roundtrip_ptr_equal_and_kernel_write():
    cp = pytest.importorskip("cupy")
    c = cp.arange(24, dtype=cp.float64).reshape(4, 6)
    a = aether.from_dlpack(c)
    assert a.ptr == c.data.ptr and a.shape == (4, 6)
    c2, c3 = cp.from_dlpack(a), cp.asarray(a)
    assert c2.data.ptr == a.ptr == c3.data.ptr
    c2 *= 2.0
    assert float(c[3, 5]) == 46.0
    del a, c2, c3
    gc.collect()


@gpu
def test_deleter_once_both_generations():
    before = _core._exports_live()
    a = aether.zeros(16, "float32", "cuda")
    cap_v, cap_l = a.__dlpack__(max_version=(1, 0)), a.__dlpack__()
    assert _core._exports_live() == before + 2
    del cap_v, cap_l                                   # never consumed: the capsule destructor releases
    gc.collect()
    assert _core._exports_live() == before
    cp = pytest.importorskip("cupy")
    view = cp.from_dlpack(a)                           # consumed: the consumer's deleter releases
    assert _core._exports_live() == before + 1
    del view
    gc.collect()
    assert _core._exports_live() == before


@gpu
def test_torch_cuda_roundtrip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA here")
    a = aether.full(8, 3.0, "float32", "cuda")
    a.synchronize()
    t = torch.from_dlpack(a)
    assert t.data_ptr() == a.ptr and float(t.sum()) == 24.0
    b = aether.from_dlpack(torch.ones(4, device="cuda"))
    assert b.shape == (4,)


@gpu
def test_import_deleter_fires_once():
    cp = pytest.importorskip("cupy")
    c = cp.zeros(8)
    a = aether.from_dlpack(c)
    del a
    gc.collect()
    assert float(c.sum()) == 0.0                       # the producer's memory is intact and still ours


@gpu
def test_dlpack_stream_codes_and_event_ordering():
    cp = pytest.importorskip("cupy")
    a = aether.zeros(8, "float64", "cuda")
    with pytest.raises(ValueError):
        a.__dlpack__(stream=0)
    for code in (None, -1, 1):
        a.__dlpack__(stream=code)
    own = cp.cuda.Stream(non_blocking=True)
    spin = cp.RawKernel(r"""extern "C" __global__ void spin(double* p, long long c) {
        long long t = clock64(); while (clock64() - t < c) {} p[0] = 1.0; }""", "spin")
    prod = cp.cuda.Stream(non_blocking=True)
    b = aether.zeros(8, "float64", "cuda", stream=prod)
    with prod:
        spin((1,), (1,), (cp.asarray(b), cp.int64(500_000_000)))      # about half a second ahead
    t = time.perf_counter()
    b.__dlpack__(stream=int(own.ptr))
    assert time.perf_counter() - t < 0.2, "the host waited for the producer's kernel"
    prod.synchronize()
    own.synchronize()


# ---- row 5: capture-legal zero/fill, two replays; a GC free inside a capture
@gpu
def test_zero_fill_capture_two_replays():
    cp = pytest.importorskip("cupy")
    s = cp.cuda.Stream(non_blocking=True)
    d = aether.empty(4096, "float64", "cuda", stream=s)
    w = aether.empty(4096, "float64", "cuda", stream=s)
    host = aether.pinned(4096, "float64")
    cd, cw = cp.asarray(d), cp.asarray(w)
    with s:
        s.begin_capture()
        d.zero(s)
        w.fill(0.1, s)                                  # per-byte strided memsets
        aether.copy(d, into=host, stream=s)
        g = s.end_capture()
    for _ in range(2):
        cd[...] = 7.0
        cp.cuda.runtime.deviceSynchronize()
        g.launch(s)
        s.synchronize()
        assert not any(memoryview(host).tolist())
        assert float(cw.min()) == 0.1 == float(cw.max())


@gpu
@needs_cupy
def test_staged_upload_then_dropping_its_stream_does_not_break_the_next_copy():
    """A pageable host to device copy on a user stream leaves the staging buffer busy; the next staged copy must
    not wait on that stream, which the caller has since destroyed (it used to segfault)."""
    out = _run("""
        import aether, cupy as cp
        s = cp.cuda.Stream(non_blocking=True)
        d = aether.asarray([1.0, 2.0, 3.0, 4.0], device="cuda", stream=s)
        s.synchronize()
        del d, s
        a = aether.zeros(8, "float64", "cuda")
        h = aether.empty(8, "float64", "cpu")
        aether.copy(a, into=h)
        print("ok")
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stdout, out.stderr[-400:])


@gpu
@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("mode", ["legacy", "stream"])
@needs_cupy
def test_gc_free_inside_global_capture(mode, blocking):
    out = _run(f"""
        import gc, aether, cupy as cp
        own = cp.cuda.Stream(non_blocking=True)
        a = aether.zeros(1024, "float64", "cuda", stream={'own' if mode == 'stream' else 'None'})
        cyc = [a]; cyc.append(cyc)
        del a, cyc
        cp.cuda.runtime.deviceSynchronize()
        gc.disable()
        cap = cp.cuda.Stream(non_blocking={not blocking})
        x = cp.zeros(8)
        with cap:
            cap.begin_capture(mode=cp.cuda.runtime.streamCaptureModeGlobal)
            x += 1
            gc.collect()
            x += 1
            g = cap.end_capture()
            g.launch(); cap.synchronize()
        assert float(x.sum()) == 16.0
        aether.empty(1, device='cuda')          # the next allocation outside the capture drains a deferred free
        print('ok')
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stdout, out.stderr[-400:])


@gpu
@pytest.mark.parametrize("mode", ["legacy", "own_stream"])
def test_early_drop_after_launch_is_stream_ordered(mode):
    cp = pytest.importorskip("cupy")
    spin = cp.RawKernel(r"""extern "C" __global__ void spin(double* p, long long c) {
        long long t = clock64(); while (clock64() - t < c) {} p[0] = 7.0; }""", "spin")
    own = cp.cuda.Stream(non_blocking=True)
    stream = own if mode == "own_stream" else cp.cuda.Stream.null
    a = aether.zeros(8, "float64", "cuda", stream=own if mode == "own_stream" else None)
    keep = cp.asarray(a)
    with stream:
        spin((1,), (1,), (keep, cp.int64(1_000_000_000)))
    del keep
    t = time.perf_counter()
    del a
    gc.collect()
    waited = time.perf_counter() - t
    stream.synchronize()
    assert waited < 0.5, f"the free blocked the host for {waited:.2f}s: it was not queued on the stream"


# ---- row 6: lifetimes at exit
@gpu
@needs_cupy
def test_arrays_alive_at_exit_and_stream_object_held():
    out = _run("""
        import aether, cupy as cp
        s = cp.cuda.Stream(non_blocking=True)
        a = aether.zeros(1024, "float64", "cuda", stream=s)
        b = aether.zeros(1024, "float64", "cuda")
        p = aether.pinned(8)
        del s                                   # the array still holds the stream object
        e = aether.zeros(512, "float32", "cuda", stream=cp.cuda.Stream(non_blocking=True))
        del e
        print('ok')
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr[-400:])


# ---- row 7: threads (the free-threaded smoke when run on 3.14t/3.15t with -m ft)
@pytest.mark.ft
def test_threads_allocate_free_export():
    errors, before = [], _core._exports_live()
    gate = threading.Barrier(8)
    kinds = ["cpu", "cuda"] if _gpu() else ["cpu"]

    def work():
        try:
            gate.wait()
            for i in range(50):
                a = aether.zeros(256, "float32", kinds[i % len(kinds)])
                cap = a.__dlpack__(max_version=(1, 0))
                b = aether.full(16, 3, "int32")
                del cap, a, b
        except BaseException as exc:                 # noqa: BLE001
            errors.append(exc)
    pool = [threading.Thread(target=work) for _ in range(8)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    gc.collect()
    assert not errors and _core._exports_live() == before


# ---- row 9: one process, three extensions
@pytest.mark.parametrize("order", [("aether", "eagle"), ("eagle", "aether")])
def test_aether_core_beside_eagle_core(order):
    code = "import importlib\n" + "".join(
        f"importlib.import_module('{'aether._core' if m == 'aether' else 'eagle._core'}')\n" for m in order) + "print('ok')"
    if _run("import eagle._core").returncode:
        pytest.skip("eagle is not importable here")
    out = _run(code)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-400:]


# ---- row 10 (informational) and 14 (plugin selection)
def test_core_size():
    assert pathlib.Path(_core.__file__).stat().st_size <= 1_500_000


@gpu
def test_alloc_free_timing_reported(capsys):
    n = 1000
    t = time.perf_counter()
    xs = [aether.empty(64, device="cuda") for _ in range(n)]
    ta = (time.perf_counter() - t) / n
    t = time.perf_counter()
    del xs
    tf = (time.perf_counter() - t) / n
    line = f"aether alloc {ta * 1e6:.1f} us  free {tf * 1e6:.1f} us"
    try:
        import cupy as cp
        t = time.perf_counter()
        ys = [cp.empty(64) for _ in range(n)]
        line += f"  cupy.empty {(time.perf_counter() - t) / n * 1e6:.1f} us"
        del ys
    except ImportError:
        pass
    with capsys.disabled():
        print("\n" + line)


@gpu
def test_cuda_array_interface_for_cupy():
    cp = pytest.importorskip("cupy")
    a = aether.zeros((2, 3), "int32", "cuda")
    v = cp.asarray(a)
    assert v.shape == (2, 3) and v.data.ptr == a.ptr


def test_the_retired_plugin_major_override_is_a_named_error():
    out = _run("""
        import aether
        try:
            aether.empty(4, device='cuda')
        except aether.Error as exc:
            print('AETHER_CUDA_MAJOR is no longer used' in str(exc))
    """, AETHER_CUDA_MAJOR="13")
    assert out.returncode == 0 and out.stdout.strip() == "True", out.stderr[-300:]


def test_without_a_driver_a_device_call_says_so():
    out = _run("""
        import aether
        try:
            aether.empty(4, device='cuda')
        except aether.CudaUnavailable as exc:
            print('libcuda' in str(exc))
    """, AETHER_CUDA_DRIVER_HIDE="libcuda.so.1")
    if not _plugin_file().exists():
        pytest.skip("no CUDA plugin in this build")
    assert out.returncode == 0 and out.stdout.strip() == "True", (out.stdout, out.stderr[-300:])


def _plugin_file():
    return pathlib.Path(aether.__file__).parent / "libaether_cuda.so"


def test_the_plugin_needs_the_driver_only():
    plugin = _plugin_file()
    if not plugin.exists():
        pytest.skip("no CUDA plugin in this build")
    dyn = subprocess.run(["readelf", "-d", str(plugin)], capture_output=True, text=True, check=True).stdout
    assert "libcudart" not in dyn and "libcuda" not in dyn, dyn
    undef = subprocess.run(["nm", "-D", "-u", str(plugin)], capture_output=True, text=True, check=True).stdout
    assert not [l for l in undef.splitlines() if " cu" in l or "cuda" in l.lower()], undef


@gpu
def test_second_device_recorded():
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") < 1:
        pytest.skip("one device visible")
    assert aether.zeros(4, device="cuda:1").device.index == 1
