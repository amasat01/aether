# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""aether arrays: the acceptance rows. GPU rows skip when no device or no cuda-bindings is present."""

import gc
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import aether
from aether import _dlpack


def _gpu():
    try:
        from aether._rt import ck, rt
        return int(ck(rt().cudaGetDeviceCount())) > 0
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu(), reason="needs cuda-bindings and a device")


def _run(code):
    return subprocess.run([sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=120)


# ---- row 1: CPU-only import
def test_import_without_cuda_bindings():
    out = _run("""
        import sys
        sys.modules['cuda'] = None; sys.modules['cuda.bindings'] = None
        import aether, numpy as np
        assert isinstance(aether.zeros(4, device='cpu'), np.ndarray)
        assert isinstance(aether.asarray([1.0, 2.0]), np.ndarray)
        try:
            aether.empty(8, device='cuda')
        except aether.CudaUnavailable as exc:
            assert 'aether-dsc[cuda12]' in str(exc)
            print('ok')
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_dtype_refusals_by_name():
    for bad in (np.complex64, "bfloat16", object):
        with pytest.raises(TypeError):
            aether.empty(4, bad, device="cpu")


@gpu
def test_surface_and_values():
    a = aether.zeros((3, 5), np.float32)
    assert (a.shape, a.dtype, a.strides, a.nbytes, a.device) == ((3, 5), np.float32, (20, 4), 60, aether.Device("cuda", 0))
    h = aether.pinned((3, 5), np.float32)
    aether.copy(aether.full((3, 5), 2.5, np.float32), into=h)
    assert (h == 2.5).all()
    for dt, val in ((np.float64, 1.0), (np.float64, 0.1), (np.int16, 7), (np.int32, -3), (np.uint8, 9), (np.bool_, True)):
        f = aether.full(33, val, dt)
        f.synchronize()
        assert (aether.asarray(f, device="cpu") == dt(val)).all(), (dt, val)
    e = aether.zeros_like(h)
    assert e.shape == h.shape and aether.empty_like(h, device="cuda").dtype == h.dtype and aether.empty(0).ptr == 0


@gpu
def test_copy_directions():
    src = np.arange(12, dtype=np.float64).reshape(3, 4)
    d = aether.asarray(src, device="cuda")
    d2 = aether.copy(d)
    back = aether.copy(d2, into=aether.pinned(src.shape))
    assert np.array_equal(back, src) and aether.asarray(d) is d
    with pytest.raises(ValueError):
        aether.copy(d, into=np.empty((4, 3)))


# ---- row 2: DLPack round trips
@gpu
def test_dlpack_cupy_roundtrip_ptr_equal():
    cp = pytest.importorskip("cupy")
    c = cp.arange(24, dtype=cp.float64).reshape(4, 6)
    a = aether.from_dlpack(c)
    assert a.ptr == c.data.ptr and a.shape == (4, 6)
    c2 = cp.from_dlpack(a)
    c3 = cp.asarray(a)
    assert c2.data.ptr == a.ptr == c3.data.ptr
    c2 *= 2.0
    assert float(c[3, 5]) == 46.0
    del a, c2, c3
    gc.collect()


@gpu
def test_dlpack_deleter_once_both_generations():
    before = _dlpack.live_exports()
    cp = pytest.importorskip("cupy")
    a = aether.zeros(16, np.float32)
    legacy = cp.from_dlpack(a)                         # cupy asks for the legacy struct or versioned
    cap_v = a.__dlpack__(max_version=(1, 0))
    cap_l = a.__dlpack__()
    assert _dlpack.live_exports() >= before + 2
    del cap_v, cap_l, legacy                           # never consumed: the capsule destructor releases
    gc.collect()
    assert _dlpack.live_exports() == before


@gpu
def test_dlpack_versioned_consumed_once():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA here")
    before = _dlpack.live_exports()
    a = aether.full(8, 3.0, np.float32)
    a.synchronize()
    t = torch.from_dlpack(a)
    assert t.data_ptr() == a.ptr and float(t.sum()) == 24.0
    del t
    gc.collect()
    assert _dlpack.live_exports() == before


def test_numpy_pinned_torch_cpu():
    torch = pytest.importorskip("torch")
    if not _gpu():
        pytest.skip("pinned memory needs a device")
    h = aether.pinned((3, 7), np.float64)
    h[:] = 4.0
    t = torch.from_dlpack(h)
    assert t.data_ptr() == h.ctypes.data and float(t.sum()) == 84.0


@gpu
def test_from_dlpack_import_deleter_fires_once():
    cp = pytest.importorskip("cupy")
    c = cp.zeros(8)
    a = aether.from_dlpack(c)
    calls = []
    real = a._owner._deleter
    a._owner._deleter = lambda address: (calls.append(address), real(address))
    del a
    gc.collect()
    assert len(calls) == 1


# ---- row 3: stream codes
@gpu
def test_dlpack_stream_codes():
    from aether._rt import ck, rt
    a = aether.zeros(8, np.float64)
    with pytest.raises(ValueError):
        a.__dlpack__(stream=0)
    a.__dlpack__(stream=-1)
    a.__dlpack__(stream=None)
    a.__dlpack__(stream=1)
    s = int(ck(rt().cudaStreamCreateWithFlags(rt().cudaStreamNonBlocking)))
    try:
        a.__dlpack__(stream=s)
    finally:
        a.synchronize()
        ck(rt().cudaStreamDestroy(s))


# ---- row 4: zero/fill inside a capture, two replays
@gpu
def test_zero_fill_capture_two_replays():
    from aether._rt import ck, rt
    r = rt()
    s = int(ck(r.cudaStreamCreateWithFlags(r.cudaStreamNonBlocking)))

    class S:
        __cuda_stream__ = lambda self: (0, s)          # noqa: E731
    d = aether.empty(4096, np.float64, stream=S())
    w = aether.empty(4096, np.float64, stream=S())
    host = aether.pinned(4096, np.float64)
    ck(r.cudaStreamBeginCapture(s, r.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
    try:
        d.zero(s)
        w.fill(0.1, s)                                  # two strided memsets
        aether.copy(d, into=host, stream=s)
    finally:
        g = ck(r.cudaStreamEndCapture(s))
    ge = ck(r.cudaGraphInstantiate(g, 0))
    for _ in range(2):
        ck(r.cudaMemset(d.ptr, 0xFF, d.nbytes))        # dirty it between replays
        ck(r.cudaGraphLaunch(ge, s))
        ck(r.cudaStreamSynchronize(s))
        assert not host.any()
        assert (aether.asarray(w, device="cpu") == 0.1).all()
    ck(r.cudaGraphExecDestroy(ge))
    ck(r.cudaGraphDestroy(g))
    del d, w
    ck(r.cudaStreamDestroy(s))


# ---- row 5: lifetimes
@gpu
def test_array_freed_before_its_stream_and_live_at_exit():
    out = _run("""
        import numpy as np, aether
        from aether._rt import ck, rt
        r = rt()
        s = int(ck(r.cudaStreamCreateWithFlags(r.cudaStreamNonBlocking)))
        a = aether.zeros(1024, np.float64, stream=s)
        b = aether.zeros(1024, np.float64, stream=s)
        c = aether.zeros(1024, np.float64)
        p = aether.pinned(8)
        del a                      # freed in order on its own stream, then the stream goes
        ck(r.cudaStreamSynchronize(s)); ck(r.cudaStreamDestroy(s))
        s2 = int(ck(r.cudaStreamCreateWithFlags(r.cudaStreamNonBlocking)))
        class S:
            __cuda_stream__ = lambda self: (0, s2)
        so = S()
        e = aether.zeros(512, np.float32, stream=so)   # holds the stream object: freed on its stream
        del e
        f = aether.zeros(512, np.float32, stream=so)   # still alive at exit, stream object alive too
        print('ok')                # b, c and p are alive at exit
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stderr[-500:])


# ---- row 6: threads (also the free-threaded smoke when run under 3.14t with -m ft)
@pytest.mark.ft
@gpu
def test_threads_allocate_free_export():
    errors = []
    before = _dlpack.live_exports()
    gate = threading.Barrier(8)

    def work():
        try:
            gate.wait()
            for _ in range(50):
                a = aether.zeros(256, np.float32)
                cap = a.__dlpack__(max_version=(1, 0))
                b = aether.full(16, 3, np.int32)
                del cap, a, b
        except BaseException as exc:             # noqa: BLE001
            errors.append(exc)
    pool = [threading.Thread(target=work) for _ in range(8)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    gc.collect()
    assert not errors and _dlpack.live_exports() == before


# ---- row 10: timing (informational) and the contract
@gpu
def test_alloc_free_timing_reported(capsys):
    n = 1000
    t = time.perf_counter()
    xs = [aether.empty(64) for _ in range(n)]
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
    a = aether.zeros((2, 3), np.int32)
    v = cp.asarray(a)
    assert v.shape == (2, 3) and v.data.ptr == a.ptr


@gpu
def test_second_device_recorded():
    from aether._rt import ck, rt
    if int(ck(rt().cudaGetDeviceCount())) < 2:
        pytest.skip("one device visible")
    assert aether.zeros(4, device="cuda:1").device.index == 1


# ---- a collector-issued free inside a global-mode capture must not break the capture
@gpu
@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("mode", ["legacy", "stream"])
def test_gc_free_inside_global_capture(mode, blocking):
    out = _run(f"""
        import gc, numpy as np, aether, aether._array as A
        from aether._rt import ck, rt
        r = rt()
        s = int(ck(r.cudaStreamCreateWithFlags(r.cudaStreamNonBlocking)))
        a = aether.zeros(1024, np.float64, stream={'s' if mode == 'stream' else 'None'})
        cyc = [a]; cyc.append(cyc)
        del a, cyc
        ck(r.cudaDeviceSynchronize())
        gc.disable()
        c = int(ck(r.cudaStreamCreateWithFlags({'r.cudaStreamDefault' if blocking else 'r.cudaStreamNonBlocking'})))
        ck(r.cudaStreamBeginCapture(c, r.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
        gc.collect()                      # the free lands inside the capture
        g = ck(r.cudaStreamEndCapture(c))     # a broken capture fails here
        ck(r.cudaGraphDestroy(g))
        aether.empty(1)                   # the next allocation outside the capture drains a deferred free
        assert not A._pending, A._pending
        print('ok')
    """)
    assert out.returncode == 0 and out.stdout.strip() == "ok", (out.returncode, out.stdout, out.stderr[-400:])


# ---- a free is ordered after the work already queued on the allocation's stream
@gpu
@pytest.mark.parametrize("mode", ["legacy", "int_handle"])
def test_early_drop_after_launch_is_stream_ordered(mode):
    cp = pytest.importorskip("cupy")
    spin = cp.RawKernel(r"""
        extern "C" __global__ void spin(double* p, long long cycles) {
            long long t = clock64(); while (clock64() - t < cycles) {}
            p[0] = 7.0; }""", "spin")
    own = cp.cuda.Stream(non_blocking=True)
    stream = own if mode == "int_handle" else cp.cuda.Stream.null
    a = aether.zeros(8, np.float64, stream=int(stream.ptr) or None)
    keep = cp.asarray(a)
    with stream:
        spin((1,), (1,), (keep, np.int64(1_000_000_000)))       # about a second on a P2000
    ptr = a.ptr
    del keep
    t = time.perf_counter()
    del a
    gc.collect()
    waited = time.perf_counter() - t
    stream.synchronize()
    assert waited < 0.5, f"the free blocked the host for {waited:.2f}s: it was not queued on the stream"
    probe = aether.zeros(8, np.float64, stream=int(stream.ptr) or None)
    stream.synchronize()
    assert (aether.asarray(probe, device="cpu") == 0).all()
    del probe                               # frees on the stream: it must still exist
    gc.collect()
    stream.synchronize()
