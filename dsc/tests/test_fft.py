# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""aether.fft rows: the host transforms (pocketfft) and the device transforms (cuFFT) against numpy.fft.

Complex data is a float array with a trailing axis of 2. Tolerance, ULP-scaled: the largest absolute error of a
transform over N points is at most (8 + 4 log2 N) * eps * ||x||_2 * scale, eps the width's machine epsilon and
scale the normalisation factor. numpy is the oracle (tests may import it; aether itself needs none)."""

import math
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

import aether
from aether import fft as F

np = pytest.importorskip("numpy")

DTYPES = ["float32", "float64"]


def _gpu():
    try:
        aether.cuda_plugin()
        return True
    except aether.CudaUnavailable:
        return False


def _cufft():
    if not _gpu():
        return False
    try:
        F.plan("c2c", "float32", (4,), (0,), device="cuda")
        return True
    except aether.FftUnavailable:
        return False


gpu = pytest.mark.skipif(not _gpu(), reason="needs a CUDA plugin that loads and a device")
gpu_fft = pytest.mark.skipif(not _cufft(), reason="needs a device and cuFFT")


def pairs(c, dtype):
    return np.ascontiguousarray(np.stack([c.real, c.imag], -1).astype(dtype))


def cplx(p):
    p = np.asarray(p, dtype=np.float64)
    return p[..., 0] + 1j * p[..., 1]


def tol(dtype, points, x, scale):
    return (8 + 4 * math.log2(points + 1)) * np.finfo(dtype).eps * float(np.linalg.norm(np.asarray(x, np.float64))) * scale


def rand_c(shape, dtype, seed=0):
    r = np.random.default_rng(seed)
    return pairs(r.uniform(-1, 1, shape) + 1j * r.uniform(-1, 1, shape), dtype)


def scale_of(norm, forward, n):
    if norm == "ortho":
        return 1 / math.sqrt(n)
    return 1 / n if forward == (norm == "forward") else 1.0


def to_host(a):
    return np.array(aether.asarray(a, device="cpu"), copy=True)


def on(x, device):
    return aether.asarray(x, device=device)


DEVICES = ["cpu", pytest.param("cuda", marks=gpu_fft)]


# ---------------------------------------------------------------- complex to complex
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_c2c_lengths_match_numpy(device, dtype):
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 31, 64, 100, 127, 1000):
        x = rand_c((3, n), dtype, n)
        for forward in (True, False):
            fn = F.fft if forward else F.ifft
            got = to_host(fn(on(x, device)))
            ref = (np.fft.fft if forward else np.fft.ifft)(cplx(x), axis=-1)
            scale = scale_of("backward", forward, n)
            err = np.abs(cplx(got) - ref).max()
            assert err <= tol(dtype, n, x, scale), (n, forward, err)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_multi_axis_and_batched_match_numpy(device, dtype):
    x = rand_c((4, 7, 6), dtype, 5)
    xc = cplx(x)
    cases = [((1,), np.fft.fft(xc, axis=1)), ((0, 2), np.fft.fftn(xc, axes=(0, 2))), ((0, 1, 2), np.fft.fftn(xc)),
             ((2, 0), np.fft.fftn(xc, axes=(2, 0)))]
    for axes, ref in cases:
        got = to_host(F.fftn(on(x, device), axes=axes))
        pts = int(np.prod([x.shape[a] for a in axes]))
        assert np.abs(cplx(got) - ref).max() <= tol(dtype, pts, x, 1.0), axes
        back = to_host(F.ifftn(on(got, device), axes=axes))
        assert np.abs(cplx(back) - xc).max() <= 2 * tol(dtype, pts, got, 1.0 / pts), axes


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_norms_match_numpy(device, norm):
    x = rand_c((2, 12), "float64", 9)
    for forward in (True, False):
        fn = F.fft if forward else F.ifft
        got = to_host(fn(on(x, device), norm=norm))
        ref = (np.fft.fft if forward else np.fft.ifft)(cplx(x), norm=norm)
        assert np.abs(cplx(got) - ref).max() <= tol("float64", 12, x, scale_of(norm, forward, 12))


# ---------------------------------------------------------------- real transforms
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rfft_irfft_match_numpy(device, dtype):
    for n in (2, 3, 7, 8, 9, 16, 17, 50, 128):
        r = np.random.default_rng(n).uniform(-1, 1, (3, n)).astype(dtype)
        got = to_host(F.rfft(on(r, device)))
        ref = np.fft.rfft(r.astype(np.float64), axis=-1)
        assert got.shape == ref.shape + (2,)
        assert np.abs(cplx(got) - ref).max() <= tol(dtype, n, r, 1.0), n
        back = to_host(F.irfft(on(got, device), n=n))
        assert np.abs(back - r).max() <= 2 * tol(dtype, n, r, 1.0 / n), n
        spec = pairs(ref, dtype)
        c2r = to_host(F.irfft(on(spec, device), n=n))
        assert np.abs(c2r - np.fft.irfft(ref, n=n, axis=-1)).max() <= 2 * tol(dtype, n, spec, 1.0 / n), n


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rfftn_irfftn_match_numpy(device, dtype):
    r = np.random.default_rng(3).uniform(-1, 1, (4, 5, 6)).astype(dtype)
    for axes in ((0, 1, 2), (1, 2), (2, 0)):
        got = to_host(F.rfftn(on(r, device), axes=axes))
        ref = np.fft.rfftn(r.astype(np.float64), axes=axes)
        pts = int(np.prod([r.shape[a] for a in axes]))
        assert got.shape == ref.shape + (2,), axes
        assert np.abs(cplx(got) - ref).max() <= tol(dtype, pts, r, 1.0), axes
        s = [r.shape[a] for a in axes]
        back = to_host(F.irfftn(on(got, device), s=s, axes=axes))
        assert np.abs(back - r).max() <= 2 * tol(dtype, pts, r, 1.0 / pts), axes
    # the input of a complex-to-real transform is left unchanged
    spec = pairs(np.fft.rfftn(r.astype(np.float64)), dtype)
    keep = spec.copy()
    F.irfftn(on(spec, device), s=r.shape, axes=(0, 1, 2))
    assert np.array_equal(spec, keep)


# ---------------------------------------------------------------- plans and errors
def test_plans_are_cached_and_carry_their_shapes():
    a = F.plan("r2c", "float32", (4, 9), (1,))
    assert a is F.plan("r2c", "float32", (4, 9), (-1,))
    assert a.input_shape == (4, 9) and a.output_shape == (4, 5, 2)
    assert F.plan("c2c", "float64", (3, 4), (0, 1), forward=False).input_shape == (3, 4, 2)


def test_misuse_is_a_named_error():
    x = aether.asarray(rand_c((8,), "float32"))
    with pytest.raises(aether.Error, match="trailing axis of 2"):
        F.fft(aether.zeros((8,), "float32"))
    with pytest.raises(aether.Error, match="norm"):
        F.fft(x, norm="bogus")
    with pytest.raises(aether.Error, match="out of range"):
        F.fft(x, axis=3)
    with pytest.raises(aether.Error, match="float32 or float64"):
        F.plan("c2c", "int32", (4,), (0,))
    p = F.plan("c2c", "float32", (8,), (0,))
    with pytest.raises(aether.Error, match="plan wants float32"):
        p.execute(aether.zeros((8, 2), "float64"), aether.zeros((8, 2), "float64"))
    with pytest.raises(aether.Error, match="wrong shape"):
        p.execute(aether.zeros((7, 2), "float32"), aether.zeros((8, 2), "float32"))
    # pinned host memory is host memory: it runs the host transform
    pin = aether.pinned((8, 2), "float32")
    pin.zero()
    assert F.fft(pin).device.kind == "cpu"


def test_the_host_path_needs_no_numpy():
    code = textwrap.dedent("""
        import sys
        for m in ('numpy', 'cupy', 'cuda'):
            sys.modules[m] = None
        import aether
        from aether import fft
        x = aether.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        y = fft.fft(x)
        assert [list(r) for r in memoryview(y).tolist()] == [[1.0, 0.0]] * 4, memoryview(y).tolist()
        print('ok')
    """)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "ok", out.stderr[-400:]


# ---------------------------------------------------------------- device only
@gpu
def test_the_plugin_still_exports_only_the_seam():
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm not available")
    import aether as a
    plugin = pathlib.Path(a.__file__).parent / "libaether_cuda.so"
    out = subprocess.run([nm, "-D", "--defined-only", str(plugin)], capture_output=True, text=True, check=True).stdout
    assert {line.split()[-1].split("@")[0] for line in out.splitlines() if line.strip()} == {"aether_memory_v1"}


@gpu
def test_missing_cufft_names_the_extra():
    code = textwrap.dedent("""
        import aether
        from aether import fft
        try:
            fft.fft(aether.zeros((8, 2), "float32", "cuda"))
        except aether.FftUnavailable as e:
            print(e)
        else:
            raise SystemExit("no error")
    """)
    import os
    env = dict(os.environ, AETHER_CUFFT_LIB="/nonexistent/libcufft.so.11")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=180, env=env)
    assert out.returncode == 0, out.stderr[-400:]
    assert "aether-dsc[cufft12]" in out.stdout or "aether-dsc[cufft13]" in out.stdout, out.stdout
    # the host transforms are unaffected
    host = subprocess.run([sys.executable, "-W", "ignore", "-c", "import aether; from aether import fft; fft.fft(aether.zeros((8, 2), 'float32')); print('ok')"],
                          capture_output=True, text=True, timeout=120, env=env)
    assert host.stdout.strip() == "ok"


@gpu_fft
def test_a_plan_built_before_a_capture_runs_inside_it_and_replays():
    cp = pytest.importorskip("cupy")
    s = cp.cuda.Stream(non_blocking=True)
    x = rand_c((16, 64), "float32", 1)
    d = aether.asarray(x, device="cuda")
    out = aether.empty((16, 64, 2), "float32", "cuda", stream=s)
    p = F.plan("c2c", "float32", (16, 64), (1,), True, "backward", "cuda", s)      # eager, outside the capture
    p2 = F.plan("c2r", "float32", (16, 64), (1,), False, "backward", "cuda", s)
    real = aether.empty((16, 64), "float32", "cuda", stream=s)
    ref = np.fft.fft(cplx(x), axis=-1)
    with s:
        s.begin_capture()
        p.execute(d, out, s)
        g = s.end_capture()
    for _ in range(2):
        cp.asarray(out)[...] = 0
        cp.cuda.runtime.deviceSynchronize()
        g.launch(s)
        s.synchronize()
        got = cp.asnumpy(cp.asarray(out))
        assert np.abs(cplx(got) - ref).max() <= tol("float32", 64, x, 1.0)
    # a complex-to-real plan (work copy, scaling kernel) also captures and replays
    r = np.random.default_rng(2).uniform(-1, 1, (16, 64)).astype("float32")
    sp = pairs(np.fft.rfft(r.astype(np.float64), axis=-1), "float32")
    dsp = aether.asarray(sp, device="cuda")
    with s:
        s.begin_capture()
        p2.execute(dsp, real, s)
        g2 = s.end_capture()
    for _ in range(2):
        cp.asarray(real)[...] = 0
        cp.cuda.runtime.deviceSynchronize()
        g2.launch(s)
        s.synchronize()
        assert np.abs(cp.asnumpy(cp.asarray(real)) - r).max() <= 2 * tol("float32", 64, r, 1 / 64)


@gpu_fft
def test_plan_creation_inside_a_capture_is_refused_by_name():
    cp = pytest.importorskip("cupy")
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        s.begin_capture()
        try:
            with pytest.raises(aether.Error, match="plan creation is not allowed inside a stream capture"):
                F.plan("c2c", "float32", (123, 45), (1,), True, "backward", "cuda", s)
        finally:
            s.end_capture()


# ---------------------------------------------------------------- host threading
_THREADED = textwrap.dedent("""
    import sys, time, hashlib
    import aether
    from aether import fft
    n, batch = 256, 10000
    x = aether.zeros((batch, n, 2), "float32")
    for i in range(0, 2 * batch * n, 977):
        x._set_flat(i, 0.001 * (i % 1013))
    y = fft.fft(x)
    r = aether.asarray([[0.5 * ((i * 7 + j) % 13) for j in range(n)] for i in range(64)])
    z = fft.rfftn(r, axes=(0, 1))
    digest = hashlib.sha256(bytes(memoryview(y).cast("B")) + bytes(memoryview(z).cast("B"))).hexdigest()
    best = 1e9
    for _ in range(15):
        t = time.perf_counter()
        fft.fft(x)
        best = min(best, time.perf_counter() - t)
    print(digest, best)
""")


def _threaded(threads):
    import os
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _THREADED], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-400:]
    digest, best = out.stdout.split()
    return digest, float(best)


def test_host_fft_gives_the_same_bits_for_any_thread_count():
    assert _threaded(1)[0] == _threaded(4)[0]


def test_host_fft_batch_speeds_up_with_threads(capsys):
    import os
    if len(os.sched_getaffinity(0)) < 4:
        pytest.skip("needs 4 cores")
    one, four = _threaded(1)[1], _threaded(4)[1]
    with capsys.disabled():
        print(f"\nfft 1e4 x 256 (f32): 1 thread / 4 threads = {one / four:.2f}x")
    assert one / four > 1.3
