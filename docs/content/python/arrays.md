# Arrays in Python: `import aether`

`aether` makes CPU and GPU arrays that other libraries read without a copy. It does no
maths: no operators, no indexing (Fourier transforms are the one exception, below). Use numpy, cupy, torch or your own kernels for the rest.

It ships in the `aether-dsc` wheel next to `aether_dsc`, and needs nothing else:
no numpy, no cuda-bindings, no extras. On Linux x86_64 and aarch64 the wheel carries one CUDA
plugin, for CUDA 12 and 13 alike. It talks to the driver API only, so a GPU user needs
the NVIDIA driver and nothing more: no CUDA toolkit, no CUDA runtime library. The first
device call loads it. With no GPU or no driver, `import aether` still works and a device
call raises `aether.CudaUnavailable`, naming what the driver said.
`aether.cuda_runtime_version()` is the CUDA version the installed driver supports.

```
pip install aether-dsc
```

Wheels: Linux x86_64 and aarch64, and macOS arm64 (CPython 3.9 to 3.15, free-threaded builds included).
The Linux wheels carry the CUDA plugin; the aarch64 one is built and load-tested without a GPU and has not run on one.
The macOS wheels run on the CPU and bundle libomp. Without a wheel for your platform, pip builds from the source
distribution: it needs a C++23 compiler, and gives you CPU arrays.

## Making arrays

```python
import aether

h = aether.zeros((3, n), "float64")                    # device="cpu" is the default
d = aether.zeros((3, n), "float64", device="cuda")     # "cuda:1" for device 1
p = aether.pinned((3, n))                              # page-locked host memory
d2 = aether.asarray(h, device="cuda")                  # copy up; zero-copy if already there
aether.copy(d, into=p, stream=s)                       # queued on stream s; without stream= it waits
```

Also `empty`, `full`, `empty_like`, `zeros_like`, `from_dlpack`. Every one of these
returns an `aether.Array`, host memory included. Dtypes are given by name or Python type:
`float64 float32 float16 bfloat16 int64 int32 int16 int8 uint64 uint32 uint16 uint8 bool`.
Others, complex included, are refused by name. `bfloat16` has no buffer-protocol format, so
`memoryview` refuses it; it moves through DLPack.

## `Array`

`shape`, `dtype`, `strides` (bytes), `ndim`, `nbytes`, `ptr`, `device`, `is_contiguous`.

- Host and pinned arrays speak the buffer protocol: `numpy.asarray(a)` and `memoryview(a)`
  share the memory, writes show on both sides.
- Every array speaks DLPack (`__dlpack__`, `__dlpack_device__`); device arrays also
  `__cuda_array_interface__`, so `cupy.from_dlpack(d)` and `torch.from_dlpack(d)` share it.
- `d.zero(stream)` and `d.fill(value, stream)` are memset nodes, legal inside a stream capture.
  `d.synchronize()` waits for the array's stream.

`stream=` takes an int handle or any object with `__cuda_stream__`.

## Lifetimes

- Memory is allocated on the stream you give (default: the legacy stream). The free is queued on
  that same stream, so it runs after the work already queued there, and it waits while a capture is
  in progress on that stream. Pass a stream object and the array keeps it alive. A bare int handle
  is yours to keep alive until the arrays on it are gone (the legacy values 0, 1 and 2 always are).
  Whatever is left at exit is freed then.
- A DLPack export keeps the array alive until the consumer's deleter runs, once. To order the
  consumer's stream after yours, pass its handle as `stream=`: aether records an event and makes it
  wait; the host never blocks.
- Arrays are immutable after construction, so distinct arrays on distinct threads need no locking,
  free-threaded Python included.

This `Array` is one buffer on one device, like the C++ `Chunk` with a `RuntimeView`. It is not the
C++ `aether::Array`, which is a host-mirrored structure of arrays.

## Fourier transforms: `aether.fft`

Batched transforms over arrays you already have, on the CPU and the GPU. There is no complex dtype: complex data is a
`float32` or `float64` array whose last axis has extent 2, `(re, im)`. `axis` counts the data axes, not the 2.

```python
import aether
import aether.fft as afft

x = aether.zeros((64, 1024, 2), "float64")   # 64 signals of 1024 points
X = afft.fft(x, axis=1)                      # same shape, forward
y = afft.ifft(X, axis=1)                     # back; scaled by 1/n
S = afft.rfft(aether.zeros((64, 1024), "float64"), axis=1)   # real input: 513 points
```

`fft`, `ifft`, `rfft`, `irfft` and the N-D `fftn`, `ifftn`, `rfftn`, `irfftn` take `norm="backward"` (default),
`"forward"` or `"ortho"`, as NumPy does. For repeated work, `aether.fft.plan(...)` builds a plan once and
`plan.execute(input, output, stream)` runs it without allocating, inside a stream capture if you like. Build the plan
before the capture starts.

The CPU transform is pocketfft with OpenMP across the batch, and gives the same bits for any thread count. The GPU
transform is cuFFT, which aether does not bundle: `pip install "aether-dsc[cufft12]"` (or `[cufft13]`). Without it a GPU
transform raises `aether.FftUnavailable`, naming the extra. The CPU and GPU results agree to rounding, not bit for bit.
