# Arrays in Python: `import aether`

`aether` makes CPU and GPU arrays that other libraries read without a copy. It does no
maths: no operators, no indexing. Use numpy, cupy, torch or your own kernels for that.

It ships in the `aether-dsc` wheel next to `aether_dsc`, and needs nothing else:
no numpy, no cuda-bindings, no extras. On Linux x86_64 and aarch64 the wheel carries two CUDA
plugins, one for CUDA 13 and one for CUDA 12. The first device call loads the first
one your driver accepts, so a GPU user installs nothing more. `AETHER_CUDA_MAJOR=12`
or `13` forces one. With no GPU or no driver, `import aether` still works and a device
call raises `aether.CudaUnavailable`, naming what each plugin said.

```
pip install aether-dsc
```

Without a wheel for your platform, pip builds from the source distribution: it needs a
C++23 compiler, and gives you CPU arrays.

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
