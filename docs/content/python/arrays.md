# Arrays in Python: `import aether`

`aether` makes CPU and GPU arrays that other libraries read without a copy. It does
no maths: no operators, no indexing. Use numpy, cupy, torch or your own kernels for that.

It ships in the `aether-dsc` wheel next to `aether_dsc`. `import aether` needs only
numpy. The CUDA runtime comes from `cuda-bindings`, loaded on the first device call:

```
pip install aether-dsc            # CPU arrays
pip install "aether-dsc[cuda12]"  # plus GPU arrays (cuda13 for a CUDA 13 toolkit)
```

Without it, `aether.empty(8, device="cuda")` raises `aether.CudaUnavailable` and names the extra.

## Making arrays

```python
import aether, numpy as np

h = aether.zeros((3, n), np.float64, device="cpu")   # a numpy array
d = aether.zeros((3, n), np.float64, device="cuda")  # an aether.Array ("cuda:1" for device 1)
p = aether.pinned((3, n))                            # numpy over page-locked memory
d2 = aether.asarray(h, device="cuda")                # copy up; zero-copy if already there
aether.copy(d, into=p, stream=s)                     # queued on stream s; without stream= it waits
```

Also `empty`, `full`, `empty_like`, `zeros_like`, `from_dlpack`. Dtypes are bool, int, uint
and float; complex and bfloat16 are refused.

## `Array`

`shape`, `dtype`, `strides` (bytes), `ndim`, `nbytes`, `ptr`, `device`. It speaks
`__dlpack__`, `__dlpack_device__` and `__cuda_array_interface__`, so
`cupy.from_dlpack(d)` and `torch.from_dlpack(d)` share its memory.
`d.zero(stream)` and `d.fill(value, stream)` are memset nodes, legal inside a stream
capture. `d.synchronize()` waits for its stream.

`stream=` takes an int handle or any object with `__cuda_stream__`.

## Lifetimes

- Memory is allocated on the stream you give (default: the legacy stream). The free is queued on
  that same stream, so it runs after the work already queued there, and it waits while a capture
  is in progress on that stream. Pass a stream object (anything with `__cuda_stream__`) and the
  array keeps it alive. A bare int handle is yours to keep alive until the arrays on it are gone;
  destroying it earlier is an error (only the legacy values 0, 1 and 2 are always safe).
  Whatever is left at exit is freed then.
- A DLPack export keeps the array alive until the consumer's deleter runs, once. Order the
  consumer's stream after yours: pass its handle as `stream=` and aether records an event
  and makes it wait; the host never blocks.
- Nothing is shared between arrays, so distinct arrays on distinct threads need no locking,
  free-threaded Python included.

This `Array` is one buffer on one device, like the C++ `Chunk` with a `RuntimeView`. It is
not the C++ `aether::Array`, which is a host-mirrored structure of arrays.
