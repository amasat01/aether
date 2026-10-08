Interoperability
==================

How does a NumPy/CuPy array become an aether ``View``?

``aether::interop::fromDLPack`` takes the raw DLPack C struct that NumPy,
CuPy or PyTorch export their arrays as, and hands back a zero-copy
``RuntimeView`` over that SAME memory plus an RAII token that releases the
producer's own array exactly once — no bytes move, and no allocation
happens on aether's side at all.

Where this surfaces
----------------------

aether itself has no Python package: it is a header-only C++ library, and
the DLPack bridge described on this page is C-ABI level only (a plain
struct with a deleter function pointer, nothing more). DLPack is the
contract every array library that reaches this bridge agrees to. The
place a Python array actually crosses it is downstream, in the RAPTOR
family's Python-facing layers:

- `eagle <https://amasat01.github.io/eagle/>`_ wraps this same bridge in
  its Python bindings and is where worked NumPy/CuPy/PyTorch interop
  examples live.
- `raptor <https://amasat01.github.io/raptor/>`_ owns the family's
  interop CERTIFICATION MATRIX — the authoritative list of which
  framework/device combinations are actually exercised by a test, not
  just plausible from the DLPack contract alone.

.. admonition:: Roadmap
   :class: note

   DLPack is a wire format many more frameworks speak — JAX,
   TensorFlow/Keras and others are foreseen consumers of this same
   bridge. Treat only what raptor's certification matrix actually
   certifies today (NumPy, CuPy, PyTorch) as supported; this page makes
   no claim that JAX or TensorFlow/Keras interop works yet.

Importing: ``fromDLPack``
-----------------------------

``aether::interop::fromDLPack`` (``aether/interop/DLPack.h``) accepts
either DLPack struct generation in circulation — the legacy
``DLManagedTensor`` and the current ``DLManagedTensorVersioned`` — and
returns a fixed-capacity ``aether::RuntimeView`` (a shape that is only
known at RUNTIME, unlike the compile-time-shaped ``View`` earlier pages
use) together with a ``DLPackOwner`` that calls the producer's deleter on
destruction. A runtime-shaped ``RuntimeView`` promotes to a statically
shaped one with ``.as<Vec3dView>()`` once you know the rank and component
count you expect. Malformed input is rejected with a thrown
``aether::Error`` — an unsupported dtype, rank above the runtime cap, a
non-zero byte offset, or an unrecognized device kind — and even a rejected
import still calls the producer's deleter exactly once before it rethrows,
so a refused import never leaks the producer's array.

Exporting: ``toDLPack``
---------------------------

The other direction, ``aether::interop::toDLPack``, wraps one of aether's
own views in a ``DLManagedTensorVersioned*`` an external consumer can
import — again zero-copy, with the exported struct's ``data`` pointer
aliasing the view's own memory and its ``shape``/``strides`` describing
the SAME layout_right SoA layout :doc:`views_and_items` introduces.

A runnable example
-------------------

The round-trip test below exports a view, then re-imports THAT SAME
export the way an external consumer would, and checks the data pointer,
rank and every element survive the trip unchanged:

.. aether-example:: tests/test_DLPackRoundtrip.cpp:177-183

.. code-block:: cpp

       aether::interop::DLPackImport reimported = aether::interop::fromDLPack(exported);
       EXPECT_EQ(reimported.view.data, static_cast<void*>(v.data()));
       EXPECT_EQ(reimported.view.rank, 2u);
       const aether::Vec3dView promoted = reimported.view.as<aether::Vec3dView>();
       for (std::size_t c = 0; c < C; ++c)
           for (std::size_t i = 0; i < N; ++i)
               EXPECT_EQ(promoted(c, i), v(c, i));

.. aether-example-end::

Note the explicit ``.as<aether::Vec3dView>()`` promotion — a
``RuntimeView`` never silently becomes a compile-time-shaped ``View``; you
always name the shape you expect. A host-only ``std::mdspan`` adapter also
exists (``aether/interop/Mdspan.h``) for a C++23 toolchain whose standard
library already ships ``<mdspan>``; this workspace's pinned toolchain does
not yet, so that adapter is untested here and included only when you ask
for it explicitly.

The buffer layer: access, ownership, lifetime
-----------------------------------------------

``aether/interop/Buffer.h`` (since 0.2.0) is the generic layer every family
consumer builds on. ``aether::interop::importDLPack`` takes either DLPack
generation and returns an ``aether::interop::BufferView``: the same zero-copy
``RuntimeView`` (the producer's ``byte_offset`` folded into its pointer),
plus

- ``access`` — ``Access::ReadWrite``, ``Access::ReadOnly`` or
  ``Access::Unknown``, exactly as the producer declared it: the versioned
  struct's read-only flag, an array interface's ``data[1]``, and
  ``Unknown`` for the legacy struct, which carries no flag. The layer never
  upgrades ``Unknown`` by itself; ``assumeWritable`` records a caller's
  explicit assertion.
- ``owner`` — the allocating library's name for memory of this family,
  ``"external"`` for anything imported; ``producer`` — the producer's type
  name.
- ``keepAlive`` — a type-erased ``std::shared_ptr<void>`` that calls the
  producer's deleter exactly once, after the last copy dies.

``fromArrayInterface`` builds the same record from raw
``__cuda_array_interface__``/``__array_interface__`` fields (pointer,
shape, byte strides, ``typestr``, read-only flag, device), so a language
binding can feed producers that do not speak DLPack. ``refusals`` /
``require`` validate a ``Requirements`` (dtype, element count or shape,
C-contiguity, alignment, device type and ordinal, writability) with one
message per failed check, in the form ``"<check>: required <want>, got
<have>"``.

``exportDLPack`` and ``exportDLPackLegacy`` hand the buffer on, zero-copy.
The versioned export sets the read-only flag unless the buffer is writable;
the legacy struct cannot carry the flag, so a read-only buffer is refused
there. Every export stores a share of the keep-alive, so a chain import ->
export -> import keeps the FIRST producer alive until the last view anywhere
in the chain is gone.

The header makes no CUDA call. Stream ordering — the DLPack 1.0 stream
codes, the export fence and the re-fence — lives in eagle, which documents
the full contract on its `Interoperability contract
<https://amasat01.github.io/eagle/content/interop_contract.html>`_ page.
