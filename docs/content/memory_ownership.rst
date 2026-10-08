Memory ownership: Chunk, Array, View
=====================================

Who owns this buffer, host or device, and how do I copy it?

An ``aether::Chunk`` owns (or, if you asked it to *borrow* foreign memory,
merely wraps) one byte allocation on exactly one ``Device``, and frees it
itself when it goes out of scope; ``aether::copy``/``copyAsync`` move
bytes between two ``Chunk``\ s along a fixed set of legal device pairs, and
refuse — by throwing — anything outside that set.

``Chunk``: one allocation, one device
----------------------------------------

``aether::Chunk`` (``aether/chunk/Chunk.h``) is a move-only, host-only
management type: allocation and deallocation are always host-side calls,
even when the bytes themselves live on a GPU. ``Chunk::allocate(device,
bytes)`` owns what it allocates and frees it on destruction or move-away;
``Chunk::borrow()`` instead wraps memory you already own (a
``std::vector``'s backing store, say) with a no-op deleter — ``owns()``
reports ``false`` and the ``Chunk`` never touches that memory at teardown.
Either way, ``Chunk`` carries no element type or shape of its own; those
live in the layers built on top (see :doc:`views_and_items`).

Copying between devices
--------------------------

``aether::copy``/``aether::copyAsync`` (``aether/chunk/Copy.h``) move raw
bytes between two ``Chunk``\ s. The legal blocking paths are CPU↔CPU,
CPU↔pinned (``kDLCUDAHost``), pinned↔CUDA, and CUDA↔CUDA — a direct
CPU↔CUDA copy is deliberately illegal; you stage it through pinned memory
yourself. ``copyAsync`` takes an ``aether::Stream`` (see
:doc:`host_vs_device` for what a stream is) and is narrower still: only
the pinned/CUDA pairs are async-capable, since ordinary pageable host
memory cannot be safely copied without blocking.

``Array``: the owning convenience
------------------------------------

:doc:`quickstart` already introduced ``aether::Array`` — this is what it
owns underneath. On a CUDA build an ``Array`` holds TWO ``Chunk``\ s, a
pinned host one and a device one, and ``.upload()``/``.download()`` are
what call ``copy``/``copyAsync`` between them; in a CPU-only build there
is only one ``Chunk``, so ``.hostView()`` and ``.deviceView()`` alias the
SAME memory and upload/download become harmless no-ops — the same call
sequence works, and means the same thing conceptually, either way.

A runnable example
-------------------

.. aether-example:: tests/test_Chunk.cpp:25-33

.. code-block:: cpp

   TEST_F(ChunkTest, CpuAllocateAndFree)
   {
       aether::Device cpu(kDLCPU);
       auto chunk = aether::Chunk::allocate(cpu, 256);
       EXPECT_NE(chunk.data(), nullptr);
       EXPECT_EQ(chunk.size(), 256u);
       EXPECT_EQ(chunk.device(), cpu);
       // Freed by the destructor at scope exit — no crash is the pass condition.
   }

.. aether-example-end::

There is no explicit ``free()`` call to make: ``chunk`` releases its 256
bytes automatically when it goes out of scope, the same rule every
move-only RAII type in this library follows.
