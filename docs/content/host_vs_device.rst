Host vs device code
====================

What does ``AETHER_DEVICEHOST`` mean, and when do I need a GPU kernel entry
point (``__global__``)?

``AETHER_DEVICEHOST()`` marks an ordinary function so it compiles correctly
and runs correctly whether it is called from plain CPU code or from code
running on the GPU; you only reach for a kernel entry point when you need
to *launch* brand-new parallel work on the GPU from the CPU, one call
spawning thousands of GPU threads at once. Almost everything in aether —
``View`` indexing, expression-template arithmetic, ``Item`` construction —
is written with ``AETHER_DEVICEHOST()`` precisely so the same source line
works unchanged on both sides; kernel entry points are the exception, not
the rule.

Vocabulary
----------

This is the one page where aether's CUDA vocabulary gets defined; every
later page assumes these terms.

- **Host** — the CPU and its ordinary system memory: the code that runs
  when your program starts, allocates buffers and decides what work to do.
- **Device** — the GPU and its own separate memory. Host code cannot read
  device memory directly (and vice versa) without an explicit copy.
- **Kernel** — a function launched *from* the host to run *on* the device.
  A single kernel launch runs the SAME function body across many threads
  at once, each thread working on a different piece of data. A kernel is
  written with a kernel entry-point qualifier (CUDA's ``__global__``) and
  cannot return a value directly to the caller.
- **Thread / block / grid** — a kernel launch is shaped as a *grid* of
  *thread blocks*, each block containing some number of *threads*. Every
  thread runs the kernel body once, over its own slice of the data; the
  block/grid shape is just bookkeeping for "how many threads, grouped how".
- **Warp** — on CUDA GPUs, threads within a block execute in
  hardware-scheduled groups of (typically) 32, called a warp. You rarely
  address a warp directly in aether-level code, but some device machinery
  is built around warp-sized groups.
- **Register-resident / lane** — a value kept in a thread's own fast,
  private on-chip storage (a *register*) rather than in memory. GPUs have
  no CPU-style SIMD registers holding several samples side by side, so
  aether's device "vectorization" instead gives one thread several
  register-resident scalars, one per *lane* — see
  :doc:`multi_device_residency` and the CPU-side analogue in
  :doc:`cpu_simd` for the two different mechanisms this same idea takes.
- **Stream** — an ordered queue of GPU work (copies, kernel launches) that
  lets independent streams' work overlap. aether exposes a stream as
  ``aether::Stream`` (a thin alias over the CUDA stream handle).
- **Pinned memory** — host memory that the CUDA driver has locked in
  physical RAM (never swapped, never moved), which is what makes a
  host↔device copy fast and asynchronous-capable; aether calls a chunk of
  it a ``kDLCUDAHost`` device kind.

The macro surface
------------------

Every qualifier aether uses is spelled ``AETHER_<NAME>()``, declared in
``aether/macros.h``:

- ``AETHER_DEVICEHOST()`` — context-aware: on the device compile
  pass it is a device qualifier, on the host pass a host qualifier. This is
  what nearly every aether function uses, including the plain example
  below.
- ``AETHER_DEVICE()`` / ``AETHER_HOST()`` / ``AETHER_DEVICEANDHOST()`` — the
  narrower, non-context-aware qualifiers for a function that is ONLY ever
  device code, ONLY ever host code, or explicitly both, respectively.
- ``AETHER_KERNEL()`` — a kernel entry point (``__global__``). You write one
  when you need to start new parallel work from the host; a kernel cannot
  itself be called the way an ordinary function is, and it returns
  ``void`` — results come back through memory a ``View`` (see
  :doc:`views_and_items`) points at.
- ``AETHER_GRID_CONSTANT()`` — marks a kernel parameter as read-only and
  shared across the whole launch, letting the compiler cache it more
  cheaply than a per-thread copy.
- ``AETHER_SHARED()`` — declares a block-local scratch variable, visible to
  every thread in the same block only.
- ``AETHER_NOINLINE()`` / ``AETHER_FORCEINLINE()`` — inlining hints, same
  idea as on the host.

In a build with no GPU toolchain at all (``AETHER_CPP_MODE``), every one of
these collapses to a plain host function — the SAME source compiles and
runs as an ordinary CPU library, which is how the runnable example below
executes even though it uses a device-aware qualifier.

Two headers, two audiences
---------------------------

``aether/device.h`` is the device-safe SLICE of the library — expression
templates, ``View``/``Item`` arithmetic — with every host-only piece
(allocation, error throwing, DLPack interop) left out, so it can be parsed
by tools that only understand device code. ``aether/aether.h`` is
everything: it includes ``aether/device.h`` and then adds the host-only
machinery on top. Day to day you just ``#include <aether/aether.h>`` and
get both; the split mostly matters to codegen tooling that consumes only
the device-safe half.

Two more device-side pieces worth knowing the names of: ``aether::cuda::
launchConfig`` (``aether/backend/cuda/Launch.h``) turns a sample count into
a grid/block shape for a launch, and ``aether::DeviceBundle``
(``aether/backend/cuda/DeviceBundle.h``) is the register-resident,
multi-lane value a kernel thread uses to process several samples at once —
the device-side counterpart to the CPU packet type in :doc:`cpu_simd`.

A runnable example
-------------------

The smoke test below defines a function with ``AETHER_DEVICEHOST()`` and
calls it directly from host code — no kernel launch needed, because the
function itself is not a kernel, just an ordinary function that also
happens to be legal to call from device code:

.. aether-example:: tests/test_Smoke.cpp:18-21

.. code-block:: cpp

   AETHER_DEVICEHOST() int answer()
   {
       return 42;
   }

.. aether-example-end::
