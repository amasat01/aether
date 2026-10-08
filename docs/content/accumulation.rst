Accumulation
==============

Many producers need to land their contributions in the same per-sample
slot — how do you add them up without a data race, and without forcing
every producer onto its own private copy?

``aether::accum`` answers that with two small, independent pieces: an
**accumulation plane** for "many writers, one running total per sample"
(:doc:`vector_matrix_cookbook`-shaped terms summed into one vector), and
an **append counter** for "many writers, an unknown number of them, one
shared destination array" — both usable from a CUDA kernel or an OpenMP
loop through the exact same call spelling.

The problem: one thread per contribution, not one thread per sample
------------------------------------------------------------------------

Most of this library's expression templates assume one thread (or OMP
task) owns a sample outright and writes it once. A force-model
accumulator breaks that assumption on purpose: several independent terms
(gravity, drag, solar pressure, ...) are often cheaper to compute as
separate kernels, each contributing to the *same* per-sample acceleration.
Two threads adding to the same ``double`` with a plain ``+=`` is a race,
and serializing every contribution defeats the reason the terms were
split apart in the first place. ``aether::AccumPlane`` exists to make that
``+=`` safe without the caller reasoning about launch order.

Accumulation planes
----------------------

``aether::AccumPlane<Real, VD, Policy>`` (``aether/accum/AccumPlane.h``,
alias of ``aether::accum::Plane``) is an owning, per-sample accumulator of
``VD`` lanes of ``Real`` (``double`` is the only scalar currently
supported), constructed once from a sample count. ``plane.hostView()``/
``plane.deviceView()`` hand a kernel the non-owning
``aether::AccumPlaneView`` it actually accumulates into — the only type a
device function ever touches — and ``zeroHost()``/``zeroDeviceAsync()`` on
the owning ``Plane`` clear the whole allocation before a step starts.

From inside a kernel or an OMP loop, ``view[i]`` returns a
``aether::accum::LaneView`` — a thin, stateless handle over one sample,
cheap to create and discard on every write — and ``view[i] +=
gravityTerm`` (equivalently, the named ``view[i].accumulate(term)``) folds
a ``VD``-component expression term into that sample's lanes, in ascending
component order so a host run and a device run accumulate the identical
bits. What the ``+=`` actually does underneath depends on the plane's
``Policy``:

- ``aether::accum::CompensatedAtomic`` (the default for ``double``) pairs
  a hardware ``atomicAdd`` with a companion residual lane, combined
  through ``aether::accum::atomic::compensatedSum`` — a Knuth two-sum, so
  the delivered value (``value + comp``) does not depend on the order
  concurrent producers happened to run in. Concurrency-safe by
  construction.
- ``aether::accum::Serialized`` is a direct ``+=`` with no atomic and no
  companion lane: cheaper, but the caller owes serialization of every
  write to one sample (a reduction already staged into one thread per
  sample, say).

Both policies read back through ``LaneView::delivered()`` / ``deliver()``,
which fold the lanes into a register-resident ``aether::Item<Real, VD>``
or write it straight into a destination expression. ``zero()`` — on the
view, per sample, or ``zeroHost()``/``zeroDeviceAsync()`` on the owning
``Plane`` for the whole allocation — always clears both lanes
unconditionally: there is no "first producer" flag to get wrong on a
replayed step.

Variable-count producers: append counters
----------------------------------------------

A plane fits "the destination count is fixed, the contributor count
varies." The complementary shape is "the DESTINATION count is unknown
ahead of time" — detecting candidate events, say, where only some samples
produce one. ``aether::accum::AppendCounter`` (``aether/accum/
AppendCounter.h``) hands out slots, one atomic ``fetch_add`` per claim,
over an ``Array``'s reserved ``spare()`` capacity: a kernel calls
``view.tryAppend()`` on the ``aether::accum::AppendCounterView``
(``counter.deviceView()``) to claim the next slot, writes its candidate
there, and the host later calls ``counter.commit(arr)`` to adopt the
claimed slots into ``arr`` and reset the counter for the next step.

``tryAppend()`` never returns a slot ``>= capacity``; past that point it
sets an overflow flag and returns the ``aether::accum::npos`` sentinel, so
every accepted write is safe even when more producers claim a slot than
the array can hold. ``commit()`` is the one-line pattern that closes the
loop: read the raw count back, clamp it to ``capacity()``, adopt that many
newly-written samples into the ``Array`` via its own ``commit()``, and
reset the counter for the next step — never throwing itself, so a
rejected overflow does not discard the samples that did fit.

A runnable example
-------------------

.. aether-example:: tests/test_AppendCounter.cpp:26-36

.. code-block:: cpp

   TEST_F(AppendCounterTest, SequentialTryAppendFillsSlotsInOrder)
   {
       AppendCounter counter(8);
       auto view = counter.deviceView();
       for (std::size_t k = 0; k < 8; ++k) {
           const aether::offset_t slot = view.tryAppend();
           EXPECT_EQ(slot, static_cast<aether::offset_t>(k));
       }
       EXPECT_EQ(counter.rawCount(), 8u);
       EXPECT_FALSE(counter.overflowed());
   }

.. aether-example-end::

This runs on the CPU host path (``AETHER_CPP_MODE``, one OMP-visible
atomic per claim); ``tests/test_AppendCounter.cu`` exercises the identical
contract from a CUDA kernel with thousands of concurrent claimants,
checking that every slot in ``[0, capacity)`` is handed out exactly once.
Both ``AccumPlane`` and ``AppendCounter`` share that same promise: the
same call spelling, the same result, whether the producers are OpenMP
tasks or CUDA threads.
