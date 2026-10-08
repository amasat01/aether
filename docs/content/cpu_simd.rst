CPU SIMD backend
==================

Does aether vectorize on CPU too?

Yes: independent of any GPU, aether has its own CPU SIMD packet layer,
``aether::simd::Packet<DataT, Width>``, which wraps the machine's real
vector registers (AVX-512, AVX2 or SSE2, whichever the build targets) and
falls back to an ordinary scalar loop when none apply — the same source
compiles either way.

``Packet``: one register, several lanes
--------------------------------------------

``aether::simd::Packet`` (``aether/backend/cpu/simd/Packet.h``) is
specialized per data type and register width — 8-wide ``double``/16-wide
``float`` on AVX-512, 4-wide/8-wide on AVX2, 2-wide/4-wide on SSE2, and a
1-wide scalar fallback everywhere else — each backed by the matching
platform intrinsic type. Every specialization offers the same small
surface: ``zero()``/``broadcast(v)``/``load()``/``maskLoad()``,
``store()``/``maskStore()``, the arithmetic operators plus ``fmadd``, and
comparison-to-mask. ``aether::simd::PreferredWidth<T>`` names the widest
packet the current build target actually supports for ``T``, so code that
wants "the best available width" rarely hard-codes a number.
``aether::simd::PacketMask`` carries the active-lane mask a partially-full
TAIL packet (a sample count not evenly divisible by the packet width)
needs, so a loop's last, partial packet is handled the same way as every
full one, not as a special case you write by hand.

Driving a loop: ``packetFor``
----------------------------------

``aether/backend/cpu/Tiled.h`` supplies the loop drivers built on top of
``Packet``: ``aether::packetFor``/``aether::packetFlatFor`` walk a
``[begin, end)`` range packet by packet — context-aware, so the same call
becomes an ``omp for nowait`` inside an existing parallel region, a fresh
``omp parallel for`` when asked explicitly, or a plain serial loop
otherwise. ``aether::packetCapture`` materializes an expression into a
register-resident packet item; ``aether::packetEval``/``aether::
packetEvalParallel`` assign an expression into a destination one packet at
a time. ``aether::packetBatchedFor``/``aether::optimalTileSize`` add an
L2-cache-aware tiled form on top of the same flat loop, sizing each tile
from the destination's L2 cache footprint rather than a fixed constant.

This is the CPU-side counterpart to the device-side, register-resident
bundles :doc:`host_vs_device` introduces — different hardware, same idea:
process several samples per unit of work, held in fast, on-chip storage
rather than read one at a time from memory.

A runnable example
-------------------

Broadcasting one scalar across every lane of a packet at the build's
preferred ``double`` width:

.. aether-example:: tests/test_PacketOps.cpp:53-61

.. code-block:: cpp

   TEST_F(PacketTest, BroadcastFillsEveryLane)
   {
       constexpr std::size_t W = PreferredWidth<double>;
       auto p = Packet<double, W>::broadcast(3.5);
       double out[W];
       Packet<double, W>::store(out, p);
       for (std::size_t k = 0; k < W; ++k)
           EXPECT_EQ(out[k], 3.5);
   }

.. aether-example-end::

Every one of the ``W`` lanes ``store()`` writes back out holds the same
``3.5`` — ``W`` itself is whatever this build's ``PreferredWidth<double>``
resolves to, 1 on a build with no wider ISA available.
