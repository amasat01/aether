Multi-device residency
========================

How do I keep a replica per GPU, or partition data across several?

Use ``aether::ReplicaSet`` when you want the SAME data copied onto every
device in a list (``rs.broadcast(src)``); use ``aether::partition_view``
instead when you want to slice ONE view's sample dimension into several
non-overlapping blocks, one per device or worker, without allocating or
copying anything at all.

Replicating: ``ReplicaSet``
-------------------------------

``aether::ReplicaSet`` (``aether/residency/Replica.h``) holds one
``aether::Chunk`` per ``Device`` in an explicit list you pass at
construction — there is no automatic device discovery or topology probing
in this v1 surface. ``rs.chunk(i)`` accesses the ``i``-th replica's
``Chunk`` directly, and ``rs.broadcast(source)`` copies one source
``Chunk``'s bytes into every replica, bit-exactly, using
``aether::copy``/``copyAsync`` under the hood (see
:doc:`memory_ownership` for the legal device-pair rules that copy obeys).

Partitioning: ``partition_view``
-------------------------------------

``aether::PartitionSpec`` / ``aether::padded_block_samples`` /
``aether::partition_view`` (``aether/residency/Partition.h``) instead slice
an EXISTING batched ``View``'s trailing sample dimension into ``parts``
blocks, each block's start rounded up to a ``pad_to``-sample granularity
(32 by default) — "real extent, padded stride": every block but possibly
the last has exactly the padded number of real samples, while block
BOUNDARIES still fall at regular, padded intervals regardless of how full
the last one actually is. No allocation and no copy happens here either —
``partition_view`` slices the original view's own memory in place.

The transport seam
-----------------------

``aether::transport`` (``aether/residency/Transport.h``) is a small C++20
concept — a type offering a blocking ``copy`` and (on a CUDA build) an
async ``copyAsync`` between two ``Chunk``\ s — that ``ReplicaSet``-style
data movement is built against. ``aether::StreamTransport`` is the one
implementation this library ships (a thin wrapper over
``aether::copy``/``copyAsync``); multi-node transports (NCCL, MPI, a
peer-to-peer fabric) are future consumers of the SAME concept, not
something this page's surface provides today.

A runnable example
-------------------

Three CPU-kind devices standing in for three distinct devices, each
getting its own 256-byte ``Chunk``:

.. aether-example:: tests/test_Replica.cpp:25-42

.. code-block:: cpp

   TEST_F(ReplicaTest, ConstructsOneChunkPerDeviceWithTheRequestedSize)
   {
       // Distinct DEVICE IDS stand in for distinct devices — Chunk::allocate
       // dispatches on device KIND, not id, so this exercises the "N chunks on
       // N devices" contract entirely within an AETHER_CPP_MODE (kDLCPU-only)
       // build.
       std::vector<aether::Device> devices{
           aether::Device(kDLCPU, 0), aether::Device(kDLCPU, 1), aether::Device(kDLCPU, 2)
       };
       aether::ReplicaSet rs(devices, 256);
       EXPECT_EQ(rs.size(), 3u);
       for (std::size_t i = 0; i < devices.size(); ++i) {
           EXPECT_EQ(rs.chunk(i).size(), 256u);
           EXPECT_EQ(rs.chunk(i).device(), devices[i]);
           EXPECT_NE(rs.chunk(i).data(), nullptr);
           EXPECT_TRUE(rs.chunk(i).owns());
       }
   }

.. aether-example-end::

On an actual multi-GPU build, ``aether::Device(kDLCUDA, 0)``,
``aether::Device(kDLCUDA, 1)``, ... would name the real devices; the CPU
build shown here exercises the identical code path against distinct
device IDs of the same kind.
