API reference — backend and interop
======================================

The device-residency surface: CPU SIMD packets, the CUDA multi-lane
carrier, the DLPack interop bridge, and multi-device/multi-stream
residency. See :doc:`../cpu_simd`, :doc:`../multi_device_residency` and
:doc:`../interop`.

CPU SIMD (``aether::simd``)
-------------------------------

.. doxygenstruct:: aether::simd::Packet
   :project: aether
   :members:

.. doxygenstruct:: aether::simd::PacketMask
   :project: aether
   :members:

.. doxygenstruct:: aether::simd::PacketTraits
   :project: aether
   :members:

``aether::DeviceBundle`` (CUDA)
-----------------------------------

.. doxygenclass:: aether::DeviceBundle
   :project: aether
   :members:

Namespace ``aether::interop`` (DLPack bridge)
--------------------------------------------------

.. doxygennamespace:: aether::interop
   :project: aether
   :members:

Multi-device residency
--------------------------

.. doxygenstruct:: aether::PartitionSpec
   :project: aether
   :members:

.. doxygenclass:: aether::PartitionedArray
   :project: aether
   :members:

.. doxygenclass:: aether::ReplicaSet
   :project: aether
   :members:

.. doxygenclass:: aether::StreamTransport
   :project: aether
   :members:
