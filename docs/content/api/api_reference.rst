API reference
=============

This section provides the complete Doxygen-generated C++ API for aether,
rendered through Breathe and split by module. For the Python side —
``aether_dsc``, the sealed header payload — see :doc:`../python/aether_dsc`.

.. note::

   The C++ API reference is generated from Doxygen XML. ``docs/Makefile``'s
   ``html``/``strict``/``linkcheck`` targets all run ``make doxygen`` first;
   invoke it directly if you only need ``_doxybuild/xml`` refreshed.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Contents
   * - :doc:`api_core`
     - ``DType``, ``Device``, ``Error``, ``SampleIndex``, ``BundleIndex``,
       ``extents``, ``layout_right``, ``layout_stride`` — the scalar,
       device and shape vocabulary every other module builds on.
   * - :doc:`api_memory`
     - ``Chunk`` (owning bytes), ``View``, ``Item``, ``RuntimeView``,
       ``TableHandle``, ``Array`` — see :doc:`../views_and_items` and
       :doc:`../memory_ownership`.
   * - :doc:`api_expr`
     - ``Expression`` (the CRTP base every leaf/node derives from) and
       ``aether::eval`` (the dual-mode runtime evaluator). See
       :doc:`../expression_templates`.
   * - :doc:`api_numerics`
     - ``aether::math`` (device-legal scalar dispatch), ``WorkingType``,
       the ``aether::banded`` module (``Band``/``BandCell8``/
       ``BandedReal``), and ``aether::accum`` (compensated/atomic
       accumulation policies). See :doc:`../banded_emulated_real`.
   * - :doc:`api_backend_interop`
     - ``aether::simd::Packet``/``PacketMask``/``PacketTraits`` (CPU SIMD),
       ``DeviceBundle`` (CUDA multi-lane carrier), ``aether::interop``
       (DLPack bridge), ``PartitionSpec``/``ReplicaSet``/
       ``StreamTransport`` (multi-device residency). See
       :doc:`../cpu_simd`, :doc:`../multi_device_residency` and
       :doc:`../interop`.

.. toctree::
   :maxdepth: 1
   :hidden:

   api_core
   api_memory
   api_expr
   api_numerics
   api_backend_interop
