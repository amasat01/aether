API reference — core
=====================

Scalar/device vocabulary and the shape machinery everything else indexes
through: data types, devices, errors, sample indexing and the layout
mappings a ``View`` (:doc:`api_memory`) is built from.

``aether::DType``
------------------

.. doxygenstruct:: aether::DType
   :project: aether
   :members:

``aether::Device``
--------------------

.. doxygenstruct:: aether::Device
   :project: aether
   :members:

``aether::Error``
-------------------

.. doxygenclass:: aether::Error
   :project: aether
   :members:

``aether::SampleIndex`` / ``aether::BundleIndex``
----------------------------------------------------

.. doxygenstruct:: aether::SampleIndex
   :project: aether
   :members:

.. doxygenstruct:: aether::BundleIndex
   :project: aether
   :members:

Shape and layout
------------------

.. doxygenclass:: aether::extents
   :project: aether
   :members: rank, rank_dynamic, static_extent, extent, extentWide, extents

.. doxygenstruct:: aether::layout_right
   :project: aether
   :members:

.. doxygenstruct:: aether::layout_stride
   :project: aether
   :members:
