API reference — memory
========================

The owning and non-owning containers: ``Chunk`` is raw owned bytes,
``View``/``Item``/``RuntimeView``/``TableHandle`` are typed, zero-copy
handles over memory, and ``Array`` is the owning convenience type built on
top of them. See :doc:`../views_and_items` and :doc:`../memory_ownership`.

``aether::Chunk``
-------------------

.. doxygenclass:: aether::Chunk
   :project: aether
   :members:

``aether::View``
-------------------

.. doxygenclass:: aether::View
   :project: aether
   :members:

``aether::Item``
-------------------

.. doxygenclass:: aether::Item
   :project: aether
   :members: element_type, element_extents, isLeaf, Rank, Size, operator=

``aether::RuntimeView``
--------------------------

.. doxygenstruct:: aether::RuntimeView
   :project: aether
   :members:

``aether::TableHandle``
--------------------------

.. doxygenclass:: aether::TableHandle
   :project: aether
   :members:

``aether::Array``
--------------------

.. doxygenclass:: aether::Array
   :project: aether
   :members:
