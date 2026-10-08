API reference — expression templates
=======================================

Every leaf and composite node in aether derives from ``Expression``; the
individual node types (``Sum``, ``CWiseScale``, ``Cross``, ``QuatRotate``,
...) live in ``aether::detail`` and are reached through the operators
documented in :doc:`../expression_templates`, not cited individually here.
``aether::eval`` is the dual-mode runtime evaluator that walks a tree at
assignment time.

``aether::Expression``
-------------------------

.. doxygenclass:: aether::Expression
   :project: aether
   :members:

Namespace ``aether::eval``
------------------------------

.. doxygennamespace:: aether::eval
   :project: aether
   :members:
