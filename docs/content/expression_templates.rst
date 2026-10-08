Expression templates
=====================

Why does ``a + b*s`` not allocate?

Because ``a + b*s`` does not compute a vector at all — it builds a tiny,
cheap object describing the computation (an *expression*), and that
description is only walked, element by element, at the moment you assign
it into a real destination. Nothing is allocated, and nothing is computed,
in between writing the expression and assigning it.

The expression base
---------------------

Every leaf (``aether::Item``, ``aether::View``) and every composite node
this library builds derives from ``aether::Expression`` (``aether/expr/
Expression.h``), a CRTP base that fixes a small, rank-generic protocol any
conforming expression type exposes: an ``element_type`` (the type it
stores), an ``element_extents`` shape, whether it ``isLeaf``, and an
``eval<Is...>(SampleIndex)`` that computes ONE element at ONE sample,
given the static element index ``Is...``. The same protocol works whether
``Es...`` names a 3-vector or a 3x3 matrix — aether has no separate
vector/matrix expression family the way some libraries do; one rank-generic
``Expression`` base covers both.

Operators build nodes, not values
------------------------------------

``a + b``, ``a - b``, ``s * e``, ``e * s`` and ``e / s`` (``aether/expr/
Operations.h``) do not touch ``a``'s or ``b``'s data — they construct a
``Sum`` or ``CWiseScale`` node (``aether/expr/nodes/Arithmetic.h``) that
simply *remembers* its operands. ``(a + b) - s*a`` therefore builds a
``Sum`` of a ``Sum`` and a ``CWiseScale``, three tiny objects deep, still
without reading a single element. This is what lets you write ordinary
arithmetic-looking C++ and get zero intermediate allocation: the compiler
sees straight through the whole chain at compile time.

Where the work actually happens
----------------------------------

An expression only produces numbers when it is assigned into something —
``Item c = a + b;`` or ``view[i] = a + b;``. That assignment is driven by
``aether::detail::RecursiveAssign`` (``aether/expr/Assign.h``), a
compile-time recursive unroll over the destination's ``element_extents``
that calls ``eval<Is...>(SampleIndex)`` on the expression tree once per
element and writes the result into the destination — this is the ONE
place an expression tree is actually walked. Every node computes in its
``working_type`` rather than its ``element_type`` (usually the same type
and free of any extra cost — see :doc:`banded_emulated_real` for the one
case where they differ and why that indirection exists at all).

A runnable example
-------------------

.. aether-example:: tests/test_ExprArithmetic.cpp:31-41

.. code-block:: cpp

   TEST_F(ExprArithmeticTest, SumAddsComponentwise)
   {
       Vec3d a, b;
       for (std::size_t i = 0; i < 3; ++i) {
           a(i) = static_cast<double>(i) + 1.0;
           b(i) = static_cast<double>(i) * 2.0 - 3.0;
       }
       Vec3d c = a + b;
       for (std::size_t i = 0; i < 3; ++i)
           EXPECT_DOUBLE_EQ(c(i), a(i) + b(i));
   }

.. aether-example-end::

``Vec3d c = a + b;`` is the assignment that actually runs the addition,
component by component. :doc:`vector_matrix_cookbook` builds on the exact
same mechanism for cross products, dot products and matrix algebra.
