Vector and matrix algebra cookbook
=====================================

How do I write cross/dot/matmul without CUDA math calls?

You call the ordinary-looking member function directly on the vector or
matrix expression — ``a.cross(b)``, ``a.dot(b)``, ``M * v``, ``a.norm()``
— and the library builds the expression tree for you; there is no CUDA
math intrinsic to reach for anywhere in this surface, on host or device.

One rank-generic operator set
--------------------------------

aether has no separate "vector algebra" and "matrix algebra" module: every
member below is declared once, on ``aether::Expression`` itself
(``aether/expr/Expression.h``), and works on ANY conforming expression —
a ``Vec3d``, a ``Mat33d``, a batched sample read off a ``View`` — because
the whole surface is generic over ``element_extents``'s rank rather than
hard-coded to "3-vector" or "3x3 matrix". Writing ``a + s*b`` on a
``Mat33d`` uses the exact same ``operator+``/``operator*`` as
:doc:`expression_templates` showed for a ``Vec3d``.

Vectors
---------

- ``a.cross(b)`` — cross product (rank-1, 3-component operands only;
  ``aether/expr/nodes/Geometric.h``).
- ``a.dot(b)`` — inner product, returning a plain scalar (``element_type``,
  not an expression) rather than something you assign further.
- ``a.norm()`` / ``a.squaredNorm()`` / ``a.cubedNorm()`` and their
  reciprocal counterparts ``a.rNorm()`` / ``a.rSquaredNorm()`` /
  ``a.rCubedNorm()`` — the reciprocal forms exist because a division is
  usually more expensive than a multiply, so a formula that needs ``1/norm``
  asks for it directly instead of computing ``norm()`` and dividing.
- ``a.maxNorm()`` — the largest-magnitude component; ``a.sum()`` — the
  component sum.
- ``a.unitVector()`` — a unit-length copy of ``a`` (no in-place
  ``normalize()``; this is always a new expression).
- ``a.segment<Off,Len>()`` / ``a.head<N>()`` / ``a.tail<N>()`` — a
  read-only, lazy view of ``Len`` consecutive components starting at
  ``Off`` (``aether/expr/nodes/Structural.h``). These are READ views only
  in the current surface — there is no ``a.segment<Off,Len>() = expr``
  write-back yet.

Matrices
----------

``aether/expr/nodes/Product.h`` adds the matrix-shaped operations:

- ``M * v`` — matrix-vector product (``MatVec``); ``A * B`` —
  matrix-matrix product (``MatMat``); ``outer(u, v)`` — outer product
  (``Outer``), all reached through natural call-site syntax, not a named
  method.
- ``a.cwiseMul(b)`` — component-wise (Hadamard) product, rank-generic like
  every arithmetic operator above.
- ``a.matDot(b)`` — the Frobenius inner product of two same-shaped
  matrices (``sum`` of the elementwise product), the matrix analogue of
  ``dot()``.
- ``a.transpose()``, ``a.row<R>()``, ``a.col<C>()``,
  ``a.block<R0,C0,BR,BC>()`` — the same read-only structural slicing
  ``segment``/``head``/``tail`` offer for vectors, extended to matrices
  (``aether/expr/nodes/Structural.h``).
- ``a.trace()``, ``a.det()``, ``a.inverse()`` — small closed-form matrix
  algebra (``aether/expr/nodes/Inverse.h``), deliberately limited to 2x2
  and 3x3: there is no general LU decomposition or linear solve in this
  surface.

An assignment like ``dest = M * v`` also gets an internal shortcut: rather
than recomputing the whole dot-product chain once per output component,
the reused operand (``v`` for ``M * v``, both matrices for ``A * B``) is
materialized into a register-resident ``Item`` once and every output
component reads from that cached copy — a structural detail you never
have to spell out yourself, since it happens automatically at the
assignment step :doc:`expression_templates` describes.

A runnable example
-------------------

The identity ``y × z = x``, ``z × x = y``, ``x × y = z`` for the standard
basis vectors, checked directly against ``.cross()``:

.. aether-example:: tests/test_ExprGeometric.cpp:34-39

.. code-block:: cpp

       Vec3d yCrossZ = y.cross(z);
       Vec3d zCrossX = z.cross(x);
       Vec3d xCrossY = x.cross(y);
       EXPECT_DOUBLE_EQ(x(0), yCrossZ(0));
       EXPECT_DOUBLE_EQ(y(1), zCrossX(1));
       EXPECT_DOUBLE_EQ(z(2), xCrossY(2));

.. aether-example-end::

Here ``x``, ``y`` and ``z`` are ``Vec3d`` values standing for the unit
basis vectors ``(1,0,0)``, ``(0,1,0)`` and ``(0,0,1)``. :doc:`quaternions`
picks up where this page leaves off, for rotating a vector rather than
combining two of them.
