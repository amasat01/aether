Quaternions
===========

How do I rotate a vector?

Build (or read) a unit quaternion as a ``Vec4d`` in ``(w, x, y, z)`` order,
and call ``.quatRotate(v)`` on it with the ``Vec3d`` you want to rotate —
one method call, no separate rotation-matrix construction step.

The quaternion surface
--------------------------

``aether/expr/nodes/Quaternion.h`` backs a small set of member functions
declared on ``aether::Expression`` itself (same pattern as
:doc:`vector_matrix_cookbook`'s geometric/matrix surface), all operating
on 4-component expressions in ``(w, x, y, z)`` order:

- ``q.quatMul(r)`` — the Hamilton product of two quaternions.
- ``q.quatConj()`` — the conjugate (negates the vector part ``x, y, z``,
  leaves ``w`` unchanged).
- ``q.quatReciprocal()`` — ``quatConj()`` scaled by ``rSquaredNorm()``
  (:doc:`vector_matrix_cookbook`'s reciprocal-norm helper); for a UNIT
  quaternion this equals the conjugate, but this form is correct for a
  non-unit one too.
- ``q.quatRotate(v)`` — rotates the ``Vec3d`` ``v`` by the quaternion
  ``q``: internally the sandwich product ``q * (0,v) * q⁻¹``, computed
  directly rather than via three separate quaternion multiplications.
- ``q.asPureQuaternion()`` / ``q.asBack3DVector()`` — convert a ``Vec3d``
  to a "pure" (zero real part) ``Vec4d`` and back, the two halves of the
  sandwich product above if you ever need to write it out by hand.

Because a rotation sandwich reassociates several multiplications and
additions, two mathematically equal ways of writing the same rotation can
legitimately disagree in the last few bits — this library's own test suite
compares quaternion results with a small absolute+relative tolerance
rather than expecting bit-for-bit equality, and the same caution applies
to any code you write against this surface.

A runnable example
-------------------

A 90° rotation about the ``z`` axis, ``q = (cos 45°, 0, 0, sin 45°)``,
applied to ``v = (1, 0, 0)``, which must land on ``(0, 1, 0)``:

.. aether-example:: tests/test_ExprQuaternion.cpp:146-149

.. code-block:: cpp

       Vec3d res = q.quatRotate(v);
       EXPECT_NEAR(res(0), 0.0, 1e-14);
       EXPECT_NEAR(res(1), 1.0, 1e-14);
       EXPECT_NEAR(res(2), 0.0, 1e-14);

.. aether-example-end::

Note the ``EXPECT_NEAR`` with an explicit tolerance rather than an exact
comparison, for exactly the reassociation reason above.
