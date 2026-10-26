Numerics contract
=================

When aether computes a sum, a dot product or a division, which exact
floating-point operations does it perform, and in what order?

This page states them. Floating-point addition is not associative, so a
different order is a different (merely close) answer. Every target that
computes against aether, whether the C++ templates or a generated language
core, follows the rules below.

Folds
-----

A fold is any reduction over the components of an item: ``sum``, ``dot``,
``squaredNorm``, ``norm``, ``maxNorm``. It is sequential and right-nested,
with the highest index outermost and no zero seed::

   e[N-1] + (e[N-2] + (... + e[0]))

so a 3-vector ``dot`` is ``(a2*b2) + ((a1*b1) + (a0*b0))``. The static forms
(``aether/expr/Reduce.h``) and the rolled-loop forms in ``aether::rt``
(``aether/expr/Rt.h``) use this same order. A matrix-vector row, a matrix
product entry and the Frobenius ``matDot`` (row-major) are folds of
``a_k*b_k + acc`` in the same order. Folds run in the working carrier of the
element type; for ``float`` and ``double`` that is the type itself.

There is no pairwise or blocked fold inside an item. The one tree in aether
is the 32-slot pair tree of the compensated reductions below.

Division by a scalar
--------------------

``expr / s`` computes ``r = 1/s`` once and multiplies every component by it
(``operator/`` in ``aether/expr/Operations.h``). It is not a correctly
rounded division per component: ``x * (1/s)`` can differ from ``x / s`` in the
last bit, and a tiny ``s`` whose reciprocal overflows gives ``inf * x``.

Per-lane linear algebra
-----------------------

``aether::linalg`` (``aether/linalg/``) factors and solves one small matrix
per lane. A factorisation is not a fold of an item, so it states its own
order:

* Every sum runs in ascending index. LU reduces each entry by one finished
  dot product (``acc += a_ik*a_kj``, then ``a_ij -= acc``), the order LAPACK's
  ``getrf`` kernels use; the triangular solves and Cholesky subtract term by
  term (``s -= l*x``).
* A division by a diagonal entry is a correctly rounded division, except LU's
  multipliers, which multiply by one reciprocal of the pivot.
* Nothing is compensated inside a factorisation.
* The pivot is the first entry of largest magnitude in its column; a zero
  pivot is the singular test. Cholesky tests ``!(d > 0)``.
* ``cholesky`` and ``eigh`` read ``0.5*a_ij + 0.5*a_ji``, so a symmetric
  input is read unchanged. ``eigh`` runs a fixed number of Jacobi sweeps
  (``eighSweeps``) in fixed order.

The same bits follow wherever the translation unit does not contract
products, as above; on the device the values agree to a few ulps and ``info``
agrees exactly.

Other fixed spellings
---------------------

* ``neg`` is ``x * -1``.
* ``norm`` is ``sqrt`` of the squared norm. There is no ``hypot``.
* Nothing promotes. A ``float`` expression stays in ``float`` and uses the
  ``f``-suffixed libm; a ``double`` expression stays in ``double``.

Compensated sums
----------------

Plain folds are not compensated. Compensation is used where many
contributions land in one slot or one value is reduced across lanes
(``aether/accum/atomic.h``):

* **Two-sum.** ``twoSumErr_(a, b, s)`` returns the exact residual
  ``(a + b) - s`` of a rounded sum, for any order and signs of ``a`` and
  ``b``. Its inputs pass an optimisation barrier, so a product feeding them
  is rounded first and never fused into the subtractions.
* **Accumulation plane.** ``AccumPlane`` with the default
  ``CompensatedAtomic`` policy adds each term to a value slot with an atomic
  add and adds the two-sum residual to a companion slot. The delivered value
  is ``value + comp``. The components of a term are folded in ascending
  component order, host and device alike (see :doc:`accumulation`).
* **Neumaier step and pair combine.** ``neumaierAdd_`` adds one term to a
  ``(hi, comp)`` pair. ``pairCombine_`` merges two pairs: a two-sum of the
  ``hi`` parts, with the ``comp`` parts added; swapping its operands gives the
  same bits.
* **The 32-slot tree.** ``reduce32Host_`` combines 32 per-lane pairs with
  ``pairCombine_`` in rounds of xor 16, 8, 4, 2, 1. The device
  ``butterfly32_`` leaves each reduced column bit-equal to it; lanes past a
  ragged end carry ``+0`` pairs.

Non-finite values
-----------------

* The compensated helpers drop the correction once the high part is not
  finite, so the pair then follows IEEE through ``hi``: a lone ``+Inf`` gives
  ``+Inf``, ``+Inf`` with ``-Inf`` gives NaN, and NaN stays NaN.
* ``isFinite`` of an expression is true only if every component is finite.
* ``maxNorm`` folds ``a > b ? a : b`` over the absolute values, so which
  operand wins against a NaN follows that expression's operand order.
* ``math::fmax`` and ``fmin`` have C ``fmax``/``fmin`` semantics: a NaN
  operand is ignored when the other is not NaN.
* ``math::sign`` of a NaN is ``0``.

Where bits are guaranteed
-------------------------

The guarantee is checked, not assumed. Against a reference set of inputs and
expected results (below), the plain-C core gives the same bits as the C++
templates on the host, under two conditions:

* the C translation unit is compiled with ``-ffp-contract=off``. GNU-mode GCC
  defaults to ``-ffp-contract=fast``, which fuses ``a*b - c*d`` (the shape of
  ``cross``, ``quat_mul`` and every row dot) into an FMA and changes the low
  bits;
* the templates take their default ``std::`` host route: no
  ``AETHER_HOST_VECTOR_MATH`` (which reroutes some functions to
  faithfully-rounded packet implementations) and no fast-math.

One result is outside the guarantee: the sign of a zero from ``min`` or
``max`` of ``+0`` and ``-0``. C leaves the sign of ``fmin(+0, -0)`` and
``fmax(+0, -0)`` unspecified, and compilers differ, so the check accepts
either zero for those two kinds. A NaN is accepted for a NaN; every other
result is a byte comparison.

Device code is not part of the bit guarantee. nvcc contracts a product and
a sum into an FMA per statement, which the host build with
``-ffp-contract=off`` does not. A core whose target has no ``float64``, or
different fused multiply-add rules, states a ULP band against the reference
set instead of bit identity.

The key cases for the plain-C core:

================  ===============================================================
Operation         Generated arithmetic
================  ===============================================================
sum, dot          right-nested, no seed: ``(a2*b2) + ((a1*b1) + (a0*b0))``
norm              ``sqrt(ssq)``, where ``ssq`` is the right-nested sum of squares
div by scalar     ``r = 1/s`` once, then ``r * x_k`` per component
neg               ``x * -1``
quaternion        a ``vec4`` laid out ``(w, x, y, z)``; a 3-vector is a ``vec3``
================  ===============================================================

Language cores
--------------

A target whose language cannot compile aether's C++23 templates needs a
language core: the contract above written out in that language. A core is
generated, so one source states each rule, and it is checked against the same
reference set as every other core. :doc:`c_core` is the first.

The reference set is the committed corpus under ``tests/c_core/`` (the
umbrella ``aether_c_corpus.h`` and the md5-fenced parts in ``corpus/``). The
``CCore`` gtest suite (``tests/test_CCore.cpp``, ``tests/test_CCore.cu``)
checks every generated function against the template expression it mirrors.
A future core (C++17 for older toolchains and MSL, OpenCL C) adds its
generator next to ``c_core`` and runs the same check.
