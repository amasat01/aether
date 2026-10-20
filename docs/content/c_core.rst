Plain-C reference core
======================

Can I get aether's arithmetic as standalone C that gives the same bits?

Yes: ``aether_dsc.c_core`` is a pure-Python generator of C99 functions that
reproduce, bit for bit, what aether's expression templates evaluate on the
host. A consumer such as hawk requests the functions it needs and inlines
the generated text into a standalone C file, with no aether header and no C++
compiler involved.

What it generates
-----------------

Each function is a ``static inline`` C99 function over small value structs.
The generator covers vectors (``vec_dot``, ``vec_norm``, ``vec_sum``,
``vec_cross``, the componentwise operators and math functions), matrices
(their elementwise mirrors plus ``mat_mv``, ``mat_mm``, ``mat_transpose`` and
``vec_outer``), quaternions (``quat_mul``, ``quat_conj``, ``quat_recip``,
``quat_rotate`` and the pure/vec3 conversions) and four scalar formulas
(``rsqrt``, ``sign``, ``clip``, ``remainder``). Every kind exists at both
real widths, ``f64`` (``double``) and ``f32`` (``float``); ``f32`` text uses
the ``f``-suffixed libm and never promotes to ``double``. ``c_core.kinds()``
lists the kinds. The format version tag is ``aether-c/1``
(``c_core.CORE_VERSION``).

The contract
------------

The generated functions follow aether's evaluation order exactly, because
floating-point addition is not associative and any other order changes the
low bits:

* Folds (dot, norm, sum) are right-nested, highest index outermost, with no
  zero seed: ``e[N-1] + (e[N-2] + (... + e[0]))``.
* Dividing by a scalar computes the reciprocal ``1/s`` once, then multiplies
  every component by it.
* ``neg`` is ``x * -1``, and ``norm`` is ``sqrt`` of the sum of squares (no
  ``hypot``).
* Nothing promotes: an ``f32`` function only handles ``f32`` values.

The key cases:

================  ===============================================================
Operation         Generated arithmetic
================  ===============================================================
sum, dot          right-nested, no seed: ``(a2*b2) + ((a1*b1) + (a0*b0))``
norm              ``sqrt(ssq)``, where ``ssq`` is the right-nested sum of squares
div by scalar     ``r = 1/s`` once, then ``r * x_k`` per component
neg               ``x * -1``
quaternion        a ``vec4`` laid out ``(w, x, y, z)``; a 3-vector is a ``vec3``
================  ===============================================================

Build conditions
~~~~~~~~~~~~~~~~

The bit-identity holds under two conditions:

* The C translation unit must be compiled with ``-ffp-contract=off``.
  GNU-mode GCC defaults to ``-ffp-contract=fast``, which fuses ``a*b - c*d``
  (the shape of ``cross``, ``quat_mul`` and every row dot) into an FMA and
  changes the low bits.
* Identity is to the templates' default ``std::`` host route: no
  ``AETHER_HOST_VECTOR_MATH`` (which reroutes some functions to
  faithfully-rounded packet implementations) and no fast-math.

Consuming it from Python
------------------------

The module uses only the standard library. Ask for a function by kind, shape
and width, then emit the C text for it and everything it depends on:

.. code-block:: python

   from aether_dsc import c_core as C

   f = C.function("vec_dot", shape=(3,), width="f64")
   print(C.emit([f.name]))

``emit`` writes the width typedef and struct types first, then the functions.
The typedefs are guarded (``#ifndef AETHER_C_F64_VEC3_T``), but the functions
are bare ``static inline`` and are emitted once per ``emit()`` call, so a
consumer concatenating several calls must de-duplicate by name (or make one
call with every name). ``f.deps`` lists the typedef and struct names a
function needs, and ``f.libm`` the libm functions it calls, so a consumer can
resolve dependencies and link ``libm`` only when required.

How it is pinned
----------------

Three checks keep the generated text and the templates from drifting apart:

* The ``CCore`` gtest suite (``tests/test_CCore.cpp`` and ``.cu``) compiles
  the corpus header and asserts that every generated function is
  ``memcmp``-equal to the corresponding template expression.
* ``tests/c_core/check_c99.sh`` compiles the corpus under a strict C99
  gate: the umbrella header, and every part on its own.
* ``tools/c_core/mint_golden.py --check`` verifies that the committed corpus
  (``tests/c_core/aether_c_corpus.h`` and the md5-fenced parts under
  ``tests/c_core/corpus/``) is exactly what the generator produces, and that
  no stale part is left behind. CI runs the last two before building; the ``CCore``
  suite runs in CI's normal build-and-test job.

Regenerating the corpus
-----------------------

The corpus is committed as one small header per width and shape
(``corpus/f64/vec3.h``, ``corpus/f32/mat2x3.h``, ...), with an umbrella
``aether_c_corpus.h`` that includes them all. It changes only when the core
does. After changing the generator, re-mint the corpus and commit it with the
change::

   PYTHONPATH=dsc python3 tools/c_core/mint_golden.py

Minting is refused under ``$CI``: a gate that regenerates its own expected
value proves nothing.
