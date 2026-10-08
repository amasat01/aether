Banded/emulated real arithmetic
==================================

What is ``BandedReal``, and when do I need it instead of native FP64?

``aether::banded::BandedReal`` is an emulated double-precision value — 53
certified significand bits assembled from three ordinary ``float`` limbs
instead of one hardware ``double`` — for target hardware that has no FP64
units at all, or only a crippled one; you build one explicitly from a
``double``, compute with it through a certified operator set, and read a
``double`` back out explicitly, never through an implicit conversion.

Why this module exists
--------------------------

Some GPUs this library targets have no FP64 hardware, or run it at a
steep fraction of the FP32 rate. ``aether::banded::Band`` delivers
double-class accuracy out of FP32 arithmetic and a handful of integer
operations, so a kernel that needs it can have it on hardware that cannot
spell ``double`` quickly. That is a portability statement, not a
performance one — this page makes no speed claim (a texture-related
benchmark exists elsewhere in this tree, but nothing certifies a number
for the banded path yet, so none is quoted here).

Three types, three jobs
----------------------------

- ``aether::banded::Band`` (``aether/banded/Band.h``) — the WORKING
  carrier: three bare ``float`` limbs (``hi``, ``lo``, ``tail``), no
  memory form, ``certifiedBits == 53``. Every chain computation happens
  here.
- ``aether::banded::BandCell8`` (``aether/banded/BandCell8.h``) — the CODEC
  WORD: one 64-bit value, whose equality is a bitwise compare (right for
  "is this still the fill value", wrong for arithmetic).
- ``aether::banded::BandedReal`` (``aether/banded/BandedReal.h``) — the
  STORAGE value type: an 8-byte codec word with NUMERIC equality and
  ordering, an explicit ``double`` egress, and a ``std::numeric_limits``
  specialization. It carries no member arithmetic at all and no implicit
  ``operator double()`` — every namespace-scope operator (``+``, ``-``,
  ``*``, ``/``) returns the working ``Band`` rather than another
  ``BandedReal``, so a chain such as ``BandedReal e = (a - b) * c / d;``
  packs exactly ONCE, at the final assignment, never in between.

None of the three is reachable through the main ``aether/aether.h``
umbrella — the codec and carrier are a few thousand lines a translation
unit computing ``sin(double)`` has no use for. Include
``aether/banded/banded.h`` explicitly when you need the types.

The working-carrier trait
------------------------------

``aether::WorkingType<T>`` (``aether/dtype/WorkingType.h``) is the general
mechanism :doc:`expression_templates` mentioned in passing: the map from a
STORAGE scalar to the scalar an expression actually computes in. For
``float``/``double`` the map is the identity and costs nothing.
``BandedReal`` is the one storage type in this library where it is not:
``aether::working_type_t<BandedReal>`` is ``Band``, so an expression tree
over ``BandedReal`` operands computes entirely in the unpacked working
carrier and encodes back to the 8-byte codec word exactly once, at the
assignment terminal — never once per node in the middle of a chain.

Certified operations
--------------------------

Landed today, all reachable as members of ``aether::banded::BandedReal``'s
namespace-scope operator set or ``aether::banded::detail::``'s free
functions: ``add``, ``sub``, ``mul``, ``div`` and unary negate (the
certified arithmetic core); the four exact sign/order operations
``abs``, ``copysign``, ``fmax``, ``fmin``; the transcendentals ``exp``,
``log`` and ``pow`` (``pow`` built as ``exp(y * log x)``); and the root
family ``sqrt``, ``rsqrt``, ``rsqrtCube``, ``cbrt`` and ``hypot``. Every
one of these is what ``aether::math``'s facade routes a banded operand
through, so — inside the domain each op is certified over — the SAME
call spelling used for a native ``float``/``double`` (``aether::math::
abs``, ``aether::math::pow``, ...) works over ``Band``/``BandedReal``
operands too.

What is still queued
--------------------------

The remaining transcendental family — trigonometric and inverse
trigonometric functions, rounding — plus new hyperbolic/exp2-log2
constructions and the expression-template/consumer integration work, are
explicitly queued behind the rest of this codebase's work, with no date
attached. This page describes what is landed today; it makes no promise
about when the rest arrives.

A runnable example
-------------------

Constructing two ``Band`` values from stored codec words and checking the
certified arithmetic core plus the sign/order facade ops against a
committed reference:

.. aether-example:: tests/test_BandGolden.cpp:91-102

.. code-block:: cpp

       for (std::size_t i = 0; i < golden::kBandChainRowCount; ++i) {
           const golden::BandChainRow& r = golden::kBandChainRows[i];
           const Band a = static_cast<Band>(BandedReal::fromBits(r.aWord));
           const Band b = static_cast<Band>(BandedReal::fromBits(r.bWord));
           expectLimbs("add", i, bd::add(a, b), r.add);
           expectLimbs("sub", i, bd::sub(a, b), r.sub);
           expectLimbs("mul", i, bd::mul(a, b), r.mul);
           expectLimbs("div", i, bd::div(a, b), r.div);
           expectLimbs("chain", i, bd::div(bd::mul(bd::sub(a, b), a), b), r.chain);
           expectLimbs("facade", i,
               bd::copysign(bd::fmax(bd::abs(a), bd::abs(b)), b), r.facade);
       }

.. aether-example-end::

``bd`` here is ``aether::banded::detail``. Everyday code more often starts
from a plain ``double`` — ``aether::banded::BandedReal::fromDouble(x)`` —
rather than from a raw stored word the way this cross-checked test does;
see the umbrella header's own worked example
(``aether/banded/banded.h``) for that shorter path.
