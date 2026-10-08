API reference — numerics
===========================

Device-legal scalar math, the working-type indirection expression trees
compute through, and the two modules built on top of it: banded emulated
double precision (for FP64-less GPUs) and compensated/atomic accumulation.

Namespace ``aether::math``
------------------------------

The device-legal scalar dispatch facade — the SAME call spelling works
over a native ``float``/``double`` and, where certified, a banded operand
(see :doc:`../banded_emulated_real`). ``fma`` and the banded-constrained
``sincos`` overload are part of this facade too, but their declarations
carry a trailing C++20 ``requires``-clause that this toolchain's Doxygen
(1.9.1) cannot parse into a clean signature — a documented upstream
limitation (``docs/doxygen_allowlist.txt``), not a gap in their own
docstrings.

The elementwise set, all ``aether::math::NAME`` on ``float``/``double``,
reachable from ``aether/aether.h``. Device compiles call the CUDA math
library function of the same C name; host compiles call ``std::``.
"vector" marks the functions that ``AETHER_HOST_VECTOR_MATH`` routes (for
``double``, in a GCC x86-64 host translation unit) to the faithfully
rounded packet math so scalar loops vectorise; the rest stay scalar on the
host.

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Group
     - Functions
     - Host vector route
   * - exponential, logarithm
     - ``exp``, ``exp2``, ``exp10``, ``expm1``, ``log``, ``log2``,
       ``log10``, ``log1p``, ``pow``
     - vector
   * - roots
     - ``sqrt``, ``rsqrt``, ``cbrt``, ``hypot``
     - vector
   * - trigonometric
     - ``sin``, ``cos``, ``sincos``, ``tan``, ``asin``, ``acos``, ``atan``,
       ``atan2``
     - vector
   * - hyperbolic
     - ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``
     - vector
   * - rounding
     - ``floor``, ``ceil``, ``trunc``, ``round`` (half away from zero, as
       in C), ``rint`` (half to even, numpy's ``round``/``rint``)
     - vector
   * - sign, selection
     - ``abs``, ``copysign``, ``fmax``, ``fmin``, ``min``, ``fdim``
     - vector / inline
   * - other
     - ``sign``, ``clip``, ``fmod``, ``remainder``, ``fma``, ``isnan``,
       ``isinf``, ``isfinite``
     - scalar
   * - special
     - ``erf``, ``erfc``
     - scalar

Where the semantics differ from C or numpy: ``remainder`` follows numpy
(result has the sign of the divisor), not C's IEEE ``remainder``; ``fmod``
is C's (sign of the dividend), as is numpy's ``fmod``. ``round`` is C's;
numpy's ``round`` is ``rint``. ``sign(NaN)`` is ``0`` (numpy returns NaN).
``clip`` propagates NaN from any argument, as numpy's does.

Device against host: the rounding, sign/selection, ``fmod``,
``remainder``, ``clip``, ``fma`` and classification functions give the same
bits; the transcendental ones agree within the CUDA Math API's documented
error plus glibc's (a few ULP).

Not provided yet: ``lgamma``, ``tgamma``, ``digamma``, Bessel functions,
and integer-specific operations.

.. doxygenfunction:: aether::math::sign
   :project: aether

.. doxygenfunction:: aether::math::isfinite
   :project: aether

.. doxygenfunction:: aether::math::isnan
   :project: aether

.. doxygenfunction:: aether::math::isinf
   :project: aether

.. doxygenfunction:: aether::math::clip
   :project: aether

.. doxygenfunction:: aether::math::remainder
   :project: aether

.. doxygenfunction:: aether::math::sin
   :project: aether

.. doxygenfunction:: aether::math::cos
   :project: aether

.. doxygenfunction:: aether::math::rsqrt
   :project: aether

.. doxygenfunction:: aether::math::rsqrtCube
   :project: aether

``aether::WorkingType``
--------------------------

.. doxygenstruct:: aether::WorkingType
   :project: aether
   :members:

Banded emulated real (``aether/banded/banded.h``)
-----------------------------------------------------

Not included from the ``aether/aether.h`` umbrella — see
:doc:`../banded_emulated_real` for why and for the three types' roles.

.. doxygenstruct:: aether::banded::Band
   :project: aether
   :members:

.. doxygenstruct:: aether::banded::BandCell8
   :project: aether
   :members:

.. doxygenstruct:: aether::banded::BandedReal
   :project: aether
   :members: toBits, cell, band, fromCell, fromBits, fromBand, operator==, operator!=, operator<, operator>, operator<=, operator>=

Namespace ``aether::accum``
-------------------------------

Compensated-sum and atomic accumulation policies for reductions.

.. doxygennamespace:: aether::accum
   :project: aether
   :members:
