How-to guides
=============

Recipes: one task each, grouped by topic and ordered from first use to advanced. No GPU needed unless the task says so.

Vector algebra
--------------

Cross, dot and matrix products, and rotating with quaternions.

:doc:`Vector and matrix cookbook <vector_matrix_cookbook>`
   Write ``cross``/``dot``/``matmul`` without reaching for a CUDA math
   intrinsic anywhere.
:doc:`Rotate a vector with a quaternion <examples/quaternion_rotation>`
   Rotate a vector 90 degrees with a unit quaternion; the page also
   lists the quaternion surface.

Random numbers
--------------

:doc:`Reproducible random numbers <examples/random_numbers>`
   Fill an array with seeded normal draws and prove the stream is
   reproducible; the page also describes the generator.

Numerics and precision
----------------------

:doc:`Banded emulated double precision <examples/banded_emulated_double>`
   Compute ``1/3`` with ``aether::banded::BandedReal`` and compare it
   to native ``double``. The ideas are in :doc:`banded_emulated_real`.

.. toctree::
   :maxdepth: 1
   :hidden:

   vector_matrix_cookbook
   examples/quaternion_rotation
   examples/random_numbers
   examples/banded_emulated_double
