How-to guides
=============

Recipes: one task each. No GPU needed unless the task says so.

:doc:`Vector/matrix cookbook <vector_matrix_cookbook>`
   Write ``cross``/``dot``/``matmul`` without reaching for a CUDA math
   intrinsic anywhere.

:doc:`Quaternions <quaternions>`
   Rotate a vector 90 degrees with a unit quaternion (with a
   :doc:`runnable notebook <examples/quaternion_rotation>`).

:doc:`Random numbers <random_numbers>`
   Fill an array with seeded normal draws and prove the stream is
   reproducible (with a :doc:`runnable notebook <examples/random_numbers>`).

:doc:`Banded emulated double precision <examples/banded_emulated_double>`
   Compute ``1/3`` with ``aether::banded::BandedReal`` and compare it
   to native ``double``.

.. toctree::
   :maxdepth: 1
   :hidden:

   vector_matrix_cookbook
   quaternions
   random_numbers
   examples/banded_emulated_double
