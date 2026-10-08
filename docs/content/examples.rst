How-to guides
=============

One task each. No GPU needed unless the task says so.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Task
   * - :doc:`Vector/matrix cookbook <vector_matrix_cookbook>`
     - Write ``cross``/``dot``/``matmul`` without reaching for a CUDA math
       intrinsic anywhere.
   * - :doc:`Quaternions <quaternions>` +
       :doc:`runnable notebook <examples/quaternion_rotation>`
     - Rotate a vector 90 degrees with a unit quaternion.
   * - :doc:`Random numbers <random_numbers>` +
       :doc:`runnable notebook <examples/random_numbers>`
     - Fill an array with seeded normal draws and prove the stream is
       reproducible.
   * - :doc:`Banded emulated double precision <examples/banded_emulated_double>`
     - Compute ``1/3`` with ``aether::banded::BandedReal`` and compare it
       to native ``double``.

.. toctree::
   :maxdepth: 1
   :hidden:

   vector_matrix_cookbook
   quaternions
   examples/quaternion_rotation
   random_numbers
   examples/random_numbers
   examples/banded_emulated_double
