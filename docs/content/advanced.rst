Advanced topics
===============

Beyond the core: several devices, CPU vectorization, emulated precision,
race-free accumulation and plain-C output.

:doc:`Multi-device residency <multi_device_residency>`
   Keep a replica per GPU, or partition data across several.

:doc:`CPU SIMD backend <cpu_simd>`
   How aether vectorizes on the CPU, with no GPU involved.

:doc:`Banded/emulated real arithmetic <banded_emulated_real>`
   ``BandedReal``: double precision from three ``float`` limbs, and when to use it.

:doc:`Accumulation <accumulation>`
   Adding many producers' contributions into one per-sample slot without a data race.

:doc:`Numerics contract <numerics_contract>`
   The fold order, division, compensated sums and non-finite rules every target computes against.

:doc:`Plain-C reference core <c_core>`
   Standalone C99 functions that reproduce aether's arithmetic bit for bit.

.. toctree::
   :maxdepth: 1
   :hidden:

   multi_device_residency
   cpu_simd
   banded_emulated_real
   accumulation
   numerics_contract
   c_core
