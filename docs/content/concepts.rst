Explanation
===========

The ideas behind aether, grouped by topic and ordered from first use to advanced. Each page answers one question; new words are in the :doc:`glossary`.

Arrays and memory
-----------------

What a view is, who owns a buffer, and why an expression does not allocate.

:doc:`Views and items <views_and_items>`
   What a ``View`` is, and why an ``Item`` lives in registers.
:doc:`Expression templates <expression_templates>`
   Why ``a + b*s`` does not allocate.
:doc:`Memory ownership <memory_ownership>`
   Who owns a buffer, host or device, and how to copy it.

Run kernels
-----------

Code that runs on both machines, with several devices, and on the CPU's vector units.

:doc:`Host vs device code <host_vs_device>`
   What ``AETHER_DEVICEHOST`` means, and when you need a GPU kernel entry point.
:doc:`CPU SIMD backend <cpu_simd>`
   How aether vectorizes on the CPU, with no GPU involved.
:doc:`Why the numerics stay domain-independent <domain_independence>`
   What aether knows, what it does not, and where mathematics earns its place.
:doc:`Multi-device residency <multi_device_residency>`
   Keep a replica per GPU, or partition data across several.

Numerics and precision
----------------------

The fold order, division, compensated sums and non-finite rules every target computes against, and how to get more than ``float`` or ``double``.

:doc:`Numerics contract <numerics_contract>`
   The rules every target computes against.
:doc:`Accumulation <accumulation>`
   Adding many producers' contributions into one per-sample slot without a data race.
:doc:`Banded/emulated real arithmetic <banded_emulated_real>`
   ``BandedReal``: double precision from three ``float`` limbs, and when to use it.

Interoperability
----------------

:doc:`Interoperability <interop>`
   How a NumPy, CuPy or PyTorch array becomes an aether ``View`` without a copy. The contract the family shares is on `raptor's interoperability page <https://amasat01.github.io/raptor/content/interop_protocols.html>`__.

Embedded C
----------

:doc:`Plain-C reference core <c_core>`
   Standalone C99 functions that reproduce aether's arithmetic bit for bit.

.. toctree::
   :maxdepth: 1
   :hidden:

   views_and_items
   expression_templates
   memory_ownership
   host_vs_device
   cpu_simd
   domain_independence
   multi_device_residency
   numerics_contract
   accumulation
   banded_emulated_real
   interop
   c_core
