Why the numerics stay domain-independent
========================================

aether knows arrays, expressions, reductions and small dense linear algebra. It does not know orbits, grids, epidemics or
circuits, and no function in it is there for one domain. A domain meets aether through the same vocabulary as every other.

When a domain needs more, the answer is a primitive that is correct for every input of its kind, never a case for that
domain. That keeps the library small, and it means a fix made for one field is a fix for all of them.

Where mathematics earns its place
---------------------------------

aether uses mathematics where it exposes structure, improves conditioning or stability, or gives a guarantee. Two examples:

- **A status contract for linear algebra.** A per-lane solve returns its answer together with an ``info`` status, so a
  singular lane is reported and not silently wrong. The tolerances are stated in :doc:`numerics_contract`.
- **Compensated sums.** ``aether::accum`` keeps the rounding error of a long sum small instead of hiding it
  (:doc:`accumulation`).

aether adds nothing whose only purpose is to work around a limit that current hardware no longer has. Code that reads like
the maths and runs on a CPU or a GPU is the aim.

The wider picture is on `Why RAPTOR is built this way <https://amasat01.github.io/why.html>`_.
