Python API — ``aether_dsc``
=============================

The ``aether-dsc`` wheel (``dsc/``) carries two Python packages. ``aether`` is the
CPU/GPU arrays (:doc:`arrays`). ``aether_dsc``, described here, is a
sealed, digest-named copy of aether's own headers (plus the few ``rtc/``
shims and eagle's plugin layout header) for consumers that compile device
kernels with NVRTC, or any file-only host compiler, without reading or
writing C++ source directly. See :doc:`../installation` to install it and
the module's own docstring below for the two ways to consume a sealed
payload.

.. autosummary::
   :toctree: generated
   :recursive:

   aether_dsc

The names the package exports at the top level:

.. autosummary::
   :toctree: generated

   aether_dsc.Payload
   aether_dsc.version
   aether_dsc.PAYLOAD_DIR
   aether_dsc.OpenMP
   aether_dsc.openmp

The OpenMP runtime
------------------

The wheel carries the one OpenMP runtime of the process (``aether/_omp/``), and ``aether._core`` links it. A
package that builds OpenMP code next to aether, such as eagle or hawk's host kernels, links that same file
instead of shipping its own: two OpenMP runtimes in one process abort on macOS and double the thread pool on
Linux. :func:`aether_dsc.openmp` returns an :class:`aether_dsc.OpenMP` (the runtime file, its directory, the
compile flags and the link flags), or ``None`` when the install carries no runtime. Wheel repair for such a
package must leave the runtime alone (``auditwheel repair --exclude libgomp.so.1``,
``delocate-wheel --exclude libomp.dylib``).

``aether_dsc.seal`` is both a function and a submodule; its page is the submodule's.
