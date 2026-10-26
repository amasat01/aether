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

``aether_dsc.seal`` is both a function and a submodule; its page is the submodule's.
