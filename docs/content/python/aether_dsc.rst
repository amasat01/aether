Python API — ``aether_dsc``
=============================

aether itself ships no Python package. ``aether_dsc`` (``dsc/``, distributed
as ``aether-dsc``) is the one Python-facing surface in this repository: a
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
