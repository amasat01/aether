Concepts
========

The core ideas behind aether. Each page answers one question; new words are
in the :doc:`glossary`.

:doc:`Views and items <views_and_items>`
   What a ``View`` is, and why an ``Item`` lives in registers.

:doc:`Expression templates <expression_templates>`
   Why ``a + b*s`` does not allocate.

:doc:`Host vs device code <host_vs_device>`
   What ``AETHER_DEVICEHOST`` means, and when you need a GPU kernel entry point.

:doc:`Memory ownership <memory_ownership>`
   Who owns a buffer, host or device, and how to copy it.

:doc:`Interoperability <interop>`
   How a NumPy, CuPy or PyTorch array becomes an aether ``View`` without a copy.

.. toctree::
   :maxdepth: 1
   :hidden:

   views_and_items
   expression_templates
   host_vs_device
   memory_ownership
   interop
