Quickstart
==========

How do I build and read one array?

``#include <aether/aether.h>``, construct an ``aether::Array<T, Es...>``
with however many samples you need, and read/write it through the
``View`` its ``.hostView()`` hands back — indexed ``(component, sample)``,
the same shape you would use from NumPy with the axes swapped.

The umbrella header
--------------------

``aether/aether.h`` pulls in the whole library: layout, views, expression
templates, arrays, random numbers, residency — everything this site's
pages cover. There is no separate install step or Python package to reach
it from (see :doc:`interop` for how aether reaches Python at all) — it is
a header-only C++23 library, so a consumer just includes
it and links nothing extra beyond the compiler's own runtime.

``aether::Array``
------------------

An ``aether::Array<T, Es...>`` (``aether/array/Array.h``) is the OWNING
convenience type: it allocates its own memory and follows the symmetry
rule — ``Array<T, Es...>`` always appends one more, dynamic dimension
(the number of samples) on top of the ``Es...`` you name, so
``Array<double, 3>`` is an array of ``N`` 3-vectors, not a single one.
``.samples()`` reports how many there are; ``.size()`` reports the total
scalar count (``samples() * 3`` for a 3-vector array).

On a build with a CUDA backend, an ``Array`` actually owns TWO buffers — a
host copy and a device copy — and ``.upload()``/``.download()`` move data
between them (see :doc:`memory_ownership`). In a CPU-only build there is
only one buffer, so those calls are harmless no-ops; the code above
compiles and behaves identically either way, which is the whole point of
the dual-mode design this library is built on.

A runnable example
-------------------

.. aether-example:: tests/test_Array.cpp:22-27

.. code-block:: cpp

   TEST_F(ArrayTest, SamplesAndSize)
   {
       aether::Array<double, 3> arr(10);
       EXPECT_EQ(arr.samples(), 10u);
       EXPECT_EQ(arr.size(), 30u);
   }

.. aether-example-end::

Ten 3-component samples, so ``samples() == 10`` and ``size() == 30``. From
here, :doc:`views_and_items` covers how to actually read and write the
elements through a ``View``, and :doc:`memory_ownership` covers what
``Array`` is doing underneath.
