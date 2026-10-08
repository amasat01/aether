Tutorials
=========

Five notebooks, read in order. Each compiles and runs a small C++ program
through the same ``run_cpp()`` helper: ``g++ -std=c++23
-DAETHER_CPP_MODE=1`` by default (no GPU or CUDA toolchain needed), and
``nvcc`` on the SAME source when a GPU is visible.

1 — :doc:`One array, two machines <tutorials/01_one_array_two_machines>`
   Build an ``aether::Array``, read/write it through a ``View``, and move
   it to the device and back with calls that are no-ops in a CPU-only
   build.

2 — :doc:`Expressions without temporaries <tutorials/02_expressions_without_temporaries>`
   Compose ``+``/``-``/``*`` into one expression tree, and see what it
   costs not to.

3 — :doc:`Device math <tutorials/03_device_math>`
   Call ``aether::math::erf``, ``sincos``, ``hypot`` the same way from
   host code or a GPU kernel.

4 — :doc:`Your own device function and kernel <tutorials/04_your_own_device_function>`
   Write ``AETHER_DEVICEHOST()`` once; dispatch it with a loop on the CPU
   or an ``AETHER_KERNEL()`` launch on the GPU.

5 — :doc:`Adding a function to aether <tutorials/05_adding_a_function>`
   The macro scaffold behind ``aether::math``'s own dispatch, proved by
   adding a new function with it.

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/01_one_array_two_machines
   tutorials/02_expressions_without_temporaries
   tutorials/03_device_math
   tutorials/04_your_own_device_function
   tutorials/05_adding_a_function
