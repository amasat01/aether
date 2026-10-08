Installation
=============

aether is a header-only C++23 library: using it from C++ is a CMake
``find_package(aether CONFIG REQUIRED)`` against an installed tree, not a
package manager step. A sealed Python payload, ``aether-dsc``, ships the
SAME headers for Python-only consumers that never touch a C++ toolchain
directly — see :doc:`python/aether_dsc`.

Requirements
------------

- CMake >= 3.20 and a C++23 compiler (GCC >= 12 or Clang >= 16).
- CUDA mode additionally needs CUDA 12.6 or newer (tested with 12.6 and 13.0); nvcc accepts only host
  compilers up to its own ceiling (GCC 13 for CUDA 12.6), so point it at
  one with ``-DCMAKE_CUDA_HOST_COMPILER=<g++-13>`` if your default
  compiler is newer.
- A GPU is OPTIONAL: build with ``AETHER_CPP_MODE=ON`` and the whole
  library — including every example on this site — runs on the CPU alone
  through its OpenMP backend.

From source (C++)
------------------

``PREFIX`` below is the install directory (inside a conda environment,
use ``${CONDA_PREFIX}``):

.. code-block:: bash

   # CUDA mode
   cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DAETHER_BUILD_TESTS=ON -B build .
   # CPU-only mode (no CUDA toolchain needed)
   cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DAETHER_CPP_MODE=ON -DAETHER_BUILD_TESTS=ON -B build .

   cmake --build build
   cmake --install build --prefix ${PREFIX}

A consuming project then links against the installed package:

.. code-block:: cmake

   find_package(aether CONFIG REQUIRED)
   target_link_libraries(your_target PRIVATE aether::aether)

aether-dsc (Python)
--------------------

``aether-dsc`` carries a sealed, digest-named copy of the same headers for
Python-only consumers — hawk's NVRTC device-kernel compiler is the
primary one, and any file-only host compiler can reach the same headers
through ``Payload.serve()``. Nothing in the blob is readable C++ source
inside a wheel's include tree; see :doc:`python/aether_dsc`.

.. code-block:: bash

   # From a source checkout
   pip install ./dsc

``aether-dsc`` on PyPI is the sealed header payload that hawk compiles against;
it is installed with hawk automatically (``pip install raptor-hawk``) and is not
something C++ users install. aether itself is used through CMake, as above.

CUDA architectures
------------------

``AETHER_CUDA_ARCHS`` picks the architectures when ``CMAKE_CUDA_ARCHITECTURES``
is not given:

- ``native`` (the default): the GPUs in this machine only, the fastest build.
  With no GPU it builds PTX for the oldest architecture the ``nvcc`` in use
  compiles (at least ``sm_60``), which the driver compiles for any newer GPU
  at load.
- ``all``: every architecture from Pascal to Blackwell that the ``nvcc`` in
  use still compiles (CUDA 13 drops the ones below Turing).
- one architecture or a list, such as ``86`` or ``"75;86"``: exactly those.

.. code-block:: bash

   cmake -DAETHER_CUDA_ARCHS=all -B build .
   cmake -DAETHER_CUDA_ARCHS="80;90" -B build .

``CMAKE_CUDA_ARCHITECTURES``, when given, still wins.

Verify
------

Build with ``AETHER_BUILD_TESTS=ON`` (either mode above), then run the
test-integrity gate directly — never the test binary bare, see
``tests/check_gate.sh``'s own header comment:

.. code-block:: bash

   tests/check_gate.sh cpp build/tests/aether_tests   # or: cuda
