Contributing
=============

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below. See ``CONTRIBUTING.md`` at the repository root for the full
statement.

Terms
-----

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (``git commit -s``); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

Tests and gates
-----------------

Build with ``AETHER_BUILD_TESTS=ON`` (CUDA or ``AETHER_CPP_MODE``, see
:doc:`installation`), then run the test-integrity gate — never the test
binary bare, since an empty ``--gtest_filter`` or a silently dropped test
both exit 0 from the binary alone:

.. code-block:: bash

   tests/check_gate.sh cpp build/tests/aether_tests   # or: cuda

It checks the executed test names against a committed manifest
(``tests/expected_tests_{cuda,cpp}.txt``), so a build that silently drops
a test is caught rather than just under-counted.

Building these docs
---------------------

.. code-block:: bash

   cd docs
   make strict    # doxygen + sphinx -W --keep-going; 0 warnings required
   make nbexec    # execute every tutorial/example notebook in place
   make nbcheck   # refuse a build if any notebook cell did not run
   make linkcheck

The C++ API reference is generated from Doxygen XML through breathe
(``make doxygen`` alone also works, if you only need ``_doxybuild/xml``
refreshed). ``docs/doxygen_allowlist.txt`` is the committed allowlist of
known, upstream Doxygen limitations — a new warning not already on that
list is a docs regression, not something to add to the list.

Docs-specific gates also run from the repository root:

.. code-block:: bash

   tests/docs/check_docs.sh

which checks the doxygen-warning allowlist, that every symbol/path cited
in a docs page actually exists in the tree, that ``sphinx-build -W``
succeeds, and that every embedded code excerpt still matches the test
file it was lifted from (``tests/docs/check_examples.py`` — no drift
between a page and the source it quotes).
