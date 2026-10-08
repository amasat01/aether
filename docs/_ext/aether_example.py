# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# docs/_ext/aether_example.py — Sphinx extension registering the two
# marker directives tests/docs/check_examples.py scans
# for: `.. aether-example:: tests/test_X.cpp:START-END` and
# `.. aether-example-end::`. Both are no-op at render time (the VISIBLE
# example is the ordinary `.. code-block:: cpp` directive the two markers
# wrap) — their only job is to name, machine-checkably, which committed
# test file+line-range a page's example was lifted from, so the gate can
# catch drift between the two. Without registering them, docutils reports
# "Unknown directive type" (fatal under `sphinx-build -W`); this extension
# is the fix, not a page content change.
from docutils.parsers.rst import Directive


class AetherExample(Directive):
    """`.. aether-example:: tests/test_X.cpp:START-END` — renders nothing;
    the argument is read only by tests/docs/check_examples.py."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    has_content = False

    def run(self):
        return []


class AetherExampleEnd(Directive):
    """`.. aether-example-end::` — closes the block above; renders nothing."""

    required_arguments = 0
    optional_arguments = 0
    has_content = False

    def run(self):
        return []


def setup(app):
    app.add_directive("aether-example", AetherExample)
    app.add_directive("aether-example-end", AetherExampleEnd)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
