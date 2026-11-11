# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A pytest plugin that records which certification rows RAN and passed.

``register_row`` fires when a test module is imported, so on its own it cannot
tell a row that ran from one that was deselected, skipped or failed.
:func:`raptor.conformance.interop.assert_rows_complete` (``ran=True``) reads
this plugin instead: a row counts on a leg only if every test registered for
that ``(row, leg)`` ran and passed in this session.

Load it from the suite's ``conftest.py``::

    pytest_plugins = ["raptor.conformance.pytest_rows"]

and mark the test that calls ``assert_rows_complete`` with
``@pytest.mark.rows_gate``: the plugin runs those tests last, so every row
test has finished by then.

Pure Python; imports pytest only through the hooks pytest hands it.
"""

from __future__ import annotations

_ACTIVE = False
#: nodeid -> the (row, leg) pairs its test registered (leg None = every leg).
_NODES: dict[str, list[tuple[str, str | None]]] = {}
_PASSED: set[str] = set()
_BAD: set[str] = set()  # failed, skipped, xfailed or deselected


def _rows_of(item):
    return getattr(getattr(item, "function", None), "raptor_rows", None)


def pytest_configure(config):
    """Start a clean record and register the ``rows_gate`` marker."""
    global _ACTIVE
    _ACTIVE = True
    _NODES.clear()
    _PASSED.clear()
    _BAD.clear()
    config.addinivalue_line(
        "markers", "rows_gate: runs after every other test (assert_rows_complete)"
    )


def pytest_itemcollected(item):
    """Remember the rows each collected test registered."""
    rows = _rows_of(item)
    if rows:
        _NODES[item.nodeid] = list(rows)


def pytest_deselected(items):
    """A deselected row test is not a run."""
    for item in items:
        if item.nodeid in _NODES:
            _BAD.add(item.nodeid)


def pytest_collection_modifyitems(items):
    """Run the ``rows_gate`` tests last."""
    items.sort(key=lambda it: it.get_closest_marker("rows_gate") is not None)


def pytest_runtest_logreport(report):
    """Record a row test as passed, or as not passed on any other outcome."""
    if report.nodeid not in _NODES:
        return
    if report.outcome != "passed":
        _BAD.add(report.nodeid)
    elif report.when == "call":
        _PASSED.add(report.nodeid)


def unfinished(pairs):
    """The ``(row, leg)`` pairs in ``pairs`` without a passing run, each as a
    one-line reason. Fails loudly when the plugin is not loaded."""
    assert _ACTIVE, (
        "raptor.conformance.pytest_rows is not loaded, so ran=True cannot see "
        "test outcomes: add it to pytest_plugins in conftest.py"
    )
    out = []
    for row, leg in pairs:
        nodes = [n for n, rows in _NODES.items()
                 if any(r == row and lg in (None, leg) for r, lg in rows)]
        if not nodes:
            out.append(f"  {row} [{leg}]: no test for this leg was collected")
            continue
        for node in nodes:
            if node in _BAD:
                out.append(f"  {row} [{leg}]: {node} did not pass "
                           "(failed, skipped or deselected)")
            elif node not in _PASSED:
                out.append(f"  {row} [{leg}]: {node} has not run yet "
                           "(mark the gate test rows_gate)")
    return out
