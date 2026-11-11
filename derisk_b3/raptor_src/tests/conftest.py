# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Register the suite's markers: a copy of the tests outside the repository has no pyproject."""

_MARKERS = (
    "ft: free-threading rows; they need a free-threaded interpreter",
    "cross_repo: eagle conformance battery; deselected only with -m 'not cross_repo'",
    "repo_local: checks the repository, not the installed distribution",
)


def pytest_configure(config):
    for line in _MARKERS:
        config.addinivalue_line("markers", line)
