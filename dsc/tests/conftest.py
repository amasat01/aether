# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Register the suite's marker where it is collected: CI copies this folder out of the repo and runs it with `-c /dev/null`,
so the marker cannot live in pyproject.toml."""


def pytest_configure(config):
    config.addinivalue_line("markers", "ft: the free-threaded smoke rows (run on 3.14t/3.15t with -m ft)")
