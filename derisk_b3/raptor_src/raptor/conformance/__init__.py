# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.conformance — the interop certification declaration and its
reusable matrix-hygiene scanners. See :mod:`raptor.conformance.interop`;
the free-threading row harness is :mod:`raptor.conformance.freethreading`;
the pytest plugin recording which rows ran is :mod:`raptor.conformance.pytest_rows`.
"""

from __future__ import annotations

from .interop import (
    CERTIFIED,
    LEGS,
    ROADMAP,
    ROW_KINDS,
    ROWS,
    UNIVERSE,
    RowDecl,
    assert_doc_rows_match,
    assert_no_skip_tokens,
    assert_rows_complete,
    certified_framework_problem,
    register_row,
    roadmap_framework,
)
from .freethreading import (
    FT_ROWS,
    FtRowDecl,
    assert_ft_rows_complete,
    declare_ft_row,
    register_ft_row,
)

__all__ = [
    "CERTIFIED",
    "LEGS",
    "ROADMAP",
    "ROW_KINDS",
    "ROWS",
    "FT_ROWS",
    "FtRowDecl",
    "assert_ft_rows_complete",
    "declare_ft_row",
    "register_ft_row",
    "UNIVERSE",
    "RowDecl",
    "assert_doc_rows_match",
    "assert_no_skip_tokens",
    "assert_rows_complete",
    "certified_framework_problem",
    "register_row",
    "roadmap_framework",
]
