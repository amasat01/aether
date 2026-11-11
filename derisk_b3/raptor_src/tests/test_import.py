# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""INVARIANT: raptor imports cleanly with zero hard deps."""

from __future__ import annotations


def test_import_raptor():
    import raptor

    assert raptor.__version__


def test_import_raptor_schema():
    import raptor.schema

    assert raptor.schema.ALL_PATTERNS == {"vector", "pure", "graph_block"}


def test_import_raptor_protocols():
    import raptor.protocols

    assert raptor.protocols.Backend is not None
