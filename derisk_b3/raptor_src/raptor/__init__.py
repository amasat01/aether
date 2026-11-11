# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor — the protocol spine of the RAPTOR family.

Pure contracts; depends on nothing. Schemas live in :mod:`raptor.schema`;
typing.Protocols live in :mod:`raptor.protocols`.
"""

from __future__ import annotations

from .version import __version__

__all__ = ["__version__"]
