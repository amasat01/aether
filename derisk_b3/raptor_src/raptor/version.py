# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor's own distribution version.

Distinct from :data:`raptor.schema.manifest.SCHEMA_VERSION` (the plugin-schema
version this package's schema documents describe) — this is raptor-the
-package's own version number.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raptor-core")
except PackageNotFoundError:  # run from a checkout without an install
    __version__ = "unknown"
