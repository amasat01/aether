# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.protocols — the family's typing.Protocols.

Pure structural contracts; zero hard deps; zero imports of any downstream
package.
"""

from __future__ import annotations

from .backend import CAPABILITIES, Backend, CapabilityError, Executable
from .launcher import KernelLauncher, Marshal
from .provider import ManifestProvider

__all__ = [
    "CAPABILITIES",
    "Backend",
    "CapabilityError",
    "Executable",
    "KernelLauncher",
    "ManifestProvider",
    "Marshal",
]
