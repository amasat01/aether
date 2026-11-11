# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernel-provider protocol.

``ManifestProvider`` (the ``compiled`` facet) duck-types hawk's ``Bundle`` —
"facet" was design jargon, not API. This protocol keeps this ONE facet: an
earlier ``symbolic`` evaluable-trace facet was deleted outright — zero
implementers, and the interpreter mechanism it declared is deliberately
forbidden; ``TorchBackend`` composes the existing engines instead, see
:mod:`raptor.protocols.backend`.

A toy implementation, just enough to satisfy the protocol::

    class DirProvider:
        def __init__(self, manifest_path):
            self._manifest_path = manifest_path
        def build(self, directory):
            # compile/emit kernels into `directory`, return the manifest path
            return self._manifest_path
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ManifestProvider(Protocol):
    """The ``compiled`` facet: emits a manifest.

    Duck-types hawk's ``Bundle`` — consumed by the launch/marshal engines."""

    def build(self, directory: Any) -> Any:
        """Compile/emit this provider's kernels into ``directory``.

        Returns the manifest path."""
        ...
