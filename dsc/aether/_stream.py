# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``stream=`` arguments: an int handle, an object with ``__cuda_stream__``, or None."""


def handle(stream):
    """The integer handle of ``stream`` (None is the legacy default stream, 0)."""
    if stream is None:
        return 0
    if isinstance(stream, int):
        return int(stream)
    proto = getattr(stream, "__cuda_stream__", None)
    if proto is not None:
        value = proto() if callable(proto) else proto
        return int(value[1])
    raise TypeError(f"stream= wants an int handle or an object with __cuda_stream__, not {type(stream).__name__}")
