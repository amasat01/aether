# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Scalar-type tags of the launch/artifact contract.

Values copied VERBATIM from ``eagle/python/eagle/dtypes.py``. No array library is imported:
the dtype of each tag is plain data (:data:`USER_DTYPE`), so this module imports
nothing — raptor has zero dependencies.
"""

from __future__ import annotations

#: the compiled scalar modes, in sidecar spelling (verbatim value from
#: ``eagle.dtypes.SCALAR_TYPES``).
SCALAR_TYPES = ("float64", "float32", "banded")


#: The user-facing array dtype NAME of Real-typed arrays per resolved ``scalar_type``, its cell size in bytes, and the
#: array-exchange standard's numeric code (DLPack: kDLFloat = 2, with the bit width). Plain data, no array library.
#: banded: an 8-byte emulated-FP64 cell; user-facing arrays are float64, values match float64 within the per-op ULP
#: contract documented in aether, not bit-identically. On the wire a banded Real plane is eagle dtype code 13 (8-byte
#: cells, kind 'V'), packed and unpacked by eagle; the user-facing dtype stays float64.
USER_DTYPE = {
    "float64": {"name": "float64", "bytes": 8, "dlpack": (2, 64, 1)},
    "float32": {"name": "float32", "bytes": 4, "dlpack": (2, 32, 1)},
    "banded": {"name": "float64", "bytes": 8, "dlpack": (2, 64, 1)},
}
