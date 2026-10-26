# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""aether: CPU and GPU arrays that every other array library can read without a copy.

Host arrays are numpy arrays. Device arrays are ``aether.Array``: one typed buffer on one
CUDA device, shared through DLPack and ``__cuda_array_interface__``. No maths lives here.
``import aether`` needs only numpy; the CUDA runtime (``cuda-bindings``) is loaded on the
first device call and, when missing, raises ``CudaUnavailable`` naming the pip extra.

Python ``aether.Array`` mirrors the C++ ``Chunk`` + ``RuntimeView`` (one buffer, one
device), not the C++ ``aether::Array`` (a host-mirrored SoA with capacity)."""

from ._array import (Array, Device, asarray, copy, empty, empty_like, from_dlpack, full, pinned, zeros,
                     zeros_like)
from ._rt import CudaError, CudaUnavailable

__all__ = ["Array", "Device", "CudaError", "CudaUnavailable", "asarray", "copy", "empty", "empty_like",
           "from_dlpack", "full", "pinned", "zeros", "zeros_like"]
