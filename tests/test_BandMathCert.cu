// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathCert.cu
 * @brief `BandMathCert` suite (CUDA mode). @see test_BandMathCert_common.h.
 *
 * The CUDA-mode twin of `test_BandMathCert.cpp`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes (see
 * `tests/bandmath/BandMathCert.h`'s file docstring — `bandToIEEE` is a
 * plain `inline` function, not `AETHER_DEVICEHOST()`), so this TU compiles
 * the shared header through nvcc's HOST pass with no `__global__` kernel
 * involved. The per-op SASS audit subjects (which DO need real device
 * kernels) live in `test_BandMathSass.cu`, a separate file — this one is
 * purely the correctness certification, unchanged whether it lands under
 * nvcc or a plain host compiler.
 */

#include "test_BandMathCert_common.h"
