// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathInvTrig.cu
 * @brief `BandMathInvTrig` suite (CUDA mode). @see
 *        test_BandMathInvTrig_common.h.
 *
 * The CUDA-mode twin of `test_BandMathInvTrig.cpp`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes (see
 * `tests/bandmath/BandMathCert.h`'s file docstring), so this TU compiles
 * the shared header through nvcc's HOST pass with no `__global__` kernel
 * involved — same reason `test_BandMathRoot.cu`'s own docstring gives. The
 * per-op SASS audit subjects (which DO need real device kernels) live in
 * `tests/sass/check_band_fp64free.sh`'s own subject list and
 * `tests/test_BandMathSass.cu`, not here.
 */

#include "test_BandMathInvTrig_common.h"
