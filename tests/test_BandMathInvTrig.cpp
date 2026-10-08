// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathInvTrig.cpp
 * @brief `BandMathInvTrig` suite (CPP_MODE). @see
 *        test_BandMathInvTrig_common.h.
 *
 * The CPP_MODE twin of `test_BandMathInvTrig.cu`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes — same
 * reason `test_BandMathRoot.cpp`'s own docstring gives.
 */

#include "test_BandMathInvTrig_common.h"
