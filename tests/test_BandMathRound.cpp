// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathRound.cpp
 * @brief `BandMathRound` suite (CPP_MODE). @see
 *        test_BandMathRound_common.h.
 *
 * The CPP_MODE twin of `test_BandMathRound.cu`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes — same
 * reason `test_BandMathCert.cpp`'s own docstring gives.
 */

#include "test_BandMathRound_common.h"
