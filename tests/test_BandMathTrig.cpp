// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathTrig.cpp
 * @brief `BandMathTrig` suite (CPP_MODE). @see
 *        test_BandMathTrig_common.h.
 *
 * The CPP_MODE twin of `test_BandMathTrig.cu`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes — see
 * `tests/bandmath/BandMathCert.h`'s file docstring.
 */

#include "test_BandMathTrig_common.h"
