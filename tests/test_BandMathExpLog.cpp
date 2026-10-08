// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathExpLog.cpp
 * @brief `BandMathExpLog` suite (CPP_MODE). @see
 *        test_BandMathExpLog_common.h.
 *
 * The CPP_MODE twin of `test_BandMathExpLog.cu`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes — see
 * `tests/bandmath/BandMathCert.h`'s file docstring.
 */

#include "test_BandMathExpLog_common.h"
