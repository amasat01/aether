// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandMathCert.cpp
 * @brief `BandMathCert` suite (CPP_MODE). @see test_BandMathCert_common.h.
 *
 * The CPP_MODE twin of `test_BandMathCert.cu`. Identical content by
 * construction: the certified round trip this suite exercises
 * (`bandFromIEEE`/op/`bandToIEEE`) is HOST-ONLY in both build modes — see
 * `tests/bandmath/BandMathCert.h`'s file docstring.
 */

#include "test_BandMathCert_common.h"
