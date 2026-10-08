// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandLevers.cpp
 * @brief `BandAccum`/`BandAccumVec` conformance (AETHER_CPP_MODE): the shared
 *        host battery.
 *
 * The suite collects `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE, so
 * exactly one translation unit registers the `BandLevers` accumulator slice
 * per mode. In CPP_MODE the host arm IS the library, so the shared battery is
 * the whole gate here; the CUDA twin adds the fp64-free SASS audit's kernel on
 * top (test_BandLevers.cu).
 *
 * See `test_BandLevers_common.h` for the scope note (a deliberate PARTIAL
 * slice of the `BandLevers` suite: the three accumulator rows only).
 */

#include "test_BandLevers_common.h"
