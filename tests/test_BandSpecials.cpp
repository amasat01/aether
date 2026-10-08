// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandSpecials.cpp
 * @brief `BandSpecialsCert` conformance (AETHER_CPP_MODE): the shared host
 *        battery.
 *
 * The suite collects `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE, so
 * exactly one translation unit registers the `BandSpecialsCert` battery per
 * mode. In CPP_MODE the host arm IS the library, so the shared battery is the
 * whole gate here.
 *
 * See `test_BandSpecials_common.h` for the rows that have no subject in
 * aether's carried surface and stay deferred.
 */

#include "test_BandSpecials_common.h"
