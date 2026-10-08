// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandedReal.cpp
 * @brief `BandedReal` conformance (AETHER_CPP_MODE): the shared host battery.
 *
 * The suite collects `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE, so
 * exactly one translation unit registers the `BandedRealCert` battery per mode.
 * In CPP_MODE the host arm IS the library, so the shared battery is the whole
 * gate here; the CUDA twin adds the device arms (the facade and the
 * comparisons inside a kernel, the limits as literal reads, and the audited
 * chain kernel the 0-FP64 SASS gate scans) on top.
 *
 * See `test_BandedReal_common.h` for the four rows that have no subject in
 * aether today and stay deferred.
 */

#include "test_BandedReal_common.h"
