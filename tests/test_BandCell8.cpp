// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandCell8.cpp
 * @brief `BandCell8` conformance (AETHER_CPP_MODE): the shared host battery.
 *
 * The suite collects `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE, so
 * exactly one translation unit registers the `BandCell8Cert` battery per mode.
 * In CPP_MODE the host arm IS the library — `floatAsInt`/`intAsFloat` run their
 * `memcpy` fallbacks and there is no device pass at all — so the shared battery
 * is the whole gate here. The CUDA twin adds the GPU arms and the
 * host-versus-device bit-identity comparison on top.
 */

#include "test_BandCell8_common.h"
