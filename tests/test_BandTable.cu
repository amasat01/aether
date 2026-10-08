// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandTable.cu
 * @brief `BandPair` conformance (CUDA mode): the shared host battery,
 *        same as the CPP_MODE pairing. This file does not register
 *        device-only rows (`DeviceDecodeMatchesHostBitExact`,
 *        `DeviceSeamMatchesHostForBothCarriers`, `TexelWordOrderIsPinned`,
 *        `TexturedAndPlainArraysFetchTheSameElement`); see
 *        `test_BandTable_common.h` for the corpus this battery covers.
 */

#include "test_BandTable_common.h"
