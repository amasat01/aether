// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_WorkingCarrier.cpp
 * @brief `WorkingCarrierTest` (AETHER_CPP_MODE): the shared host battery.
 *
 * The suite collects `test_*.cu` in CUDA mode and `test_*.cpp` in CPP_MODE, so
 * exactly one translation unit registers the `WorkingCarrierTest` battery per
 * mode. Every row is HOST-ONLY by design: the subject is the expression layer's
 * TYPE protocol (`aether::WorkingType`, `Expression::working_type`, the store
 * terminal) and the codec words it produces, all of which are the same source on
 * both arms and already certified device-side by `test_BandedReal.cu`'s facade
 * and audited-chain kernels. Adding a kernel here would add a NEW audit subject
 * for no new claim.
 *
 * See `test_WorkingCarrier_common.h` for the corpus and the row-by-row
 * coverage, including the one row that has no aether subject.
 */

#include "test_WorkingCarrier_common.h"
