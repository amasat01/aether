// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandCell8Window.cu
 * @brief Tier-1 storage window vs the carrier egress (CUDA mode): the shared
 *        host battery.
 *
 * The CUDA twin of `test_BandCell8Window.cpp`, and deliberately NOT more than
 * that. The subject is the `double` egress terminals, which are HOST ONLY on
 * both sides (`bandToIEEE` and `BandedReal::toDouble` are plain `inline` host
 * functions — the 0-FP64 portability warrant keeps `double` out of device
 * code), so there is no device arm to add that would not be a different claim.
 * The one DEVICEHOST participant, `cell8BandIsStorable`, already has its
 * host-versus-device bit-identity arm in `test_BandCell8.cu`.
 *
 * @see `test_BandCell8Window_common.h` for what the suite pins and why.
 */

#include "test_BandCell8Window_common.h"
