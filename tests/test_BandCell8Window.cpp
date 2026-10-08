// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandCell8Window.cpp
 * @brief Tier-1 storage window vs the carrier egress (AETHER_CPP_MODE): the
 *        shared host battery.
 *
 * The CPP_MODE twin of `test_BandCell8Window.cu`. Every claim in this suite is
 * a HOST claim by construction — `double` egress does not exist on device (the
 * 0-FP64 portability warrant) — so the two twins carry the identical rows and
 * the CUDA build adds nothing.
 *
 * @see `test_BandCell8Window_common.h` for what the suite pins and why.
 */

#include "test_BandCell8Window_common.h"
