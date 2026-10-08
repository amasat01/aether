// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandSpecials.cu
 * @brief `BandSpecialsCert` (CUDA mode): the shared host battery
 *        (`test_BandSpecials_common.h`).
 *
 * Two device arms for this suite are NOT carried here:
 *
 *  - `GatedOpsAreBitIdenticalOnDevice` bundles `sqrt_` into the SAME kernel
 *    and struct as the certified ops, and its entire body exists to bound
 *    `sqrt_`'s host/device seed divergence (`rsqrtCore`'s one
 *    `__CUDA_ARCH__` fork). `sqrt_` needs `RsqrtCore`, not carried here.
 *    There is no core-only subject left once it is removed.
 *  - `Cell8SpecialsAreBitIdenticalOnDevice` calls `bd::bandToIEEE` FROM
 *    WITHIN A KERNEL. `bandToIEEE` (`aether/banded/Band.h`) is HOST ONLY
 *    (it reconstructs the finite path through `double`, exactly as
 *    `detail::bandToDouble`/`doubleFromCell8` already do for the storage
 *    egress terminal) -- a device-safe, FP32-only egress terminal would be
 *    new numerical design work, not a spelling-swap port.
 *
 * Both stay `deferred-with-addon`.
 * The shared host battery below still runs, and is listed, in this
 * CUDA-mode binary.
 *
 * GPU EXECUTION NOTE: this file is built and its tests listed here; the
 * CUDA suite that actually launches these kernels runs separately.
 */

#include "test_BandSpecials_common.h"
