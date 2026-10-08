// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file random.h
 * @brief Umbrella header for the `aether::random` module — reproducible,
 *        counter-based RNG and i.i.d. distribution sampling, exposed as
 *        composable aether expressions.
 *
 * Include `aether/aether.h` (which pulls this in) and use a
 * `aether::random::Generator`:
 * @code
 * aether::random::Generator rng(0xC0FFEEu);
 * out[i] = rng.normal<double, 3>(0.0, 1.0);   // per-component, composes
 * @endcode
 *
 * aether owns its Philox4x32-10 outright (`detail/Philox.h`):
 *
 *  - a seed names one stream — host draws and device draws are bit-identical
 *    (and the reason the cpp-mode golden test is a meaningful anchor for the
 *    device arm too);
 *  - nothing here includes `<curand_kernel.h>`, so the module compiles with
 *    no CUDA toolchain at all;
 *  - the device stream still matches the real cuRAND device stream bit-for-
 *    bit for `randomBits64` and `uniform01`, because the implementation
 *    reproduces cuRAND's key/counter layout and output ordering exactly
 *    (certified by `tools/random/curand_host_probe.cpp` against libcurand).
 *    The normal stream is tolerance-matched against that reference, by
 *    design: Box-Muller here goes through `aether::math::{log,sqrt,sincos}`
 *    rather than cuRAND's `sincospi`/`__sincosf`.
 *
 * Every draw is a pure function of `(seed, global sample id, sub-counter)`:
 * no per-sample state array, no atomic counter, no persisted generator state.
 * A parallel fill and a serial fill therefore agree bit-for-bit, and a
 * re-evaluated expression re-draws the same value.
 *
 * Also carries `lognormal`/`exponential`/`bernoulli`/`uniformInt`/
 * `multivariateNormal` (`Generator` factory methods,
 * `detail/DistributionLeaf.h` + `detail/MultivariateNormalLeaf.h`),
 * `LowerTriangular`/`cholesky` (`LowerTriangular.h`), the `free.h`
 * one-liners, and the CPU-SIMD packet path (`detail/PacketSupport.h`) —
 * one ADL-found `packetGet` covering every per-component leaf, packing `W`
 * independent scalar draws lane-wise so packet output is bit-identical to
 * scalar output by construction. `multivariateNormal` is a plain
 * per-component leaf — see `detail/MultivariateNormalLeaf.h`'s own
 * docstring for the trade its `eval<I>` makes (`O(VD^2)` redundant draws
 * per sample, documented as the revival lever for a future multi-load
 * specialization).
 */

#include "aether/random/Generator.h"
#include "aether/random/LowerTriangular.h"
#include "aether/random/free.h"
#include "aether/random/detail/PacketSupport.h"
