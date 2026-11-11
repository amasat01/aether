// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Fetch.h
 * @brief `aether::texture_handle_t` and `aether::dtype::Fetch<T>`: the
 *        opaque texture-object handle type and the texel-fetch trait
 *        `view/TableHandle.h`'s texture carrier and
 *        `chunk/TextureBinding.h` both key on.
 *
 * Only the two forms actually needed (a Chebyshev-coefficient fill kernel):
 *   - `double` -> `int2` (a two-word fetch; `TableHandle`'s texture arm
 *     decodes it via `__hiloint2double`, `x` = low word / `y` = high word).
 *   - `float`  -> `float` (a direct fetch, no reinterpretation) — the
 *     texture hardware's native float channel format needs no
 *     reinterpretation step at all.
 *
 * No `<cuda_runtime.h>`/`<vector_types.h>` include, deliberately: `int2`
 * and `tex1Dfetch` are available to a CUDA-language translation unit (a real
 * nvcc `.cu` compile, either the host or the device pass) with no include
 * at all, precisely because this header stays reachable from
 * `aether/device.h` (the NVRTC-parseable umbrella) and every unnecessary
 * include is a candidate to widen that surface. `texture_handle_t` is
 * spelled as the raw `unsigned long long` `cudaTextureObject_t` is itself
 * typedef'd to (`texture_types.h`) rather than the CUDA name, for the
 * identical reason — the alias is that type (a typedef, not a distinct
 * one), so it binds with zero conversion to every real
 * `cudaTextureObject_t`-typed CUDA API parameter, letting
 * `chunk/TextureBinding.h` (which does have `<cuda_runtime.h>` in scope, a
 * host-only header) hand one to this device-safe surface with no cast
 * anywhere.
 */

#include "aether/macros.h"

#if !defined(AETHER_CPP_MODE) && !AETHER_DEVICE_COMPILER
// The zero-include property above holds for a CUDA-LANGUAGE TU only.
// A CUDA-mode build may also compile TUs with the HOST compiler (`g++`),
// which has no implicit CUDA vocabulary at all — yet `Fetch<double>` must
// still exist there, with the same members, or the two compilers disagree
// about the member set (an ODR violation no linker diagnoses). `int2` is
// therefore pulled from its real header on that path only, so the nvcc /
// NVRTC (`aether/device.h`) route keeps the include-free shape the probe
// described above certified.
#include <vector_types.h>
#endif

namespace aether {

/**
 * @brief Opaque texture-object handle, bit-for-bit `cudaTextureObject_t`.
 *        `0` is the "no texture bound" sentinel, matching
 *        `cudaTextureObject_t`'s own convention.
 */
using texture_handle_t = unsigned long long;

namespace dtype {

/**
 * @brief Texel-fetch trait: `Fetch<T>::%type` is the `tex1Dfetch<...>`
 *        template argument for element type `T`; `needsConversion` selects
 *        whether the fetched texel already IS the element bit-for-bit
 *        (`false`) or needs decoding (`true`, e.g. `__hiloint2double`).
 *
 * The primary template is intentionally left UNDEFINED: instantiating it
 * for an element type outside the certified set is a compile error at the
 * point of use, rather than a silent primary-template degrade (a dummy
 * `int`, `needsConversion = false`) — there is no texture-carrying use for
 * an uncertified type here, so failing loudly is more honest than silently
 * texturing garbage.
 */
template<class T>
struct Fetch;

#ifndef AETHER_CPP_MODE

/** @brief `double` -> `int2` two-word fetch. */
template<>
struct Fetch<double> {
    using type = int2;
    static constexpr bool needsConversion = true;
};

/** @brief `float` -> `float` direct fetch, no conversion. */
template<>
struct Fetch<float> {
    using type = float;
    static constexpr bool needsConversion = false;
};

#endif // AETHER_CPP_MODE

} // namespace dtype
} // namespace aether
