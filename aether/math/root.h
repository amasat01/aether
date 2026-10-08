// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file root.h
 * @brief `aether::math` — the ROOT family: `sqrt` gains its BANDED
 *        companion, and `%rsqrt`/`%rsqrtCube`/`%cbrt`/`%hypot` are 3-way
 *        (`float`/`double`/`Band`) dispatch entries.
 *
 * Own independent include/define/undef cycle over `MathDispatch.h`, exactly
 * like `explog.h`'s — `math.h` includes this AFTER its own
 * `MathDispatchUndef.h` pass, so the macros this file (re-)defines never
 * collide with `math.h`'s own invocations, and this file's
 * `AETHER_MATH_UNARY_BAND(sqrt, sqrt)` is simply a SECOND overload of
 * `aether::math::sqrt` (Band-constrained; `math.h`'s own `sqrt` entry stays
 * float/double-only and untouched) landing in the same namespace scope by
 * the time both headers have been `#include`d.
 *
 * `%rsqrt`/`%rsqrtCube` have no `std::`/CUDA-intrinsic-uniform spelling
 * (`std::rsqrt` does not exist; CUDA's `rsqrtf`/`::%rsqrt` are device-only
 * hardware approximations), so they are hand-written templates — same shape
 * as `math.h`'s own `fma`/`sign`/`isfinite` — rather than routed through the
 * `AETHER_MATH_UNARY` macro's `(FLOATFN, DOUBLEFN, STDEXPR)` triple. This
 * scalar (non-`Band`) leg is a plain convenience overload so the `_BAND`
 * companion macro has a name to attach to; it carries none of the `Band`
 * overload's certification (that is the `Band` overload's job, via
 * `BandedFacade`) and is not claimed host/device bit-identical.
 *
 * `%cbrt`/`%hypot` DO have standard `std::`/CUDA spellings (`std::cbrt`/
 * `cbrtf`/`::%cbrt`, `std::hypot`/`hypotf`/`::%hypot`), so they use the same
 * `AETHER_MATH_UNARY`/`AETHER_MATH_BINARY` scaffold `math.h`'s `abs`/`fmax`/…
 * do.
 *
 * Every Band-typed entry point routes through `BandedFacade<T>` (additions
 * to `aether/banded/BandedRealOps.h`), so this file names NO
 * `aether::banded::` symbol directly — same header-diet discipline
 * `math.h`'s own docstring describes.
 */

#include <cmath>
#include <type_traits>

#include "aether/macros.h"
#include "aether/math/detail/MathDispatch.h"

namespace aether {
namespace math {

/** @brief …`sqrt` on a banded operand: forwards to the certified
 *         `RsqrtCore.h::sqrt_` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(sqrt, sqrt)

/**
 * @brief Reciprocal square root, `1/sqrt(x)`. Device routes to the hardware
 *        SFU intrinsic (`rsqrtf`/`::%rsqrt`); host computes `1/sqrt(x)`
 *        (under `AETHER_HOST_VECTOR_MATH`, the faithfully rounded packet
 *        `rsqrt` for `double`) — NOT claimed bit-identical between the
 *        routes: the seed itself differs by construction, exactly as the
 *        certified `Band` family's own `%rsqrt` is not.
 */
template<class T>
AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T rsqrt(T x)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "aether::math::rsqrt: T must be float or double");
#if defined(AETHER_DEVICE_COMPILE)
    if constexpr (std::is_same_v<T, float>) {
        return rsqrtf(x);
    } else {
        return ::rsqrt(x);
    }
#else
    return AETHER_MATH_HOST_ROUTE_OR(rsqrt, T(1) / std::sqrt(x), x);
#endif
}
/** @brief …`%rsqrt` on a banded operand: forwards to `BandRoot.h`'s
 *         `rsqrtIeee` (the `%rsqrt(+Inf)=+0` fix) via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(rsqrt, rsqrt)

/**
 * @brief `x^(-3/2)`. No CUDA/`std::` intrinsic exists; the scalar (non-
 *        `Band`) leg composes this file's own `%rsqrt` rather than
 *        duplicating a seed/refine schedule — unlike the certified `Band`
 *        overload (`RsqrtCore.h::%rsqrtCube`), which fuses to avoid
 *        materializing the root, there is no certified-bound reason to fuse
 *        a plain `float`/`double` composition.
 */
template<class T>
AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T rsqrtCube(T x)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "aether::math::rsqrtCube: T must be float or double");
    const T r = rsqrt(x);
    return r * r * r;
}
/** @brief …`%rsqrtCube` on a banded operand: forwards to the certified,
 *         FUSED `RsqrtCore.h::%rsqrtCube` via `BandedFacade` — not this
 *         file's composed scalar body. */
AETHER_MATH_UNARY_BAND(rsqrtCube, rsqrtCube)

/** @brief Cube root. */
AETHER_MATH_UNARY(cbrt, cbrtf, ::cbrt, AETHER_MATH_HOST_ROUTE(cbrt, x))
/** @brief …`%cbrt` on a banded operand: forwards to `BandRoot.h`'s
 *         `%cbrt` via `BandedFacade`. */
AETHER_MATH_UNARY_BAND(cbrt, cbrt)

/** @brief `sqrt(a^2 + b^2)`, overflow-/underflow-safe. */
AETHER_MATH_BINARY(hypot, hypotf, ::hypot, AETHER_MATH_HOST_ROUTE(hypot, a, b))
/** @brief …`%hypot` on a banded operand: forwards to `BandRoot.h`'s
 *         `%hypot` via `BandedFacade`. */
AETHER_MATH_BINARY_BAND(hypot, hypot)

} // namespace math
} // namespace aether

#include "aether/math/detail/MathDispatchUndef.h"
