// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file HostVectorMathRoute.h
 * @brief `AETHER_MATH_HOST_ROUTE(name, args...)` — the host-leg spelling the
 *        `aether::math` facade uses for its transcendental functions.
 *
 * Default (no `AETHER_HOST_VECTOR_MATH`, or an nvcc pass): expands to
 * `std::name(args...)`, token for token what the facade spelled before, so
 * the default build is unchanged. With `AETHER_HOST_VECTOR_MATH` defined in
 * a GCC host compile: expands to `aether::math::detail::hostvec::name(...)`,
 * the auto-vectoriser hook in `HostVectorMath.h`.
 *
 * `AETHER_MATH_HOST_ABI` is empty by default; under the hook it is an ABI
 * tag on the host facade functions, so a hooked instantiation
 * (`aether::math::pow<double>` calling the vector-ABI symbol) never shares
 * a mangled name — and so never gets merged at link time — with the
 * default instantiation in another TU or library.
 */

#if defined(AETHER_HOST_VECTOR_MATH) && !defined(__CUDACC__)
#include "aether/math/detail/HostVectorMath.h"
#define AETHER_MATH_HOST_ROUTE(NAME, ...) ::aether::math::detail::hostvec::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ROUTE_OR(NAME, FALLBACK, ...) ::aether::math::detail::hostvec::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ABI [[gnu::abi_tag("aether_hvm")]]
#else
#define AETHER_MATH_HOST_ROUTE(NAME, ...) std::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ROUTE_OR(NAME, FALLBACK, ...) FALLBACK
#define AETHER_MATH_HOST_ABI
#endif
