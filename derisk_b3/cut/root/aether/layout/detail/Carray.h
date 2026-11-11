// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Carray.h
 * @brief `aether::detail::Carray<T,N>`: a trivial, `AETHER_DEVICEHOST()`-safe
 *        fixed-size array aggregate.
 *
 * `std::array` is FORBIDDEN in device-visible headers (host-annotation risk
 * under nvcc — its member functions are not reliably usable from device
 * code across toolchain versions). `Carray` is the internal replacement:
 * a plain aggregate over `T v[N]`, nothing more.
 */

#include <cstddef>

#include "aether/macros.h"

namespace aether::detail {

/**
 * @brief Trivial fixed-size array aggregate. Aggregate-initializable
 *        (`Carray<std::size_t, 3>{a, b, c}`), trivially copyable whenever
 *        `T` is.
 */
template<class T, std::size_t N>
struct Carray {
    T v[N];

    AETHER_DEVICEHOST() constexpr T& operator[](std::size_t i) { return v[i]; }
    AETHER_DEVICEHOST() constexpr const T& operator[](std::size_t i) const { return v[i]; }

    AETHER_DEVICEHOST() constexpr std::size_t size() const { return N; }

    AETHER_DEVICEHOST() constexpr T* data() { return v; }
    AETHER_DEVICEHOST() constexpr const T* data() const { return v; }
};

/**
 * @brief Zero-size specialization. `T v[0]` is not standard C++ (a GCC/Clang
 *        extension, warned on under `-Wpedantic`), so the empty case is a
 *        stateless empty aggregate instead. No `operator[]` is provided —
 *        by construction (L2's `rank_dynamic() == 0` path) nothing ever
 *        reads an element of a zero-size `Carray`.
 */
template<class T>
struct Carray<T, 0> {
    AETHER_DEVICEHOST() constexpr std::size_t size() const { return 0; }

    AETHER_DEVICEHOST() constexpr T* data() { return nullptr; }
    AETHER_DEVICEHOST() constexpr const T* data() const { return nullptr; }
};

} // namespace aether::detail
