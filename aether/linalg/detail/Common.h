// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Common.h
 * @brief Shared constants and helpers of `aether::linalg`: the size limit,
 *        the working epsilon, the symmetrising read and the operand access.
 *
 * An operand is a raw row-major pointer or a per-sample operand: a proxy
 * `view[i]` of a plane, or an `Item`. A proxy is read and written element by
 * element through `rt::detail::at` (`aether/expr/Rt.h`), so a lane never
 * copies a plane it only reads.
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Rt.h"
#include "aether/macros.h"

namespace aether {
namespace linalg {

/** @brief Largest supported dimension; `eigh`'s sweep table ends here. */
inline constexpr std::size_t max_dim = 32;

namespace detail {

/** @brief `T` is `float` or `double` and `1 <= N <= max_dim`. */
template<class T, std::size_t N>
inline constexpr bool supported_v = (std::is_same_v<T, float> || std::is_same_v<T, double>) && N >= 1 && N <= max_dim;

/** @brief Machine epsilon (spacing of 1), spelled out so device code needs no `<limits>`. */
template<class T>
inline constexpr T epsilon_v = static_cast<T>(std::is_same_v<T, float> ? 1.1920928955078125e-07 : 2.220446049250313e-16);

/** @brief `P` is a per-sample operand (`view[i]` or an `Item`): it names its element type and extents. */
template<class P>
concept sample_operand = requires {
    typename std::remove_cvref_t<P>::element_type;
    typename std::remove_cvref_t<P>::element_extents;
};

/** @brief `P` is a matrix operand of the library: a row-major pointer (or array) or a per-sample operand. */
template<class P>
concept operand = std::is_pointer_v<std::decay_t<P>> || sample_operand<P>;

/** @brief The value type behind a row-major pointer or a per-sample operand. */
template<class A>
struct value_of {
    using type = rt::detail::elem_t<A>;
};
template<class T>
struct value_of<T*> {
    using type = std::remove_cv_t<T>;
};
template<class A>
using value_t = typename value_of<std::decay_t<A>>::type;

/**
 * @brief Entry `(r, c)` of an operand whose rows are `W` wide: `a[r * W + c]` of
 *        a pointer, `(r, c)` of a rank-2 operand, `r` of a rank-1 one.
 */
template<std::size_t W, class A>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() decltype(auto) elemAt(A&& a, std::size_t r, std::size_t c)
{
    if constexpr (std::is_pointer_v<std::decay_t<A>>) {
        return (a[r * W + c]);
    } else if constexpr (rt::detail::ext_t<A>::Rank == 1) {
        return rt::detail::at(a, r);
    } else {
        return rt::detail::at(a, r, c);
    }
}

/**
 * @brief Entry `(i, j)` of `(A + A^T) / 2`. Exact when `A` is symmetric, so a
 *        symmetric input is read back unchanged.
 */
template<std::size_t N, class A>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() value_t<A> symAt(const A& a, std::size_t i, std::size_t j)
{
    using T = value_t<A>;
    return T(0.5) * elemAt<N>(a, i, j) + T(0.5) * elemAt<N>(a, j, i);
}

/** @brief `x[0 .. n) = 0`. */
template<class T>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() void zeroFill(T* x, std::size_t n)
{
    AETHER_NO_UNROLL()
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = T(0);
    }
}

} // namespace detail
} // namespace linalg
} // namespace aether
