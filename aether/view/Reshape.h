// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Reshape.h
 * @brief `aether::reshaped<Ms...>(view)`: replaces a `layout_right` view's
 *        trailing dynamic sample mode with modes `Ms...`, where exactly one
 *        of `Ms...` is `dyn` and the product of the static ones must divide
 *        `view.samples()`. Returns a `View` over the same data with a
 *        recomputed `layout_right` mapping — no copy, no reallocation.
 *
 * The `reshaped()` free function itself (the only `<string>`/`err::`-using
 * content) lives in `aether/view/MakeReshape.h`; the compile-time index
 * metaprogramming below (`detail::SizeSeq`/`SeqConcat`/`DropLast`/
 * `BuildExtents`) stays here, so this header is standalone under NVRTC.
 * Include `aether/view/MakeReshape.h` (or `aether/aether.h`, which does)
 * wherever `reshaped()` is called.
 */

#include <cstddef>

#include "aether/layout/Extents.h"

namespace aether {

namespace detail {

/** @brief A bare compile-time list of `std::size_t` — a splice-able tag type. */
template<std::size_t...>
struct SizeSeq { };

template<class Seq1, class Seq2>
struct SeqConcat;
template<std::size_t... A, std::size_t... B>
struct SeqConcat<SizeSeq<A...>, SizeSeq<B...>> {
    using type = SizeSeq<A..., B...>;
};

/** @brief `prefix` = `Es...` without its last element; `last` = that element. */
template<std::size_t... Es>
struct DropLast;

template<std::size_t E0, std::size_t... Rest>
struct DropLast<E0, Rest...> {
    using Sub    = DropLast<Rest...>;
    using prefix = typename SeqConcat<SizeSeq<E0>, typename Sub::prefix>::type;
    static constexpr std::size_t last = Sub::last;
};

template<std::size_t E0>
struct DropLast<E0> {
    using prefix                      = SizeSeq<>;
    static constexpr std::size_t last = E0;
};

/** @brief `extents<Prefix..., Suffix...>` from a `SizeSeq<Prefix...>` and a `Suffix...` pack. */
template<class PrefixSeq, std::size_t... Suffix>
struct BuildExtents;
template<std::size_t... Prefix, std::size_t... Suffix>
struct BuildExtents<SizeSeq<Prefix...>, Suffix...> {
    using type = extents<Prefix..., Suffix...>;
};

} // namespace detail

} // namespace aether
