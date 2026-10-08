// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file MakeReshape.h
 * @brief `aether::reshaped<Ms...>(view)`: reshapes the trailing dynamic
 *        sample mode of a `layout_right` view into `Ms...`.
 *
 * `reshaped()` is the only `<string>`/`err::`-using content split out of
 * `Reshape.h`; the compile-time index metaprogramming it builds on
 * (`detail::SizeSeq`/`SeqConcat`/`DropLast`/`BuildExtents`) stays there.
 * Include this header explicitly (or `aether/aether.h`, which does)
 * wherever `reshaped()` is called.
 */

#include <cstddef>
#include <string>

#include "aether/err/Error.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/view/Reshape.h"
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Reshape the trailing (SAMPLE) mode of a `layout_right` view into
 *        `Ms...`. Exactly one of `Ms...` must be `dyn`; the product of the
 *        rest must divide `view.samples()` (throws `aether::Error`
 *        otherwise, host-side).
 */
template<std::size_t... Ms, class T, std::size_t... Es>
View<T, typename detail::BuildExtents<typename detail::DropLast<Es...>::prefix, Ms...>::type, layout_right> reshaped(
    const View<T, extents<Es...>, layout_right>& view)
{
    using SrcExt = extents<Es...>;
    static_assert(SrcExt::Rank >= 1, "reshaped: the source view must have rank >= 1");

    using DropSrc = detail::DropLast<Es...>;
    static_assert(DropSrc::last == dyn,
        "reshaped: the source view's trailing mode must be the dynamic SAMPLE mode");

    using PrefixExt = typename detail::BuildExtents<typename DropSrc::prefix>::type;
    static_assert(PrefixExt::RankDynamic == 0,
        "reshaped: every mode preceding the SAMPLE mode must be static");

    constexpr std::size_t nDyn = ((Ms == dyn ? std::size_t{ 1 } : std::size_t{ 0 }) + ... + std::size_t{ 0 });
    static_assert(nDyn == 1, "reshaped<Ms...>: exactly one of Ms... must be dyn");

    constexpr std::size_t staticProduct = ((Ms == dyn ? std::size_t{ 1 } : Ms) * ... * std::size_t{ 1 });

    const std::size_t samples = view.samples();
    if (samples % staticProduct != 0) {
        err::fail("reshaped", "view", samples,
            "static mode product " + std::to_string(staticProduct) + " does not divide samples()="
                + std::to_string(samples));
    }
    const std::size_t dynExtent = samples / staticProduct;

    using ResultExt = typename detail::BuildExtents<typename DropSrc::prefix, Ms...>::type;
    static_assert(ResultExt::RankDynamic == 1,
        "reshaped: internal invariant — the result extents must carry exactly one dynamic mode");

    ResultExt resultExt(dynExtent);
    typename layout_right::mapping<ResultExt> resultMap(resultExt);

    return View<T, ResultExt, layout_right>(view.data(), resultMap, view.device());
}

} // namespace aether
