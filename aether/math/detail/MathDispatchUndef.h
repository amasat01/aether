// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file MathDispatchUndef.h
 * @brief Undefs the `AETHER_MATH_UNARY`/`AETHER_MATH_BINARY` dispatch-macro
 *        scaffold from `MathDispatch.h` (and its `*_BAND` banded
 *        companions), so it never leaks into consumer translation units.
 *        Paired `#include` at the bottom of any facade header that
 *        included `MathDispatch.h`.
 *
 * No `#pragma once` here either: this file is meant to be re-included every
 * time a facade header wants to close out its `MathDispatch.h` scope.
 */

#undef AETHER_MATH_UNARY
#undef AETHER_MATH_BINARY
#undef AETHER_MATH_UNARY_BAND
#undef AETHER_MATH_BINARY_BAND
