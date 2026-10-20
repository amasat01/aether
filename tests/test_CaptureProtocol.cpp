// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Capture-protocol invariant (compile-time only, host build): no expression
// node holds a leaf operand by value. A leaf (`View`, `Item`, `Constant`) is
// held by `const&`; only composite nodes are held by value. A nested `select`
// over value leaves held by value is miscompiled by GCC's points-to analysis
// (see tests/hosttu/check_select_pta.sh), so the rule is a build failure here.

#include <type_traits>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/expr/nodes/Constant.h>

namespace aether_tests {
namespace {

using aether::Vec3d;

// One row per operand shape: the node's members must be references when the
// operand is a leaf. Binary nodes: both members; unary nodes: expr_.
using Item3 = Vec3d;
using Cst3 = aether::detail::Constant<double, aether::extents<3>>;
using CstB = aether::detail::Constant<bool, aether::extents<>>;

#define AETHER_LEAF_REF(member) std::is_reference_v<decltype(std::declval<Node&>().member)>

template<class Node>
constexpr bool binaryByRef = AETHER_LEAF_REF(l_) && AETHER_LEAF_REF(r_);
template<class Node>
constexpr bool unaryByRef = AETHER_LEAF_REF(expr_);
template<class Node>
constexpr bool selectByRef = AETHER_LEAF_REF(cond_) && AETHER_LEAF_REF(a_) && AETHER_LEAF_REF(b_);

#undef AETHER_LEAF_REF

template<class L, class R>
using SumT = decltype(std::declval<const L&>() + std::declval<const R&>());
template<class E>
using ScaleT = decltype(2.0 * std::declval<const E&>());
template<class C, class A, class B>
using SelT = decltype(aether::select(std::declval<const C&>(), std::declval<const A&>(), std::declval<const B&>()));

static_assert(binaryByRef<SumT<Item3, Item3>>);
static_assert(binaryByRef<SumT<Cst3, Cst3>>);
static_assert(binaryByRef<SumT<Item3, Cst3>>);
static_assert(unaryByRef<ScaleT<Item3>>);
static_assert(unaryByRef<ScaleT<Cst3>>);
static_assert(selectByRef<SelT<CstB, Cst3, Item3>>);
static_assert(selectByRef<SelT<CstB, Cst3, Cst3>>);
// a composite operand stays by value, a leaf next to it still by reference
static_assert(std::is_same_v<aether::detail::SelectOperand<Item3>, const Item3&>);
static_assert(std::is_same_v<aether::detail::SelectOperand<Cst3>, const Cst3&>);
static_assert(std::is_same_v<aether::detail::SelectOperand<SumT<Item3, Item3>>, const SumT<Item3, Item3>>);
static_assert(std::is_reference_v<decltype(std::declval<SelT<CstB, Cst3, SumT<Item3, Item3>>&>().a_)>);
static_assert(!std::is_reference_v<decltype(std::declval<SelT<CstB, Cst3, SumT<Item3, Item3>>&>().b_)>);

} // namespace
} // namespace aether_tests
