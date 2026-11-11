// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Product.h
 * @brief Matrix product/Hadamard expression nodes: `MatVec`
 *        (rank-2 x rank-1 -> rank-1), `MatMat` (rank-2 x rank-2 -> rank-2),
 *        `Outer` (rank-1 x rank-1 -> rank-2), `CWiseMul` (rank-generic
 *        Hadamard product) and `MatFrobeniusFold` (the compile-time fold
 *        backing `Expression::matDot()`), plus the `operator*`/`outer()`
 *        free functions that construct `MatVec`/`MatMat`/`Outer` from
 *        natural call-site syntax (`M * v`, `A * B`, `outer(u, v)`).
 *
 * Also carries the "fused multi-load" assignment specializations:
 * `to = M * v` / `to = A * B` cache the reused operand (the vector for MatVec;
 * both matrices for MatMat) in a register-resident `Item` once via a
 * normal recursive `assign()` call (which itself cascades the same fused
 * specialization if that operand is itself a `MatVec`/`MatMat`), then
 * writes every output component from the cache instead of recomputing the
 * whole dot-product chain per row/component. Hooked through
 * `aether::detail::AssignDispatch` (`aether/expr/Assign.h`'s indirection —
 * see that header's docstring for why a class-template partial
 * specialization, not a function overload, is the only sound hook point
 * for a driver keyed on the from-expression's type).
 */

#include <cstddef>
#include <type_traits>

#include "aether/expr/Assign.h"
#include "aether/expr/Expression.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/macros.h"
#include "aether/view/Item.h"

namespace aether {
namespace detail {

// ---------------------------------------------------------------------------
// Compile-time dot-product folds (shared by the plain per-component eval()
// path AND the fused multi-load write path below).
// ---------------------------------------------------------------------------

/** @brief `sum_{c=0}^{C} m.eval<Row,c>(i) * v.eval<c>(i)` — one row of `M*v`. */
template<class M, class V, std::size_t Row, std::size_t C>
struct MatVecRowDot {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename M::element_type> eval(const M& m, const V& v, const SampleIndex& i)
    {
        return evalW<Row, C>(m, i) * evalW<C>(v, i) + MatVecRowDot<M, V, Row, C - 1>::eval(m, v, i);
    }
};
template<class M, class V, std::size_t Row>
struct MatVecRowDot<M, V, Row, 0> {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename M::element_type> eval(const M& m, const V& v, const SampleIndex& i)
    {
        return evalW<Row, 0>(m, i) * evalW<0>(v, i);
    }
};

/** @brief `sum_{k=0}^{K} l.eval<Row,k>(i) * r.eval<k,Col>(i)` — one component of `L*R`. */
template<class L, class R, std::size_t Row, std::size_t Col, std::size_t K>
struct MatMatDot {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename L::element_type> eval(const L& l, const R& r, const SampleIndex& i)
    {
        return evalW<Row, K>(l, i) * evalW<K, Col>(r, i) + MatMatDot<L, R, Row, Col, K - 1>::eval(l, r, i);
    }
};
template<class L, class R, std::size_t Row, std::size_t Col>
struct MatMatDot<L, R, Row, Col, 0> {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename L::element_type> eval(const L& l, const R& r, const SampleIndex& i)
    {
        return evalW<Row, 0>(l, i) * evalW<0, Col>(r, i);
    }
};

/** @brief Row-major linear fold `sum_{k=0}^{Rows*Cols-1} l.eval<k/Cols,k%Cols>(i) * r.eval<k/Cols,k%Cols>(i)`
 *         — backs `Expression::matDot()` (Frobenius inner product). */
template<class L, class R, std::size_t Rows, std::size_t Cols, std::size_t K>
struct MatFrobeniusFold {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename L::element_type> eval(const L& l, const R& r, const SampleIndex& i)
    {
        constexpr std::size_t Row = K / Cols;
        constexpr std::size_t Col = K % Cols;
        return evalW<Row, Col>(l, i) * evalW<Row, Col>(r, i)
            + MatFrobeniusFold<L, R, Rows, Cols, K - 1>::eval(l, r, i);
    }
};
template<class L, class R, std::size_t Rows, std::size_t Cols>
struct MatFrobeniusFold<L, R, Rows, Cols, 0> {
    AETHER_DEVICEHOST() static constexpr working_type_t<typename L::element_type> eval(const L& l, const R& r, const SampleIndex& i)
    {
        return evalW<0, 0>(l, i) * evalW<0, 0>(r, i);
    }
};

// ---------------------------------------------------------------------------
// Product expression nodes.
// ---------------------------------------------------------------------------

/**
 * @brief Matrix-vector product `M * v` (rank-2 x rank-1 -> rank-1 result;
 *        a VECTOR expression, so it composes with the whole vector algebra —
 *        `.dot()`, `.norm()`, further `+`, ...).
 */
template<class M, class V>
class MatVec : public Expression<MatVec<M, V>, typename M::element_type> {
    static_assert(M::element_extents::Rank == 2, "MatVec: M (left operand) must be a rank-2 (matrix) expression");
    static_assert(V::element_extents::Rank == 1, "MatVec: V (right operand) must be a rank-1 (vector) expression");
    static_assert(M::element_extents::static_extent(1) == V::element_extents::static_extent(0),
        "MatVec: M's column count must equal V's length");
    static_assert(std::is_same_v<typename M::element_type, typename V::element_type>, "MatVec: operand element_type must match");

public:
    using element_type = typename M::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<M::element_extents::static_extent(0)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<M::isLeaf, const M&, const M> m_;
    typename std::conditional_t<V::isLeaf, const V&, const V> v_;

    AETHER_DEVICEHOST() constexpr MatVec(const M& m, const V& v)
        : m_{ m }
        , v_{ v }
    {
    }

    template<std::size_t Row>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return MatVecRowDot<M, V, Row, M::element_extents::static_extent(1) - 1>::eval(m_, v_, i);
    }
};

/**
 * @brief Matrix-matrix product `L * R` (rank-2 x rank-2 -> rank-2 result).
 */
template<class L, class R>
class MatMat : public Expression<MatMat<L, R>, typename L::element_type> {
    static_assert(L::element_extents::Rank == 2, "MatMat: L (left operand) must be a rank-2 (matrix) expression");
    static_assert(R::element_extents::Rank == 2, "MatMat: R (right operand) must be a rank-2 (matrix) expression");
    static_assert(L::element_extents::static_extent(1) == R::element_extents::static_extent(0),
        "MatMat: L's column count must equal R's row count");
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>, "MatMat: operand element_type must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<L::element_extents::static_extent(0), R::element_extents::static_extent(1)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr MatMat(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t Row, std::size_t Col>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return MatMatDot<L, R, Row, Col, L::element_extents::static_extent(1) - 1>::eval(l_, r_, i);
    }
};

/**
 * @brief Vector outer product `u v^T` (rank-1 x rank-1 -> rank-2 result).
 *        Reached via the free `outer(u, v)` function below (aether has one
 *        namespace, so `aether::outer` is the only spelling needed).
 */
template<class U, class V>
class Outer : public Expression<Outer<U, V>, typename U::element_type> {
    static_assert(U::element_extents::Rank == 1, "outer: U (left operand) must be a rank-1 (vector) expression");
    static_assert(V::element_extents::Rank == 1, "outer: V (right operand) must be a rank-1 (vector) expression");
    static_assert(std::is_same_v<typename U::element_type, typename V::element_type>, "outer: operand element_type must match");

public:
    using element_type = typename U::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = extents<U::element_extents::static_extent(0), V::element_extents::static_extent(0)>;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<U::isLeaf, const U&, const U> u_;
    typename std::conditional_t<V::isLeaf, const V&, const V> v_;

    AETHER_DEVICEHOST() constexpr Outer(const U& u, const V& v)
        : u_{ u }
        , v_{ v }
    {
    }

    template<std::size_t Row, std::size_t Col>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<Row>(u_, i) * evalW<Col>(v_, i);
    }
};

/**
 * @brief Hadamard (component-wise) product `L .* R` — rank-generic (works
 *        for any matching `element_extents`, vector OR matrix), same shape
 *        as `Sum` (`aether/expr/nodes/Arithmetic.h`). Reached via
 *        `Expression::cwiseMul()`.
 */
template<class L, class R>
class CWiseMul : public Expression<CWiseMul<L, R>, typename L::element_type> {
    static_assert(std::is_same_v<typename L::element_type, typename R::element_type>, "cwiseMul: operand element_type must match");
    static_assert(std::is_same_v<typename L::element_extents, typename R::element_extents>,
        "cwiseMul: operand element_extents must match");

public:
    using element_type = typename L::element_type;
    using working_type = working_type_t<element_type>;
    using element_extents = typename L::element_extents;
    static constexpr bool isLeaf = false;

    typename std::conditional_t<L::isLeaf, const L&, const L> l_;
    typename std::conditional_t<R::isLeaf, const R&, const R> r_;

    AETHER_DEVICEHOST() constexpr CWiseMul(const L& l, const R& r)
        : l_{ l }
        , r_{ r }
    {
    }

    template<std::size_t... Is>
    AETHER_DEVICEHOST() constexpr working_type eval(const SampleIndex& i) const
    {
        return evalW<Is...>(l_, i) * evalW<Is...>(r_, i);
    }
};

// ---------------------------------------------------------------------------
// Fused multi-load assignment — see this
// file's docstring for why the hook is a partial specialization of
// aether::detail::AssignDispatch (aether/expr/Assign.h).
// ---------------------------------------------------------------------------

/**
 * @brief `true` for a register-resident LEAF (`Item`, `PacketItem`, the
 *        fused caches themselves, ...) that is NOT a `View` — i.e. an
 *        operand the fused multi-load specializations below may bind
 *        DIRECTLY (`const E&`) instead of copying it into a cache.
 *
 * The fused cache exists to avoid RE-LOADING an operand FROM MEMORY (a
 * `View`) once per output row/component — it must never DUPLICATE registers
 * an operand already occupies. Swapping `Item`->`FusedMatCache` alone
 * measured ZERO change (`twinAetherMatMat33Kernel` stayed at 56
 * registers): the twin kernel pre-materializes its `MatMat` operands into
 * `Item<double,3,3>` (`A = a[i].get(); B = b[i].get();`) BEFORE building
 * `A * B`, so the fused path was copying an ALREADY register-resident leaf
 * into a second buffer regardless of that buffer's type — pure duplication,
 * not a memory-reload savings. This trait lets `AssignDispatch<MatVec<...>>`
 * / `AssignDispatch<MatMat<...>>` skip the copy entirely for that case,
 * while still caching a `View` (genuine per-row/component memory reload) or
 * a composite/lazy expression (genuine RECOMPUTE avoidance) exactly as
 * before.
 */
template<class E>
struct is_view : std::false_type { };
template<class T, class Extents, class Layout, bool Volatile>
struct is_view<View<T, Extents, Layout, Volatile>> : std::true_type { };

template<class E>
inline constexpr bool is_register_leaf_v = E::isLeaf && !is_view<E>::value;

/**
 * @brief Minimal register-resident caches used by the fused multi-load
 *        `AssignDispatch` specializations below — ONLY for operands that
 *        are NOT already register-resident (`!is_register_leaf_v`, i.e. a
 *        `View` or a composite/lazy expression). Deliberately NOT `Item<T,
 *        Es...>`: leaner — no Expression CRTP base, no
 *        Zeros/Ones/Identity factories, no general Carray-offset
 *        `operator()` — just raw storage plus the `eval<Is...>` leaf-
 *        protocol pair `assign()`/`MatVecRowDot`/`MatMatDot` actually need.
 *        `element_type`/`element_extents` satisfy `assign()`'s static_asserts
 *        (aether/expr/Assign.h) without deriving from `Expression` at all.
 */
/* ★ The SLOTS hold the WORKING carrier, while `element_type` keeps
 * naming the STORAGE scalar. Both halves are load-bearing. `element_type` is
 * what `assign()`'s static_asserts match the SOURCE expression on, so it must
 * stay the tree's own scalar; the slot type is what `detail::storeWorking`
 * (`aether/expr/Assign.h`) keys the encode on, so making it the carrier is what
 * keeps a staged operand UNPACKED across the whole fold instead of paying an
 * encode into the buffer and a decode straight back out. `working_type_t<T>` IS
 * `T` for every native dtype, so this matches the previous layout there,
 * byte for byte. */
template<class T, std::size_t N>
struct FusedVecCache {
    using element_type = T;
    using working_type = working_type_t<T>;
    using element_extents = extents<N>;

    working_type data_[N];

    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr working_type& eval(const SampleIndex&)
    {
        return data_[I];
    }
    template<std::size_t I>
    AETHER_DEVICEHOST() constexpr const working_type& eval(const SampleIndex&) const
    {
        return data_[I];
    }
};

template<class T, std::size_t Rows, std::size_t Cols>
struct FusedMatCache {
    using element_type = T;
    using working_type = working_type_t<T>;
    using element_extents = extents<Rows, Cols>;

    working_type data_[Rows * Cols];

    template<std::size_t R, std::size_t C>
    AETHER_DEVICEHOST() constexpr working_type& eval(const SampleIndex&)
    {
        return data_[R * Cols + C];
    }
    template<std::size_t R, std::size_t C>
    AETHER_DEVICEHOST() constexpr const working_type& eval(const SampleIndex&) const
    {
        return data_[R * Cols + C];
    }
};

/**
 * @brief `to = M * v`: if `v` is already register-resident (`Item`,
 *        `PacketItem`, ... — `is_register_leaf_v<V>`), bind it DIRECTLY, no
 *        copy. Otherwise (a `View` — genuine per-row memory reload avoided;
 *        or a composite expression — genuine per-row RECOMPUTE avoided)
 *        cache it in a lean `FusedVecCache` ONCE, then stream `M`'s rows
 *        against the (possibly cached) vector. `+=`/`-=` fall back to the
 *        plain per-element path (no fused Add/Sub variant exists here
 *        either).
 */
template<class M, class V, AssignOp Op>
struct AssignDispatch<MatVec<M, V>, Op> {
    template<class ExprL>
    AETHER_DEVICEHOST() static constexpr void eval(const SampleIndex& i, ExprL& to, const MatVec<M, V>& from)
    {
        if constexpr (Op == AssignOp::Set) {
            if constexpr (is_register_leaf_v<V>) {
                // Already register-resident — bind directly, no cache copy.
                writeRows_<M::element_extents::static_extent(0) - 1>(i, to, from.m_, from.v_);
            } else {
                using VC = FusedVecCache<typename V::element_type, V::element_extents::static_extent(0)>;
                VC vc;
                assign(i, vc, from.v_); // cascades the fused path again if from.v_ is itself composite
                writeRows_<M::element_extents::static_extent(0) - 1>(i, to, from.m_, vc);
            }
        } else {
            RecursiveAssign<typename ExprL::element_extents, Op, 0>::eval(i, to, from);
        }
    }

private:
    template<std::size_t Row, class ExprL, class VC>
    AETHER_DEVICEHOST() static constexpr void writeRows_(const SampleIndex& i, ExprL& to, const M& m, const VC& vc)
    {
        storeWorking<AssignOp::Set>(
            to.template eval<Row>(i), MatVecRowDot<M, VC, Row, M::element_extents::static_extent(1) - 1>::eval(m, vc, i));
        if constexpr (Row > 0) {
            writeRows_<Row - 1>(i, to, m, vc);
        }
    }
};

/**
 * @brief `to = L * R`: cache each operand ONLY when it is NOT already
 *        register-resident (`!is_register_leaf_v` — a `View`, genuine
 *        per-component memory reload avoided; or a composite expression,
 *        genuine per-component RECOMPUTE avoided). An `Item`/`PacketItem`
 *        operand is bound DIRECTLY, no copy — `L` and `R` are judged
 *        INDEPENDENTLY, so e.g. `Item * View` caches only the `View` side.
 *        `each l component is reused Cols times, each r component Rows
 *        times` is exactly why a genuinely memory- or recompute-backed
 *        operand still gets cached ONCE up front.
 */
template<class L, class R, AssignOp Op>
struct AssignDispatch<MatMat<L, R>, Op> {
    template<class ExprL>
    AETHER_DEVICEHOST() static constexpr void eval(const SampleIndex& i, ExprL& to, const MatMat<L, R>& from)
    {
        if constexpr (Op == AssignOp::Set) {
            constexpr std::size_t Cols = MatMat<L, R>::element_extents::static_extent(1);
            constexpr std::size_t Last = MatMat<L, R>::element_extents::static_extent(0) * Cols - 1;
            constexpr bool LReg = is_register_leaf_v<L>;
            constexpr bool RReg = is_register_leaf_v<R>;
            if constexpr (LReg && RReg) {
                emitAll_<Last, Cols>(i, to, from.l_, from.r_);
            } else if constexpr (LReg && !RReg) {
                using RC = FusedMatCache<typename R::element_type, R::element_extents::static_extent(0), R::element_extents::static_extent(1)>;
                RC rc;
                assign(i, rc, from.r_); // cascades on nesting if from.r_ is itself composite
                emitAll_<Last, Cols>(i, to, from.l_, rc);
            } else if constexpr (!LReg && RReg) {
                using LC = FusedMatCache<typename L::element_type, L::element_extents::static_extent(0), L::element_extents::static_extent(1)>;
                LC lc;
                assign(i, lc, from.l_); // cascades on nesting if from.l_ is itself composite
                emitAll_<Last, Cols>(i, to, lc, from.r_);
            } else {
                using LC = FusedMatCache<typename L::element_type, L::element_extents::static_extent(0), L::element_extents::static_extent(1)>;
                using RC = FusedMatCache<typename R::element_type, R::element_extents::static_extent(0), R::element_extents::static_extent(1)>;
                LC lc;
                RC rc;
                assign(i, lc, from.l_); // cascades on nesting, same as the MatVec specialization above
                assign(i, rc, from.r_);
                emitAll_<Last, Cols>(i, to, lc, rc);
            }
        } else {
            RecursiveAssign<typename ExprL::element_extents, Op, 0>::eval(i, to, from);
        }
    }

private:
    /**
     * @brief DESTINATION STAGING.
     *
     * When `to` is a memory-backed `View`, the product is computed into a
     * register-resident `FusedMatCache` FIRST and only then copied out; the
     * direct component-by-component write is kept for a destination that is
     * ALREADY register-resident (an `Item`, or an outer fused cache in the
     * cascade case), where staging would be a pure register-to-register copy.
     *
     * WHY (measured, sm_61): `writeAll_`
     * stores each output component to `to` AS SOON AS its fold completes, so
     * a `View` destination interleaves 9 GLOBAL STORES with the operand
     * reads. `to`, `l` and `r` are plain `double*`-backed views with no
     * `restrict`/noalias information, so NVVM may not sink any operand load
     * BELOW the first of those stores — it therefore hoists ALL 18 operand
     * loads above them and every loaded double stays live at once
     * (`twinAetherMatMat33Kernel`: 18 `LDG` then 9 `DMUL`/18 `DFMA`, 56
     * registers). With the stores moved behind the whole fold the PTX
     * recovers a load/compute interleave (6 loads, fold, 3 loads, fold,
     * ...) and the kernel drops to 48 registers.
     *
     * Staging alone left a 48-vs-40 residual, and that half was never this
     * node's: at `-maxrregcount=42` the staged kernel already compiled with
     * 0 bytes stack and 0 bytes spills (the SAME kernel BEFORE staging
     * spills — 24 B stack, 20 B spill stores, 20 B spill loads), i.e. 48 was
     * ptxas ELECTING to issue all 18 operand `LDG`s up front because
     * registers were free. The remaining gap was a per-address COST
     * question — a 32-bit index chain versus aether's cheap 64-bit `IADD`
     * pointer chain — closed separately by `aether::offset_t` narrowing
     * (`aether/index/Offset.h`). With both landed the kernel sits at 40
     * registers. NEITHER HALF REACHES THE BUDGET ALONE — measured, same TU,
     * sm_61: 56 with neither, 48 with staging only, 54 with the narrowing
     * only, 40 with both (budget 42). Keep them together.
     */
    template<std::size_t Last, std::size_t Cols, class ExprL, class LC, class RC>
    AETHER_DEVICEHOST() static constexpr void emitAll_(const SampleIndex& i, ExprL& to, const LC& lc, const RC& rc)
    {
        if constexpr (is_view<ExprL>::value) {
            using OC = FusedMatCache<typename ExprL::element_type, ExprL::element_extents::static_extent(0),
                ExprL::element_extents::static_extent(1)>;
            OC oc;
            writeAll_<Last, Cols>(i, oc, lc, rc);
            RecursiveAssign<typename ExprL::element_extents, AssignOp::Set, 0>::eval(i, to, oc);
        } else {
            writeAll_<Last, Cols>(i, to, lc, rc);
        }
    }

    template<std::size_t K, std::size_t Cols, class ExprL, class LC, class RC>
    AETHER_DEVICEHOST() static constexpr void writeAll_(const SampleIndex& i, ExprL& to, const LC& lc, const RC& rc)
    {
        constexpr std::size_t Row = K / Cols;
        constexpr std::size_t Col = K % Cols;
        storeWorking<AssignOp::Set>(to.template eval<Row, Col>(i),
            MatMatDot<LC, RC, Row, Col, LC::element_extents::static_extent(1) - 1>::eval(lc, rc, i));
        if constexpr (K > 0) {
            writeAll_<K - 1, Cols>(i, to, lc, rc);
        }
    }
};

} // namespace detail

// ---------------------------------------------------------------------------
// operator*/outer() — natural call-site syntax (`M * v`, `A * B`, `outer(u, v)`).
// Scalar*expr / expr*scalar / expr/scalar already work rank-generically via
// aether/expr/Operations.h's existing `CWiseScale` overloads — no
// matrix-specific scalar leg is needed here.
// ---------------------------------------------------------------------------

/** @brief `M * v` — matrix-vector product (rank-2 x rank-1 -> rank-1 lazy expression). */
template<class L, class R>
    requires(aether_expression<L> && aether_expression<R> && L::element_extents::Rank == 2 && R::element_extents::Rank == 1)
AETHER_DEVICEHOST() constexpr detail::MatVec<L, R> operator*(const L& l, const R& r)
{
    return detail::MatVec<L, R>(l, r);
}

/** @brief `A * B` — matrix-matrix product (rank-2 x rank-2 -> rank-2 lazy expression). */
template<class L, class R>
    requires(aether_expression<L> && aether_expression<R> && L::element_extents::Rank == 2 && R::element_extents::Rank == 2)
AETHER_DEVICEHOST() constexpr detail::MatMat<L, R> operator*(const L& l, const R& r)
{
    return detail::MatMat<L, R>(l, r);
}

/** @brief `outer(u, v)` — vector outer product `u v^T` (rank-1 x rank-1 -> rank-2 lazy expression). */
template<class U, class V>
    requires(aether_expression<U> && aether_expression<V> && U::element_extents::Rank == 1 && V::element_extents::Rank == 1)
AETHER_DEVICEHOST() constexpr detail::Outer<U, V> outer(const U& u, const V& v)
{
    return detail::Outer<U, V>(u, v);
}

} // namespace aether
