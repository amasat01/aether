// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Expression.h
 * @brief `aether::Expression<Derived,T>`: the CRTP base every
 *        expression-template node/leaf derives from, plus the
 *        `aether::aether_expression<E>` concept that gates every operator
 *        and assignment path in `expr/`.
 *
 * Rank-generic element protocol: a conforming expression type `E` exposes
 *   - `using element_type = T;`                       the storage scalar
 *     type - what a leaf holds and what an assignment terminal writes back.
 *   - `using working_type = working_type_t<T>;`        the on-register carrier
 *     (`aether/dtype/WorkingType.h`) - what every node's `eval()` returns and
 *     computes in. `working_type` is `element_type` for every native dtype, so
 *     this is a no-op there by construction; for an emulated scalar it is the
 *     difference between one encode per node and one encode per store.
 *   - `using element_extents = extents<...all-static...>;`  the element's
 *     own shape (rank-1 for a vector, rank-2 for a matrix, ...) — never
 *     carries a `dyn` mode; the batch/sample dimension is threaded through
 *     the `SampleIndex` argument instead, never through `element_extents`.
 *   - `static constexpr bool isLeaf;`                  true for storage
 *     (`Item`, `View`), false for composite nodes (`Sum`, `CWiseScale`, ...).
 *   - `template<std::size_t... Is> AETHER_DEVICEHOST() ... eval(SampleIndex) const;`
 *     `Is...` is the static element multi-index (matching `element_extents`'s
 *     rank); the sample rides the `SampleIndex` argument.
 *
 * CRTP, not deducing-this: nvcc 12.9's device-code ceiling is C++20, which
 * has no deducing-this — every device-visible expression-template idiom in
 * this workspace stays on the classic CRTP pattern.
 *
 * `Expression` also carries the geometric (`cross`/`unitVector`), reduction
 * (`dot`/`norm`/`squaredNorm`/`cubedNorm`/`rNorm`/`rSquaredNorm`/
 * `rCubedNorm`/`maxNorm`/`sum`), quaternion (`quatMul`/`quatConj`/
 * `quatReciprocal`/`quatRotate`/`asPureQuaternion`/`asBack3DVector`) and
 * slicing (`segment`/`head`/`tail`) member functions — all of them inline,
 * directly in the class body here: this class forward-declares its node
 * class templates (`Cross`, `QuatConj`, ...) in its own `detail` namespace
 * immediately above the `Expression` class body, then defines member
 * functions whose bodies construct those (still-incomplete-at-this-point)
 * node types. This works because a class template's ordinary member
 * function bodies are only instantiated when actually ODR-used (never
 * merely by `Expression<Derived,T>` itself being instantiated as a CRTP
 * base) — by the time any consumer TU actually calls `.cross()`/`.dot()`/
 * `.quatMul()`/`.segment()`, the real node definitions (`aether/expr/nodes/
 * {Geometric,Quaternion,Structural}.h`) have been `#include`d (via the
 * `aether.h` umbrella, or directly). The alternative — declaring these
 * members here and defining them out-of-line in each node header (the
 * pattern `aether/expr/Assign.h` uses for `Item`'s expression ctor) — was
 * considered and rejected for this batch: it would require every dependent
 * return type (e.g. `tail<N>()`'s `Segment<Derived, Extent-N, N>`) to be
 * repeated token-identically between the in-class declaration and the
 * out-of-class definition (the nvcc trap `aether/view/Item.h` already
 * documents for `requires`-clauses), for no benefit here since none of these
 * members need to be visible before their owning node header is available
 * anyway. `aether/expr/Reduce.h` is the one exception: its
 * `detail::GenericReduce`/`GenericBinaryReduce` helpers are dependency-free
 * (no reference back to `Expression`), so this file `#include`s it directly
 * instead of forward-declaring yet more templates.
 */

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "aether/dtype/WorkingType.h"
#include "aether/expr/Reduce.h"
#include "aether/index/SampleIndex.h"
#include "aether/macros.h"
#include "aether/math/math.h"

namespace aether {
namespace detail {

// Forward declarations only — full definitions live in
// aether/expr/nodes/{Geometric,Quaternion,Structural}.h. These types are
// only ever named as dependent template-ids inside Expression's member
// function bodies, so they need not be complete until actually called.
template<class L, class R>
class Cross;
template<class E>
class UnitVector;
template<class L, class R>
class QuatMul;
template<class E>
class QuatConj;
template<class Q, class V>
class QuatRotate;
template<class E>
class AsPureQuaternion;
template<class E, std::size_t Off, std::size_t Len>
class Segment;
// Full definitions in aether/expr/nodes/Structural.h (Transpose/
// Row/Col/Block) and aether/expr/nodes/Product.h (CWiseMul, MatFrobeniusFold;
// MatVec/MatMat/Outer are reached via operator*()/outer(), not a member
// function, so they need no forward declaration here).
template<class E>
class Transpose;
template<class E, std::size_t R>
class Row;
template<class E, std::size_t C>
class Col;
template<class E, std::size_t R0, std::size_t C0, std::size_t BR, std::size_t BC>
class Block;
template<class L, class R>
class CWiseMul;
template<class L, class R, std::size_t Rows, std::size_t Cols, std::size_t K>
struct MatFrobeniusFold;
// Full definition in aether/expr/nodes/Inverse.h. trace()/
// det() need no forward declaration (eager scalars, no node type).
template<class E>
class Inverse;

/**
 * @brief Read one component of an expression in the working carrier — the
 *        single spelling every node uses for an operand read.
 *
 * A leaf (`Item`, `View`, a distribution leaf, a fused cache) hands back a
 * reference to what it stores; a composite node hands back the carrier
 * already. `toWorkingValue` is idempotent, so one call is correct for both
 * and no `E::isLeaf` branch is needed here. For every native dtype this is a
 * copy of a `const double&`.
 */
template<std::size_t... Is, class E>
AETHER_DEVICEHOST() AETHER_FORCEINLINE() constexpr working_type_t<typename E::element_type> evalW(
    const E& e, const SampleIndex& i)
{
    return toWorkingValue(e.template eval<Is...>(i));
}

} // namespace detail

/**
 * @brief CRTP base for every expression-template leaf/node. A tag base
 *        (`element_type` alias only — no state) that anchors the
 *        `aether_expression` concept via `std::derived_from`, plus the
 *        geometric/reduction/quaternion/slicing member-function surface
 *        described in the file docstring above. Adding members here does
 *        not change any already-derived type's ABI (still an empty base —
 *        every new member is either state-free or template-only).
 */
template<class Derived, class T>
class Expression {
public:
    /** @brief The STORAGE scalar type (matches `Derived::element_type`). */
    using element_type = T;
    /** @brief The on-register carrier every node's `eval()` returns.
     *         `== element_type` for every native dtype. */
    using working_type = working_type_t<T>;

    // -------------------------------------------------------------------
    // Reductions (aether/expr/Reduce.h backs these; rank-1 only).
    // Sample-free: evaluated at `SampleIndex::make(0)`, matching `Item`'s own
    // expression-ctor convention — intended for an already-materialized
    // (Item-level) expression; batched access goes through `view[i].get()`
    // first.
    // -------------------------------------------------------------------

    /** @brief Dot product with another rank-1 expression of the same shape. */
    template<class E>
    AETHER_DEVICEHOST() constexpr element_type dot(const Expression<E, typename E::element_type>& other) const
    {
        static_assert(Derived::element_extents::Rank == 1, "dot: only rank-1 expressions are supported");
        static_assert(std::is_same_v<typename Derived::element_extents, typename E::element_extents>,
            "dot: operand element_extents must match");
        return detail::fromWorkingValue<element_type>(
            detail::GenericBinaryReduce<Derived::element_extents::static_extent(0), Derived, E,
                detail::SumOp<working_type>, detail::TimesOp<working_type>, working_type>::
                eval(static_cast<const Derived&>(*this), static_cast<const E&>(other), SampleIndex::make(0)));
    }

    /** @brief Squared L2 norm — sum of squares of components. */
    AETHER_DEVICEHOST() constexpr element_type squaredNorm() const
    {
        return detail::fromWorkingValue<element_type>(squaredNormWorking_());
    }

    /**
     * @brief Squared L2 norm left IN THE WORKING CARRIER - the shared body the
     *        four norm accessors below fold on top of, so a `norm()` or a
     *        `rCubedNorm()` pays ONE encode at its own return rather than one
     *        per intermediate. Internal (trailing underscore); the public
     *        spelling is `squaredNorm()` directly above.
     */
    AETHER_DEVICEHOST() constexpr working_type squaredNormWorking_() const
    {
        static_assert(Derived::element_extents::Rank == 1, "squaredNorm: only rank-1 expressions are supported");
        return detail::GenericReduce<Derived::element_extents::static_extent(0), Derived, detail::SumOp<working_type>,
            detail::SquareOp<working_type>, working_type>::eval(static_cast<const Derived&>(*this), SampleIndex::make(0));
    }

    /** @brief L2 norm. */
    AETHER_DEVICEHOST() constexpr element_type norm() const
    {
        return detail::fromWorkingValue<element_type>(math::sqrt(squaredNormWorking_()));
    }

    /** @brief Reciprocal of the L2 norm (accurate division, not a fast rsqrt approximation). */
    AETHER_DEVICEHOST() constexpr element_type rNorm() const
    {
        return detail::fromWorkingValue<element_type>(element_type{ 1 } / math::sqrt(squaredNormWorking_()));
    }

    /** @brief Reciprocal of the squared L2 norm. */
    AETHER_DEVICEHOST() constexpr element_type rSquaredNorm() const
    {
        return detail::fromWorkingValue<element_type>(element_type{ 1 } / squaredNormWorking_());
    }

    /** @brief Cubed L2 norm (`‖v‖³`). */
    AETHER_DEVICEHOST() constexpr element_type cubedNorm() const
    {
        const working_type s = squaredNormWorking_();
        return detail::fromWorkingValue<element_type>(s * math::sqrt(s));
    }

    /** @brief Reciprocal of the cubed L2 norm (`1/‖v‖³`). */
    AETHER_DEVICEHOST() constexpr element_type rCubedNorm() const
    {
        const working_type s = squaredNormWorking_();
        return detail::fromWorkingValue<element_type>(element_type{ 1 } / (s * math::sqrt(s)));
    }

    /** @brief L-infinity (max-abs) norm. */
    AETHER_DEVICEHOST() constexpr element_type maxNorm() const
    {
        static_assert(Derived::element_extents::Rank == 1, "maxNorm: only rank-1 expressions are supported");
        return detail::fromWorkingValue<element_type>(
            detail::GenericReduce<Derived::element_extents::static_extent(0), Derived, detail::MaxOp<working_type>,
                detail::AbsOp<working_type>, working_type>::eval(static_cast<const Derived&>(*this), SampleIndex::make(0)));
    }

    /** @brief Sum of all components. */
    AETHER_DEVICEHOST() constexpr element_type sum() const
    {
        static_assert(Derived::element_extents::Rank == 1, "sum: only rank-1 expressions are supported");
        return detail::fromWorkingValue<element_type>(
            detail::GenericReduce<Derived::element_extents::static_extent(0), Derived, detail::SumOp<working_type>,
                detail::SelfOp<working_type>, working_type>::eval(static_cast<const Derived&>(*this), SampleIndex::make(0)));
    }

    // -------------------------------------------------------------------
    // Geometric (aether/expr/nodes/Geometric.h).
    // -------------------------------------------------------------------

    /** @brief Cross product with another 3-vector expression; lazy (evaluate by assigning to a 3-shaped target). */
    template<class E>
    AETHER_DEVICEHOST() constexpr auto cross(const Expression<E, typename E::element_type>& other) const
    {
        return detail::Cross<Derived, E>(static_cast<const Derived&>(*this), static_cast<const E&>(other));
    }

    /** @brief Unit-vector (normalized) view of this expression; lazy. */
    AETHER_DEVICEHOST() constexpr auto unitVector() const { return detail::UnitVector<Derived>(static_cast<const Derived&>(*this)); }

    // -------------------------------------------------------------------
    // Quaternions (aether/expr/nodes/Quaternion.h); extents<4>/<3>.
    // -------------------------------------------------------------------

    /** @brief Quaternion (Hamilton) product `*this ⊗ other`; lazy. */
    template<class E>
    AETHER_DEVICEHOST() constexpr auto quatMul(const Expression<E, typename E::element_type>& other) const
    {
        return detail::QuatMul<Derived, E>(static_cast<const Derived&>(*this), static_cast<const E&>(other));
    }

    /** @brief Quaternion conjugate (negate the vector part); lazy. */
    AETHER_DEVICEHOST() constexpr auto quatConj() const { return detail::QuatConj<Derived>(static_cast<const Derived&>(*this)); }

    /** @brief Quaternion reciprocal (`q⁻¹ = q* / ‖q‖²`); lazy, reuses `quatConj()` and `rSquaredNorm()`. */
    AETHER_DEVICEHOST() constexpr auto quatReciprocal() const { return quatConj() * rSquaredNorm(); }

    /** @brief Rodrigues-formula rotation of 3-vector `vec` by this unit quaternion; lazy. */
    template<class E>
    AETHER_DEVICEHOST() constexpr auto quatRotate(const Expression<E, typename E::element_type>& vec) const
    {
        return detail::QuatRotate<Derived, E>(static_cast<const Derived&>(*this), static_cast<const E&>(vec));
    }

    /** @brief View this 3-vector as a pure quaternion `[0,x,y,z]`; lazy. */
    AETHER_DEVICEHOST() constexpr auto asPureQuaternion() const
    {
        return detail::AsPureQuaternion<Derived>(static_cast<const Derived&>(*this));
    }

    /** @brief View this quaternion's vector part (drop the scalar, index 0); reuses `tail<3>`. */
    AETHER_DEVICEHOST() constexpr auto asBack3DVector() const { return tail<3>(); }

    // -------------------------------------------------------------------
    // Slicing (aether/expr/nodes/Structural.h); rank-1, read-only.
    // -------------------------------------------------------------------

    /** @brief Read-only view of `Len` consecutive components starting at `Off`. */
    template<std::size_t Off, std::size_t Len>
    AETHER_DEVICEHOST() constexpr auto segment() const
    {
        return detail::Segment<Derived, Off, Len>(static_cast<const Derived&>(*this));
    }

    /** @brief Read-only view of the first `N` components. */
    template<std::size_t N>
    AETHER_DEVICEHOST() constexpr auto head() const
    {
        return segment<0, N>();
    }

    /** @brief Read-only view of the last `N` components. */
    template<std::size_t N>
    AETHER_DEVICEHOST() constexpr auto tail() const
    {
        return segment<Derived::element_extents::static_extent(0) - N, N>();
    }

    // -------------------------------------------------------------------
    // Matrix structural views (aether/expr/nodes/Structural.h) and
    // Hadamard/Frobenius (aether/expr/nodes/Product.h). `row`/`col`/`block`/
    // `transpose` are rank-2-only (static_assert lives in the node
    // constructor, same forward-declared-and-defined-on-first-call scheme as
    // the members above). `cwiseMul` is rank-generic (same shape as
    // `+`/`-`); MatVec/MatMat (`A*x`/`A*B`) and `outer(u,v)` are not
    // members — they are reached via `operator*`/free `outer()` in
    // `aether/expr/nodes/Product.h`.
    // -------------------------------------------------------------------

    /** @brief Read-only transpose of this rank-2 (matrix) expression; lazy. */
    AETHER_DEVICEHOST() constexpr auto transpose() const { return detail::Transpose<Derived>(static_cast<const Derived&>(*this)); }

    /** @brief Read-only view of row `R` of this rank-2 expression (a rank-1/vector expression); lazy. */
    template<std::size_t R>
    AETHER_DEVICEHOST() constexpr auto row() const
    {
        return detail::Row<Derived, R>(static_cast<const Derived&>(*this));
    }

    /** @brief Read-only view of column `C` of this rank-2 expression (a rank-1/vector expression); lazy. */
    template<std::size_t C>
    AETHER_DEVICEHOST() constexpr auto col() const
    {
        return detail::Col<Derived, C>(static_cast<const Derived&>(*this));
    }

    /** @brief Read-only view of a `BR x BC` block starting at `(R0, C0)` of this rank-2 expression; lazy. */
    template<std::size_t R0, std::size_t C0, std::size_t BR, std::size_t BC>
    AETHER_DEVICEHOST() constexpr auto block() const
    {
        return detail::Block<Derived, R0, C0, BR, BC>(static_cast<const Derived&>(*this));
    }

    /** @brief Hadamard (component-wise) product with another expression of the same `element_extents`; lazy. */
    template<class E>
    AETHER_DEVICEHOST() constexpr auto cwiseMul(const Expression<E, typename E::element_type>& other) const
    {
        return detail::CWiseMul<Derived, E>(static_cast<const Derived&>(*this), static_cast<const E&>(other));
    }

    /** @brief Frobenius inner product `sum_{r,c} this(r,c) * other(r,c)` of two rank-2 expressions; evaluates eagerly (sample-free, `SampleIndex::make(0)`, matching `dot()`'s convention). */
    template<class E>
    AETHER_DEVICEHOST() constexpr element_type matDot(const Expression<E, typename E::element_type>& other) const
    {
        static_assert(Derived::element_extents::Rank == 2, "matDot: only rank-2 (matrix) expressions are supported");
        static_assert(std::is_same_v<typename Derived::element_extents, typename E::element_extents>,
            "matDot: operand element_extents must match");
        constexpr std::size_t Rows = Derived::element_extents::static_extent(0);
        constexpr std::size_t Cols = Derived::element_extents::static_extent(1);
        return detail::fromWorkingValue<element_type>(
            detail::MatFrobeniusFold<Derived, E, Rows, Cols, Rows * Cols - 1>::eval(
                static_cast<const Derived&>(*this), static_cast<const E&>(other), SampleIndex::make(0)));
    }

    // -------------------------------------------------------------------
    // Small-matrix closed forms (aether/expr/nodes/Inverse.h backs
    // `inverse()`; `trace()`/`det()` are eager scalars, same sample-free,
    // `SampleIndex::make(0)` convention as `dot()`/`matDot()` above,
    // intended for an already-materialized (Item-level) expression).
    // Square rank-2, 2x2 or 3x3 only: a non-square or non-2x2/3x3 shape
    // hard `static_assert`s. General LU/solve is not implemented.
    // -------------------------------------------------------------------

    /** @brief Trace (sum of the diagonal) of a square 2x2 or 3x3 rank-2 expression. */
    AETHER_DEVICEHOST() constexpr element_type trace() const
    {
        static_assert(Derived::element_extents::Rank == 2, "trace: only rank-2 (matrix) expressions are supported");
        constexpr std::size_t N = Derived::element_extents::static_extent(0);
        static_assert(N == Derived::element_extents::static_extent(1), "trace: matrix must be square");
        static_assert(N == 2 || N == 3, "trace: only 2x2 and 3x3 closed forms are implemented");
        const Derived& self = static_cast<const Derived&>(*this);
        const SampleIndex i0 = SampleIndex::make(0);
        if constexpr (N == 2) {
            return detail::fromWorkingValue<element_type>(
                detail::evalW<0, 0>(self, i0) + detail::evalW<1, 1>(self, i0));
        } else {
            return detail::fromWorkingValue<element_type>(
                detail::evalW<0, 0>(self, i0) + detail::evalW<1, 1>(self, i0) + detail::evalW<2, 2>(self, i0));
        }
    }

    /** @brief Determinant of a square 2x2 or 3x3 rank-2 expression (closed-form cofactor expansion). */
    AETHER_DEVICEHOST() constexpr element_type det() const
    {
        static_assert(Derived::element_extents::Rank == 2, "det: only rank-2 (matrix) expressions are supported");
        constexpr std::size_t N = Derived::element_extents::static_extent(0);
        static_assert(N == Derived::element_extents::static_extent(1), "det: matrix must be square");
        static_assert(N == 2 || N == 3, "det: only 2x2 and 3x3 closed forms are implemented");
        const Derived& self = static_cast<const Derived&>(*this);
        const SampleIndex i0 = SampleIndex::make(0);
        if constexpr (N == 2) {
            const working_type a = detail::evalW<0, 0>(self, i0);
            const working_type b = detail::evalW<0, 1>(self, i0);
            const working_type c = detail::evalW<1, 0>(self, i0);
            const working_type d = detail::evalW<1, 1>(self, i0);
            return detail::fromWorkingValue<element_type>(a * d - b * c);
        } else {
            const working_type a = detail::evalW<0, 0>(self, i0);
            const working_type b = detail::evalW<0, 1>(self, i0);
            const working_type c = detail::evalW<0, 2>(self, i0);
            const working_type d = detail::evalW<1, 0>(self, i0);
            const working_type e = detail::evalW<1, 1>(self, i0);
            const working_type f = detail::evalW<1, 2>(self, i0);
            const working_type g = detail::evalW<2, 0>(self, i0);
            const working_type h = detail::evalW<2, 1>(self, i0);
            const working_type k = detail::evalW<2, 2>(self, i0);
            return detail::fromWorkingValue<element_type>(
                a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g));
        }
    }

    /**
     * @brief Closed-form inverse of a square 2x2 or 3x3 rank-2 expression;
     *        lazy — each component is an INDEPENDENT recompute of the
     *        source expression's components plus the determinant (matches
     *        `Cross`'s own per-component recompute convention,
     *        `aether/expr/nodes/Geometric.h`'s docstring), so it is safe to
     *        assign into a batched destination (each `eval<R,C>(i)` uses the
     *        `i` it is actually called with, unlike `trace()`/`det()` above).
     */
    AETHER_DEVICEHOST() constexpr auto inverse() const { return detail::Inverse<Derived>(static_cast<const Derived&>(*this)); }
};

namespace detail {

/** @brief Structural half of `aether_expression` — the member surface every
 *         conforming expression type must expose. Split out so the
 *         `std::derived_from` check below short-circuits cleanly when `E`
 *         does not even have an `element_type` (concept `&&` is guaranteed
 *         to evaluate left-to-right with short-circuit on failure). */
template<class E>
concept aether_expression_structural = requires {
    typename E::element_type;
    typename E::element_extents;
    { E::element_extents::Rank } -> std::convertible_to<std::size_t>;
    { E::isLeaf } -> std::convertible_to<bool>;
};

/**
 * @brief `true` for a leaf that IS its value rather than a window onto
 *        storage — `Item` (`aether/view/Item.h`) and `Constant`
 *        (`aether/expr/nodes/Constant.h`) specialise it. Every other leaf
 *        (a `View`, a table, a fused cache) stays `false`.
 */
template<class E>
inline constexpr bool isValueLeaf = false;

/**
 * @brief How `Select` holds an operand: a storage leaf by `const&` (the
 *        capture protocol every node uses), a composite node or a value leaf
 *        BY VALUE.
 *
 * A value leaf held by reference is read through the caller's own object.
 * When that object is a `const` local (`const auto c = constant(...)`, the
 * shape emitted kernels use), GCC will not scalarise it (a store into a
 * read-only declaration disqualifies it), so it stays on the stack and its
 * end-of-life clobber blocks if-conversion of a host sample loop around the
 * `select`. A by-value copy is an ordinary aggregate the compiler keeps in
 * registers; the selected values are the same.
 */
template<class E>
using SelectOperand = std::conditional_t<E::isLeaf && !isValueLeaf<std::remove_cv_t<E>>, const E&, const E>;

} // namespace detail

/**
 * @brief Gates every operator/assignment path in `expr/`. `E` must expose
 *        the element protocol described above and publicly, unambiguously
 *        derive from `Expression<E, typename E::element_type>` (the CRTP
 *        anchor) — checked modulo top-level cv-qualification on `E` itself:
 *        a leaf's declared CRTP base always names its own non-const type
 *        (`View<...>` derives from `Expression<View<...>, T>`, never from
 *        `Expression<const View<...>, T>`, a distinct instantiation), so
 *        `E = const View<...>` would fail `std::derived_from` without this
 *        `remove_cv_t`. Stripping `E`'s cv before naming both sides of the
 *        check makes a `const` leaf conform too, and is a no-op for every
 *        already-conforming non-const `E`. Device-safe: pure compile-time
 *        predicate, no runtime component.
 */
template<class E>
concept aether_expression = detail::aether_expression_structural<E>
    && std::derived_from<std::remove_cv_t<E>, Expression<std::remove_cv_t<E>, typename E::element_type>>;

} // namespace aether
