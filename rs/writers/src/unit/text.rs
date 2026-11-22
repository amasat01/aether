//! Fixed text both unit writers use, spelled as the translation unit carries it; the markers `@QUAL@`, `@ARM@`, `@ROW@`, `@HALF@`, `@OFFSETS@` and `@HELPERS@` are filled in by the writer.
/// The includes every unit starts with.
pub const UNIT_INCLUDES: &str = r####"#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <aether/expr/nodes/Geometric.h>
#include <aether/expr/nodes/Inverse.h>
#include <aether/expr/nodes/Product.h>
#include <aether/expr/nodes/Quaternion.h>
#include <aether/expr/nodes/Structural.h>
#include <aether/expr/Reduce.h>
#include <aether/math/math.h>

#include "plugin/gref_layout.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
"####;
/// The include a body that draws random numbers adds.
pub const RANDOM_INCLUDE: &str = r####"#include "aether/random/detail/Backend.h"
"####;
/// The include after which the extra includes are placed.
pub const RANDOM_INCLUDE_MARKER: &str = r####"#include <aether/math/math.h>
"####;
/// The include of the run commits.
pub const RUN_INCLUDE: &str = r####"#include <aether/accum/run.h>
"####;
/// The include of the atomic element updates.
pub const ELEMENT_ATOMIC_INCLUDE: &str = r####"#include <aether/accum/ElementAtomic.h>
"####;
/// The include of the compensated sums.
pub const TWO_SUM_INCLUDE: &str = r####"#include <aether/accum/TwoSum.h>
"####;
/// The include of the linear algebra library.
pub const LINALG_INCLUDE: &str = r####"#include <aether/linalg/linalg.h>
"####;
/// The include of the machine facts.
pub const FACTS_INCLUDE: &str = r####"#include "raptor_facts.h"
"####;
/// The include of the team library.
pub const TEAM_INCLUDE: &str = r####"#include <aether/linalg/team.h>
"####;
/// The include of the geometry library.
pub const RAY_INCLUDE: &str = r####"#include <aether/geometry/geometry.h>
"####;
/// The include of the runtime expression library.
pub const RT_INCLUDE: &str = r####"#include <aether/expr/Rt.h>
"####;
/// The two-sum residual helper.
pub const TWO_SUM_ERR: &str = r####"
template <class E>
@QUAL@ E twoSumErr_(E a, E b, E s)
{
    // aether's ONE two-sum residual: exact for the rounded a, b and s, NaN
    // for a non-finite s, +0 for a zero residual (aether/accum/TwoSum.h).
    return aether::accum::twoSumErr(a, b, s);
}
"####;
/// The compensated add helper.
pub const ACCUM_ADD_COMPENSATED: &str = r####"
template <class ViewT, class T>
@QUAL@ void accum_add_compensated(ViewT& target, ViewT& comp,
                                       const aether::SampleIndex& at, T term)
{
    auto& slot = target(at.global());
    auto& compSlot = comp(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add_compensated: a scattered compensated "
                  "accumulate commits through the target's atomic add, and "
                  "only a floating accumulate plane has one");
    E t = static_cast<E>(term);
    AETHER_FP_BARRIER(t);  // rounded once: never fused into `old + t`
@ARM@
}
"####;
/// The compensated add at an element of a plane with axes: the term is added at the element the view's call form names, and the
/// correction at the same element of the companion plane.
pub const ACCUM_ADD_COMPENSATED_AT: &str = r####"
template <class E, class T>
@QUAL@ void accum_add_compensated_at(E& slot, E& compSlot, T term)
{
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add_compensated_at: a scattered compensated "
                  "accumulate commits through the target's atomic add, and "
                  "only a floating accumulate plane has one");
    E t = static_cast<E>(term);
    AETHER_FP_BARRIER(t);  // rounded once: never fused into `old + t`
@ARM@
}
"####;
/// The scattered add at an element of a plane with axes: the element is named by the view's call form, so the helper takes it.
pub const ACCUM_ADD_AT: &str = r####"
template <class E, class T>
@QUAL@ void accum_add_at(E& slot, T term)
{
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add_at: a scattered accumulate commits through the "
                  "target's atomic add, and only a floating accumulate plane has one");
@ARM@
}
"####;
/// The value of rank above zero a plane with axes holds at one position: each component is the view's call form with the
/// component's indices ahead of the axis indices (the storage order is the value's components, then the axes).
pub const VIEW_ITEM: &str = r####"
template <class T, std::size_t... Es, class ViewT, class... Axes>
@QUAL@ aether::Item<T, Es...> view_item(const ViewT& view, Axes... axes)
{
    constexpr std::size_t kRank = sizeof...(Es);
    constexpr std::size_t kExtents[kRank] = {Es...};
    aether::Item<T, Es...> out;
    for (std::size_t flat = 0; flat < (Es * ...); ++flat) {
        std::size_t component[kRank];
        std::size_t rest = flat;
        for (std::size_t axis = kRank; axis-- > 0;) {
            component[axis] = rest % kExtents[axis];
            rest /= kExtents[axis];
        }
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            out(component[K]...) = view(component[K]..., axes...);
        }(std::make_index_sequence<kRank>{});
    }
    return out;
}
"####;
/// The scattered or at an element of a plane with axes: the element is named by the view's call form, so the helper takes it.
pub const ACCUM_OR_AT: &str = r####"
template <class E, class T>
@QUAL@ void accum_or_at(E& slot, T bits)
{
    static_assert(std::is_integral_v<E> && sizeof(E) == 4,
                  "hawk_abi::accum_or_at: a scattered OR commits through the "
                  "target's atomic OR, on a 32-bit integer plane");
@ARM@
}
"####;
/// The atomic or helper.
pub const ACCUM_OR: &str = r####"
template <class ViewT, class T>
@QUAL@ void accum_or(ViewT& target, const aether::SampleIndex& at, T bits)
{
    auto& slot = target(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_integral_v<E> && sizeof(E) == 4,
                  "hawk_abi::accum_or: a scattered OR commits through the "
                  "target's atomic OR, on a 32-bit integer plane");
@ARM@
}
"####;
/// The helper that folds a companion into its target.
pub const FOLD_COMPANION: &str = r####"
template <class ViewT>
@QUAL@ void fold_companion(ViewT& target, ViewT& comp, const aether::SampleIndex& at)
{
    auto& t = target(at.global());
    auto& c = comp(at.global());
    using E = std::remove_reference_t<decltype(t)>;
    const E s = t + c;
    const E e = twoSumErr_(t, c, s);
    t = s;
    c = aether::accum::isFiniteBits(s) ? e : E(0);
}
"####;
/// The helper that folds a companion into a target with axes, element by element in storage order: such a plane is
/// contiguous, so its element at a flat position is the data's.
pub const FOLD_COMPANION_AXES: &str = r####"
template <class ViewT>
@QUAL@ auto& flat_element(ViewT& view, const aether::SampleIndex& at)
{
    if constexpr (ViewT::rank() == 1) {
        return view(at.global());
    } else {
        return view.data()[at.global()];
    }
}
template <class ViewT>
@QUAL@ void fold_companion(ViewT& target, ViewT& comp, const aether::SampleIndex& at)
{
    auto& t = flat_element(target, at);
    auto& c = flat_element(comp, at);
    using E = std::remove_reference_t<decltype(t)>;
    const E s = t + c;
    const E e = twoSumErr_(t, c, s);
    t = s;
    c = aether::accum::isFiniteBits(s) ? e : E(0);
}
"####;
/// The compensated store helper.
pub const STORE_COMPENSATED: &str = r####"
template <class ViewT, class CompT, class T>
@QUAL@ void store_compensated(ViewT& target, CompT& comp,
                                   const aether::SampleIndex& at, T term)
{
    using E = typename ViewT::element_type;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::store_compensated: an own-column compensated "
                  "commit accumulates into a floating target plane only");
    if constexpr (ViewT::extents_type::Rank == 1) {
        // a Scalar own-column target: ONE (component, sample) pair
        // collapses to a plain per-sample index.
        auto& slot = target(at.global());
        auto& compSlot = comp(at.global());
        const E old = slot;
        E t = static_cast<E>(term);
        AETHER_FP_BARRIER(t);  // rounded once: never fused into `old + t`
        const E s = old + t;
        slot = s;
        compSlot = compSlot + twoSumErr_(old, t, s);
    } else {
        static_assert(ViewT::extents_type::Rank == 2,
                      "hawk_abi::store_compensated: built for a Scalar or "
                      "Vector[N] own-column target only");
        // a Vector[N] own-column target: the SAME twoSumErr_ recurrence,
        // per component -- `target`/`comp` are (component, sample) views.
        // `term` is any same-width expression over per-sample values (a
        // bare `Item`, or arithmetic such as `(-k * norm(v)) * v`); it is
        // evaluated ONCE into an `aether::Item` so each component can be
        // read by index, and component `k` of the target accumulates
        // against component `k` of the term alone.
        constexpr std::size_t N = ViewT::extents_type::static_extent(0);
        const aether::Item<typename T::element_type, N> termItem(term);
        for (std::size_t k = 0; k < N; ++k) {
            auto& slot = target(k, at.global());
            auto& compSlot = comp(k, at.global());
            const E old = slot;
            E t = static_cast<E>(termItem(k));
            AETHER_FP_BARRIER(t);
            const E s = old + t;
            slot = s;
            compSlot = compSlot + twoSumErr_(old, t, s);
        }
    }
}
"####;
/// The spacing helper.
pub const SPACING: &str = r####"
template <class T>
@QUAL@ T spacing(T x)
{
    const T a = x < T(0) ? -x : x;
@ARM@
    return x < T(0) ? -up : up;
}
"####;
/// The mix-count helper.
pub const MIX_COUNT: &str = r####"
template <class ViewT>
@QUAL@ void mix_count(const ViewT& mix, int K, int k)
{
    auto* cells = const_cast<std::uint32_t*>(
        reinterpret_cast<const std::uint32_t*>(mix.data()));
@ARM@
}
"####;
/// The integer helpers.
pub const INT_HELPERS: &str = r####"
template <class T> struct int_traits;
template <> struct int_traits<std::int32_t> {
    using U = std::uint32_t;
    static constexpr bool is_signed = true;
    static constexpr std::int32_t lo = -2147483647 - 1, hi = 2147483647;
    static constexpr unsigned bits = 32u;
};
template <> struct int_traits<std::uint32_t> {
    using U = std::uint32_t;
    static constexpr bool is_signed = false;
    static constexpr std::uint32_t lo = 0u, hi = 4294967295u;
    static constexpr unsigned bits = 32u;
};
template <> struct int_traits<long long> {
    using U = unsigned long long;
    static constexpr bool is_signed = true;
    static constexpr long long lo = -9223372036854775807LL - 1, hi = 9223372036854775807LL;
    static constexpr unsigned bits = 64u;
};

template <class T> @QUAL@ T int_add(T a, T b)
{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}
template <class T> @QUAL@ T int_sub(T a, T b)
{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}
template <class T> @QUAL@ T int_mul(T a, T b)
{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}
template <class T> @QUAL@ T int_neg(T a)
{
    using U = typename int_traits<T>::U;
    return static_cast<T>(U(0) - static_cast<U>(a));
}
template <class T> @QUAL@ T int_abs(T a)
{
    if constexpr (int_traits<T>::is_signed)
        return a < T(0) ? int_neg(a) : a;
    else
        return a;
}
template <class T> @QUAL@ T int_min(T a, T b) { return b < a ? b : a; }
template <class T> @QUAL@ T int_max(T a, T b) { return a < b ? b : a; }
// One unsigned compare is the whole in-range test (a negative index wraps past n), so an in-range read pays one
// predictable branch: a min/max or nested-compare spelling costs more per read on the host gather loop.
template <class T> @QUAL@ T int_clamp_index(T i, T n)
{
    using U = typename int_traits<T>::U;
    return U(i) < U(n) ? i : (i < T(0) ? T(0) : n - T(1));
}
template <class T> @QUAL@ T int_floordiv(T a, T b)
{
    if (b == T(0))
        return T(0);
    if constexpr (int_traits<T>::is_signed) {
        if (b == T(-1))
            return int_neg(a);
        const T q = static_cast<T>(a / b);
        return (a % b != T(0) && ((a < T(0)) != (b < T(0)))) ? static_cast<T>(q - T(1)) : q;
    } else {
        return static_cast<T>(a / b);
    }
}
template <class T> @QUAL@ T int_mod(T a, T b)
{
    if (b == T(0))
        return T(0);
    if constexpr (int_traits<T>::is_signed) {
        if (b == T(-1))
            return T(0);
        const T r = static_cast<T>(a % b);
        return (r != T(0) && ((r < T(0)) != (b < T(0)))) ? static_cast<T>(r + b) : r;
    } else {
        return static_cast<T>(a % b);
    }
}
template <class T> @QUAL@ T int_and(T a, T b) { return static_cast<T>(a & b); }
template <class T> @QUAL@ T int_or(T a, T b) { return static_cast<T>(a | b); }
template <class T> @QUAL@ T int_xor(T a, T b) { return static_cast<T>(a ^ b); }
template <class T> @QUAL@ T int_invert(T a) { return static_cast<T>(~a); }
template <class T> @QUAL@ T int_shl(T a, T b)
{
    using U = typename int_traits<T>::U;
    if (static_cast<U>(b) < U(int_traits<T>::bits))
        return static_cast<T>(static_cast<U>(a) << static_cast<U>(b));
    return T(0);
}
template <class T> @QUAL@ T int_shr(T a, T b)
{
    using U = typename int_traits<T>::U;
    if (static_cast<U>(b) < U(int_traits<T>::bits))
        return static_cast<T>(a >> b);
    if constexpr (int_traits<T>::is_signed)
        return a < T(0) ? T(-1) : T(0);
    else
        return T(0);
}
template <class T, class R> @QUAL@ T int_from_real(R x)
{
    if (aether::math::isnan(x))
        return T(0);
    if (x <= static_cast<R>(int_traits<T>::lo))
        return int_traits<T>::lo;
    if (x >= static_cast<R>(int_traits<T>::hi))
        return int_traits<T>::hi;
    return static_cast<T>(x);
}
"####;
/// The include of the edge-rule helpers of an indexed access.
pub const EDGE_INCLUDE: &str = "#include \"aether/view/Edge.h\"\n";
/// The count clamp helper.
pub const CLAMP_COUNT: &str = r####"@QUAL@ int hawk_clamp_count(long long n)
{
    return n < 2147483647LL ? static_cast<int>(n) : 2147483647;
}

"####;
/// The finished-count helper.
pub const FINISH_COUNT: &str = r####"
template <class ViewT>
@QUAL@ void finish_count(const ViewT& counter)
{
    // The finish epilogue's count (hawk.ir.nodes.FINISHED_PLANE): ONE
    // relaxed increment of a uint32 cell per NEWLY finished sample, at most N
    // over a whole run. Read only after the launch, in stream / team order.
    auto* cell = const_cast<std::uint32_t*>(
        reinterpret_cast<const std::uint32_t*>(counter.data()));
@ARM@
}
"####;
/// The row sum of sample sums.
pub const SAMPLE_SUM: &str = r####"
template <class E>
@QUAL@ E ssExtreme_(int op, E a, E b)
{
    // max (op 1) or min (op 2) that ignores a NaN operand, as hawk.math.max/min.
    if (a != a) return b;
    if (b != b) return a;
    return op == 1 ? (b > a ? b : a) : (b < a ? b : a);
}

template <int OP, class E>
@QUAL@ E ssIdentity_()
{
    // What a slot past the last sample holds: 0, -inf, +inf.
    if constexpr (OP == 0) return E(0);
    else if constexpr (OP == 1) return -static_cast<E>(__builtin_inf());
    else return static_cast<E>(__builtin_inf());
}

template <int OP = 0, class E>
@QUAL@ void pairCombine_(E& s, E& c, E s2, E c2)
{
    if constexpr (OP != 0) {
        s = ssExtreme_(OP, s, s2);
        return;
    }
    // Compensated combine of two partials (s, c) and (s2, c2), kept verbatim
    // with aether/accum/atomic.h. Symmetric: either side may hold either
    // partial. A non-finite sum drops its compensation, so Inf and NaN
    // follow IEEE through s.
    const E t = s + s2;
    const E e = aether::accum::twoSumErrFinite(s, s2, t);
    c = aether::accum::isFiniteBits(t) ? (c + c2) + e : E(0);
    s = t;
}

template <int N, int OP = 0, class E, class Store>
@QUAL@ void sampleSumRow_(E* terms, Store&& store)
{
    // One row: @ROW@ slots of N terms (row-major), slot l combined with slot
    // l + off for off = @OFFSETS@ -- the pairs of sampleSumWarp_ --
    // then store(column, sum, compensation). Leaves the slots zeroed.
    E comp[@ROW@ * N] = {};
    for (int off = @HALF@; off >= 1; off >>= 1)
        for (int l = 0; l < off; ++l)
            for (int k = 0; k < N; ++k)
                pairCombine_<OP>(terms[l * N + k], comp[l * N + k],
                             terms[(l + off) * N + k], comp[(l + off) * N + k]);
    for (int k = 0; k < N; ++k) store(k, terms[k], comp[k]);
    for (int k = 0; k < @ROW@ * N; ++k) terms[k] = ssIdentity_<OP, E>();
}

"####;
/// The helpers every unit shares.
pub const ABI_HELPERS: &str = r####"namespace hawk_abi {

template <class T, std::size_t N>
@QUAL@ aether::View<T, aether::extents<N, aether::dyn>, aether::layout_right>
vec_view(const eagle::plugin::GRefMirror& m)
{
    // A GRef plane is SoA and C-contiguous by eagle's packing rule (marshal.py):
    // unit sample stride, component pitch = m.compStride_ (the full plane's
    // sample count even for a partition slice). layout_right over that ONE
    // runtime pitch lets aether fold the per-component offsets.
    using Ext = aether::extents<N, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}

template <class T, std::size_t R, std::size_t C>
@QUAL@ aether::View<T, aether::extents<R, C, aether::dyn>, aether::layout_right>
mat_view(const eagle::plugin::GRefMirror& m)
{
    using Ext = aether::extents<R, C, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}

template <class T>
@QUAL@ aether::View<T, aether::extents<aether::dyn>, aether::layout_right>
scalar_view(const eagle::plugin::ScalarHandle& m)
{
    using Ext = aether::extents<aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data), Map(Ext(static_cast<std::size_t>(m.samples))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType), m.deviceId));
}

template <class ViewT, class T>
@QUAL@ void accum_add(ViewT& target, const aether::SampleIndex& at, T term)
{
    // The scattered commit (hawk/ext): ONE body string spelled per
    // target here -- the device's own atomic add, or an atomic_ref under the
    // OpenMP host team -- because a scattered read-modify-write has no correct
    // use under any parallel launch. One subscript on the target.
    auto& slot = target(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add: a scattered accumulate commits through the "
                  "target's atomic add, and only a floating accumulate plane has one");
@ARM@
}
@HELPERS@
}  // namespace hawk_abi
"####;
