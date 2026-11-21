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
/// The include of the integer helpers.
pub const INTEGER_INCLUDE: &str = r####"#include "aether/integer/integer.h"
"####;
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
