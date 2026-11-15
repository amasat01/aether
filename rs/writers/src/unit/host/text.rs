//! Fixed text only the host unit carries, and the host arms of the helpers.
/// The blend helpers of the host text.
pub const BLEND_HELPERS: &str = r####"#include <bit>
#include <type_traits>

namespace hawk_abi {

template <class A, class B>
static inline std::common_type_t<A, B> blend(bool m, A a, B b)
{
    using T = std::common_type_t<A, B>;
    if constexpr (std::is_floating_point_v<T>) {
        using U = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
        const U k = U{0} - static_cast<U>(m);
        return std::bit_cast<T>((std::bit_cast<U>(static_cast<T>(a)) & k) |
                                (std::bit_cast<U>(static_cast<T>(b)) & ~k));
    } else {
        return m ? static_cast<T>(a) : static_cast<T>(b);
    }
}

template <class T, std::size_t... Es, class A, class B>
static inline aether::Item<T, Es...> select_item(bool m, const A& a, const B& b)
{
    const aether::Item<T, Es...> x = a;
    const aether::Item<T, Es...> y = b;
    aether::Item<T, Es...> r;
    for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c)
        r.data()[c] = blend(m, x.data()[c], y.data()[c]);
    return r;
}

}  // namespace hawk_abi
"####;
/// Host spelling of the atomic add.
pub const ARM_ACCUM_ADD: &str = r####"    std::atomic_ref<E>(slot).fetch_add(static_cast<E>(term), std::memory_order_relaxed);"####;
/// Host spelling of the compensated atomic add.
pub const ARM_ACCUM_ADD_COMPENSATED: &str = r####"    const E old = std::atomic_ref<E>(slot).fetch_add(t, std::memory_order_relaxed);
    const E s = old + t;
    std::atomic_ref<E>(compSlot).fetch_add(aether::accum::twoSumErrAtomic(old, t, s),
                                           std::memory_order_relaxed);"####;
/// Host spelling of the atomic or.
pub const ARM_ACCUM_OR: &str = r####"    std::atomic_ref<E>(slot).fetch_or(static_cast<E>(bits), std::memory_order_relaxed);"####;
/// Host spelling of the finished-count commit.
pub const ARM_FINISH_COUNT: &str =
    r####"    std::atomic_ref<std::uint32_t>(*cell).fetch_add(1u, std::memory_order_relaxed);"####;
/// Host spelling of the mix-count commit.
pub const ARM_MIX_COUNT: &str = r####"    (void)K;
    std::atomic_ref<std::uint32_t>(cells[k]).fetch_add(1u, std::memory_order_relaxed);"####;
/// Host spelling of the spacing helper.
pub const ARM_SPACING: &str = r####"    const T up = std::nextafter(a, a * T(2) + T(1)) - a;"####;
