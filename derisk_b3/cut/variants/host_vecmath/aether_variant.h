// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// Variant "host_vecmath": the named input of the components layer. Every device / host / instruction-set decision of the
// components sits HERE, spelled once per variant; the components call these names unconditionally.
#pragma once
#define AETHER_VARIANT_NAME "host_vecmath"
#define AETHER_DEVICE_COMPILER 0
#define AETHER_DEVICE(...)
#define AETHER_HOST(...)
#define AETHER_DEVICEANDHOST(...)
#define AETHER_DEVICEHOST(...)
#define AETHER_SHARED(...)
#define AETHER_GRID_CONSTANT(...)
#define AETHER_NOINLINE(...) __attribute__((noinline))
#define AETHER_FORCEINLINE(...) inline
#define AETHER_BITCOPY(dst, src, n) __builtin_memcpy((dst), (src), (n))
#define AETHER_NO_UNROLL() _Pragma("GCC unroll 1")
#define AETHER_DETAIL_UNROLL_INLINE __attribute__((always_inline))
// instruction-set arm of the host variant: an empty asm pins a rounded value (no FMA contraction across it)
#if defined(__x86_64__) || defined(__i386__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif defined(__aarch64__) || defined(__arm__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define AETHER_FP_BARRIER(x) __asm__("" : "+r"(x))
#endif
#define AETHER_VIEW_LOAD_RO(data, map, ...) return (data)[::aether::detail::elementOffset((map), __VA_ARGS__)];
#define AETHER_TABLE_READ_BODY(T, tex, ptr, idx, off) return (ptr)[(idx) + (off)];
#define AETHER_MATH_IMPL_FMA(T, a, b, c) \
    return std::fma(a, b, c);
#define AETHER_MATH_IMPL_ISFINITE(T, x) \
    return std::isfinite(x);
#define AETHER_MATH_IMPL_ISNAN(T, x) \
    return std::isnan(x);
#define AETHER_MATH_IMPL_ISINF(T, x) \
    return std::isinf(x);
#define AETHER_MATH_IMPL_SIN(T, x) \
    return AETHER_MATH_HOST_ROUTE(sin, x);
#define AETHER_MATH_IMPL_COS(T, x) \
    return AETHER_MATH_HOST_ROUTE(cos, x);
#define AETHER_MATH_IMPL_SINCOS(T, x, sinOut, cosOut) \
    *sinOut = AETHER_MATH_HOST_ROUTE(sin, x); \
    *cosOut = AETHER_MATH_HOST_ROUTE(cos, x);
#define AETHER_MATH_IMPL_RSQRT(T, x) \
    return AETHER_MATH_HOST_ROUTE_OR(rsqrt, T(1) / std::sqrt(x), x);
// D3 hooks: the host spells the element offset at full width; `as<>` is a member of the runtime view.
#define AETHER_ELEMENT_OFFSET_BODY(map, idxs) \
    if constexpr (isLayoutRightMapping<Map>) { \
        static_assert(sizeof...(Idxs) == Map::extents_type::Rank, \
            "layout_right::mapping::operator(): wrong number of indices"); \
        return wideRightOffset(map, std::make_index_sequence<sizeof...(Idxs)>{}, \
            static_cast<std::size_t>(static_cast<offset_t>(idxs))...); \
    } else if constexpr (isLayoutStrideMapping<Map>) { \
        static_assert(sizeof...(Idxs) == Map::Rank, \
            "layout_stride::mapping::operator(): wrong number of indices"); \
        return wideStrideOffset(map, std::make_index_sequence<sizeof...(Idxs)>{}, \
            static_cast<std::size_t>(static_cast<offset_t>(idxs))...); \
    } else { \
        return map(idxs...); \
    }
#define AETHER_RUNTIME_VIEW_AS_MEMBER \
    template<class StaticViewT> \
    StaticViewT as() const;
// linalg team / random hooks
#define AETHER_TEAM_SYNC_BODY(barrier, ctx) \
    if (barrier != nullptr) { \
        barrier(ctx); \
    }
#define AETHER_TEAM_HERE_BODY(t)
#define AETHER_MULHILO32_BODY(a, b, hi) \
    const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b); \
    hi = static_cast<std::uint32_t>(product >> 32); \
    return static_cast<std::uint32_t>(product);
