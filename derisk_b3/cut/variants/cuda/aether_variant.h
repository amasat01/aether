// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// Variant "cuda": the named input of the components layer. Every device / host / instruction-set decision of the
// components sits HERE, spelled once per variant; the components call these names unconditionally.
#pragma once
#define AETHER_VARIANT_NAME "cuda"
// NOT PARSED here (no nvcc on this machine); text moved verbatim from the original headers.
#define AETHER_DEVICE_COMPILER 1
#define AETHER_VIEW_LOAD_RO(data, map, ...) return __ldg(&(data)[(map)(__VA_ARGS__)]);
// moved from macros.h (all of it: qualifiers, __CUDA_ARCH__ arms, AETHER_DEVICE_COMPILE): see variants/cuda/macros_device_arm.h
#include "macros_device_arm.h"
// TableHandle tex1Dfetch body, moved verbatim:
//   AETHER_TABLE_READ_BODY(T, tex, ptr, idx, off) -> see table_read_device_arm.h
#include "table_read_device_arm.h"
// moved from expr/Assign.h: forced inline on the GCC host pass only; empty on the device pass (device codegen untouched)
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
#define AETHER_DETAIL_UNROLL_INLINE __attribute__((always_inline))
#else
#define AETHER_DETAIL_UNROLL_INLINE
#endif
#define AETHER_MATH_IMPL_FMA(T, a, b, c) \
    if constexpr (std::is_same_v<T, float>) { \
        return fmaf(a, b, c); \
    } else { \
        return ::fma(a, b, c); \
    }
#define AETHER_MATH_IMPL_ISFINITE(T, x) \
    return ::isfinite(x);
#define AETHER_MATH_IMPL_ISNAN(T, x) \
    return ::isnan(x);
#define AETHER_MATH_IMPL_ISINF(T, x) \
    return ::isinf(x);
#define AETHER_MATH_IMPL_SIN(T, x) \
    if constexpr (std::is_same_v<T, float>) { \
        return sinf(x); \
    } else { \
        return detail::cwSin(x); \
    }
#define AETHER_MATH_IMPL_COS(T, x) \
    if constexpr (std::is_same_v<T, float>) { \
        return cosf(x); \
    } else { \
        return detail::cwCos(x); \
    }
#define AETHER_MATH_IMPL_SINCOS(T, x, sinOut, cosOut) \
    if constexpr (std::is_same_v<T, float>) { \
        sincosf(x, sinOut, cosOut); \
    } else { \
        detail::cwSinCos(x, *sinOut, *cosOut); \
    }
#define AETHER_MATH_IMPL_RSQRT(T, x) \
    if constexpr (std::is_same_v<T, float>) { \
        return rsqrtf(x); \
    } else { \
        return ::rsqrt(x); \
    }
// D3 hooks: the device keeps the narrow `offset_t` chain; `as<>` stays a member (a variant that must drop it defines the macro empty).
#define AETHER_ELEMENT_OFFSET_BODY(map, idxs) \
    return map(idxs...);
#define AETHER_RUNTIME_VIEW_AS_MEMBER \
    template<class StaticViewT> \
    StaticViewT as() const;
// linalg team / random hooks
#if defined(__CUDA_ARCH__)
#define AETHER_TEAM_SYNC_BODY(barrier, ctx) \
    __syncthreads();
#define AETHER_TEAM_HERE_BODY(t) \
    t.rank = static_cast<int>(threadIdx.x); \
    t.size = static_cast<int>(blockDim.x);
#else
#define AETHER_TEAM_SYNC_BODY(barrier, ctx) \
    if (barrier != nullptr) { \
        barrier(ctx); \
    }
#define AETHER_TEAM_HERE_BODY(t)
#endif
#if defined(AETHER_DEVICE_COMPILE)
#define AETHER_MULHILO32_BODY(a, b, hi) \
    hi = __umulhi(a, b); \
    return a * b;
#else
#define AETHER_MULHILO32_BODY(a, b, hi) \
    const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b); \
    hi = static_cast<std::uint32_t>(product >> 32); \
    return static_cast<std::uint32_t>(product);
#endif
