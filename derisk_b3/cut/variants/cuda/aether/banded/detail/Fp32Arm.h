// Variant cuda: NOT PARSED here (no nvcc); text moved verbatim from banded/detail/Fp32.h, device arm.
// Inline PTX pins every rounding; it is never fused into a neighbouring FFMA nor out-of-lined into libdevice.
#pragma once
#define AETHER_BAND_FP_BARRIER(x) ((void)0)

#define AETHER_BAND_IMPL_FLOAT_AS_INT(f) \
    return __float_as_int(f);
#define AETHER_BAND_IMPL_INT_AS_FLOAT(i) \
    return __int_as_float(i);
#define AETHER_BAND_IMPL_FMA(a, b, c) \
    float r; \
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); \
    return r;
#define AETHER_BAND_IMPL_MUL(a, b) \
    float r; \
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); \
    return r;
#define AETHER_BAND_IMPL_ADD(a, b) \
    float r; \
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); \
    return r;
#define AETHER_BAND_IMPL_SUB(a, b) \
    float r; \
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); \
    return r;
#define AETHER_BAND_IMPL_ROUND32(x) \
    return __float2int_rn(x);
#define AETHER_BAND_IMPL_ROUND64(x) \
    return __float2ll_rn(x);
