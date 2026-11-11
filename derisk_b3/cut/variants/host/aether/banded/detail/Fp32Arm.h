// Variant host: the host half of banded/detail/Fp32.h (the rounding guard and the memcpy / builtin bodies).
#pragma once
#include <cstring>

// The empty asm pins a rounded value: gcc and clang fuse across statements by default, so the guard is explicit.
#if defined(__x86_64__) || defined(__i386__)
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif defined(__aarch64__) || defined(__arm__)
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define AETHER_BAND_FP_BARRIER(x) __asm__("" : "+r"(x))
#endif

#define AETHER_BAND_IMPL_FLOAT_AS_INT(f) \
    int r; \
    std::memcpy(&r, &f, sizeof(int)); \
    return r;
#define AETHER_BAND_IMPL_INT_AS_FLOAT(i) \
    float r; \
    std::memcpy(&r, &i, sizeof(float)); \
    return r;
#define AETHER_BAND_IMPL_FMA(a, b, c) \
    return __builtin_fmaf(a, b, c);
#define AETHER_BAND_IMPL_MUL(a, b) \
    float r = a * b; \
    AETHER_BAND_FP_BARRIER(r); \
    return r;
#define AETHER_BAND_IMPL_ADD(a, b) \
    float x = a, y = b; \
    AETHER_BAND_FP_BARRIER(x); \
    AETHER_BAND_FP_BARRIER(y); \
    return x + y;
#define AETHER_BAND_IMPL_SUB(a, b) \
    float x = a, y = b; \
    AETHER_BAND_FP_BARRIER(x); \
    AETHER_BAND_FP_BARRIER(y); \
    return x - y;
#define AETHER_BAND_IMPL_ROUND32(x) \
    return static_cast<std::int32_t>(__builtin_lrintf(x));
#define AETHER_BAND_IMPL_ROUND64(x) \
    return static_cast<std::int64_t>(__builtin_llrintf(x));
