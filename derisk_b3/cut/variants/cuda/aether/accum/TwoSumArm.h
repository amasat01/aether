// Variant cuda: NOT PARSED here (no nvcc); text moved verbatim from accum/TwoSum.h, device arm.
#pragma once
#define AETHER_FLOAT_BITS_F64(x) __double_as_longlong(x)
#define AETHER_FLOAT_BITS_F32(x) __float_as_int(x)
#define AETHER_BITS_FLOAT_F64(b) __longlong_as_double(static_cast<long long>(b))
#define AETHER_BITS_FLOAT_F32(b) __int_as_float(static_cast<int>(b))
#define AETHER_TWOSUM_ORDERED 1
#define AETHER_TWOSUM_ACCUM_VIA_ERR 1
