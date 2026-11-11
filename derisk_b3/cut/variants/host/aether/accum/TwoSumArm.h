// Variant host: bit reinterpretation by std::bit_cast; Knuth's two-sum unless the ordered form is selected.
#pragma once
#include <bit>
#include <cstdint>
#define AETHER_FLOAT_BITS_F64(x) std::bit_cast<std::int64_t>(x)
#define AETHER_FLOAT_BITS_F32(x) std::bit_cast<std::int32_t>(x)
#define AETHER_BITS_FLOAT_F64(b) std::bit_cast<E>(b)
#define AETHER_BITS_FLOAT_F32(b) std::bit_cast<E>(b)
// The integer compare of the ordered form is a blend on a CPU: kept only where measured faster (the named host fact).
#if defined(AETHER_TWOSUM_FAST_HOST)
#define AETHER_TWOSUM_ORDERED 1
#else
#define AETHER_TWOSUM_ORDERED 0
#endif
#define AETHER_TWOSUM_ACCUM_VIA_ERR 0
