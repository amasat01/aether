// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Packet-math kernels on aarch64: pow alone, the one function aether routes there, at
// widths 1 and 2. Under GCC they call the route's own symbols (aether_hvm_pow,
// _ZGVnN2vv_aether_hvm_pow and, when it exists, the masked variant), and this TU is built
// with -ffp-contract=fast like a calling kernel, so the check covers the contraction
// guard around them. Clang has no route and no NEON packet here: width 1 only, on pm::vpow.
// Any other function returns its first argument, which the faithful check rejects.

#include "tests/packetmath/PacketMathKernels.h"

#if !defined(__aarch64__)
#error "pm_kernels_arm.cpp is for aarch64"
#endif

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__clang__)
#include "aether/backend/cpu/simd/math/PacketExpLog.h"
#else
#if !defined(AETHER_HOST_VECTOR_MATH)
#error "pm_kernels_arm.cpp must be built with -DAETHER_HOST_VECTOR_MATH under GCC"
#endif
#include "aether/math/math.h"
#endif

namespace aether_pm_test {
namespace {

bool always() { return true; }
bool never() { return false; }

#if defined(__clang__)
double powW1(double a, double b) { return ::aether::simd::pm::vpow<double>(a, b); }
#else
double powW1(double a, double b) { return ::aether_hvm_pow(a, b); }
float64x2_t powW2(float64x2_t a, float64x2_t b)
{
    namespace abi = ::aether::math::detail::hvm_abi;
    const float64x2_t r = abi::pow_n2(a, b);
#if AETHER_HVM_MASKED
    // The masked variant with every lane on must agree bit for bit; a NaN fails the check.
    const float64x2_t all = vreinterpretq_f64_u64(vdupq_n_u64(~std::uint64_t{0}));
    const float64x2_t m = abi::pow_m2(a, b, all);
    if (std::memcmp(&r, &m, sizeof r) != 0)
        return vdupq_n_f64(std::numeric_limits<double>::quiet_NaN());
#endif
    return r;
}
#endif

void unaryNone(int, const double* x, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = x[i];
}

void binaryW1(int fn, const double* x, const double* y, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = fn == kPow ? powW1(x[i], y[i]) : x[i];
}

#if !defined(__clang__)
void binaryW2(int fn, const double* x, const double* y, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; i += 2) {
        const float64x2_t a = vld1q_f64(x + i);
        vst1q_f64(out + i, fn == kPow ? powW2(a, vld1q_f64(y + i)) : a);
    }
}
#endif

} // namespace

const Kernels& kernelsW1()
{
    static const Kernels k{ 1, &always, &unaryNone, &binaryW1 };
    return k;
}

const Kernels& kernelsW2()
{
#if defined(__clang__)
    static const Kernels k{ 2, &never, &unaryNone, &binaryW1 };
#else
    static const Kernels k{ 2, &always, &unaryNone, &binaryW2 };
#endif
    return k;
}

const Kernels& kernelsW4()
{
    static const Kernels k{ 4, &never, &unaryNone, &binaryW1 };
    return k;
}

const Kernels& kernelsW8()
{
    static const Kernels k{ 8, &never, &unaryNone, &binaryW1 };
    return k;
}

} // namespace aether_pm_test
