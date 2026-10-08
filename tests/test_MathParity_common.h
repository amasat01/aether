// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_MathParity_common.h
 * @brief The numpy-parity elementwise set of `aether::math`: `expm1`,
 *        `log1p`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `rint`,
 *        `remainder`, `clip`, `isnan`, `isinf`, `erf`, `erfc`, plus the
 *        existing `exp2`/`log2`/`log10`/`pow`/`cbrt`/`hypot`/`ceil`/`trunc`/
 *        `round`/`fmod`/`fdim`/`copysign`/`fma` they sit beside.
 *
 * Host route: each entry is a thin wrapper over `std::` (no
 * `AETHER_HOST_VECTOR_MATH` here; the hook has its own suite in
 * `test_PacketMath_common.h`), so the host checks are bit for bit against
 * `std::` over a corpus with zeros of both signs, subnormals, infinities,
 * NaN and domain edges. `remainder`/`fmod`/`clip`/`round`/`rint` are checked
 * against numpy's answers (numpy 2.4, the values written out below).
 *
 * Device route (the `.cu` twin, CUDA builds only): the same corpus through
 * a kernel, compared to the host. The exact functions (rounding, `fmod`,
 * `remainder`, `copysign`, `fdim`, `clip`, `isnan`/`isinf`/`isfinite`,
 * `fma`) must match bit for bit — each is a single correctly rounded or
 * exact operation, so FMA contraction (`-fmad`) has nothing to fuse. The
 * transcendental ones are within the CUDA Math API's documented maximum
 * error plus glibc's (`deviceUlp` below).
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "tools/ulp_oracle/ulp.h"

namespace aether_tests {
namespace math_parity {

inline bool sameBits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0 || (std::isnan(a) && std::isnan(b));
}

inline bool sameBitsF(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
}

/// Specials, domain edges and two random spreads (seeded).
inline std::vector<double> corpus()
{
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double den = std::numeric_limits<double>::denorm_min();
    std::vector<double> v = { 0.0, -0.0, den, -den, 1e-310, -1e-310, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5,
        -3.5, 0.49999999999999994, -0.49999999999999994, 1.0000000000000002, 0.9999999999999999,
        -0.9999999999999999, 1e-300, -1e-300, 1e-20, -1e-20, 1e300, -1e300, 22.0, -22.0, 700.0, -700.0, 709.8, 710.0,
        -745.0, 0x1p28, 0x1p52, -0x1p52, 4503599627370495.5, -4503599627370495.5, 9007199254740993.0, 30.0, -30.0,
        inf, -inf, nan };
    std::mt19937_64 g(20261004);
    std::uniform_real_distribution<double> u(-10.0, 10.0);
    std::uniform_real_distribution<double> e(-300.0, 300.0);
    for (int i = 0; i < 4000; ++i)
        v.push_back(u(g));
    for (int i = 0; i < 2000; ++i) {
        const double m = std::pow(10.0, e(g));
        v.push_back(i % 2 ? m : -m);
    }
    return v;
}

/// Second operand for the binary functions: the corpus rotated, so every
/// special meets every other kind of value somewhere.
inline std::vector<double> corpusB()
{
    std::vector<double> a = corpus();
    std::vector<double> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        b[i] = a[(i * 7 + 3) % a.size()];
    return b;
}

enum Fn : int {
    kExpm1,
    kLog1p,
    kSinh,
    kCosh,
    kAsinh,
    kAcosh,
    kAtanh,
    kErf,
    kErfc,
    kExp2,
    kLog2,
    kLog10,
    kCbrt,
    kCeil,
    kTrunc,
    kRound,
    kRint,
    kAbs,
    kSign,
    kIsnan,
    kIsinf,
    kIsfinite,
    kPow,
    kHypot,
    kFmod,
    kRemainder,
    kFdim,
    kCopysign,
    kClip,
    kFma,
    kNumFn
};

inline const char* fnName(int f)
{
    static const char* const n[kNumFn] = { "expm1", "log1p", "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc",
        "exp2", "log2", "log10", "cbrt", "ceil", "trunc", "round", "rint", "abs", "sign", "isnan", "isinf", "isfinite",
        "pow", "hypot", "fmod", "remainder", "fdim", "copysign", "clip", "fma" };
    return n[f];
}

/// The aether facade, one function per index — the SAME body on host and device.
AETHER_DEVICEHOST() inline double evalAether(int f, double x, double y)
{
    namespace m = aether::math;
    switch (f) {
    case kExpm1: return m::expm1(x);
    case kLog1p: return m::log1p(x);
    case kSinh: return m::sinh(x);
    case kCosh: return m::cosh(x);
    case kAsinh: return m::asinh(x);
    case kAcosh: return m::acosh(x);
    case kAtanh: return m::atanh(x);
    case kErf: return m::erf(x);
    case kErfc: return m::erfc(x);
    case kExp2: return m::exp2(x);
    case kLog2: return m::log2(x);
    case kLog10: return m::log10(x);
    case kCbrt: return m::cbrt(x);
    case kCeil: return m::ceil(x);
    case kTrunc: return m::trunc(x);
    case kRound: return m::round(x);
    case kRint: return m::rint(x);
    case kAbs: return m::abs(x);
    case kSign: return m::sign(x);
    case kIsnan: return m::isnan(x) ? 1.0 : 0.0;
    case kIsinf: return m::isinf(x) ? 1.0 : 0.0;
    case kIsfinite: return m::isfinite(x) ? 1.0 : 0.0;
    case kPow: return m::pow(x, y);
    case kHypot: return m::hypot(x, y);
    case kFmod: return m::fmod(x, y);
    case kRemainder: return m::remainder(x, y);
    case kFdim: return m::fdim(x, y);
    case kCopysign: return m::copysign(x, y);
    case kClip: return m::clip(x, y, y + 1.0);
    case kFma: return m::fma(x, y, 0.75);
    default: return 0.0;
    }
}

/// The `std::` reference for the host route (numpy semantics spelled out
/// for `remainder` and `clip`, the aether convention for `sign`).
inline double evalStd(int f, double x, double y)
{
    switch (f) {
    case kExpm1: return std::expm1(x);
    case kLog1p: return std::log1p(x);
    case kSinh: return std::sinh(x);
    case kCosh: return std::cosh(x);
    case kAsinh: return std::asinh(x);
    case kAcosh: return std::acosh(x);
    case kAtanh: return std::atanh(x);
    case kErf: return std::erf(x);
    case kErfc: return std::erfc(x);
    case kExp2: return std::exp2(x);
    case kLog2: return std::log2(x);
    case kLog10: return std::log10(x);
    case kCbrt: return std::cbrt(x);
    case kCeil: return std::ceil(x);
    case kTrunc: return std::trunc(x);
    case kRound: return std::round(x);
    case kRint: return std::rint(x);
    case kAbs: return std::fabs(x);
    case kSign: return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    case kIsnan: return std::isnan(x) ? 1.0 : 0.0;
    case kIsinf: return std::isinf(x) ? 1.0 : 0.0;
    case kIsfinite: return std::isfinite(x) ? 1.0 : 0.0;
    case kPow: return std::pow(x, y);
    case kHypot: return std::hypot(x, y);
    case kFmod: return std::fmod(x, y);
    case kRemainder: {
        // numpy's npy_remainder, written out independently of the facade.
        if (std::isnan(x) || std::isnan(y) || std::isinf(x) || y == 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        double r = std::fmod(x, y);
        if (r == 0.0)
            return std::copysign(0.0, y);
        if (std::signbit(r) != std::signbit(y))
            r += y;
        return r;
    }
    case kFdim: return std::fdim(x, y);
    case kCopysign: return std::copysign(x, y);
    case kClip: {
        const double hi = y + 1.0;
        if (std::isnan(x) || std::isnan(y))
            return std::numeric_limits<double>::quiet_NaN();
        const double t = x >= y ? x : y;
        return std::isnan(hi) ? hi : (t <= hi ? t : hi);
    }
    case kFma: return std::fma(x, y, 0.75);
    default: return 0.0;
    }
}

inline bool isBinary(int f) { return f >= kPow; }

/// Device-vs-host bound in ULP: 0 = bit for bit. Transcendentals: the CUDA
/// Math API's documented double-precision maximum error plus glibc's own.
inline std::int64_t deviceUlp(int f)
{
    switch (f) {
    case kExpm1: return 2;
    case kLog1p: return 2;
    case kSinh: return 3;
    case kCosh: return 2;
    case kAsinh: return 4;
    case kAcosh: return 4;
    case kAtanh: return 3;
    case kErf: return 3;
    case kErfc: return 6;
    case kExp2: return 2;
    case kLog2: return 2;
    case kLog10: return 2;
    case kCbrt: return 4; // glibc's cbrt is up to 3 ULP off (see PacketMath.h)
    case kPow: return 3;
    case kHypot: return 2;
    default: return 0;
    }
}

class MathParityTest : public ::testing::Test { };

TEST_F(MathParityTest, HostFacadeMatchesStdBitForBitOverTheCorpus)
{
    const auto xs = corpus();
    const auto ys = corpusB();
    for (int f = 0; f < kNumFn; ++f) {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const double y = isBinary(f) ? ys[i] : 0.0;
            ASSERT_TRUE(sameBits(evalAether(f, xs[i], y), evalStd(f, xs[i], y)))
                << fnName(f) << "(" << xs[i] << ", " << y << ")";
        }
    }
}

TEST_F(MathParityTest, FloatOverloadsMatchStd)
{
    const float xs[] = { 0.0f, -0.0f, 0.3f, -0.3f, 2.5f, -2.5f, 0.999f, 1.0f, 7.0f, -7.0f,
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() };
    namespace m = aether::math;
    for (float x : xs) {
        EXPECT_TRUE(sameBitsF(m::expm1(x), std::expm1(x))) << x;
        EXPECT_TRUE(sameBitsF(m::log1p(x), std::log1p(x))) << x;
        EXPECT_TRUE(sameBitsF(m::sinh(x), std::sinh(x))) << x;
        EXPECT_TRUE(sameBitsF(m::cosh(x), std::cosh(x))) << x;
        EXPECT_TRUE(sameBitsF(m::asinh(x), std::asinh(x))) << x;
        EXPECT_TRUE(sameBitsF(m::acosh(x), std::acosh(x))) << x;
        EXPECT_TRUE(sameBitsF(m::atanh(x), std::atanh(x))) << x;
        EXPECT_TRUE(sameBitsF(m::erf(x), std::erf(x))) << x;
        EXPECT_TRUE(sameBitsF(m::erfc(x), std::erfc(x))) << x;
        EXPECT_TRUE(sameBitsF(m::rint(x), std::rint(x))) << x;
        EXPECT_TRUE(sameBitsF(m::remainder(x, 2.0f), float(evalStd(kRemainder, x, 2.0)))) << x;
        EXPECT_EQ(m::isnan(x), std::isnan(x)) << x;
        EXPECT_EQ(m::isinf(x), std::isinf(x)) << x;
        EXPECT_TRUE(sameBitsF(m::clip(x, -1.0f, 0.0f), float(evalStd(kClip, double(x), -1.0)))) << x;
    }
}

TEST_F(MathParityTest, RemainderAndFmodMatchNumpy)
{
    const double inf = std::numeric_limits<double>::infinity();
    // { a, b, numpy.remainder(a, b), numpy.fmod(a, b) }
    const std::array<std::array<double, 4>, 20> rows = { {
        { 5.0, 3.0, 2.0, 2.0 },
        { -5.0, 3.0, 1.0, -2.0 },
        { 5.0, -3.0, -1.0, 2.0 },
        { -5.0, -3.0, -2.0, -2.0 },
        { 0.0, 3.0, 0.0, 0.0 },
        { -0.0, 3.0, 0.0, -0.0 },
        { 0.0, -3.0, -0.0, 0.0 },
        { 6.0, 3.0, 0.0, 0.0 },
        { -6.0, 3.0, 0.0, -0.0 },
        { 6.0, -3.0, -0.0, 0.0 },
        { 1.0, inf, 1.0, 1.0 },
        { -1.0, inf, inf, -1.0 },
        { 1.0, -inf, -inf, 1.0 },
        { -1.0, -inf, -1.0, -1.0 },
        { 1e300, 1e-300, 4.891554850853602e-301, 4.891554850853602e-301 },
        { -1e-320, 1.0, 1.0, -1e-320 },
        { -0.0, -inf, -0.0, -0.0 },
        { 0.0, inf, 0.0, 0.0 },
        { -0.0, inf, 0.0, -0.0 },
        { 7.5, 2.0, 1.5, 1.5 },
    } };
    for (const auto& r : rows) {
        EXPECT_TRUE(sameBits(aether::math::remainder(r[0], r[1]), r[2])) << "remainder(" << r[0] << ", " << r[1] << ")";
        EXPECT_TRUE(sameBits(aether::math::fmod(r[0], r[1]), r[3])) << "fmod(" << r[0] << ", " << r[1] << ")";
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(aether::math::remainder(inf, 3.0)));
    EXPECT_TRUE(std::isnan(aether::math::remainder(5.0, 0.0)));
    EXPECT_TRUE(std::isnan(aether::math::remainder(nan, 1.0)));
    EXPECT_TRUE(std::isnan(aether::math::remainder(1.0, nan)));
}

TEST_F(MathParityTest, ClipRoundRintSignMatchNumpy)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    namespace m = aether::math;
    EXPECT_TRUE(std::isnan(m::clip(nan, 0.0, 1.0)));
    EXPECT_TRUE(std::isnan(m::clip(0.5, nan, 1.0)));
    EXPECT_TRUE(std::isnan(m::clip(0.5, 0.0, nan)));
    EXPECT_EQ(m::clip(0.5, 2.0, 1.0), 1.0);  // lo > hi -> hi
    EXPECT_EQ(m::clip(-3.0, -1.0, 1.0), -1.0);
    EXPECT_EQ(m::clip(3.0, -1.0, 1.0), 1.0);
    EXPECT_TRUE(sameBits(m::clip(-0.0, 0.0, 1.0), -0.0));
    EXPECT_TRUE(sameBits(m::clip(0.0, -0.0, 1.0), 0.0));
    // round = C (half away from zero); rint = numpy round/rint (half to even).
    EXPECT_EQ(m::round(2.5), 3.0);
    EXPECT_EQ(m::round(-2.5), -3.0);
    EXPECT_EQ(m::rint(2.5), 2.0);
    EXPECT_EQ(m::rint(3.5), 4.0);
    EXPECT_TRUE(sameBits(m::rint(-0.5), -0.0));
    EXPECT_TRUE(sameBits(m::trunc(-0.7), -0.0));
    EXPECT_TRUE(sameBits(m::ceil(-0.7), -0.0));
    // sign(NaN) is 0 in aether (numpy: NaN) — the documented divergence.
    EXPECT_EQ(m::sign(nan), 0.0);
    // domain edges
    EXPECT_EQ(m::log1p(-1.0), -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(m::log1p(-2.0)));
    EXPECT_TRUE(sameBits(m::acosh(1.0), 0.0));
    EXPECT_TRUE(std::isnan(m::acosh(0.5)));
    EXPECT_EQ(m::atanh(1.0), std::numeric_limits<double>::infinity());
    EXPECT_EQ(m::atanh(-1.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(m::erf(30.0), 1.0);
    EXPECT_TRUE(sameBits(m::erfc(30.0), std::erfc(30.0)));
    EXPECT_EQ(m::erfc(-30.0), 2.0);
}

#if defined(__CUDACC__)

__global__ void mathParityKernel(int f, const double* x, const double* y, double* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = evalAether(f, x[i], y[i]);
}

TEST_F(MathParityTest, DeviceMatchesHostWithinDocumentedUlp)
{
    const auto xs = corpus();
    const auto ys = corpusB();
    const int n = static_cast<int>(xs.size());
    double *dx = nullptr, *dy = nullptr, *dout = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dy, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx, xs.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dy, ys.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<double> got(n);
    for (int f = 0; f < kNumFn; ++f) {
        mathParityKernel<<<(n + 127) / 128, 128>>>(f, dx, dy, dout, n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(got.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
        const std::int64_t bound = deviceUlp(f);
        std::int64_t worst = 0;
        for (int i = 0; i < n; ++i) {
            const double host = evalAether(f, xs[i], ys[i]);
            if (bound == 0) {
                ASSERT_TRUE(sameBits(host, got[i])) << fnName(f) << "(" << xs[i] << ", " << ys[i] << "): host " << host
                                                    << " device " << got[i];
            } else {
                const std::int64_t d = aether_tools::ulp::ulpDistanceAbs(host, got[i]);
                ASSERT_LE(d, bound) << fnName(f) << "(" << xs[i] << ", " << ys[i] << "): host " << host << " device "
                                    << got[i];
                worst = d > worst ? d : worst;
            }
        }
        RecordProperty(fnName(f), static_cast<int>(worst));
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
}

#endif

} // namespace math_parity
} // namespace aether_tests
