// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Accuracy gate for the CPU packet math (aether/backend/cpu/simd/math/) and
// the AETHER_HOST_VECTOR_MATH hook. Shared by test_PacketMath.cpp (C++ build)
// and test_PacketMath.cu (CUDA build); the packet code itself lives in the
// host-compiled kernel TUs under tests/packetmath/ (see PacketMathKernels.h),
// this file only builds the corpus, computes the glibc std:: reference and
// checks ULP distances.
//
// For every function and every width (AVX-512 8, AVX2 4, SSE2 2, scalar 1):
//   * FAITHFUL ROUNDING: error < 1 ULP against the exact result (mpmath at
//     128 bits) over the committed hard-case set tests/packetmath/
//     faithful_hard.bin (specials, domain edges, a stride of the full
//     corpus and its worst rows; see tools/packet_math_ref/). The full
//     >= 1M-input-per-function corpus is the `make faithful-gate` target;
//   * a planted 2-ULP error turns that check red (non-vacuity);
//   * max ULP distance to std:: over the corpus <= the documented bound
//     (aether/backend/cpu/simd/math/PacketMath.h's table): 1 where glibc is
//     itself within 1 ULP of the exact result (two faithful results are at
//     most 1 apart), more where glibc's own error is larger (log10, sinh,
//     tanh, acosh, atanh: 2; cbrt: 3), and 1 for rsqrt against the doubly
//     rounded 1/std::sqrt;
//   * NaN / infinity class and the sign of zero results match std:: exactly.
// Plus: width 4 and width 8 lanes are bit-identical to width 1 (same FMA
// semantics), and the hook's vectorised loops are bit-identical to the
// per-element scalar symbol.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "aether/math/math.h"
#include "tests/packetmath/FaithfulCheck.h"
#include "tests/packetmath/PacketMathKernels.h"
#include "tools/ulp_oracle/ulp.h"

namespace {

using namespace aether_pm_test;

constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

const std::vector<double>& specialValues()
{
    static const std::vector<double> v = {
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 1.5, -1.5, 2.5, -2.5, 10.0, -10.0, 0.1, -0.1,
        kInf, -kInf, kNaN, -kNaN, 5e-324, -5e-324, 2.2250738585072014e-308, -2.2250738585072014e-308,
        2.2250738585072009e-308, 1.7976931348623157e308, -1.7976931348623157e308, 1e-300, 1e300, -1e300,
        709.78, 709.79, 710.0, -708.4, -745.1, -745.2, -746.0, 1023.99, 1024.0, -1074.0, -1075.0, 307.0, 308.5,
        -323.5, 22.0, -22.0, 0.49999999999999994, -0.49999999999999994, 0.9999999999999999,
        1.0000000000000002, 0x1p-27, -0x1p-27, 0x1p-60, 0x1p28, 0x1p30, 0x1p52, -0x1p52, 0x1p53,
        4503599627370495.5, -4503599627370495.5, 4503599627370497.0, 9007199254740993.0,
        3.141592653589793, 1.5707963267948966, 0.7853981633974483, 524288.0, 524289.5, 1e6, -1e6, 1e22,
        0.41421356237309503, 2.414213562373095,
    };
    return v;
}

// Uniform (or log-uniform over a positive range) samples plus the specials,
// padded to a multiple of 8 so every width sees whole packets.
std::vector<double> corpus(double lo, double hi, bool logScale, int n, unsigned seed, bool withSpecials = true)
{
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> v;
    v.reserve(n + 128);
    for (int i = 0; i < n; ++i) {
        const double t = u(g);
        v.push_back(logScale ? std::exp(std::log(lo) + t * (std::log(hi) - std::log(lo))) : lo + t * (hi - lo));
    }
    if (withSpecials)
        v.insert(v.end(), specialValues().begin(), specialValues().end());
    while (v.size() % 8)
        v.push_back(0.75);
    return v;
}

std::vector<double> concat(std::vector<double> a, const std::vector<double>& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

struct UnaryCase {
    Unary fn;
    const char* name;
    double (*ref)(double);
    std::int64_t bound;
    std::vector<double> xs;
};

struct BinaryCase {
    Binary fn;
    const char* name;
    double (*ref)(double, double);
    std::int64_t bound;
    std::vector<double> xs, ys;
};

const std::vector<UnaryCase>& unaryCases()
{
    static const std::vector<UnaryCase> cases = [] {
        const auto trig = concat(concat(corpus(-10, 10, false, 60000, 21), corpus(-6e5, 6e5, false, 20000, 22)),
            corpus(1e-30, 1e-2, true, 10000, 23, false));
        const auto unit = concat(corpus(-1.0, 1.0, false, 60000, 28), corpus(1e-20, 1.0, true, 10000, 29, false));
        const auto hyp = concat(concat(corpus(-30, 30, false, 60000, 30), corpus(1e-20, 1, true, 10000, 31, false)),
            corpus(-712, 712, false, 10000, 32, false));
        // log: the whole range, a dense band around 1 (the table-driven path hands
        // |x - 1| < 1/16 to the double-double path, so both sides of that seam
        // are scored) and the subnormals down to 5e-324.
        const auto logs = concat(concat(corpus(1e-310, 1e308, true, 60000, 5), corpus(0.5, 2.0, false, 40000, 6, false)),
            concat(corpus(0.9, 1.1, false, 20000, 51, false), corpus(5e-324, 2.3e-308, true, 20000, 52, false)));
        // exp: the whole range plus the overflow edge (709.78), the underflow edge (-745.13),
        // the subnormal results between -708.4 and -745.13 and the table-path seam at +-700.
        const auto exps = concat(concat(corpus(-750, 750, false, 80000, 1), corpus(709.0, 710.5, false, 10000, 53, false)),
            concat(corpus(-746.0, -708.0, false, 20000, 54, false), corpus(-701.0, 701.0, false, 10000, 55, false)));
        const auto rnd = concat(corpus(-1e6, 1e6, false, 40000, 37), corpus(-4, 4, false, 20000, 38, false));
        std::vector<UnaryCase> c = {
            { kExp, "exp", [](double x) { return std::exp(x); }, 1, exps },
            { kExp2, "exp2", [](double x) { return std::exp2(x); }, 1, corpus(-1100, 1100, false, 80000, 2) },
            { kExp10, "exp10", [](double x) { return std::pow(10.0, x); }, 1, corpus(-330, 330, false, 80000, 3) },
            { kExpm1, "expm1", [](double x) { return std::expm1(x); }, 1,
                concat(concat(corpus(-3, 3, false, 60000, 4), corpus(-50, 720, false, 20000, 9, false)),
                    corpus(1e-20, 1e-3, true, 10000, 10, false)) },
            { kLog, "log", [](double x) { return std::log(x); }, 1, logs },
            { kLog2, "log2", [](double x) { return std::log2(x); }, 1, logs },
            { kLog10, "log10", [](double x) { return std::log10(x); }, 2, logs },
            { kLog1p, "log1p", [](double x) { return std::log1p(x); }, 1,
                concat(corpus(-0.999, 3, false, 60000, 7), corpus(1e-30, 1e300, true, 20000, 8, false)) },
            { kSin, "sin", [](double x) { return std::sin(x); }, 1, trig },
            { kCos, "cos", [](double x) { return std::cos(x); }, 1, trig },
            { kSinCosSin, "sincos.sin", [](double x) { return std::sin(x); }, 1, trig },
            { kSinCosCos, "sincos.cos", [](double x) { return std::cos(x); }, 1, trig },
            { kTan, "tan", [](double x) { return std::tan(x); }, 1, trig },
            { kAtan, "atan", [](double x) { return std::atan(x); }, 1,
                concat(corpus(-50, 50, false, 60000, 24), corpus(1e-300, 1e300, true, 20000, 25, false)) },
            { kAsin, "asin", [](double x) { return std::asin(x); }, 1, unit },
            { kAcos, "acos", [](double x) { return std::acos(x); }, 1, unit },
            { kSinh, "sinh", [](double x) { return std::sinh(x); }, 2, hyp },
            { kCosh, "cosh", [](double x) { return std::cosh(x); }, 1, hyp },
            { kTanh, "tanh", [](double x) { return std::tanh(x); }, 2, hyp },
            { kAsinh, "asinh", [](double x) { return std::asinh(x); }, 1,
                concat(corpus(1e-300, 1e300, true, 40000, 33), corpus(-5, 5, false, 40000, 34, false)) },
            { kAcosh, "acosh", [](double x) { return std::acosh(x); }, 2,
                concat(corpus(1, 1e300, true, 40000, 35), corpus(1, 3, false, 40000, 36, false)) },
            { kAtanh, "atanh", [](double x) { return std::atanh(x); }, 2, unit },
            { kFloor, "floor", [](double x) { return std::floor(x); }, 0, rnd },
            { kCeil, "ceil", [](double x) { return std::ceil(x); }, 0, rnd },
            { kTrunc, "trunc", [](double x) { return std::trunc(x); }, 0, rnd },
            { kRound, "round", [](double x) { return std::round(x); }, 0, rnd },
            { kRint, "rint", [](double x) { return std::rint(x); }, 0, rnd },
            { kAbs, "fabs", [](double x) { return std::fabs(x); }, 0, rnd },
            { kSqrt, "sqrt", [](double x) { return std::sqrt(x); }, 0, logs },
            { kRsqrt, "rsqrt", [](double x) { return 1.0 / std::sqrt(x); }, 1, logs },
            { kCbrt, "cbrt", [](double x) { return std::cbrt(x); }, 3,
                concat(corpus(1e-310, 1e308, true, 60000, 39), corpus(-10, 10, false, 20000, 40, false)) },
        };
        return c;
    }();
    return cases;
}

// Every pair of special values, plus random pairs.
void specialGrid(std::vector<double>& xs, std::vector<double>& ys)
{
    for (double x : specialValues())
        for (double y : specialValues()) {
            xs.push_back(x);
            ys.push_back(y);
        }
}

void padPairs(std::vector<double>& xs, std::vector<double>& ys)
{
    while (xs.size() % 8) {
        xs.push_back(1.25);
        ys.push_back(0.75);
    }
}

const std::vector<BinaryCase>& binaryCases()
{
    static const std::vector<BinaryCase> cases = [] {
        std::vector<BinaryCase> c;
        {
            // pow: |y log x| spread up to ~700 (the over/underflow range), the
            // [0.7, 1.4] base band with |y| up to 2000, integer and odd-integer
            // exponents on negative bases, and the special grid.
            auto x = corpus(1e-300, 1e300, true, 60000, 11, false);
            auto y = corpus(-1, 1, false, 60000, 12, false);
            for (std::size_t i = 0; i < x.size(); ++i)
                y[i] = y[i] * 700.0 / std::fmax(1e-3, std::fabs(std::log(x[i])));
            x = concat(x, corpus(0.7, 1.4, false, 20000, 13, false));
            y = concat(y, corpus(-2000, 2000, false, 20000, 14, false));
            auto nx = corpus(-50, -1e-3, false, 20000, 15, false);
            auto ny = corpus(-40, 40, false, 20000, 16, false);
            for (double& v : ny)
                v = std::round(v);
            x = concat(x, nx);
            y = concat(y, ny);
            x.resize(std::min(x.size(), y.size()));
            y.resize(x.size());
            specialGrid(x, y);
            padPairs(x, y);
            c.push_back({ kPow, "pow", [](double a, double b) { return std::pow(a, b); }, 1, x, y });
        }
        {
            auto y = corpus(-1e3, 1e3, false, 60000, 26, false);
            auto x = corpus(-1e3, 1e3, false, 60000, 27, false);
            specialGrid(y, x);
            padPairs(y, x);
            c.push_back({ kAtan2, "atan2", [](double a, double b) { return std::atan2(a, b); }, 1, y, x });
        }
        {
            auto x = concat(corpus(1e-300, 1e300, true, 40000, 41, false), corpus(-10, 10, false, 40000, 43, false));
            auto y = concat(corpus(1e-300, 1e300, true, 40000, 42, false), corpus(-10, 10, false, 40000, 44, false));
            specialGrid(x, y);
            padPairs(x, y);
            c.push_back({ kHypot, "hypot", [](double a, double b) { return std::hypot(a, b); }, 1, x, y });
        }
        std::vector<double> gx, gy;
        specialGrid(gx, gy);
        padPairs(gx, gy);
        c.push_back({ kFmax, "fmax", [](double a, double b) { return std::fmax(a, b); }, 0, gx, gy });
        c.push_back({ kFmin, "fmin", [](double a, double b) { return std::fmin(a, b); }, 0, gx, gy });
        c.push_back({ kFdim, "fdim", [](double a, double b) { return std::fdim(a, b); }, 0, gx, gy });
        c.push_back({ kCopysign, "copysign", [](double a, double b) { return std::copysign(a, b); }, 0, gx, gy });
        return c;
    }();
    return cases;
}

// ULP distance with exact class/zero-sign agreement folded in: a zero result
// whose sign differs from std::'s counts as a mismatch, EXCEPT where C leaves
// the sign of an equal-magnitude choice unspecified (fmax/fmin of +-0).
std::int64_t distance(double ref, double got, bool zeroSignMatters)
{
    const std::int64_t d = aether_tools::ulp::ulpDistanceAbs(ref, got);
    if (d == 0 && zeroSignMatters && ref == 0.0 && got == 0.0 && std::signbit(ref) != std::signbit(got))
        return aether_tools::ulp::kMismatch;
    return d;
}

const Kernels& kernelsFor(std::size_t w)
{
    switch (w) {
    case 1: return kernelsW1();
    case 2: return kernelsW2();
    case 4: return kernelsW4();
    default: return kernelsW8();
    }
}

void checkUnaryWidth(std::size_t w)
{
    const Kernels& k = kernelsFor(w);
    if (!k.available())
        GTEST_SKIP() << "width " << w << " needs AVX-512F, which this CPU does not report";
    for (const UnaryCase& c : unaryCases()) {
        std::vector<double> got(c.xs.size());
        k.unary(c.fn, c.xs.data(), got.data(), c.xs.size());
        std::int64_t worst = 0;
        std::size_t at = 0;
        for (std::size_t i = 0; i < c.xs.size(); ++i) {
            const std::int64_t d = distance(c.ref(c.xs[i]), got[i], true);
            if (d > worst) {
                worst = d;
                at = i;
            }
        }
        std::printf("[packet-math] W=%zu %-11s max ULP %lld (n=%zu)\n", w, c.name, static_cast<long long>(worst),
            c.xs.size());
        EXPECT_LE(worst, c.bound) << c.name << " at x=" << c.xs[at] << " got " << got[at] << " want "
                                  << c.ref(c.xs[at]);
    }
}

void checkBinaryWidth(std::size_t w)
{
    const Kernels& k = kernelsFor(w);
    if (!k.available())
        GTEST_SKIP() << "width " << w << " needs AVX-512F, which this CPU does not report";
    for (const BinaryCase& c : binaryCases()) {
        std::vector<double> got(c.xs.size());
        k.binary(c.fn, c.xs.data(), c.ys.data(), got.data(), c.xs.size());
        const bool zeroSign = c.fn != kFmax && c.fn != kFmin;
        std::int64_t worst = 0;
        std::size_t at = 0;
        for (std::size_t i = 0; i < c.xs.size(); ++i) {
            const std::int64_t d = distance(c.ref(c.xs[i], c.ys[i]), got[i], zeroSign);
            if (d > worst) {
                worst = d;
                at = i;
            }
        }
        std::printf("[packet-math] W=%zu %-11s max ULP %lld (n=%zu)\n", w, c.name, static_cast<long long>(worst),
            c.xs.size());
        EXPECT_LE(worst, c.bound) << c.name << " at x=" << c.xs[at] << " y=" << c.ys[at] << " got " << got[at]
                                  << " want " << c.ref(c.xs[at], c.ys[at]);
    }
}

// Same bits, or both NaN (C leaves NaN payload/sign unspecified, and GCC
// may commute the operands of a NaN-propagating `x + y` differently in the
// scalar and the vector code).
std::size_t bitMismatches(const std::vector<double>& a, const std::vector<double>& b)
{
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0 && !(std::isnan(a[i]) && std::isnan(b[i])))
            ++bad;
    return bad;
}

// Lanes of width w are bit-identical to width 1 (both FMA builds).
void checkLanesMatchScalar(std::size_t w)
{
    const Kernels& k = kernelsFor(w);
    if (!k.available())
        GTEST_SKIP() << "width " << w << " needs AVX-512F, which this CPU does not report";
    const Kernels& s = kernelsW1();
    for (const UnaryCase& c : unaryCases()) {
        std::vector<double> a(c.xs.size()), b(c.xs.size());
        k.unary(c.fn, c.xs.data(), a.data(), c.xs.size());
        s.unary(c.fn, c.xs.data(), b.data(), c.xs.size());
        EXPECT_EQ(0u, bitMismatches(a, b)) << c.name;
    }
    for (const BinaryCase& c : binaryCases()) {
        std::vector<double> a(c.xs.size()), b(c.xs.size());
        k.binary(c.fn, c.xs.data(), c.ys.data(), a.data(), c.xs.size());
        s.binary(c.fn, c.xs.data(), c.ys.data(), b.data(), c.xs.size());
        EXPECT_EQ(0u, bitMismatches(a, b)) << c.name;
    }
}

} // namespace

namespace {

const std::map<std::string, GoldenRows>& hardSet()
{
    static const std::map<std::string, GoldenRows> rows = [] {
        std::map<std::string, GoldenRows> m;
        loadHardSet(AETHER_PM_HARD_SET, m);
        return m;
    }();
    return rows;
}

void checkFaithfulWidth(std::size_t w)
{
    const Kernels& k = kernelsFor(w);
    if (!k.available())
        GTEST_SKIP() << "width " << w << " needs AVX-512F, which this CPU does not report";
    ASSERT_FALSE(hardSet().empty()) << "cannot read " << AETHER_PM_HARD_SET;
    for (const FaithfulFn& f : faithfulFns()) {
        const auto it = hardSet().find(f.file);
        ASSERT_NE(it, hardSet().end()) << f.file << " missing from " << AETHER_PM_HARD_SET;
        const GoldenRows& g = it->second;
        const FaithfulResult r = faithfulScore(k, f, g);
        std::printf("[packet-math] W=%zu %-11s max error %.4f ULP vs exact (n=%zu)\n", w, f.name, r.worst, g.size());
        EXPECT_LT(r.worst, 1.0) << f.name << " at x=" << g.x[r.at] << (f.binary ? " y=" : "")
                                << (f.binary ? std::to_string(g.y[r.at]) : std::string()) << " got " << r.got
                                << " want " << g.hi[r.at] << " (frac " << g.frac[r.at] << ")";
    }
}

} // namespace

TEST(PacketMathFaithful, HardSetWidth1) { checkFaithfulWidth(1); }
TEST(PacketMathFaithful, HardSetWidth2Sse2) { checkFaithfulWidth(2); }
TEST(PacketMathFaithful, HardSetWidth4Avx2) { checkFaithfulWidth(4); }
TEST(PacketMathFaithful, HardSetWidth8Avx512) { checkFaithfulWidth(8); }

// Non-vacuity: the same scoring with one result moved 2 ULP away from the
// exact value must report an error >= 1 ULP for every function.
TEST(PacketMathFaithful, PlantedTwoUlpErrorIsCaught)
{
    ASSERT_FALSE(hardSet().empty()) << "cannot read " << AETHER_PM_HARD_SET;
    for (const FaithfulFn& f : faithfulFns()) {
        const GoldenRows& g = hardSet().at(f.file);
        const FaithfulResult r = faithfulScore(kernelsW1(), f, g, static_cast<std::ptrdiff_t>(g.size() / 2));
        EXPECT_GE(r.worst, 1.0) << "planted 2-ULP error in " << f.name << " not caught";
    }
}

TEST(PacketMathAccuracy, UnaryWidth1) { checkUnaryWidth(1); }
TEST(PacketMathAccuracy, UnaryWidth2Sse2) { checkUnaryWidth(2); }
TEST(PacketMathAccuracy, UnaryWidth4Avx2) { checkUnaryWidth(4); }
TEST(PacketMathAccuracy, UnaryWidth8Avx512) { checkUnaryWidth(8); }
TEST(PacketMathAccuracy, BinaryWidth1) { checkBinaryWidth(1); }
TEST(PacketMathAccuracy, BinaryWidth2Sse2) { checkBinaryWidth(2); }
TEST(PacketMathAccuracy, BinaryWidth4Avx2) { checkBinaryWidth(4); }
TEST(PacketMathAccuracy, BinaryWidth8Avx512) { checkBinaryWidth(8); }
TEST(PacketMathAccuracy, Width4LanesBitIdenticalToWidth1) { checkLanesMatchScalar(4); }
TEST(PacketMathAccuracy, Width8LanesBitIdenticalToWidth1) { checkLanesMatchScalar(8); }

// pm_hook.cpp is built with -ffp-contract=fast, the kernel TUs with
// contraction off: every routed function's facade loop must still equal the
// width-1 kernel bit for bit over its own accuracy corpus. Without the
// contraction guard in HostVectorMath.h gcc fuses the argument reductions
// and error-free transformations, and sin, cos, tan, atan, asin, acos and
// atan2 move by an ULP on some inputs (measured: up to 486 of
// 65536 inputs per function) — this test then fails.
TEST(PacketMathHostVectorHook, HookUnderContractionMatchesKernels)
{
    int covered = 0;
    for (const UnaryCase& c : unaryCases()) {
        std::vector<double> hook(c.xs.size()), ref(c.xs.size());
        if (!hookUnaryLoop(c.fn, c.xs.data(), hook.data(), c.xs.size()))
            continue;
        kernelsW1().unary(c.fn, c.xs.data(), ref.data(), c.xs.size());
        EXPECT_EQ(0u, bitMismatches(hook, ref)) << c.name;
        ++covered;
    }
    for (const BinaryCase& c : binaryCases()) {
        std::vector<double> hook(c.xs.size()), ref(c.xs.size());
        if (!hookBinaryLoop(c.fn, c.xs.data(), c.ys.data(), hook.data(), c.xs.size()))
            continue;
        kernelsW1().binary(c.fn, c.xs.data(), c.ys.data(), ref.data(), c.xs.size());
        EXPECT_EQ(0u, bitMismatches(hook, ref)) << c.name;
        ++covered;
    }
    EXPECT_EQ(covered, 29 + 5); // every routed function with a loop of its own
}

// The hook's vectorised loops (vector-ABI variants + scalar remainder) give
// exactly what the scalar symbol gives element by element, for a length that
// is not a multiple of any vector width.
TEST(PacketMathHostVectorHook, PowLoopBitIdenticalToScalarSymbol)
{
    const auto x = corpus(1e-3, 1e3, true, 4093, 51, false);
    const auto y = corpus(-8, 8, false, 4093, 52, false);
    const std::size_t n = 4093;
    std::vector<double> out(n);
    hookPowLoop(x.data(), y.data(), out.data(), n);
    std::int64_t worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double s = hookPowScalar(x[i], y[i]);
        ASSERT_TRUE(std::memcmp(&s, &out[i], sizeof(double)) == 0 || (std::isnan(s) && std::isnan(out[i]))) << "i=" << i;
        worst = std::max(worst, distance(std::pow(x[i], y[i]), out[i], true));
    }
    EXPECT_LE(worst, 1);
}

TEST(PacketMathHostVectorHook, ConditionalPowLoopUsesMaskedVariantCorrectly)
{
    const auto x = corpus(0.01, 4.0, false, 4093, 53, false);
    const auto y = corpus(-3, 3, false, 4093, 54, false);
    const std::size_t n = 4093;
    std::vector<double> out(n);
    hookPowLoopConditional(x.data(), y.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const double s = x[i] > 1.0 ? hookPowScalar(x[i], y[i]) : x[i];
        ASSERT_TRUE(std::memcmp(&s, &out[i], sizeof(double)) == 0 || (std::isnan(s) && std::isnan(out[i]))) << "i=" << i;
    }
}

TEST(PacketMathHostVectorHook, SinExpLoopBitIdenticalToScalarSymbols)
{
    const auto x = corpus(-20, 20, false, 4093, 55, false);
    const std::size_t n = 4093;
    std::vector<double> out(n);
    hookSinExpLoop(x.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const double s = hookSinScalar(x[i]) + hookExpScalar(x[i]);
        ASSERT_TRUE(std::memcmp(&s, &out[i], sizeof(double)) == 0 || (std::isnan(s) && std::isnan(out[i]))) << "i=" << i;
    }
}

// sqrt, rsqrt and floor go through the vector-ABI symbols and abs stays
// inline: the vectorised loop matches the scalar symbols bit for bit, and
// std:: exactly (rsqrt: within 1 ULP of the doubly rounded 1/std::sqrt).
TEST(PacketMathHostVectorHook, RootFloorAbsLoopMatchesScalarSymbolsAndStd)
{
    const auto x = concat(corpus(1e-300, 1e300, true, 2000, 60, false), corpus(-50, 50, false, 2093, 61));
    const std::size_t n = x.size();
    std::vector<double> sq(n), rs(n), fl(n), ab(n);
    hookRootFloorAbsLoop(x.data(), sq.data(), rs.data(), fl.data(), ab.data(), n);
    const auto same = [](double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0 || (std::isnan(a) && std::isnan(b)); };
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(same(hookSqrtScalar(x[i]), sq[i])) << "sqrt i=" << i;
        ASSERT_TRUE(same(hookRsqrtScalar(x[i]), rs[i])) << "rsqrt i=" << i;
        ASSERT_TRUE(same(hookFloorScalar(x[i]), fl[i])) << "floor i=" << i;
        ASSERT_TRUE(same(std::sqrt(x[i]), sq[i])) << "sqrt vs std x=" << x[i];
        ASSERT_TRUE(same(std::floor(x[i]), fl[i])) << "floor vs std x=" << x[i];
        ASSERT_TRUE(same(std::fabs(x[i]), ab[i])) << "abs vs std x=" << x[i];
        ASSERT_LE(distance(1.0 / std::sqrt(x[i]), rs[i], true), 1) << "rsqrt vs std x=" << x[i];
    }
}

// The functions routed to the hook beyond the original transcendental set
// (expm1, log1p, the hyperbolics, the rounding family, fdim, copysign): over
// each function's own accuracy corpus the vectorised facade loop matches its
// scalar symbol bit for bit and std:: within the documented bound.
TEST(PacketMathHostVectorHook, ParityLoopsMatchScalarSymbolsAndStd)
{
    static const char* const names[kNumHookParity] = { "expm1", "log1p", "sinh", "cosh", "asinh", "acosh", "atanh",
        "ceil", "trunc", "round", "rint", "fdim", "copysign" };
    const auto same = [](double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0 || (std::isnan(a) && std::isnan(b)); };
    int covered = 0;
    for (int fn = 0; fn < kNumHookParity; ++fn) {
        const UnaryCase* u = nullptr;
        const BinaryCase* b = nullptr;
        for (const UnaryCase& c : unaryCases())
            if (std::string(c.name) == names[fn])
                u = &c;
        for (const BinaryCase& c : binaryCases())
            if (std::string(c.name) == names[fn])
                b = &c;
        ASSERT_TRUE(u != nullptr || b != nullptr) << names[fn];
        const std::vector<double>& xs = u ? u->xs : b->xs;
        const std::vector<double> ys = u ? std::vector<double>(xs.size(), 0.0) : b->ys;
        std::vector<double> out(xs.size());
        hookParityLoop(fn, xs.data(), ys.data(), out.data(), xs.size());
        std::int64_t worst = 0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            ASSERT_TRUE(same(hookParityScalar(fn, xs[i], ys[i]), out[i])) << names[fn] << " i=" << i;
            const double ref = u ? u->ref(xs[i]) : b->ref(xs[i], ys[i]);
            worst = std::max(worst, distance(ref, out[i], true));
        }
        EXPECT_LE(worst, u ? u->bound : b->bound) << names[fn];
        ++covered;
    }
    EXPECT_EQ(covered, int(kNumHookParity));
}

// Under the hook fmax/fmin are inline selects; they must give std::'s
// results bit for bit, except the sign of a +0/-0 tie, which C leaves
// unspecified (glibc's out-of-line fmax returns the second operand, GCC's
// inline expansion of std::fmax the first) — and NaN payloads.
TEST(PacketMathHostVectorHook, FmaxFminMatchStdBitForBit)
{
    std::vector<double> x, y;
    specialGrid(x, y);
    const auto rx = corpus(-5, 5, false, 2000, 58, false);
    const auto ry = corpus(-5, 5, false, 2000, 59, false);
    x.insert(x.end(), rx.begin(), rx.end());
    y.insert(y.end(), ry.begin(), ry.end());
    std::vector<double> mx(x.size()), mn(x.size());
    hookFmaxFminLoop(x.data(), y.data(), mx.data(), mn.data(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double a = std::fmax(x[i], y[i]);
        const double b = std::fmin(x[i], y[i]);
        const bool zeroTie = x[i] == 0.0 && y[i] == 0.0;
        ASSERT_TRUE(std::memcmp(&a, &mx[i], sizeof(double)) == 0 || (std::isnan(a) && std::isnan(mx[i]))
            || (zeroTie && mx[i] == 0.0))
            << "fmax(" << x[i] << ", " << y[i] << ")";
        ASSERT_TRUE(std::memcmp(&b, &mn[i], sizeof(double)) == 0 || (std::isnan(b) && std::isnan(mn[i]))
            || (zeroTie && mn[i] == 0.0))
            << "fmin(" << x[i] << ", " << y[i] << ")";
    }
}

// Without the macro (this TU), the facade is std:: — the hook is opt-in.
TEST(PacketMathHostVectorHook, DefaultFacadeStaysStd)
{
    const auto x = corpus(1e-3, 1e3, true, 1000, 56);
    const auto y = corpus(-8, 8, false, 1000, 57);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double a = aether::math::pow(x[i], y[i]);
        const double b = std::pow(x[i], y[i]);
        const double c = aether::math::sin(x[i]);
        const double d = std::sin(x[i]);
        ASSERT_EQ(0, std::memcmp(&a, &b, sizeof(double)));
        ASSERT_EQ(0, std::memcmp(&c, &d, sizeof(double)));
    }
}
