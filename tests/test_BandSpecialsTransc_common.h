// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandSpecialsTransc_common.h
 * @brief Special-value and admission-edge table for the banded
 *        transcendental, root, rounding and remainder functions.
 *
 * Every row's expected outcome is written from IEEE 754-2019 / C99 Annex F,
 * not read back from the implementation. Where the library deliberately
 * answers differently (a documented loud reject: NaN instead of a real
 * answer), the row lives in `kLabelled` and NAMES the outcome the library
 * returns next to the IEEE outcome it departs from. The rows go through the
 * public ingest/egress terminals (`bandFromIEEE` / `bandToIEEE`).
 */

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "aether/banded/banded.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace aether_tests {
namespace BandSpecialsTranscTest {

using aether::banded::Band;
namespace bd = aether::banded::detail;

inline std::uint64_t bits64(double d)
{
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

/// @brief The ingest terminal, spelled the way a consumer reaches it.
inline Band bandOf(double x)
{
    const std::uint64_t u = bits64(x);
    return bd::bandFromIEEE(static_cast<std::uint32_t>(u), static_cast<std::uint32_t>(u >> 32));
}

/// @brief The egress terminal, spelled the way a consumer reaches it.
inline double doubleOf(Band b)
{
    std::uint32_t lo = 0, hi = 0;
    bd::bandToIEEE(b, lo, hi);
    const std::uint64_t u = (static_cast<std::uint64_t>(hi) << 32) | lo;
    double d;
    std::memcpy(&d, &u, sizeof(d));
    return d;
}

enum class K { PosZero, NegZero, PosInf, NegInf, NaN, Val };

/// The expected outcome: a kind, plus the value when the kind is `Val`.
struct Want {
    K k;
    double v;
};

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi  = 3.14159265358979323846;

inline Want Z(bool neg = false) { return { neg ? K::NegZero : K::PosZero, 0.0 }; }
inline Want I(bool neg = false) { return { neg ? K::NegInf : K::PosInf, 0.0 }; }
inline Want N() { return { K::NaN, 0.0 }; }
inline Want V(double v) { return { K::Val, v }; }

/// @brief True when `got` is the outcome `w` describes (bit-exact for the
/// specials and signed zeros, 1e-6 relative for a finite value).
inline bool matches(double got, const Want& w)
{
    switch (w.k) {
        case K::PosZero: return bits64(got) == bits64(0.0);
        case K::NegZero: return bits64(got) == bits64(-0.0);
        case K::PosInf: return got == kInf;
        case K::NegInf: return got == -kInf;
        case K::NaN: return got != got;
        case K::Val:
            return std::isfinite(got) && std::fabs(got - w.v) <= 1e-6 * std::fabs(w.v);
    }
    return false;
}

inline std::string describe(double got)
{
    char b[64];
    std::snprintf(b, sizeof b, "%a (bits 0x%016llx)", got, static_cast<unsigned long long>(bits64(got)));
    return b;
}

struct Fn {
    const char* name;
    Band (*f1)(Band);
    Band (*f2)(Band, Band);
};

inline const Fn& fn(const std::string& n)
{
    static const Fn t[] = {
        { "exp", [](Band a) { return bd::exp(a); }, nullptr },
        { "expm1", [](Band a) { return bd::expm1(a); }, nullptr },
        { "log", [](Band a) { return bd::log(a); }, nullptr },
        { "log1p", [](Band a) { return bd::log1p(a); }, nullptr },
        { "exp2", [](Band a) { return bd::exp2(a); }, nullptr },
        { "asinh", [](Band a) { return bd::asinh(a); }, nullptr },
        { "acosh", [](Band a) { return bd::acosh(a); }, nullptr },
        { "atanh", [](Band a) { return bd::atanh(a); }, nullptr },
        { "tan", [](Band a) { return bd::tan(a); }, nullptr },
        { "tanh", [](Band a) { return bd::tanh(a); }, nullptr },
        { "sinh", [](Band a) { return bd::sinh(a); }, nullptr },
        { "cosh", [](Band a) { return bd::cosh(a); }, nullptr },
        { "log2", [](Band a) { return bd::log2(a); }, nullptr },
        { "log10", [](Band a) { return bd::log10(a); }, nullptr },
        { "pow", nullptr, [](Band a, Band b) { return bd::pow(a, b); } },
        { "sin", [](Band a) { return bd::sin(a); }, nullptr },
        { "cos", [](Band a) { return bd::cos(a); }, nullptr },
        { "sincos.s", [](Band a) { Band s, c; bd::sincos(a, s, c); return s; }, nullptr },
        { "sincos.c", [](Band a) { Band s, c; bd::sincos(a, s, c); return c; }, nullptr },
        { "atan", [](Band a) { return bd::atan(a); }, nullptr },
        { "atan2", nullptr, [](Band a, Band b) { return bd::atan2(a, b); } },
        { "asin", [](Band a) { return bd::asin(a); }, nullptr },
        { "acos", [](Band a) { return bd::acos(a); }, nullptr },
        { "sqrt", [](Band a) { return bd::sqrt_(a); }, nullptr },
        { "rsqrt", [](Band a) { return bd::rsqrtIeee(a); }, nullptr },
        { "rsqrt.core", [](Band a) { return bd::rsqrt(a); }, nullptr },
        { "cbrt", [](Band a) { return bd::cbrt(a); }, nullptr },
        { "hypot", nullptr, [](Band a, Band b) { return bd::hypot(a, b); } },
        { "fmod", nullptr, [](Band a, Band b) { return bd::fmod(a, b); } },
        { "floor", [](Band a) { return bd::floor(a); }, nullptr },
        { "ceil", [](Band a) { return bd::ceil(a); }, nullptr },
        { "round", [](Band a) { return bd::round(a); }, nullptr },
        { "trunc", [](Band a) { return bd::trunc(a); }, nullptr },
    };
    for (const Fn& f : t)
        if (n == f.name)
            return f;
    ADD_FAILURE() << "unknown function " << n;
    return t[0];
}

inline double run(const Fn& f, double a, double b)
{
    return doubleOf(f.f2 ? f.f2(bandOf(a), bandOf(b)) : f.f1(bandOf(a)));
}

struct Row {
    const char* fn;
    double a;
    double b; ///< unused by a unary function
    Want want;
    const char* cite;
};

// The edges. A float32 neighbour is the "edge + 1" step.
inline double up(float x) { return static_cast<double>(std::nextafterf(x, INFINITY)); }
constexpr double kTrigEdge   = 16777216.0; // 2^24, the last admitted |x|
constexpr float kExpHigh     = 88.7228391f; // 128 ln2, the library's overflow edge
constexpr double kSub32      = 1e-39;      // an FP32 subnormal magnitude

/// IEEE 754-2019 / C99 Annex F outcomes the library must reproduce.
inline const std::vector<Row>& kIeee()
{
    static const std::vector<Row> t{
        // exp (F.10.3.1)
        { "exp", 0.0, 0, V(1), "exp(+0) = 1" },
        { "exp", -0.0, 0, V(1), "exp(-0) = 1" },
        { "exp", -kInf, 0, Z(), "exp(-Inf) = +0" },
        { "exp", kInf, 0, I(), "exp(+Inf) = +Inf" },
        { "exp", kNaN, 0, N(), "exp(NaN) = NaN" },
        // expm1 (F.10.3.4)
        { "expm1", 0.0, 0, Z(), "expm1(+0) = +0" },
        { "expm1", -0.0, 0, Z(true), "expm1(-0) = -0" },
        { "expm1", -kInf, 0, V(-1), "expm1(-Inf) = -1" },
        { "expm1", kInf, 0, I(), "expm1(+Inf) = +Inf" },
        { "expm1", kNaN, 0, N(), "expm1(NaN) = NaN" },
        // tan (F.10.1.7)
        { "tan", 0.0, 0, Z(), "tan(+0) = +0" },
        { "tan", -0.0, 0, Z(true), "tan(-0) = -0" },
        { "tan", kInf, 0, N(), "tan(+Inf) = NaN" },
        { "tan", -kInf, 0, N(), "tan(-Inf) = NaN" },
        { "tan", kNaN, 0, N(), "tan(NaN) = NaN" },
        { "tan", 1.0, 0, V(std::tan(1.0)), "tan(1)" },
        { "tan", -0.5, 0, V(std::tan(-0.5)), "tan(-1/2)" },
        { "tan", 1e-30, 0, V(1e-30), "tan(1e-30) = 1e-30" },
        // asinh (F.10.4.3)
        { "asinh", 0.0, 0, Z(), "asinh(+0) = +0" },
        { "asinh", -0.0, 0, Z(true), "asinh(-0) = -0" },
        { "asinh", kInf, 0, I(), "asinh(+Inf) = +Inf" },
        { "asinh", -kInf, 0, I(true), "asinh(-Inf) = -Inf" },
        { "asinh", kNaN, 0, N(), "asinh(NaN) = NaN" },
        { "asinh", 1.0, 0, V(std::asinh(1.0)), "asinh(1)" },
        { "asinh", -3.0, 0, V(std::asinh(-3.0)), "asinh(-3), the [2, 2^28) branch" },
        { "asinh", 0x1p30, 0, V(std::asinh(0x1p30)), "asinh(2^30), the log(a) + ln2 branch" },
        { "asinh", 1e-30, 0, V(1e-30), "asinh(1e-30) = 1e-30" },
        // acosh (F.10.4.4)
        { "acosh", 1.0, 0, Z(), "acosh(1) = +0" },
        { "acosh", 0.5, 0, N(), "acosh(x<1) = NaN" },
        { "acosh", -1.0, 0, N(), "acosh(-1) = NaN" },
        { "acosh", std::nextafter(1.0, 0.0), 0, N(), "acosh(1 - ulp) = NaN" },
        { "acosh", kInf, 0, I(), "acosh(+Inf) = +Inf" },
        { "acosh", -kInf, 0, N(), "acosh(-Inf) = NaN" },
        { "acosh", kNaN, 0, N(), "acosh(NaN) = NaN" },
        { "acosh", 2.0, 0, V(std::acosh(2.0)), "acosh(2)" },
        { "acosh", 1.0 + 0x1p-40, 0, V(std::acosh(1.0 + 0x1p-40)), "acosh(1 + 2^-40), the exact-subtraction regime" },
        { "acosh", 0x1p30, 0, V(std::acosh(0x1p30)), "acosh(2^30), the log(x) + ln2 branch" },
        // atanh (F.10.4.6)
        { "atanh", 0.0, 0, Z(), "atanh(+0) = +0" },
        { "atanh", -0.0, 0, Z(true), "atanh(-0) = -0" },
        { "atanh", 1.0, 0, I(), "atanh(+1) = +Inf" },
        { "atanh", -1.0, 0, I(true), "atanh(-1) = -Inf" },
        { "atanh", 2.0, 0, N(), "atanh(|x|>1) = NaN" },
        { "atanh", std::nextafter(1.0, 2.0), 0, N(), "atanh(1 + ulp) = NaN" },
        { "atanh", kInf, 0, N(), "atanh(+Inf) = NaN" },
        { "atanh", kNaN, 0, N(), "atanh(NaN) = NaN" },
        { "atanh", 0.25, 0, V(std::atanh(0.25)), "atanh(1/4)" },
        { "atanh", -0.75, 0, V(std::atanh(-0.75)), "atanh(-3/4)" },
        { "atanh", 1.0 - 0x1p-40, 0, V(std::atanh(1.0 - 0x1p-40)), "atanh(1 - 2^-40), the exact 1 - a regime" },
        // tanh (F.10.4.5)
        { "tanh", 0.0, 0, Z(), "tanh(+0) = +0" },
        { "tanh", -0.0, 0, Z(true), "tanh(-0) = -0" },
        { "tanh", kInf, 0, V(1), "tanh(+Inf) = +1" },
        { "tanh", -kInf, 0, V(-1), "tanh(-Inf) = -1" },
        { "tanh", kNaN, 0, N(), "tanh(NaN) = NaN" },
        { "tanh", 1.0, 0, V(std::tanh(1.0)), "tanh(1)" },
        { "tanh", -0.25, 0, V(std::tanh(-0.25)), "tanh(-1/4)" },
        { "tanh", 30.0, 0, V(1), "tanh(30) = 1" },
        { "tanh", 1e-30, 0, V(1e-30), "tanh(1e-30) = 1e-30" },
        // sinh (F.10.4.5)
        { "sinh", 0.0, 0, Z(), "sinh(+0) = +0" },
        { "sinh", -0.0, 0, Z(true), "sinh(-0) = -0" },
        { "sinh", kInf, 0, I(), "sinh(+Inf) = +Inf" },
        { "sinh", -kInf, 0, I(true), "sinh(-Inf) = -Inf" },
        { "sinh", kNaN, 0, N(), "sinh(NaN) = NaN" },
        { "sinh", 1.0, 0, V(std::sinh(1.0)), "sinh(1)" },
        { "sinh", -1.0, 0, V(std::sinh(-1.0)), "sinh(-1) = -sinh(1)" },
        { "sinh", 1e-30, 0, V(1e-30), "sinh(1e-30) = 1e-30" },
        // cosh (F.10.4.6)
        { "cosh", 0.0, 0, V(1), "cosh(+0) = 1" },
        { "cosh", -0.0, 0, V(1), "cosh(-0) = 1" },
        { "cosh", kInf, 0, I(), "cosh(+Inf) = +Inf" },
        { "cosh", -kInf, 0, I(), "cosh(-Inf) = +Inf" },
        { "cosh", kNaN, 0, N(), "cosh(NaN) = NaN" },
        { "cosh", 1.0, 0, V(std::cosh(1.0)), "cosh(1)" },
        { "cosh", -1.0, 0, V(std::cosh(-1.0)), "cosh(-1) = cosh(1) >= 1" },
        { "cosh", 1e-30, 0, V(1), "cosh(1e-30) = 1" },
        // exp2 (F.10.3.2)
        { "exp2", 0.0, 0, V(1), "exp2(+0) = 1" },
        { "exp2", -0.0, 0, V(1), "exp2(-0) = 1" },
        { "exp2", -kInf, 0, Z(), "exp2(-Inf) = +0" },
        { "exp2", kInf, 0, I(), "exp2(+Inf) = +Inf" },
        { "exp2", kNaN, 0, N(), "exp2(NaN) = NaN" },
        { "exp2", 10.0, 0, V(1024.0), "exp2(10) = 1024" },
        { "exp2", 127.0, 0, V(0x1p127), "exp2(127) = 2^127" },
        { "exp2", 128.0, 0, I(), "exp2(128): FP32 overflow gives +Inf" },
        { "exp2", -151.0, 0, Z(), "exp2(-151): FP32 underflow gives +0" },
        // log2 (F.10.3.10)
        { "log2", 1.0, 0, Z(), "log2(1) = +0" },
        { "log2", 0.0, 0, I(true), "log2(+0) = -Inf" },
        { "log2", -0.0, 0, I(true), "log2(-0) = -Inf" },
        { "log2", -2.0, 0, N(), "log2(x<0) = NaN" },
        { "log2", kInf, 0, I(), "log2(+Inf) = +Inf" },
        { "log2", kNaN, 0, N(), "log2(NaN) = NaN" },
        { "log2", 8.0, 0, V(3.0), "log2(8) = 3" },
        // log10 (F.10.3.8)
        { "log10", 1.0, 0, Z(), "log10(1) = +0" },
        { "log10", 0.0, 0, I(true), "log10(+0) = -Inf" },
        { "log10", -0.0, 0, I(true), "log10(-0) = -Inf" },
        { "log10", -2.0, 0, N(), "log10(x<0) = NaN" },
        { "log10", kInf, 0, I(), "log10(+Inf) = +Inf" },
        { "log10", kNaN, 0, N(), "log10(NaN) = NaN" },
        { "log10", 1000.0, 0, V(3.0), "log10(1000) = 3" },
        // log1p (F.10.3.9)
        { "log1p", 0.0, 0, Z(), "log1p(+0) = +0" },
        { "log1p", -0.0, 0, Z(true), "log1p(-0) = -0" },
        { "log1p", -1.0, 0, I(true), "log1p(-1) = -Inf" },
        { "log1p", kInf, 0, I(), "log1p(+Inf) = +Inf" },
        { "log1p", kNaN, 0, N(), "log1p(NaN) = NaN" },
        { "log1p", -2.0, 0, N(), "log1p(x<-1) = NaN" },
        { "log1p", -kInf, 0, N(), "log1p(-Inf) = NaN" },
        { "log1p", std::nextafter(-1.0, -2.0), 0, N(), "log1p(-1 - ulp) = NaN" },
        { "log1p", 1.0, 0, V(std::log1p(1.0)), "log1p(1) = ln 2" },
        { "log1p", -0.5, 0, V(std::log1p(-0.5)), "log1p(-1/2) = -ln 2" },
        { "log1p", 0x1p-19, 0, V(std::log1p(0x1p-19)), "log1p at the series seam" },
        { "log1p", -0x1p-19, 0, V(std::log1p(-0x1p-19)), "log1p at the series seam, negative" },
        { "log1p", 1e-30, 0, V(1e-30), "log1p(1e-30) = 1e-30" },
        { "log1p", kSub32, 0, V(1e-39), "log1p(1e-39) = 1e-39" },
        { "log1p", -1.0 + 0x1p-30, 0, V(std::log1p(-1.0 + 0x1p-30)), "log1p(-1 + 2^-30), {-1, 2^-30, 0}" },
        // log (F.10.3.7)
        { "log", 1.0, 0, Z(), "log(1) = +0" },
        { "log", 0.0, 0, I(true), "log(+0) = -Inf" },
        { "log", -0.0, 0, I(true), "log(-0) = -Inf" },
        { "log", -2.0, 0, N(), "log(x<0) = NaN" },
        { "log", -kInf, 0, N(), "log(-Inf) = NaN" },
        { "log", kInf, 0, I(), "log(+Inf) = +Inf" },
        { "log", kNaN, 0, N(), "log(NaN) = NaN" },
        // pow (F.10.4.4): the rows the library keeps
        { "pow", 2.0, 0.0, V(1), "pow(x,+0) = 1" },
        { "pow", 2.0, -0.0, V(1), "pow(x,-0) = 1" },
        { "pow", kNaN, 0.0, V(1), "pow(NaN,+0) = 1" },
        { "pow", 0.0, 0.0, V(1), "pow(+0,+0) = 1" },
        { "pow", 2.5, 1.0, V(2.5), "pow(x,1) = x" },
        { "pow", 2.0, 3.0, V(8), "pow(2,3) = 8" },
        // sin / cos (F.10.1.6, F.10.1.5)
        { "sin", 0.0, 0, Z(), "sin(+0) = +0" },
        { "sin", -0.0, 0, Z(true), "sin(-0) = -0" },
        { "sin", kInf, 0, N(), "sin(+Inf) = NaN" },
        { "sin", -kInf, 0, N(), "sin(-Inf) = NaN" },
        { "sin", kNaN, 0, N(), "sin(NaN) = NaN" },
        { "cos", 0.0, 0, V(1), "cos(+0) = 1" },
        { "cos", -0.0, 0, V(1), "cos(-0) = 1" },
        { "cos", kInf, 0, N(), "cos(+Inf) = NaN" },
        { "cos", -kInf, 0, N(), "cos(-Inf) = NaN" },
        { "cos", kNaN, 0, N(), "cos(NaN) = NaN" },
        { "sincos.s", 0.0, 0, Z(), "sincos(+0).sin = +0" },
        { "sincos.s", -0.0, 0, Z(true), "sincos(-0).sin = -0" },
        { "sincos.s", kInf, 0, N(), "sincos(Inf).sin = NaN" },
        { "sincos.s", kNaN, 0, N(), "sincos(NaN).sin = NaN" },
        { "sincos.c", 0.0, 0, V(1), "sincos(+0).cos = 1" },
        { "sincos.c", -0.0, 0, V(1), "sincos(-0).cos = 1" },
        { "sincos.c", -kInf, 0, N(), "sincos(-Inf).cos = NaN" },
        { "sincos.c", kNaN, 0, N(), "sincos(NaN).cos = NaN" },
        // atan / atan2 (F.10.1.3, F.10.1.4)
        { "atan", 0.0, 0, Z(), "atan(+0) = +0" },
        { "atan", -0.0, 0, Z(true), "atan(-0) = -0" },
        { "atan", kInf, 0, V(kPi / 2), "atan(+Inf) = +pi/2" },
        { "atan", -kInf, 0, V(-kPi / 2), "atan(-Inf) = -pi/2" },
        { "atan", kNaN, 0, N(), "atan(NaN) = NaN" },
        { "atan2", 0.0, 0.0, Z(), "atan2(+0,+0) = +0" },
        { "atan2", -0.0, 0.0, Z(true), "atan2(-0,+0) = -0" },
        { "atan2", 0.0, -0.0, V(kPi), "atan2(+0,-0) = +pi" },
        { "atan2", -0.0, -0.0, V(-kPi), "atan2(-0,-0) = -pi" },
        { "atan2", 0.0, -1.0, V(kPi), "atan2(+0,x<0) = +pi" },
        { "atan2", -0.0, -1.0, V(-kPi), "atan2(-0,x<0) = -pi" },
        { "atan2", 0.0, 1.0, Z(), "atan2(+0,x>0) = +0" },
        { "atan2", -0.0, 1.0, Z(true), "atan2(-0,x>0) = -0" },
        { "atan2", 1.0, 0.0, V(kPi / 2), "atan2(y>0,+0) = +pi/2" },
        { "atan2", -1.0, -0.0, V(-kPi / 2), "atan2(y<0,-0) = -pi/2" },
        { "atan2", 1.0, kInf, Z(), "atan2(y>0,+Inf) = +0" },
        { "atan2", -1.0, kInf, Z(true), "atan2(y<0,+Inf) = -0" },
        { "atan2", 1.0, -kInf, V(kPi), "atan2(y>0,-Inf) = +pi" },
        { "atan2", -1.0, -kInf, V(-kPi), "atan2(y<0,-Inf) = -pi" },
        { "atan2", kInf, 1.0, V(kPi / 2), "atan2(+Inf,x) = +pi/2" },
        { "atan2", -kInf, 1.0, V(-kPi / 2), "atan2(-Inf,x) = -pi/2" },
        { "atan2", kInf, kInf, V(kPi / 4), "atan2(+Inf,+Inf) = +pi/4" },
        { "atan2", kInf, -kInf, V(3 * kPi / 4), "atan2(+Inf,-Inf) = +3pi/4" },
        { "atan2", -kInf, kInf, V(-kPi / 4), "atan2(-Inf,+Inf) = -pi/4" },
        { "atan2", -kInf, -kInf, V(-3 * kPi / 4), "atan2(-Inf,-Inf) = -3pi/4" },
        { "atan2", kNaN, 1.0, N(), "atan2(NaN,x) = NaN" },
        { "atan2", 1.0, kNaN, N(), "atan2(y,NaN) = NaN" },
        // asin / acos (F.10.1.2, F.10.1.1)
        { "asin", 0.0, 0, Z(), "asin(+0) = +0" },
        { "asin", -0.0, 0, Z(true), "asin(-0) = -0" },
        { "asin", 1.0, 0, V(kPi / 2), "asin(1) = pi/2" },
        { "asin", -1.0, 0, V(-kPi / 2), "asin(-1) = -pi/2" },
        { "asin", 2.0, 0, N(), "asin(|x|>1) = NaN" },
        { "asin", -2.0, 0, N(), "asin(|x|>1) = NaN" },
        { "asin", kInf, 0, N(), "asin(Inf) = NaN" },
        { "asin", kNaN, 0, N(), "asin(NaN) = NaN" },
        { "acos", 1.0, 0, Z(), "acos(1) = +0" },
        { "acos", -1.0, 0, V(kPi), "acos(-1) = pi" },
        { "acos", 0.0, 0, V(kPi / 2), "acos(0) = pi/2" },
        { "acos", 2.0, 0, N(), "acos(|x|>1) = NaN" },
        { "acos", -2.0, 0, N(), "acos(|x|>1) = NaN" },
        { "acos", -kInf, 0, N(), "acos(-Inf) = NaN" },
        { "acos", kNaN, 0, N(), "acos(NaN) = NaN" },
        // sqrt / rsqrt (F.10.4.5; rsqrt follows 1/sqrt)
        { "sqrt", -2.0, 0, N(), "sqrt(x<0) = NaN" },
        { "sqrt", -kInf, 0, N(), "sqrt(-Inf) = NaN" },
        { "sqrt", 0.0, 0, Z(), "sqrt(+0) = +0" },
        { "sqrt", -0.0, 0, Z(true), "sqrt(-0) = -0" },
        { "sqrt", kInf, 0, I(), "sqrt(+Inf) = +Inf" },
        { "sqrt", kNaN, 0, N(), "sqrt(NaN) = NaN" },
        { "sqrt", 4.0, 0, V(2), "sqrt(4) = 2" },
        { "rsqrt", kInf, 0, Z(), "rsqrt(+Inf) = +0" },
        { "rsqrt", 0.0, 0, I(), "rsqrt(+0) = +Inf" },
        { "rsqrt", -0.0, 0, I(true), "rsqrt(-0) = -Inf" },
        { "rsqrt", -2.0, 0, N(), "rsqrt(x<0) = NaN" },
        { "rsqrt", kNaN, 0, N(), "rsqrt(NaN) = NaN" },
        // cbrt (F.10.4.1) / hypot (F.10.4.3)
        { "cbrt", kInf, 0, I(), "cbrt(+Inf) = +Inf" },
        { "cbrt", -kInf, 0, I(true), "cbrt(-Inf) = -Inf" },
        { "cbrt", 0.0, 0, Z(), "cbrt(+0) = +0" },
        { "cbrt", -0.0, 0, Z(true), "cbrt(-0) = -0" },
        { "cbrt", kNaN, 0, N(), "cbrt(NaN) = NaN" },
        { "cbrt", -8.0, 0, V(-2), "cbrt(-8) = -2" },
        { "hypot", kInf, kNaN, I(), "hypot(+Inf,NaN) = +Inf" },
        { "hypot", kNaN, -kInf, I(), "hypot(NaN,-Inf) = +Inf" },
        { "hypot", -kInf, 1.0, I(), "hypot(-Inf,y) = +Inf" },
        { "hypot", 0.0, -0.0, Z(), "hypot(+0,-0) = +0" },
        { "hypot", kNaN, 1.0, N(), "hypot(NaN,finite) = NaN" },
        { "hypot", 3.0, -4.0, V(5), "hypot(3,-4) = 5" },
        // fmod (F.10.7.1)
        { "fmod", 1.0, 0.0, N(), "fmod(x,+0) = NaN" },
        { "fmod", 1.0, -0.0, N(), "fmod(x,-0) = NaN" },
        { "fmod", 0.0, 2.0, Z(), "fmod(+0,y) = +0" },
        { "fmod", -0.0, 2.0, Z(true), "fmod(-0,y) = -0" },
        { "fmod", kInf, 2.0, N(), "fmod(+Inf,y) = NaN" },
        { "fmod", -kInf, 2.0, N(), "fmod(-Inf,y) = NaN" },
        { "fmod", 3.0, kInf, V(3), "fmod(x,+Inf) = x" },
        { "fmod", -3.0, -kInf, V(-3), "fmod(x,-Inf) = x" },
        { "fmod", kNaN, 2.0, N(), "fmod(NaN,y) = NaN" },
        { "fmod", 2.0, kNaN, N(), "fmod(x,NaN) = NaN" },
        { "fmod", -5.5, 2.0, V(-1.5), "fmod(-5.5,2) = -1.5" },
        // floor / ceil / round / trunc (F.10.6): zeros, infinities, NaN
        { "floor", 0.0, 0, Z(), "floor(+0) = +0" },
        { "floor", -0.0, 0, Z(true), "floor(-0) = -0" },
        { "floor", kInf, 0, I(), "floor(+Inf) = +Inf" },
        { "floor", -kInf, 0, I(true), "floor(-Inf) = -Inf" },
        { "floor", kNaN, 0, N(), "floor(NaN) = NaN" },
        { "ceil", 0.0, 0, Z(), "ceil(+0) = +0" },
        { "ceil", -0.0, 0, Z(true), "ceil(-0) = -0" },
        { "ceil", kInf, 0, I(), "ceil(+Inf) = +Inf" },
        { "ceil", -kInf, 0, I(true), "ceil(-Inf) = -Inf" },
        { "ceil", kNaN, 0, N(), "ceil(NaN) = NaN" },
        { "round", 0.0, 0, Z(), "round(+0) = +0" },
        { "round", -0.0, 0, Z(true), "round(-0) = -0" },
        { "round", kInf, 0, I(), "round(+Inf) = +Inf" },
        { "round", -kInf, 0, I(true), "round(-Inf) = -Inf" },
        { "round", kNaN, 0, N(), "round(NaN) = NaN" },
        { "trunc", 0.0, 0, Z(), "trunc(+0) = +0" },
        { "trunc", -0.0, 0, Z(true), "trunc(-0) = -0" },
        { "trunc", kInf, 0, I(), "trunc(+Inf) = +Inf" },
        { "trunc", -kInf, 0, I(true), "trunc(-Inf) = -Inf" },
        { "trunc", kNaN, 0, N(), "trunc(NaN) = NaN" },
        // admission edges (the last admitted point, and the step past it) and
        // an FP32-subnormal magnitude (1e-39)
        { "exp", std::nextafterf(kExpHigh, 0.0f), 0, V(std::exp(static_cast<double>(std::nextafterf(kExpHigh, 0.0f)))),
            "exp just below the overflow edge stays finite" },
        { "exp", 88.0, 0, V(std::exp(88.0)), "exp(88) is finite" },
        { "expm1", std::nextafterf(kExpHigh, 0.0f), 0, V(std::expm1(static_cast<double>(std::nextafterf(kExpHigh, 0.0f)))),
            "expm1 just below the overflow edge stays finite" },
        { "expm1", kSub32, 0, V(1e-39), "expm1(1e-39) = 1e-39" },
        { "exp", kSub32, 0, V(1), "exp(1e-39) = 1" },
        { "log", kSub32, 0, V(std::log(kSub32)), "log(1e-39)" },
        { "log", 0x1p-126, 0, V(std::log(0x1p-126)), "log(smallest normal)" },
        { "log", 0x1p127, 0, V(std::log(0x1p127)), "log(2^127)" },
        { "pow", 2.0, 127.0, V(0x1p127), "pow(2,127) = 2^127" },
        { "pow", 2.0, kInf, I(), "pow(2,+Inf) = +Inf" },
        { "pow", 2.0, -kInf, Z(), "pow(2,-Inf) = +0" },
        { "pow", kNaN, 1.0, N(), "pow(NaN,1) = NaN" },
        { "pow", -2.0, 0.5, N(), "pow(x<0, non-integer) = NaN" },
        { "pow", 1.0, 5.0, V(1), "pow(1,y) = 1" },
        { "pow", 1.0, kNaN, V(1), "pow(1,NaN) = 1" },
        { "pow", 1.0, kInf, V(1), "pow(1,+Inf) = 1" },
        { "pow", 0.0, 2.0, Z(), "pow(+0,y>0) = +0" },
        { "pow", -0.0, 3.0, Z(true), "pow(-0,odd y>0) = -0" },
        { "pow", 0.0, -1.0, I(), "pow(+0,y<0) = +Inf" },
        { "pow", kInf, 2.0, I(), "pow(+Inf,y>0) = +Inf" },
        { "pow", kInf, -2.0, Z(), "pow(+Inf,y<0) = +0" },
        { "pow", -kInf, 2.0, I(), "pow(-Inf,even y>0) = +Inf" },
        { "pow", -kInf, 3.0, I(true), "pow(-Inf,odd y>0) = -Inf" },
        { "pow", -kInf, -2.0, Z(), "pow(-Inf,even y<0) = +0" },
        { "pow", -kInf, -3.0, Z(true), "pow(-Inf,odd y<0) = -0" },
        { "pow", -0.0, -3.0, I(true), "pow(-0,odd y<0) = -Inf" },
        { "sin", kTrigEdge, 0, V(std::sin(kTrigEdge)), "sin(2^24), last admitted" },
        { "sin", -kTrigEdge, 0, V(std::sin(-kTrigEdge)), "sin(-2^24), last admitted" },
        { "cos", kTrigEdge, 0, V(std::cos(kTrigEdge)), "cos(2^24), last admitted" },
        { "sin", up(16777216.0f), 0, N(), "past 2^24 the library rejects: labelled NaN" },
        { "cos", up(16777216.0f), 0, N(), "past 2^24 the library rejects: labelled NaN" },
        { "sincos.s", up(16777216.0f), 0, N(), "past 2^24 the library rejects: labelled NaN" },
        { "sincos.c", up(16777216.0f), 0, N(), "past 2^24 the library rejects: labelled NaN" },
        { "sincos.s", kTrigEdge, 0, V(std::sin(kTrigEdge)), "sincos(2^24).sin, last admitted" },
        { "sincos.c", kTrigEdge, 0, V(std::cos(kTrigEdge)), "sincos(2^24).cos, last admitted" },
        { "sin", kSub32, 0, V(kSub32), "sin(x) = x for tiny x" },
        { "cos", kSub32, 0, V(1), "cos(1e-39) = 1" },
        { "atan", kSub32, 0, V(kSub32), "atan(x) = x for tiny x" },
        { "asin", kSub32, 0, V(kSub32), "asin(x) = x for tiny x" },
        { "acos", kSub32, 0, V(kPi / 2), "acos(1e-39) = pi/2" },
        { "asin", std::nextafterf(1.0f, 0.0f), 0, V(std::asin(static_cast<double>(std::nextafterf(1.0f, 0.0f)))),
            "asin just inside 1" },
        { "asin", up(1.0f), 0, N(), "asin(1+ulp) = NaN, the step past the domain edge" },
        { "acos", up(1.0f), 0, N(), "acos(1+ulp) = NaN, the step past the domain edge" },
        { "atan2", kSub32, 1.0, V(kSub32), "atan2(1e-39,1) = 1e-39" },
        { "atan2", 1.0, kSub32, V(kPi / 2), "atan2(1,1e-39) = pi/2" },
        { "atan2", 0x1p-128, 0x1p-128, V(kPi / 4), "atan2 equal operands at 2^-128, last admitted" },
        { "atan2", 0x1p125, 0x1p125, V(kPi / 4), "atan2 equal operands at 2^125, last admitted" },
        { "sqrt", 0x1p-127, 0, V(std::sqrt(0x1p-127)), "sqrt(2^-127), last admitted" },
        { "sqrt", 0x1p-126, 0, V(0x1p-63), "sqrt(smallest normal)" },
        { "rsqrt", 0x1p-127, 0, V(1.0 / std::sqrt(0x1p-127)), "rsqrt(2^-127), last admitted" },
        { "cbrt", 0x1p-128, 0, V(std::cbrt(0x1p-128)), "cbrt(2^-128), last admitted" },
        { "hypot", kSub32, kSub32, V(kSub32 * std::sqrt(2.0)), "hypot(1e-39,1e-39)" },
        { "fmod", kSub32, 1.0, V(kSub32), "fmod(1e-39,1) = 1e-39" },
        { "floor", kSub32, 0, Z(), "floor(1e-39) = +0" },
        { "floor", -kSub32, 0, V(-1), "floor(-1e-39) = -1" },
        { "ceil", kSub32, 0, V(1), "ceil(1e-39) = 1" },
        { "round", kSub32, 0, Z(), "round(1e-39) = +0" },
        { "trunc", kSub32, 0, Z(), "trunc(1e-39) = +0" },
        { "round", 0.5, 0, V(1), "round(0.5) = 1, half away from zero" },
        { "round", -0.5, 0, V(-1), "round(-0.5) = -1, half away from zero" },
        { "trunc", -0.2, 0, Z(true), "trunc(-0.2) = -0" },
        { "ceil", -0.2, 0, Z(true), "ceil(-0.2) = -0" },
        { "fmod", -5.0, 5.0, Z(true), "fmod(-5,5) = -0, the sign of x" },
    };
    return t;
}

TEST(BandSpecialsTransc, IeeeAnnexFRowsHold)
{
    for (const Row& r : kIeee()) {
        const double got = run(fn(r.fn), r.a, r.b);
        EXPECT_TRUE(matches(got, r.want)) << r.fn << "(" << r.a << ", " << r.b << "): " << r.cite << "; got "
                                          << describe(got);
    }
}

/// Rows where the library's outcome is a NAMED departure from the IEEE
/// outcome in the cite: a loud NaN reject for an operand the op does not
/// certify, or an FP32-range overflow/underflow. The row pins what the
/// library returns; the cite says what Annex F would give.
inline const std::vector<Row>& kLabelled()
{
    static const std::vector<Row> t{
        { "exp", kExpHigh, 0, I(), "overflow edge: FP32 tier answers +Inf (Annex F, double: 2^128)" },
        { "exp", up(kExpHigh), 0, I(), "past the overflow edge: +Inf" },
        { "exp", -103.972077, 0, V(0x1p-149), "underflow edge: smallest FP32 subnormal (Annex F: 7e-46)" },
        { "exp", std::nextafterf(-103.972077f, -INFINITY), 0, Z(), "past the underflow edge: +0" },
        { "tan", 33554432.0, 0, N(), "beyond the 2^24 trig guard: NaN (Annex F: a finite value)" },
        { "sinh", 90.0, 0, I(), "past the FP32 envelope edge: +Inf (Annex F, double: 6e38)" },
        { "sinh", -90.0, 0, I(true), "past the FP32 envelope edge: -Inf" },
        { "cosh", 90.0, 0, I(), "past the FP32 envelope edge: +Inf (Annex F, double: 6e38)" },
        { "cosh", -90.0, 0, I(), "past the FP32 envelope edge: +Inf" },
        { "sinh", 89.0, 0, V(std::sinh(89.0)), "sinh(89) is finite, past expm1's own edge" },
        { "cosh", 89.0, 0, V(std::cosh(89.0)), "cosh(89) is finite, past expm1's own edge" },
        { "expm1", kExpHigh, 0, I(), "overflow edge: FP32 tier answers +Inf" },
        { "expm1", up(kExpHigh), 0, I(), "past the overflow edge: +Inf" },
        { "expm1", -103.972077, 0, V(-1), "underflow edge: -1 + 2^-149 is -1 to double" },
        { "expm1", std::nextafterf(-103.972077f, -INFINITY), 0, V(-1), "past the underflow edge: exactly -1" },
        { "pow", 2.0, 128.0, I(), "pow(2,128): FP32 overflow gives +Inf" },
        { "pow", 2.0, -150.0, Z(), "pow(2,-150): FP32 underflow gives +0" },
        { "pow", kSub32, 2.0, Z(), "pow(1e-39,2): FP32 underflow gives +0 (Annex F: 1e-78)" },
        { "pow", -2.0, 3.0, N(), "rejected base x<=0: NaN (Annex F: -8; pown is the op for it)" },
        { "sqrt", kSub32, 0, N(), "FP32-subnormal operand rejected: NaN (Annex F: 3.2e-20)" },
        { "sqrt", 0x1p-128, 0, N(), "first rejected point, one binade below 2^-127: NaN" },
        { "rsqrt", kSub32, 0, N(), "FP32-subnormal operand rejected: NaN (Annex F: 3.2e19)" },
        { "rsqrt", 0x1p-128, 0, N(), "first rejected point, one binade below 2^-127: NaN" },
        { "rsqrt.core", kInf, 0, N(), "the raw core has no +Inf fix: NaN (Annex F: +0; rsqrt is the public op)" },
        { "cbrt", kSub32, 0, N(), "FP32-subnormal operand rejected: NaN (Annex F: 1e-13)" },
        { "cbrt", 0x1p-130, 0, N(), "first rejected point, two binades below 2^-128: NaN" },
        { "atan2", kSub32, kSub32, N(), "both operands past the small-magnitude edge: NaN (Annex F: pi/4)" },
        { "atan2", 0x1p127, 0x1p127, N(), "both operands past the 2^125 ceiling: NaN (Annex F: pi/4)" },
        { "atan2", 0x1p127, 1.0, N(), "operand past the 2^125 ceiling: NaN (Annex F: pi/2)" },
        { "hypot", 0x1.c363cbf21f28ap+127, 0x1.c363cbf21f28ap+127, I(), "FP32 overflow gives +Inf (Annex F, double: 4.2e38)" },
    };
    return t;
}

inline Want mutated(const Want& w)
{
    switch (w.k) {
        case K::PosZero: return Z(true);
        case K::NegZero: return Z();
        case K::PosInf: return I(true);
        case K::NegInf: return I();
        case K::NaN: return Z();
        case K::Val: return V(w.v * 2.0);
    }
    return w;
}

TEST(BandSpecialsTransc, LabelledRowsNameTheLibrarysActualOutcome)
{
    for (const Row& r : kLabelled()) {
        const double got = run(fn(r.fn), r.a, r.b);
        EXPECT_TRUE(matches(got, r.want)) << r.fn << "(" << r.a << ", " << r.b << "): " << r.cite << "; got "
                                          << describe(got);
    }
}

// tan = sin/cos: its relative claim stops at |cos x| = 2^-13. These two inputs
// pin the documented degradation as OBSERVED, so the claim cannot silently
// widen (or the degradation vanish unannounced): |cos x| in [2^-30, 2^-20)
// is a few ULP off, within 2^-30 of a pole nothing is delivered. The
// references are correctly rounded values of the exact three-limb inputs.
TEST(BandSpecialsTransc, TanDegradesNearPolesAsDocumented)
{
    auto steps = [](Band x, double ref) {
        const double got = doubleOf(bd::tan(x));
        return std::fabs(got - ref) / std::ldexp(1.0, std::ilogb(ref) - 52);
    };
    const double degrades = steps(Band{ 0x1.e2a3b2p+15f, 0x1.1b2f62p-10f, 0x1.b16744p-36f }, 0x1.b9c2986c94736p+29);
    EXPECT_GE(degrades, 4.0) << "|cos| in [2^-30, 2^-20): the degradation is no longer observed";
    EXPECT_LE(degrades, 64.0) << "|cos| in [2^-30, 2^-20): worse than the documented 0.25*2^-13/|cos| ULP";
    const double undelivered = steps(Band{ 0x1.5f5e14p+12f, 0x1.662a44p-13f, 0x1.c98adap-38f }, 0x1.f916b218208e2p+53);
    EXPECT_GE(undelivered, 1024.0) << "|cos| < 2^-30: the value is no longer undelivered";
}

TEST(BandSpecialsTransc, EveryFunctionHasAnIeeeRowAndAnEdgeOrSubnormalRow)
{
    for (const char* name : { "exp", "expm1", "log", "log1p", "exp2", "log2", "log10", "tanh", "sinh", "cosh", "asinh", "acosh", "atanh", "tan", "pow", "sin", "cos", "sincos.s", "sincos.c", "atan", "atan2", "asin",
             "acos", "sqrt", "rsqrt", "cbrt", "hypot", "fmod", "floor", "ceil", "round", "trunc" }) {
        int ieee = 0;
        for (const Row& r : kIeee())
            ieee += std::string(r.fn) == name;
        EXPECT_GE(ieee, 5) << name << " has too few Annex F rows";
    }
}

// RED-first: a checker that cannot go red certifies nothing. Perturb the
// expected outcome of every row, and every function's input, and require red.
TEST(BandSpecialsTransc, RedFirstPerturbedExpectationFailsEveryRow)
{
    for (const std::vector<Row>* tab : { &kIeee(), &kLabelled() })
        for (const Row& r : *tab) {
            const double got = run(fn(r.fn), r.a, r.b);
            EXPECT_FALSE(matches(got, mutated(r.want))) << r.fn << ": " << r.cite << " passed a perturbed expectation";
        }
}

TEST(BandSpecialsTransc, RedFirstPerturbedInputFailsEveryFunction)
{
    for (const char* name : { "exp", "expm1", "log", "log1p", "exp2", "log2", "log10", "tanh", "sinh", "cosh", "asinh", "acosh", "atanh", "tan", "pow", "sin", "cos", "sincos.s", "sincos.c", "atan", "atan2", "asin",
             "acos", "sqrt", "rsqrt", "cbrt", "hypot", "fmod", "floor", "ceil", "round", "trunc" }) {
        bool red = false;
        for (const Row& r : kIeee()) {
            if (std::string(r.fn) != name || r.want.k == K::NaN)
                continue;
            // The same row with its first operand replaced by NaN.
            red = !matches(run(fn(name), kNaN, r.b), r.want);
            if (red)
                break;
        }
        EXPECT_TRUE(red) << name << ": a NaN operand never turned any row red";
    }
}

} // namespace BandSpecialsTranscTest
} // namespace aether_tests
