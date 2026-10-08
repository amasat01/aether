// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Faithful-rounding check of the CPU packet math against an exact reference,
// shared by the gtest hard-case gate (test_PacketMath_common.h) and the full-
// corpus checker (faithful_check_main.cpp).
//
// Reference rows come from tools/packet_math_ref/gen_golden.py: per input,
// `hi` = the exact result rounded to nearest and `frac` = (|exact| - |hi|)
// in units of the gap from `hi` to its neighbour on the exact value's side
// (|frac| <= 0.5; 0 = the exact result is `hi`). A result is faithfully
// rounded — error < 1 ULP — iff it is `hi` (error |frac|) or that neighbour
// (error 1 - |frac|). NaN must give NaN, an exact infinity that infinity, an
// exact or underflowed zero a zero of the right sign; an overflowing exact
// result (hi = inf, frac = -0.5) accepts inf or DBL_MAX.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "tests/packetmath/PacketMathKernels.h"

namespace aether_pm_test {

/// One golden function: its name in the files, its kernel, and its arity.
struct FaithfulFn {
    const char* name;  ///< kernel label (sincos.sin, ...)
    const char* file;  ///< golden file stem (sin, ...)
    int fn;            ///< Unary / Binary enumerator
    bool binary;
};

inline const std::vector<FaithfulFn>& faithfulFns()
{
    static const std::vector<FaithfulFn> v = {
        { "exp", "exp", kExp, false },
        { "exp2", "exp2", kExp2, false },
        { "exp10", "exp10", kExp10, false },
        { "expm1", "expm1", kExpm1, false },
        { "log", "log", kLog, false },
        { "log2", "log2", kLog2, false },
        { "log10", "log10", kLog10, false },
        { "log1p", "log1p", kLog1p, false },
        { "sin", "sin", kSin, false },
        { "cos", "cos", kCos, false },
        { "sincos.sin", "sin", kSinCosSin, false },
        { "sincos.cos", "cos", kSinCosCos, false },
        { "tan", "tan", kTan, false },
        { "atan", "atan", kAtan, false },
        { "asin", "asin", kAsin, false },
        { "acos", "acos", kAcos, false },
        { "sinh", "sinh", kSinh, false },
        { "cosh", "cosh", kCosh, false },
        { "tanh", "tanh", kTanh, false },
        { "asinh", "asinh", kAsinh, false },
        { "acosh", "acosh", kAcosh, false },
        { "atanh", "atanh", kAtanh, false },
        { "cbrt", "cbrt", kCbrt, false },
        { "rsqrt", "rsqrt", kRsqrt, false },
        { "sqrt", "sqrt", kSqrt, false },
        { "floor", "floor", kFloor, false },
        { "ceil", "ceil", kCeil, false },
        { "trunc", "trunc", kTrunc, false },
        { "round", "round", kRound, false },
        { "rint", "rint", kRint, false },
        { "fabs", "fabs", kAbs, false },
        { "pow", "pow", kPow, true },
        { "atan2", "atan2", kAtan2, true },
        { "hypot", "hypot", kHypot, true },
        { "fmax", "fmax", kFmax, true },
        { "fmin", "fmin", kFmin, true },
        { "fdim", "fdim", kFdim, true },
        { "copysign", "copysign", kCopysign, true },
    };
    return v;
}

/// Reference rows of one function, columns split out.
struct GoldenRows {
    std::vector<double> x, y, hi, frac;
    std::size_t size() const { return x.size(); }
};

/// Error of `got` in ULPs against the reference row (hi, frac): < 1 iff
/// faithfully rounded; +inf for a class mismatch.
inline double faithfulError(double got, double hi, double frac)
{
    constexpr double kBad = std::numeric_limits<double>::infinity();
    if (std::isnan(hi))
        return std::isnan(got) ? 0.0 : kBad;
    if (std::isnan(got))
        return kBad;
    if (frac == 0.0) { // exact: only hi itself, zero sign included
        if (std::memcmp(&got, &hi, sizeof(double)) == 0)
            return 0.0;
        if (std::isinf(hi) || std::isinf(got) || (got == 0.0 && hi == 0.0))
            return kBad;
    }
    if (got == 0.0 && hi == 0.0 && std::signbit(got) != std::signbit(hi))
        return kBad;
    if (got == hi)
        return std::fabs(frac);
    // Neighbour of hi on the side of the exact value (magnitude-wise).
    const double away = std::copysign(std::numeric_limits<double>::infinity(), hi);
    const double nb = std::nextafter(hi, frac > 0.0 ? away : std::copysign(0.0, hi));
    if (got == nb && !(got == 0.0 && std::signbit(got) != std::signbit(hi)))
        return 1.0 - std::fabs(frac);
    if (std::isinf(got) || std::isinf(hi) || std::signbit(got) != std::signbit(hi))
        return kBad;
    // Not faithful: report roughly how far (ordered-integer distance).
    std::int64_t bg, bh;
    std::memcpy(&bg, &got, sizeof(double));
    std::memcpy(&bh, &hi, sizeof(double));
    const double d = static_cast<double>((bg & 0x7FFFFFFFFFFFFFFFLL) - (bh & 0x7FFFFFFFFFFFFFFFLL));
    const bool sameSide = (d > 0) == (frac > 0.0);
    return sameSide ? std::fabs(d) - std::fabs(frac) : std::fabs(d) + std::fabs(frac);
}

/// Worst error of one function at one width.
struct FaithfulResult {
    double worst = 0.0;
    std::size_t at = 0;
    std::size_t bad = 0; ///< rows with error >= 1
    double got = 0.0;
};

/// Runs `k` on the rows and scores every result; `plant` >= 0 perturbs that
/// row's result by 2 ULP away from the exact value (non-vacuity probe).
inline FaithfulResult faithfulScore(const Kernels& k, const FaithfulFn& f, const GoldenRows& g,
    std::ptrdiff_t plant = -1)
{
    const std::size_t n = g.size();
    const std::size_t padded = (n + 7) / 8 * 8;
    std::vector<double> x(padded, 1.25), y(padded, 0.75), out(padded);
    std::copy(g.x.begin(), g.x.end(), x.begin());
    if (f.binary) {
        std::copy(g.y.begin(), g.y.end(), y.begin());
        k.binary(f.fn, x.data(), y.data(), out.data(), padded);
    } else {
        k.unary(f.fn, x.data(), out.data(), padded);
    }
    if (plant >= 0 && static_cast<std::size_t>(plant) < n) {
        // 2 ULP from hi, on the side away from the exact value, in the first
        // row from `plant` on whose reference is a normal number.
        std::size_t p = static_cast<std::size_t>(plant);
        for (std::size_t i = 0; i < n && !std::isnormal(g.hi[p]); ++i)
            p = (p + 1) % n;
        const double h = g.hi[p];
        const double dir = g.frac[p] > 0.0 ? 0.0 : std::copysign(std::numeric_limits<double>::infinity(), h);
        out[p] = std::nextafter(std::nextafter(h, dir), dir);
    }
    FaithfulResult r;
    for (std::size_t i = 0; i < n; ++i) {
        double e = faithfulError(out[i], g.hi[i], g.frac[i]);
        // C leaves the sign of fmax/fmin(+0, -0) unspecified.
        if (f.binary && (f.fn == kFmax || f.fn == kFmin) && g.x[i] == 0.0 && g.y[i] == 0.0 && out[i] == 0.0)
            e = 0.0;
        if (e >= 1.0)
            ++r.bad;
        if (e > r.worst || i == 0) {
            r.worst = e > r.worst ? e : r.worst;
            if (e >= r.worst) {
                r.at = i;
                r.got = out[i];
            }
        }
    }
    return r;
}

/// Reads `<dir>/<stem>.bin` (gen_golden.py layout). False when unreadable.
inline bool loadGolden(const std::string& path, bool binary, GoldenRows& g)
{
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    const std::size_t cols = binary ? 4 : 3;
    std::vector<double> buf(cols * 65536);
    g = GoldenRows{};
    for (;;) {
        const std::size_t got = std::fread(buf.data(), sizeof(double) * cols, 65536, fp);
        for (std::size_t i = 0; i < got; ++i) {
            const double* r = &buf[i * cols];
            g.x.push_back(r[0]);
            if (binary)
                g.y.push_back(r[1]);
            g.hi.push_back(r[cols - 2]);
            g.frac.push_back(r[cols - 1]);
        }
        if (got < 65536)
            break;
    }
    std::fclose(fp);
    return g.size() > 0;
}

/// Reads the committed hard-case set (tools/packet_math_ref/make_hard_set.py
/// layout: per function a 16-byte name, uint32 columns, uint32 rows, rows).
inline bool loadHardSet(const std::string& path, std::map<std::string, GoldenRows>& out)
{
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    out.clear();
    for (;;) {
        char name[17] = {};
        std::uint32_t hdr[2];
        if (std::fread(name, 1, 16, fp) != 16 || std::fread(hdr, sizeof(hdr), 1, fp) != 1)
            break;
        const std::size_t cols = hdr[0], rows = hdr[1];
        std::vector<double> buf(cols * rows);
        if (std::fread(buf.data(), sizeof(double), buf.size(), fp) != buf.size()) {
            std::fclose(fp);
            return false;
        }
        GoldenRows& g = out[name];
        for (std::size_t i = 0; i < rows; ++i) {
            const double* r = &buf[i * cols];
            g.x.push_back(r[0]);
            if (cols == 4)
                g.y.push_back(r[1]);
            g.hi.push_back(r[cols - 2]);
            g.frac.push_back(r[cols - 1]);
        }
    }
    std::fclose(fp);
    return !out.empty();
}

} // namespace aether_pm_test
