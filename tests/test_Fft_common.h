// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_Fft_common.h
 * @brief `Fft` suite: `aether::fft` (pocketfft over interleaved (re, im) pairs) against a direct O(n^2) DFT
 *        accumulated in `long double`.
 *
 * Tolerance, ULP-scaled: the largest absolute error of a transform of `N` points is bounded by
 * `(8 + 4 log2 N) * eps * ||x||_2 * scale`, `eps` the machine epsilon of the width and `scale` the
 * normalisation factor. Both widths, even and odd lengths (including primes), 1-D / 2-D / batched, the three
 * norms, the real transforms and their round trips, and the error paths.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "aether/fft/Fft.h"

namespace {

using Idx = std::vector<std::size_t>;
using LD = long double;
using CLD = std::complex<LD>;

template <class T> std::vector<T> randomVec(std::size_t n, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<T> v(n);
    for (auto& x : v) x = T(u(gen));
    return v;
}

std::size_t prodOf(const Idx& s) {
    std::size_t n = 1;
    for (auto v : s) n *= v;
    return n;
}

/// naive DFT of dense row-major complex data over `axes` (one axis at a time), unscaled
template <class T> std::vector<CLD> naive(const T* pairs, const Idx& shape, const Idx& axes, bool forward) {
    std::size_t total = prodOf(shape);
    std::vector<CLD> cur(total);
    for (std::size_t i = 0; i < total; ++i) cur[i] = CLD(pairs[2 * i], pairs[2 * i + 1]);
    const LD pi = std::acos(LD(-1));
    for (std::size_t ax : axes) {
        std::size_t n = shape[ax], inner = 1;
        for (std::size_t k = ax + 1; k < shape.size(); ++k) inner *= shape[k];
        std::size_t outer = total / (n * inner);
        std::vector<CLD> next(total);
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t in = 0; in < inner; ++in)
                for (std::size_t k = 0; k < n; ++k) {
                    CLD acc = 0;
                    for (std::size_t j = 0; j < n; ++j) {
                        LD ang = (forward ? -2 : 2) * pi * LD((j * k) % n) / LD(n);
                        acc += cur[(o * n + j) * inner + in] * CLD(std::cos(ang), std::sin(ang));
                    }
                    next[(o * n + k) * inner + in] = acc;
                }
        cur = std::move(next);
    }
    return cur;
}

template <class T> double bound(std::size_t points, const std::vector<T>& x, double scale) {
    double norm2 = 0;
    for (T v : x) norm2 += double(v) * double(v);
    return (8 + 4 * std::log2(double(points) + 1)) * double(std::numeric_limits<T>::epsilon()) * std::sqrt(norm2) * scale;
}

template <class T> double scaleOf(aether::fft::Norm norm, bool forward, std::size_t n) {
    using aether::fft::Norm;
    if (norm == Norm::Ortho) return 1.0 / std::sqrt(double(n));
    return forward == (norm == Norm::Forward) ? 1.0 / double(n) : 1.0;
}

template <class T> void checkC2c(const Idx& shape, const Idx& axes, bool forward, aether::fft::Norm norm, unsigned seed) {
    std::size_t total = prodOf(shape);
    auto in = randomVec<T>(2 * total, seed);
    std::vector<T> out(2 * total);
    aether::fft::c2c<T>(in.data(), out.data(), shape, axes, forward, norm);
    std::size_t points = 1;
    for (auto a : axes) points *= shape[a];
    double scale = scaleOf<T>(norm, forward, points);
    auto ref = naive<T>(in.data(), shape, axes, forward);
    double tol = bound(points, in, scale), worst = 0;
    for (std::size_t i = 0; i < total; ++i) {
        worst = std::max(worst, double(std::abs(CLD(out[2 * i], out[2 * i + 1]) - ref[i] * LD(scale))));
    }
    EXPECT_LE(worst, tol) << "points " << points << " forward " << forward;
}

template <class T> void c2cLengths() {
    for (std::size_t n : {1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 31, 64, 100, 127})
        for (bool forward : {true, false}) checkC2c<T>({n}, {0}, forward, aether::fft::Norm::Backward, unsigned(n));
}

template <class T> void multiAxis() {
    checkC2c<T>({6, 5}, {0, 1}, true, aether::fft::Norm::Backward, 11);
    checkC2c<T>({6, 5}, {1, 0}, false, aether::fft::Norm::Backward, 12);
    checkC2c<T>({4, 7, 3}, {1}, true, aether::fft::Norm::Backward, 13);       // a middle axis, batched on both sides
    checkC2c<T>({4, 7, 3}, {0, 2}, false, aether::fft::Norm::Backward, 14);
    checkC2c<T>({3, 4, 5}, {0, 1, 2}, true, aether::fft::Norm::Backward, 15);
    checkC2c<T>({9, 8}, {1}, true, aether::fft::Norm::Backward, 16);          // batch of 9 length-8 transforms
}

template <class T> void norms() {
    using aether::fft::Norm;
    for (Norm nm : {Norm::Backward, Norm::Forward, Norm::Ortho})
        for (bool forward : {true, false}) checkC2c<T>({12}, {0}, forward, nm, 21);
}

template <class T> void roundTrip() {
    Idx shape{5, 12};
    std::size_t total = prodOf(shape);
    auto x = randomVec<T>(2 * total, 31);
    std::vector<T> y(2 * total), z(2 * total);
    aether::fft::c2c<T>(x.data(), y.data(), shape, Idx{0, 1}, true);
    aether::fft::c2c<T>(y.data(), z.data(), shape, Idx{0, 1}, false);
    double worst = 0;
    for (std::size_t i = 0; i < 2 * total; ++i) worst = std::max(worst, std::abs(double(z[i]) - double(x[i])));
    EXPECT_LE(worst, bound(total, x, 1.0 / double(total)) * 2);
    // in place gives the same as out of place
    std::vector<T> inplace = x;
    aether::fft::c2c<T>(inplace.data(), inplace.data(), shape, Idx{0, 1}, true);
    EXPECT_EQ(inplace, y);
}

// real transforms: r2c of real data equals c2c of the same data with zero imaginary parts, on the kept half
template <class T> void checkR2c(const Idx& shape, const Idx& axes, unsigned seed) {
    std::size_t total = prodOf(shape);
    auto re = randomVec<T>(total, seed);
    Idx half = shape;
    half[axes.back()] = shape[axes.back()] / 2 + 1;
    std::vector<T> got(2 * prodOf(half));
    aether::fft::r2c<T>(re.data(), got.data(), shape, axes);
    std::vector<T> pairs(2 * total, T(0));
    for (std::size_t i = 0; i < total; ++i) pairs[2 * i] = re[i];
    auto ref = naive<T>(pairs.data(), shape, axes, true);
    std::size_t points = 1;
    for (auto a : axes) points *= shape[a];
    double tol = bound(points, pairs, 1.0), worst = 0;
    // map each half-array element to its full-array index
    Idx idx(shape.size(), 0);
    for (std::size_t h = 0; h < prodOf(half); ++h) {
        std::size_t rem = h, full = 0, stride = 1;
        for (std::size_t d = shape.size(); d-- > 0;) {
            std::size_t c = rem % half[d];
            rem /= half[d];
            full += c * stride;
            stride *= shape[d];
        }
        worst = std::max(worst, double(std::abs(CLD(got[2 * h], got[2 * h + 1]) - ref[full])));
    }
    EXPECT_LE(worst, tol);

    // c2r of that restores the data (backward norm: the 1/n is inside)
    std::vector<T> back(total);
    aether::fft::c2r<T>(got.data(), back.data(), shape, axes);
    double rt = 0;
    for (std::size_t i = 0; i < total; ++i) rt = std::max(rt, std::abs(double(back[i]) - double(re[i])));
    EXPECT_LE(rt, bound(points, pairs, 1.0 / double(points)) * 2);
}

template <class T> void realTransforms() {
    for (std::size_t n : {2, 3, 7, 8, 9, 16, 17, 50}) checkR2c<T>({n}, {0}, unsigned(n));
    checkR2c<T>({4, 6}, {0, 1}, 41);
    checkR2c<T>({5, 7}, {0, 1}, 42);
    checkR2c<T>({3, 8}, {1}, 43);        // batched along the first axis
    checkR2c<T>({4, 5, 6}, {2, 0}, 44);  // the last listed axis is the halved one
}

}  // namespace

TEST(Fft, C2cLengthsF32) { c2cLengths<float>(); }
TEST(Fft, C2cLengthsF64) { c2cLengths<double>(); }
TEST(Fft, MultiAxisF32) { multiAxis<float>(); }
TEST(Fft, MultiAxisF64) { multiAxis<double>(); }
TEST(Fft, NormsF32) { norms<float>(); }
TEST(Fft, NormsF64) { norms<double>(); }
TEST(Fft, RoundTripF32) { roundTrip<float>(); }
TEST(Fft, RoundTripF64) { roundTrip<double>(); }
TEST(Fft, RealTransformsF32) { realTransforms<float>(); }
TEST(Fft, RealTransformsF64) { realTransforms<double>(); }

TEST(Fft, ErrorsAreAetherErrors) {
    std::vector<double> a(8), b(8);
    EXPECT_THROW(aether::fft::c2c<double>(a.data(), b.data(), Idx{4}, Idx{}, true), aether::Error);
    EXPECT_THROW(aether::fft::c2c<double>(a.data(), b.data(), Idx{4}, Idx{1}, true), aether::Error);
    EXPECT_THROW(aether::fft::c2c<double>(a.data(), b.data(), Idx{2, 2}, Idx{0, 0}, true), aether::Error);
}

TEST(Fft, AdjointOfUnscaledForwardIsUnscaledInverse) {
    // <F x, y> == <x, F^H y> with F the unscaled forward: F^H is the inverse in the Forward norm
    using aether::fft::Norm;
    std::size_t n = 10;
    auto x = randomVec<double>(2 * n, 51), y = randomVec<double>(2 * n, 52);
    std::vector<double> fx(2 * n), fhy(2 * n);
    aether::fft::c2c<double>(x.data(), fx.data(), Idx{n}, Idx{0}, true, Norm::Backward);
    aether::fft::c2c<double>(y.data(), fhy.data(), Idx{n}, Idx{0}, false, Norm::Forward);
    CLD lhs = 0, rhs = 0;
    for (std::size_t i = 0; i < n; ++i) {
        lhs += CLD(fx[2 * i], fx[2 * i + 1]) * std::conj(CLD(y[2 * i], y[2 * i + 1]));
        rhs += CLD(x[2 * i], x[2 * i + 1]) * std::conj(CLD(fhy[2 * i], fhy[2 * i + 1]));
    }
    EXPECT_LT(double(std::abs(lhs - rhs)), 1e-12);
}

// The chunking depends on the shape only: one thread and four threads give the same bits, on every route.
template <class T> void threadInvariant() {
    auto run = [](int threads) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        std::vector<std::vector<T>> outs;
        for (Idx shape : {Idx{37, 100}, Idx{9, 20, 11}}) {
            std::size_t total = prodOf(shape);
            auto x = randomVec<T>(2 * total, 77);
            for (Idx axes : {Idx{shape.size() - 1}, Idx{0}, Idx{0, shape.size() - 1}}) {
                std::vector<T> y(2 * total);
                aether::fft::c2c<T>(x.data(), y.data(), shape, axes, true);
                outs.push_back(y);
                auto re = randomVec<T>(total, 78);
                Idx half = shape;
                half[axes.back()] = shape[axes.back()] / 2 + 1;
                std::vector<T> spec(2 * prodOf(half)), back(total);
                aether::fft::r2c<T>(re.data(), spec.data(), shape, axes);
                aether::fft::c2r<T>(spec.data(), back.data(), shape, axes);
                outs.push_back(spec);
                outs.push_back(back);
            }
        }
        return outs;
    };
    auto one = run(1), four = run(4);
    ASSERT_EQ(one.size(), four.size());
    for (std::size_t i = 0; i < one.size(); ++i)
        EXPECT_EQ(0, std::memcmp(one[i].data(), four[i].data(), one[i].size() * sizeof(T))) << "case " << i;
}

TEST(Fft, ThreadCountDoesNotChangeTheBitsF32) { threadInvariant<float>(); }
TEST(Fft, ThreadCountDoesNotChangeTheBitsF64) { threadInvariant<double>(); }
