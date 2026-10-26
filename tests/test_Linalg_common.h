// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::linalg (aether/linalg/): LU, Cholesky and Jacobi eigh on small static
// matrices. The host battery shared by test_Linalg.cpp and test_Linalg.cu:
// residual and invariant checks over random matrices at both precisions,
// LAPACK reference values (linalg/golden_linalg.h), the singular and
// not-positive-definite policy (info, zero fill), and the symmetrised read.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "linalg/golden_linalg.h"

namespace aether_tests {
namespace linalg {
namespace {

namespace al = aether::linalg;
namespace gold = aether_test_linalg;

using Mat = std::vector<double>;

/** @brief splitmix64: a small deterministic stream, uniform in [-1, 1). */
struct Stream {
    std::uint64_t s;
    explicit Stream(std::uint64_t seed) : s(seed) {}
    double next()
    {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * 0x1.0p-52 - 1.0;
    }
};

template<class T>
constexpr double eps()
{
    return static_cast<double>(al::detail::epsilon_v<T>);
}

template<class T>
std::vector<T> cast(const Mat& m)
{
    return std::vector<T>(m.begin(), m.end());
}

template<class T>
bool isPlusZero(T x)
{
    return x == T(0) && !std::signbit(x);
}

template<class T>
bool allPlusZero(const T* x, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (!isPlusZero(x[i]))
            return false;
    return true;
}

Mat randomMatrix(std::size_t n, Stream& rng)
{
    Mat a(n * n);
    for (double& v : a)
        v = rng.next();
    return a;
}

/** @brief B B^T + n I. */
Mat randomSpd(std::size_t n, Stream& rng)
{
    const Mat b = randomMatrix(n, rng);
    Mat s(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = i == j ? static_cast<double>(n) : 0.0;
            for (std::size_t k = 0; k < n; ++k)
                acc += b[i * n + k] * b[j * n + k];
            s[i * n + j] = acc;
        }
    return s;
}

Mat randomSymmetric(std::size_t n, Stream& rng)
{
    const Mat a = randomMatrix(n, rng);
    Mat s(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            s[i * n + j] = 0.5 * a[i * n + j] + 0.5 * a[j * n + i];
    return s;
}

/** @brief Orthogonal matrix: modified Gram-Schmidt on a random one, twice. */
Mat randomOrthogonal(std::size_t n, Stream& rng)
{
    Mat q = randomMatrix(n, rng);
    for (int pass = 0; pass < 2; ++pass)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < j; ++k) {
                double d = 0;
                for (std::size_t i = 0; i < n; ++i)
                    d += q[i * n + j] * q[i * n + k];
                for (std::size_t i = 0; i < n; ++i)
                    q[i * n + j] -= d * q[i * n + k];
            }
            double nrm = 0;
            for (std::size_t i = 0; i < n; ++i)
                nrm += q[i * n + j] * q[i * n + j];
            nrm = std::sqrt(nrm);
            for (std::size_t i = 0; i < n; ++i)
                q[i * n + j] /= nrm;
        }
    return q;
}

/** @brief C = A B, row-major, accumulated in double. */
Mat matmul(const Mat& a, const Mat& b, std::size_t n, std::size_t m, std::size_t k)
{
    Mat c(n * k, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            double acc = 0;
            for (std::size_t l = 0; l < m; ++l)
                acc += a[i * m + l] * b[l * k + j];
            c[i * k + j] = acc;
        }
    return c;
}

Mat transpose(const Mat& a, std::size_t n, std::size_t m)
{
    Mat t(a.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            t[j * n + i] = a[i * m + j];
    return t;
}

double maxAbs(const Mat& a)
{
    double m = 0;
    for (double v : a)
        m = std::max(m, std::fabs(v));
    return m;
}

double maxDiff(const Mat& a, const Mat& b)
{
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

template<class T>
Mat widen(const T* x, std::size_t n)
{
    return Mat(x, x + n);
}

/** @brief ||A x - b||_max <= 8 n eps ||A||_inf ||x||_inf, every right-hand side. */
template<class T>
void expectSolves(const Mat& a, const Mat& x, const Mat& b, std::size_t n, std::size_t k)
{
    const Mat r = matmul(a, x, n, n, k);
    const double tol = 8.0 * static_cast<double>(n) * eps<T>() * static_cast<double>(n) * maxAbs(a) * std::max(maxAbs(x), 1.0);
    EXPECT_LE(maxDiff(r, b), tol) << "n=" << n << " k=" << k;
}

template<std::size_t N, class T>
void checkLuReconstructs(std::uint64_t seed)
{
    Stream rng(seed);
    const Mat a = randomMatrix(N, rng);
    const std::vector<T> at = cast<T>(a);
    const Mat ar = widen(at.data(), N * N);
    const al::Lu<T, N> f = al::lu<N>(at.data());
    ASSERT_EQ(f.info, 0);
    Mat l(N * N, 0.0), u(N * N, 0.0), pa = ar;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            if (j < i)
                l[i * N + j] = static_cast<double>(f.a[i * N + j]);
            else
                u[i * N + j] = static_cast<double>(f.a[i * N + j]);
            if (i == j)
                l[i * N + j] = 1.0;
        }
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t p = static_cast<std::size_t>(f.piv[k]);
        ASSERT_GE(p, k);
        ASSERT_LT(p, N);
        for (std::size_t j = 0; j < N; ++j)
            std::swap(pa[k * N + j], pa[p * N + j]);
    }
    const Mat lu = matmul(l, u, N, N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < i; ++j)
            EXPECT_LE(std::fabs(l[i * N + j]), 1.0) << "partial pivoting bounds every multiplier";
    EXPECT_LE(maxDiff(lu, pa), 8.0 * static_cast<double>(N) * eps<T>() * std::max(maxAbs(lu), maxAbs(ar)) * static_cast<double>(N)) << "n=" << N;
}

template<std::size_t N, std::size_t K, class T>
void checkLuSolve(std::uint64_t seed)
{
    Stream rng(seed);
    const std::vector<T> a = cast<T>(randomMatrix(N, rng));
    std::vector<T> b(N * K), x(N * K), xt(N * K);
    for (T& v : b)
        v = static_cast<T>(rng.next());
    const al::Lu<T, N> f = al::lu<N>(a.data());
    ASSERT_EQ(f.info, 0);
    al::luSolve<K>(f, b.data(), x.data());
    expectSolves<T>(widen(a.data(), N * N), widen(x.data(), N * K), widen(b.data(), N * K), N, K);
    al::luSolveT<K>(f, b.data(), xt.data());
    expectSolves<T>(transpose(widen(a.data(), N * N), N, N), widen(xt.data(), N * K), widen(b.data(), N * K), N, K);
    std::vector<T> y = b;
    al::luSolve<K>(f, y.data(), y.data());
    EXPECT_EQ(std::memcmp(y.data(), x.data(), sizeof(T) * N * K), 0) << "in-place solve differs";
}

template<std::size_t N, class T>
void checkCholesky(std::uint64_t seed)
{
    Stream rng(seed);
    const std::vector<T> s = cast<T>(randomSpd(N, rng));
    const al::Cholesky<T, N> f = al::cholesky<N>(s.data());
    ASSERT_EQ(f.info, 0);
    const Mat l = widen(f.l, N * N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j)
            EXPECT_TRUE(isPlusZero(f.l[i * N + j])) << "upper part of the factor is zero";
    const Mat llt = matmul(l, transpose(l, N, N), N, N, N);
    const Mat sr = widen(s.data(), N * N);
    EXPECT_LE(maxDiff(llt, sr), 8.0 * static_cast<double>(N) * eps<T>() * maxAbs(sr)) << "n=" << N;
}

template<std::size_t N, std::size_t K, class T>
void checkChoSolve(std::uint64_t seed)
{
    Stream rng(seed);
    const std::vector<T> s = cast<T>(randomSpd(N, rng));
    std::vector<T> b(N * K), x(N * K);
    for (T& v : b)
        v = static_cast<T>(rng.next());
    const al::Cholesky<T, N> f = al::cholesky<N>(s.data());
    ASSERT_EQ(f.info, 0);
    al::choSolve<K>(f, b.data(), x.data());
    expectSolves<T>(widen(s.data(), N * N), widen(x.data(), N * K), widen(b.data(), N * K), N, K);
}

template<std::size_t N, class T>
void checkEigh(const Mat& s, const Mat& expectedW, bool allowClose)
{
    const std::vector<T> st = cast<T>(s);
    const Mat sr = widen(st.data(), N * N);
    const al::Eigh<T, N> r = al::eigh<N>(st.data());
    EXPECT_EQ(r.info & 1, 0) << "not converged, n=" << N;
    if (!allowClose) {
        EXPECT_EQ(r.info & 2, 0) << "n=" << N;
    }
    const Mat v = widen(r.v, N * N);
    double fro = 0;
    for (double e : sr)
        fro += e * e;
    fro = std::sqrt(fro);
    Mat vl = v;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            vl[i * N + j] *= static_cast<double>(r.w[j]);
    EXPECT_LE(maxDiff(matmul(sr, v, N, N, N), vl), 16.0 * static_cast<double>(N) * eps<T>() * fro) << "S V = V diag(w), n=" << N;
    Mat vtv = matmul(transpose(v, N, N), v, N, N, N);
    for (std::size_t i = 0; i < N; ++i)
        vtv[i * N + i] -= 1.0;
    EXPECT_LE(maxAbs(vtv), 16.0 * static_cast<double>(N) * eps<T>()) << "V^T V = I, n=" << N;
    for (std::size_t j = 0; j + 1 < N; ++j)
        EXPECT_LE(r.w[j], r.w[j + 1]) << "ascending, n=" << N;
    if (!expectedW.empty()) {
        std::vector<double> sorted = expectedW;
        std::sort(sorted.begin(), sorted.end());
        double scale = 0;
        for (double e : sorted)
            scale = std::max(scale, std::fabs(e));
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_NEAR(static_cast<double>(r.w[j]), sorted[j], 16.0 * static_cast<double>(N) * eps<T>() * scale) << "w[" << j << "], n=" << N;
    }
    for (std::size_t j = 0; j < N; ++j) {
        std::size_t big = 0;
        for (std::size_t k = 1; k < N; ++k)
            if (std::fabs(r.v[k * N + j]) > std::fabs(r.v[big * N + j]))
                big = k;
        EXPECT_GT(r.v[big * N + j], T(0)) << "column " << j << " sign, n=" << N;
    }
}

/** @brief Q diag(w) Q^T, symmetrised. */
Mat withSpectrum(const Mat& w, Stream& rng)
{
    const std::size_t n = w.size();
    const Mat q = randomOrthogonal(n, rng);
    Mat qw = q;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            qw[i * n + j] *= w[j];
    Mat a = matmul(qw, transpose(q, n, n), n, n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            a[i * n + j] = a[j * n + i] = 0.5 * a[i * n + j] + 0.5 * a[j * n + i];
    return a;
}

template<std::size_t N, class T>
void checkEighHardSpectra(std::uint64_t seed)
{
    Stream rng(seed);
    constexpr bool wide = sizeof(T) == sizeof(double);
    Mat random(N), clustered(N, 1.0), graded(N), integers(N), rank1(N, 0.0), huge(N);
    for (std::size_t i = 0; i < N; ++i) {
        random[i] = rng.next();
        integers[i] = std::round(2.0 * rng.next());
        graded[i] = N > 1 ? std::pow(10.0, -12.0 * static_cast<double>(i) / static_cast<double>(N - 1)) : 1.0;
        huge[i] = rng.next() * (wide ? 1e150 : 1e18);
    }
    clustered[0] = 1.0 + (wide ? 1e-12 : 1e-5);
    rank1[0] = 1.0;
    for (const Mat* w : { &random, &clustered, &graded, &integers, &rank1, &huge })
        checkEigh<N, T>(withSpectrum(*w, rng), *w, true);
}

} // namespace

// ---------------------------------------------------------------- LU

TEST(LinalgLu, ReconstructsPermutedMatrixF64)
{
    checkLuReconstructs<1, double>(1);
    checkLuReconstructs<2, double>(2);
    checkLuReconstructs<3, double>(3);
    checkLuReconstructs<5, double>(4);
    checkLuReconstructs<8, double>(5);
    checkLuReconstructs<16, double>(6);
    checkLuReconstructs<32, double>(7);
}

TEST(LinalgLu, ReconstructsPermutedMatrixF32)
{
    checkLuReconstructs<2, float>(11);
    checkLuReconstructs<3, float>(12);
    checkLuReconstructs<8, float>(13);
    checkLuReconstructs<32, float>(14);
}

TEST(LinalgLu, SolveResidualVectorAndMatrixRhs)
{
    checkLuSolve<1, 1, double>(21);
    checkLuSolve<3, 1, double>(22);
    checkLuSolve<4, 3, double>(23);
    checkLuSolve<9, 9, double>(24);
    checkLuSolve<32, 2, double>(25);
    checkLuSolve<3, 1, float>(26);
    checkLuSolve<8, 4, float>(27);
    checkLuSolve<32, 1, float>(28);
}

TEST(LinalgLu, MatchesLapackGetrf)
{
    const al::Lu<double, 4> f = al::lu<4>(gold::kA4);
    EXPECT_EQ(f.info, 0);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(f.piv[i], gold::kPiv4[i]) << "pivot " << i;
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(f.a[i], gold::kLu4[i], 64 * eps<double>() * 8.0) << "packed LU entry " << i;
}

TEST(LinalgLu, SolveDetAndInverseMatchReference)
{
    double x[4], inv[16];
    EXPECT_EQ(al::solve<4>(gold::kA4, gold::kB4, x), 0);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(x[i], gold::kX4[i], 64 * eps<double>()) << i;
    EXPECT_NEAR(al::det<4>(gold::kA4), gold::kDet4, 64 * eps<double>() * gold::kDet4);
    EXPECT_EQ(al::inv<4>(gold::kA4, inv), 0);
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(inv[i], gold::kInv4[i], 64 * eps<double>()) << i;
}

TEST(LinalgLu, DeterminantSignFollowsRowExchanges)
{
    const double swap2[4] = { 0, 1, 1, 0 };
    EXPECT_EQ(al::det<2>(swap2), -1.0);
    const double cyc3[9] = { 0, 1, 0, 0, 0, 1, 1, 0, 0 };
    EXPECT_EQ(al::det<3>(cyc3), 1.0);
    const double scale[1] = { -4.0 };
    EXPECT_EQ(al::det<1>(scale), -4.0);
}

TEST(LinalgLu, PivotTiesGoToTheLowestRow)
{
    const double a[4] = { 2, 1, -2, 1 };
    const al::Lu<double, 2> f = al::lu<2>(a);
    EXPECT_EQ(f.piv[0], 0);
    EXPECT_EQ(f.info, 0);
    const double b[4] = { 1, 1, 3, 1 };
    EXPECT_EQ(al::lu<2>(b).piv[0], 1) << "the larger magnitude wins";
}

namespace {

template<std::size_t N>
void expectSingularPolicy(const double* a, int lapackInfo)
{
    const al::Lu<double, N> f = al::lu<N>(a);
    EXPECT_EQ(f.info, lapackInfo);
    double b[N], x[N], inv[N * N];
    for (std::size_t i = 0; i < N; ++i) {
        b[i] = static_cast<double>(i + 1);
        x[i] = -7.0;
    }
    al::luSolve(f, b, x);
    EXPECT_TRUE(allPlusZero(x, N)) << "solve is zero-filled";
    for (std::size_t i = 0; i < N; ++i)
        x[i] = -7.0;
    al::luSolveT(f, b, x);
    EXPECT_TRUE(allPlusZero(x, N)) << "transposed solve is zero-filled";
    EXPECT_TRUE(isPlusZero(al::luDet(f))) << "det of a singular matrix is 0";
    EXPECT_EQ(al::inv<N>(a, inv), lapackInfo);
    EXPECT_TRUE(allPlusZero(inv, N * N)) << "inverse is zero-filled";
    EXPECT_EQ(al::solve<N>(a, b, x), lapackInfo);
    EXPECT_TRUE(allPlusZero(x, N));
}

} // namespace

TEST(LinalgLu, SingularInfoMatchesLapackAndEverythingIsZero)
{
    expectSingularPolicy<2>(gold::kSingA, gold::kSingAInfo);
    expectSingularPolicy<2>(gold::kSingB, gold::kSingBInfo);
    expectSingularPolicy<3>(gold::kSingC, gold::kSingCInfo);
    expectSingularPolicy<3>(gold::kSingD, gold::kSingDInfo);
    const double zero1[1] = { 0.0 };
    expectSingularPolicy<1>(zero1, 1);
}

TEST(LinalgLu, FailedFactorKeepsThePartialFactorisation)
{
    const al::Lu<double, 3> f = al::lu<3>(gold::kSingD);
    EXPECT_EQ(f.info, 2);
    EXPECT_EQ(f.a[0], 1.0);
    EXPECT_EQ(f.a[8], 1.0) << "elimination carries on past the zero pivot";
}

TEST(LinalgLu, SizeOneIsAScalarDivision)
{
    const double a[1] = { 4.0 }, b[1] = { 2.0 };
    double x[1];
    EXPECT_EQ(al::solve<1>(a, b, x), 0);
    EXPECT_EQ(x[0], 0.5);
    EXPECT_EQ(al::lu<1>(a).piv[0], 0);
}

// ---------------------------------------------------------------- Cholesky

TEST(LinalgCholesky, ReconstructsTheMatrix)
{
    checkCholesky<1, double>(31);
    checkCholesky<2, double>(32);
    checkCholesky<3, double>(33);
    checkCholesky<8, double>(34);
    checkCholesky<32, double>(35);
    checkCholesky<3, float>(36);
    checkCholesky<16, float>(37);
    checkCholesky<32, float>(38);
}

TEST(LinalgCholesky, MatchesLapackPotrf)
{
    const al::Cholesky<double, 3> f = al::cholesky<3>(gold::kS3);
    EXPECT_EQ(f.info, 0);
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_NEAR(f.l[i], gold::kChol3[i], 16 * eps<double>() * 8.0) << i;
    double l[9];
    al::choleskyFactor(f, l);
    EXPECT_EQ(std::memcmp(l, f.l, sizeof l), 0);
}

TEST(LinalgCholesky, ChoSolveResidualVectorAndMatrixRhs)
{
    checkChoSolve<1, 1, double>(41);
    checkChoSolve<4, 1, double>(42);
    checkChoSolve<5, 5, double>(43);
    checkChoSolve<32, 3, double>(44);
    checkChoSolve<6, 2, float>(45);
    checkChoSolve<32, 1, float>(46);
    double x[3];
    al::choSolve(al::cholesky<3>(gold::kS3), gold::kB3, x);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x[i], gold::kChoX3[i], 1e-12) << i;
}

TEST(LinalgCholesky, ReadsTheSymmetricPartOfTheInput)
{
    Stream rng(51);
    constexpr std::size_t N = 6;
    Mat a = randomSpd(N, rng);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < i; ++j)
            a[i * N + j] += 0.37 * rng.next();
    Mat sym(N * N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            sym[i * N + j] = 0.5 * a[i * N + j] + 0.5 * a[j * N + i];
    const al::Cholesky<double, N> f = al::cholesky<N>(a.data());
    const al::Cholesky<double, N> g = al::cholesky<N>(sym.data());
    EXPECT_EQ(f.info, 0);
    EXPECT_EQ(std::memcmp(f.l, g.l, sizeof f.l), 0) << "cholesky(A) is bitwise cholesky(sym(A))";
}

namespace {

template<std::size_t N>
void expectNotPositiveDefinite(const double* a, int lapackInfo)
{
    const al::Cholesky<double, N> f = al::cholesky<N>(a);
    EXPECT_EQ(f.info, lapackInfo);
    double l[N * N], b[N], x[N];
    al::choleskyFactor(f, l);
    EXPECT_TRUE(allPlusZero(l, N * N)) << "the projected factor is zero-filled";
    for (std::size_t i = 0; i < N; ++i) {
        b[i] = 1.0;
        x[i] = -7.0;
    }
    al::choSolve(f, b, x);
    EXPECT_TRUE(allPlusZero(x, N)) << "solve is zero-filled";
    for (std::size_t j = static_cast<std::size_t>(lapackInfo - 1); j < N; ++j)
        for (std::size_t i = j; i < N; ++i)
            EXPECT_TRUE(isPlusZero(f.l[i * N + j])) << "columns from the failed one on stay zero";
}

} // namespace

TEST(LinalgCholesky, NotPositiveDefiniteInfoMatchesLapack)
{
    expectNotPositiveDefinite<2>(gold::kNpdA, gold::kNpdAInfo);
    expectNotPositiveDefinite<2>(gold::kNpdB, gold::kNpdBInfo);
    expectNotPositiveDefinite<3>(gold::kNpdC, gold::kNpdCInfo);
    expectNotPositiveDefinite<2>(gold::kNpdD, gold::kNpdDInfo);
    const double nan1[1] = { std::numeric_limits<double>::quiet_NaN() };
    expectNotPositiveDefinite<1>(nan1, 1);
}

TEST(LinalgCholesky, FailedFactorKeepsTheCompletedColumns)
{
    const al::Cholesky<double, 3> f = al::cholesky<3>(gold::kNpdC);
    EXPECT_EQ(f.info, 2);
    EXPECT_EQ(f.l[0], 2.0);
    EXPECT_EQ(f.l[3], 1.0);
}

TEST(LinalgCholesky, SizeOneIsASquareRoot)
{
    const double a[1] = { 9.0 };
    const al::Cholesky<double, 1> f = al::cholesky<1>(a);
    EXPECT_EQ(f.info, 0);
    EXPECT_EQ(f.l[0], 3.0);
    double x[1];
    const double b[1] = { 18.0 };
    al::choSolve(f, b, x);
    EXPECT_EQ(x[0], 2.0);
}

TEST(LinalgCholesky, FromAFactorIsTheFactorItWasMadeOf)
{
    Stream rng(41);
    constexpr std::size_t N = 7;
    const Mat a = randomSpd(N, rng);
    const al::Cholesky<double, N> f = al::cholesky<N>(a.data());
    double l[N * N];
    al::choleskyFactor(f, l);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j)
            l[i * N + j] = std::numeric_limits<double>::quiet_NaN();    // above the diagonal: never read
    al::Cholesky<double, N> g;
    al::choleskyFrom(g, l);
    EXPECT_EQ(g.info, 0);
    EXPECT_EQ(std::memcmp(f.l, g.l, sizeof f.l), 0);
    double b0[N], x[N], y[N];
    for (std::size_t i = 0; i < N; ++i)
        b0[i] = 1.0 + static_cast<double>(i);
    al::choSolve(f, b0, x);
    al::choSolve(g, b0, y);
    EXPECT_EQ(std::memcmp(x, y, sizeof x), 0);
}

TEST(LinalgCholesky, FromAnUnusableDiagonalFailsAtItsFirstBadColumn)
{
    constexpr std::size_t N = 4;
    const double bad[] = { 0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity() };
    for (const double d : bad) {
        double l[N * N] = { 2.0, 0.0, 0.0, 0.0,  0.5, 3.0, 0.0, 0.0,  0.25, 0.5, 1.0, 0.0,  0.1, 0.2, 0.3, 4.0 };
        l[2 * N + 2] = d;
        l[3 * N + 3] = -1.0;                   // a later bad column does not change the first
        al::Cholesky<double, N> f;
        al::choleskyFrom(f, l);
        EXPECT_EQ(f.info, 3);
        EXPECT_TRUE(allPlusZero(f.l, N * N)) << "a failed factor is all zeros";
        const double b0[N] = { 1.0, 1.0, 1.0, 1.0 };
        double x[N] = { -7.0, -7.0, -7.0, -7.0 };
        al::choSolve(f, b0, x);
        EXPECT_TRUE(allPlusZero(x, N));
    }
}

// ---------------------------------------------------------------- triangular

TEST(LinalgTriangular, SolvesAgainstTheFactorAndItsTranspose)
{
    constexpr std::size_t N = 7, K = 4;
    Stream rng(61);
    const Mat spd = randomSpd(N, rng);
    const al::Cholesky<double, N> f = al::cholesky<N>(spd.data());
    Mat b(N * K);
    for (double& v : b)
        v = rng.next();
    Mat x(N * K);
    al::triLower<N, K>(f.l, b.data(), x.data());
    const Mat l = widen(f.l, N * N);
    EXPECT_LE(maxDiff(matmul(l, x, N, N, K), b), 1e-13);
    al::triUpperT<N, K>(f.l, b.data(), x.data());
    EXPECT_LE(maxDiff(matmul(transpose(l, N, N), x, N, N, K), b), 1e-13);
}

TEST(LinalgTriangular, InPlaceEqualsOutOfPlaceAndIgnoresTheUpperPart)
{
    constexpr std::size_t N = 5, K = 5;
    Stream rng(62);
    Mat l = randomSpd(N, rng);
    Mat clean = l;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) {
            l[i * N + j] = 1e9;
            clean[i * N + j] = 0.0;
        }
    Mat b(N * K);
    for (double& v : b)
        v = rng.next();
    Mat out(N * K), same = b;
    al::triLower<N, K>(l.data(), b.data(), out.data());
    al::triLower<N, K>(l.data(), same.data(), same.data());
    EXPECT_EQ(std::memcmp(out.data(), same.data(), sizeof(double) * N * K), 0);
    EXPECT_LE(maxDiff(matmul(clean, out, N, N, K), b), 1e-12);
    al::triUpperT<N, K>(l.data(), b.data(), out.data());
    same = b;
    al::triUpperT<N, K>(l.data(), same.data(), same.data());
    EXPECT_EQ(std::memcmp(out.data(), same.data(), sizeof(double) * N * K), 0);
    EXPECT_LE(maxDiff(matmul(transpose(clean, N, N), out, N, N, K), b), 1e-12);
}

// ---------------------------------------------------------------- eigh

TEST(LinalgEigh, SweepTableIsFixed)
{
    static_assert(al::eighSweeps(1) == 2 && al::eighSweeps(2) == 2);
    static_assert(al::eighSweeps(3) == 6 && al::eighSweeps(4) == 6);
    static_assert(al::eighSweeps(5) == 8 && al::eighSweeps(8) == 8);
    static_assert(al::eighSweeps(9) == 12 && al::eighSweeps(16) == 12);
    static_assert(al::eighSweeps(17) == 15 && al::eighSweeps(32) == 15);
    SUCCEED();
}

TEST(LinalgEigh, InvariantsOnRandomSymmetricMatrices)
{
    Stream rng(71);
    checkEigh<2, double>(randomSymmetric(2, rng), {}, false);
    checkEigh<3, double>(randomSymmetric(3, rng), {}, false);
    checkEigh<4, double>(randomSymmetric(4, rng), {}, false);
    checkEigh<8, double>(randomSymmetric(8, rng), {}, false);
    checkEigh<16, double>(randomSymmetric(16, rng), {}, false);
    checkEigh<32, double>(randomSymmetric(32, rng), {}, false);
    checkEigh<2, float>(randomSymmetric(2, rng), {}, false);
    checkEigh<5, float>(randomSymmetric(5, rng), {}, false);
    checkEigh<16, float>(randomSymmetric(16, rng), {}, false);
    checkEigh<32, float>(randomSymmetric(32, rng), {}, false);
}

TEST(LinalgEigh, MatchesLapackEigh)
{
    const al::Eigh<double, 4> r4 = al::eigh<4>(gold::kE4);
    EXPECT_EQ(r4.info, 0);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(r4.w[i], gold::kE4W[i], 64 * eps<double>()) << i;
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(r4.v[i], gold::kE4V[i], 64 * eps<double>()) << i;
    const al::Eigh<double, 3> r3 = al::eigh<3>(gold::kE3);
    EXPECT_EQ(r3.info, 0);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(r3.w[i], gold::kE3W[i], 64 * eps<double>()) << i;
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_NEAR(r3.v[i], gold::kE3V[i], 64 * eps<double>()) << i;
}

TEST(LinalgEigh, HardSpectraConvergeWithinTheFixedSweeps)
{
    checkEighHardSpectra<2, double>(81);
    checkEighHardSpectra<3, double>(82);
    checkEighHardSpectra<4, double>(83);
    checkEighHardSpectra<8, double>(84);
    checkEighHardSpectra<16, double>(85);
    checkEighHardSpectra<32, double>(86);
    checkEighHardSpectra<2, float>(87);
    checkEighHardSpectra<3, float>(88);
    checkEighHardSpectra<4, float>(89);
    checkEighHardSpectra<8, float>(90);
    checkEighHardSpectra<16, float>(91);
    checkEighHardSpectra<32, float>(92);
}

TEST(LinalgEigh, CloseEigenvaluesSetInfoBitTwo)
{
    const double eye[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const al::Eigh<double, 3> r = al::eigh<3>(eye);
    EXPECT_EQ(r.info, 2);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(r.w[i], 1.0);
    const double apart[4] = { 1, 0, 0, 3 };
    EXPECT_EQ(al::eigh<2>(apart).info, 0);
}

TEST(LinalgEigh, NaNInputSetsInfoBitOne)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double a[9] = { 2, 1, 0, 1, nan, 1, 0, 1, 2 };
    EXPECT_NE(al::eigh<3>(a).info & 1, 0);
}

TEST(LinalgEigh, ReadsTheSymmetricPartOfTheInput)
{
    Stream rng(93);
    constexpr std::size_t N = 6;
    const Mat a = randomMatrix(N, rng);
    Mat sym(N * N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            sym[i * N + j] = 0.5 * a[i * N + j] + 0.5 * a[j * N + i];
    const al::Eigh<double, N> f = al::eigh<N>(a.data());
    const al::Eigh<double, N> g = al::eigh<N>(sym.data());
    EXPECT_EQ(f.info, g.info);
    EXPECT_EQ(std::memcmp(f.w, g.w, sizeof f.w), 0);
    EXPECT_EQ(std::memcmp(f.v, g.v, sizeof f.v), 0) << "eigh(A) is bitwise eigh(sym(A))";
}

TEST(LinalgEigh, SizeOneIsTheScalar)
{
    const double a[1] = { -3.5 };
    const al::Eigh<double, 1> r = al::eigh<1>(a);
    EXPECT_EQ(r.w[0], -3.5);
    EXPECT_EQ(r.v[0], 1.0);
    EXPECT_EQ(r.info, 0);
}

} // namespace linalg
} // namespace aether_tests
