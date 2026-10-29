// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::linalg::team (aether/linalg/team/): one matrix worked on by a team of
// threads. The host battery shared by test_LinalgTeam.cpp and test_LinalgTeam.cu:
// the team at size 1 and at host-thread teams of several sizes (a std::barrier
// is the team barrier), bitwise across sizes, a ULP contract against the
// one-thread family at N <= 32, and a long-double reference at N = 33..512.
#pragma once

#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace linalg_team {
namespace {

namespace al = aether::linalg;
namespace tl = aether::linalg::team;

using Real = long double;

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

/** @brief Run `fn(Team)` on a team of `size` host threads (the size-1 twin runs inline). */
template<class F>
void runTeam(int size, F&& fn)
{
    if (size == 1) {
        fn(tl::Team{});
        return;
    }
    std::barrier<> bar(size);
    const auto trampoline = +[](void* c) { static_cast<std::barrier<>*>(c)->arrive_and_wait(); };
    std::vector<std::thread> threads;
    for (int r = 0; r < size; ++r) {
        threads.emplace_back([&, r] {
            tl::Team t;
            t.rank = r;
            t.size = size;
            t.barrier = trampoline;
            t.ctx = &bar;
            fn(t);
        });
    }
    for (auto& t : threads)
        t.join();
}

/** @brief Well-conditioned general matrix: uniform entries plus a diagonal of `2 sqrt(N)`. */
template<class T>
std::vector<T> generalMatrix(std::size_t n, std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<T> a(n * n);
    for (T& v : a)
        v = static_cast<T>(rng.next());
    for (std::size_t i = 0; i < n; ++i)
        a[i * n + i] += static_cast<T>(2.0 * std::sqrt(static_cast<double>(n)));
    return a;
}

/** @brief Symmetric positive definite: `B B^T / n + I`, formed in double and rounded. */
template<class T>
std::vector<T> spdMatrix(std::size_t n, std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<double> b(n * n);
    for (double& v : b)
        v = rng.next();
    std::vector<T> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                s += b[i * n + k] * b[j * n + k];
            a[i * n + j] = a[j * n + i] = static_cast<T>(s / static_cast<double>(n) + (i == j ? 1.0 : 0.0));
        }
    return a;
}

template<class T>
std::vector<T> randomVector(std::size_t n, std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<T> v(n);
    for (T& x : v)
        x = static_cast<T>(rng.next());
    return v;
}

/** @brief Solve `A x = b` in long double (Gaussian elimination, partial pivoting): the reference. */
template<class T>
std::vector<Real> referenceSolve(const std::vector<T>& a, const std::vector<T>& b, std::size_t n)
{
    std::vector<Real> m(a.begin(), a.end()), x(b.begin(), b.end());
    for (std::size_t j = 0; j < n; ++j) {
        std::size_t p = j;
        for (std::size_t i = j + 1; i < n; ++i)
            if (std::fabs(m[i * n + j]) > std::fabs(m[p * n + j]))
                p = i;
        if (p != j) {
            for (std::size_t c = 0; c < n; ++c)
                std::swap(m[j * n + c], m[p * n + c]);
            std::swap(x[j], x[p]);
        }
        for (std::size_t i = j + 1; i < n; ++i) {
            const Real f = m[i * n + j] / m[j * n + j];
            for (std::size_t c = j; c < n; ++c)
                m[i * n + c] -= f * m[j * n + c];
            x[i] -= f * x[j];
        }
    }
    for (std::size_t r = n; r-- > 0;) {
        for (std::size_t c = r + 1; c < n; ++c)
            x[r] -= m[r * n + c] * x[c];
        x[r] /= m[r * n + r];
    }
    return x;
}

/** @brief `max |x - ref| / max |ref|`. */
template<class T>
double forwardError(const std::vector<T>& x, const std::vector<Real>& ref)
{
    Real d = 0, s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        d = std::max(d, std::fabs(static_cast<Real>(x[i]) - ref[i]));
        s = std::max(s, std::fabs(ref[i]));
    }
    return static_cast<double>(d / s);
}

/** @brief `max |A x - b| / (|A| |x| + |b|)` in long double (infinity norms). */
template<class T>
double backwardResidual(const std::vector<T>& a, const std::vector<T>& x, const std::vector<T>& b, std::size_t n)
{
    Real worst = 0, na = 0, nx = 0, nb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        Real row = 0, acc = 0;
        for (std::size_t j = 0; j < n; ++j) {
            acc += static_cast<Real>(a[i * n + j]) * static_cast<Real>(x[j]);
            row += std::fabs(static_cast<Real>(a[i * n + j]));
        }
        na = std::max(na, row);
        nx = std::max(nx, std::fabs(static_cast<Real>(x[i])));
        nb = std::max(nb, std::fabs(static_cast<Real>(b[i])));
        worst = std::max(worst, std::fabs(acc - static_cast<Real>(b[i])));
    }
    return static_cast<double>(worst / (na * nx + nb));
}

template<class T>
bool bitwiseEqual(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

/** @brief `max |a - b|`. */
template<class T>
double maxAbsDiff(const std::vector<T>& a, const std::vector<T>& b)
{
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return d;
}

/** @brief `max |a|`, at least `1e-300` so a zero reference still has a scale. */
template<class T>
double maxAbs(const std::vector<T>& a)
{
    double d = 1e-300;
    for (const T& x : a)
        d = std::max(d, std::fabs(static_cast<double>(x)));
    return d;
}

/** @brief Two results of one operation by different call shapes agree to `tol` (relative to the largest entry). */
template<class T>
bool ulpClose(const std::vector<T>& a, const std::vector<T>& b, double tol)
{
    return a.size() == b.size() && maxAbsDiff(a, b) <= tol * maxAbs(b);
}

template<class T>
std::vector<T> absOf(std::vector<T> v)
{
    for (T& x : v)
        x = std::fabs(x);
    return v;
}

/** @brief The scratch and results of one team LU + solve. */
template<class T>
struct LuRun {
    std::vector<T> factor, x, inverse;
    std::vector<std::int32_t> piv;
    std::int32_t info = -1;
    T det = T(0);
    std::vector<T> xt;
};

template<std::size_t N, class T>
LuRun<T> luTeam(const std::vector<T>& a, const std::vector<T>& b, std::size_t k, int size)
{
    LuRun<T> r;
    r.factor.assign(N * N, T(0));
    r.piv.assign(N, 0);
    r.x.assign(N * k, T(0));
    r.xt.assign(N * k, T(0));
    r.inverse.assign(N * N, T(0));
    std::int32_t info = -1;
    const tl::TeamLu<T, N> f{r.factor.data(), r.piv.data(), &info};
    runTeam(size, [&](const tl::Team& t) {
        tl::lu<N>(t, f, a.data());
        tl::luSolve<N>(t, f, b.data(), r.x.data(), k);
        tl::luSolveT<N>(t, f, b.data(), r.xt.data(), k);
        tl::luInverse<N>(t, f, r.inverse.data());
    });
    r.info = info;
    r.det = tl::luDet<N>(f);
    return r;
}

template<class T>
struct CholRun {
    std::vector<T> l, x, lowerSolve, upperSolve;
    std::int32_t info = -1;
};

template<std::size_t N, class T>
CholRun<T> cholTeam(const std::vector<T>& a, const std::vector<T>& b, int size, bool inPlace = false)
{
    CholRun<T> r;
    r.l.assign(N * N, T(0));
    r.x.assign(N, T(0));
    r.lowerSolve.assign(N, T(0));
    r.upperSolve.assign(N, T(0));
    if (inPlace)
        r.l = a;
    std::int32_t info = -1;
    const tl::TeamCholesky<T, N> f{r.l.data(), &info};
    runTeam(size, [&](const tl::Team& t) {
        tl::cholesky<N>(t, f, inPlace ? r.l.data() : a.data());
        tl::choSolve<N>(t, f, b.data(), r.x.data());
        tl::triLower<N>(t, f, b.data(), r.lowerSolve.data());
        tl::triUpperT<N>(t, f, b.data(), r.upperSolve.data());
    });
    r.info = info;
    return r;
}

template<class T>
struct EighRun {
    std::vector<T> w, v;
    std::int32_t info = -1;
};

template<std::size_t N, class T>
EighRun<T> eighTeam(const std::vector<T>& a, int size)
{
    EighRun<T> r;
    r.w.assign(N, T(0));
    r.v.assign(N * N, T(0));
    std::vector<T> work(tl::eighWorkSize(N), T(0));
    std::int32_t info = -1;
    const tl::TeamEigh<T, N> f{r.w.data(), r.v.data(), work.data(), &info};
    runTeam(size, [&](const tl::Team& t) { tl::eigh<N>(t, f, a.data()); });
    r.info = info;
    return r;
}

template<class T>
struct QrRun {
    std::vector<T> x, factor, tau;
    std::int32_t info = -1;
};

template<std::size_t N, class T>
QrRun<T> qrTeam(const std::vector<T>& a, std::vector<T> b, std::size_t m, std::size_t k, int size)
{
    QrRun<T> r;
    r.factor.assign(m * N, T(0));
    r.tau.assign(N, T(0));
    r.x.assign(N * k, T(0));
    std::vector<T> work(tl::qrWorkSize(m, std::max(N, k)), T(0));
    std::int32_t info = -1;
    const tl::TeamQr<T, N> f{r.factor.data(), r.tau.data(), work.data(), &info};
    runTeam(size, [&](const tl::Team& t) {
        tl::qr<N>(t, f, a.data(), m);
        tl::qrSolve<N>(t, f, m, b.data(), r.x.data(), k);
    });
    r.info = info;
    return r;
}

/** @brief Invoke `fn(integral_constant<N>)` for each size in the list. */
template<std::size_t... Ns, class F>
void forSizes(F&& fn)
{
    (fn(std::integral_constant<std::size_t, Ns>{}), ...);
}

constexpr std::size_t kK = 3;  // right-hand sides

// ---------------------------------------------------------------- LU

template<class T, std::size_t N>
void checkLuAgainstReference()
{
    const std::vector<T> a = generalMatrix<T>(N, 11 + N), b = randomVector<T>(N * kK, 12 + N);
    const LuRun<T> r = luTeam<N>(a, b, kK, 1);
    ASSERT_EQ(r.info, 0) << "N=" << N;
    const double tol = 50.0 * static_cast<double>(N) * eps<T>();
    for (std::size_t c = 0; c < kK; ++c) {
        std::vector<T> bc(N), xc(N), xtc(N), at(N * N);
        for (std::size_t i = 0; i < N; ++i) {
            bc[i] = b[i * kK + c];
            xc[i] = r.x[i * kK + c];
            xtc[i] = r.xt[i * kK + c];
        }
        EXPECT_LE(forwardError(xc, referenceSolve(a, bc, N)), tol) << "solve N=" << N << " col " << c;
        EXPECT_LE(backwardResidual(a, xc, bc, N), tol) << "solve N=" << N << " col " << c;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                at[i * N + j] = a[j * N + i];
        EXPECT_LE(forwardError(xtc, referenceSolve(at, bc, N)), tol) << "solveT N=" << N << " col " << c;
    }
    // inverse: A * inv(A) = I, and the determinant against the long-double product of the reference pivots
    Real worst = 0;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            Real s = 0;
            for (std::size_t k = 0; k < N; ++k)
                s += static_cast<Real>(a[i * N + k]) * static_cast<Real>(r.inverse[k * N + j]);
            worst = std::max(worst, std::fabs(s - (i == j ? Real(1) : Real(0))));
        }
    EXPECT_LE(static_cast<double>(worst), tol) << "inverse N=" << N;
    if (std::isfinite(r.det) && r.det != T(0)) {  // large N overflows the determinant itself
        Real logdet = 0;
        for (std::size_t k = 0; k < N; ++k)
            logdet += std::log(std::fabs(static_cast<Real>(r.factor[k * N + k])));
        EXPECT_NEAR(static_cast<double>(std::log(std::fabs(static_cast<Real>(r.det)))), static_cast<double>(logdet), 1e-3) << "det N=" << N;
    }
}

TEST(LinalgTeamLu, MatchesALongDoubleReferenceF64)
{
    forSizes<33, 64, 71, 128, 512>([](auto n) { checkLuAgainstReference<double, decltype(n)::value>(); });
}

TEST(LinalgTeamLu, MatchesALongDoubleReferenceF32)
{
    forSizes<33, 64, 71, 128, 512>([](auto n) { checkLuAgainstReference<float, decltype(n)::value>(); });
}

template<class T>
void checkLuVsOneThread()
{
    forSizes<4, 16, 32>([](auto n) {
        constexpr std::size_t N = decltype(n)::value;
        const std::vector<T> a = generalMatrix<T>(N, 21 + N), b = randomVector<T>(N * kK, 22 + N);
        const LuRun<T> r = luTeam<N>(a, b, kK, 1);
        const al::Lu<T, N> f = al::lu<N>(a.data());
        EXPECT_EQ(r.info, f.info);
        std::vector<T> x1(N * kK);
        al::luSolve<kK>(f, b.data(), x1.data());
        const double tol = 50.0 * static_cast<double>(N) * eps<T>();
        for (std::size_t c = 0; c < kK; ++c) {
            std::vector<T> bc(N), xt(N), xo(N);
            for (std::size_t i = 0; i < N; ++i) {
                bc[i] = b[i * kK + c];
                xt[i] = r.x[i * kK + c];
                xo[i] = x1[i * kK + c];
            }
            EXPECT_LE(backwardResidual(a, xt, bc, N), tol) << "team N=" << N;
            EXPECT_LE(backwardResidual(a, xo, bc, N), tol) << "one-thread N=" << N;
            EXPECT_LE(forwardError(xt, std::vector<Real>(xo.begin(), xo.end())), tol) << "team vs one-thread N=" << N;
        }
        EXPECT_NEAR(static_cast<double>(r.det), static_cast<double>(al::luDet(f)),
            tol * std::max(1.0, std::fabs(static_cast<double>(al::luDet(f)))));
    });
}

TEST(LinalgTeamLu, UlpContractAgainstTheOneThreadFormF64)
{
    checkLuVsOneThread<double>();
}

TEST(LinalgTeamLu, UlpContractAgainstTheOneThreadFormF32)
{
    checkLuVsOneThread<float>();
}

TEST(LinalgTeamLu, AgreesAcrossTeamSizes)
{
    constexpr std::size_t N = 71;
    const std::vector<double> a = generalMatrix<double>(N, 31), b = randomVector<double>(N * kK, 32);
    const double tol = 50.0 * N * eps<double>();
    // Host teams of 32..256 threads replay one compiled path: bitwise, the determinism of that path.
    const LuRun<double> ref = luTeam<N>(a, b, kK, 32);
    for (int size : {64, 128, 256}) {
        const LuRun<double> r = luTeam<N>(a, b, kK, size);
        EXPECT_TRUE(bitwiseEqual(r.factor, ref.factor)) << "factor, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.piv, ref.piv)) << "pivots, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.x, ref.x)) << "solve, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.xt, ref.xt)) << "solveT, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.inverse, ref.inverse)) << "inverse, team " << size;
        EXPECT_EQ(r.info, ref.info);
    }
    // The size-1 twin is another call site: the ULP contract.
    const LuRun<double> one = luTeam<N>(a, b, kK, 1);
    EXPECT_TRUE(ulpClose(one.factor, ref.factor, tol)) << "factor, size 1";
    EXPECT_TRUE(ulpClose(one.x, ref.x, tol)) << "solve, size 1";
    EXPECT_TRUE(ulpClose(one.xt, ref.xt, tol)) << "solveT, size 1";
    EXPECT_TRUE(ulpClose(one.inverse, ref.inverse, tol)) << "inverse, size 1";
    EXPECT_EQ(one.info, ref.info);
}

TEST(LinalgTeamLu, PivotTiesGoToTheLowestRowLikeTheOneThreadForm)
{
    constexpr std::size_t N = 6;
    std::vector<double> a(N * N, 1.0);  // every column ties: the pivot is the first row each time
    for (std::size_t i = 0; i < N; ++i)
        a[i * N + i] = 3.0 + static_cast<double>(i);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < N; ++i)
            a[i * N + j] = (i == 2 || i == 4) ? 5.0 + static_cast<double>(j) : a[i * N + j];
    const std::vector<double> b = randomVector<double>(N, 5);
    const LuRun<double> r = luTeam<N>(a, b, 1, 4);
    const al::Lu<double, N> f = al::lu<N>(a.data());
    for (std::size_t k = 0; k < N; ++k)
        EXPECT_EQ(r.piv[k], f.piv[k]) << "step " << k;
}

TEST(LinalgTeamLu, SingularInfoMatchesTheOneThreadFormAndSolvesAreZero)
{
    constexpr std::size_t N = 30;  // the one-thread form is the reference for `info`: N <= 32
    std::vector<double> a = generalMatrix<double>(N, 41);
    for (std::size_t j = 0; j < N; ++j)
        a[5 * N + j] = 0.0;  // a zero row
    const std::vector<double> b = randomVector<double>(N, 42);
    for (int size : {1, 8}) {
        const LuRun<double> r = luTeam<N>(a, b, 1, size);
        EXPECT_EQ(r.info, al::lu<N>(a.data()).info) << "team " << size;
        EXPECT_GT(r.info, 0);
        EXPECT_EQ(r.det, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(r.x[i], 0.0);
            EXPECT_EQ(r.xt[i], 0.0);
        }
        for (double v : r.inverse)
            EXPECT_EQ(v, 0.0);
    }
}

// ---------------------------------------------------------------- Cholesky

template<class T, std::size_t N>
void checkCholAgainstReference()
{
    const std::vector<T> a = spdMatrix<T>(N, 51 + N), b = randomVector<T>(N, 52 + N);
    const CholRun<T> r = cholTeam<N>(a, b, 1);
    ASSERT_EQ(r.info, 0) << "N=" << N;
    const double tol = 50.0 * static_cast<double>(N) * eps<T>();
    EXPECT_LE(forwardError(r.x, referenceSolve(a, b, N)), tol) << "choSolve N=" << N;
    EXPECT_LE(backwardResidual(a, r.x, b, N), tol) << "choSolve N=" << N;
    // L is lower with a zero upper part, and L L^T reproduces A
    Real worst = 0, scale = 0;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            if (j > i) {
                ASSERT_EQ(r.l[i * N + j], T(0));
            }
            Real s = 0;
            for (std::size_t k = 0; k <= std::min(i, j); ++k)
                s += static_cast<Real>(r.l[i * N + k]) * static_cast<Real>(r.l[j * N + k]);
            worst = std::max(worst, std::fabs(s - static_cast<Real>(a[i * N + j])));
            scale = std::max(scale, std::fabs(static_cast<Real>(a[i * N + j])));
        }
    EXPECT_LE(static_cast<double>(worst / scale), tol) << "reconstruction N=" << N;
}

TEST(LinalgTeamCholesky, MatchesALongDoubleReferenceF64)
{
    forSizes<33, 64, 71, 128, 512>([](auto n) { checkCholAgainstReference<double, decltype(n)::value>(); });
}

TEST(LinalgTeamCholesky, MatchesALongDoubleReferenceF32)
{
    forSizes<33, 64, 71, 128, 512>([](auto n) { checkCholAgainstReference<float, decltype(n)::value>(); });
}

template<class T>
void checkCholVsOneThread()
{
    forSizes<4, 16, 32>([](auto n) {
        constexpr std::size_t N = decltype(n)::value;
        const std::vector<T> a = spdMatrix<T>(N, 61 + N), b = randomVector<T>(N, 62 + N);
        const CholRun<T> r = cholTeam<N>(a, b, 1);
        const al::Cholesky<T, N> f = al::cholesky<N>(a.data());
        ASSERT_EQ(r.info, f.info);
        std::vector<T> x1(N);
        al::choSolve(f, b.data(), x1.data());
        const double tol = 50.0 * static_cast<double>(N) * eps<T>();
        for (std::size_t i = 0; i < N * N; ++i)
            EXPECT_NEAR(static_cast<double>(r.l[i]), static_cast<double>(f.l[i]), tol * std::max(1.0, std::fabs(static_cast<double>(f.l[i]))));
        EXPECT_LE(forwardError(r.x, std::vector<Real>(x1.begin(), x1.end())), tol) << "N=" << N;
    });
}

TEST(LinalgTeamCholesky, UlpContractAgainstTheOneThreadFormF64)
{
    checkCholVsOneThread<double>();
}

TEST(LinalgTeamCholesky, UlpContractAgainstTheOneThreadFormF32)
{
    checkCholVsOneThread<float>();
}

TEST(LinalgTeamCholesky, AgreesAcrossTeamSizesAndInPlace)
{
    constexpr std::size_t N = 71;
    const std::vector<double> a = spdMatrix<double>(N, 71), b = randomVector<double>(N, 72);
    const double tol = 50.0 * N * eps<double>();
    // Host teams of 32..256 threads replay one compiled path: bitwise, the determinism of that path.
    const CholRun<double> ref = cholTeam<N>(a, b, 32);
    for (int size : {64, 128, 256}) {
        const CholRun<double> r = cholTeam<N>(a, b, size);
        EXPECT_TRUE(bitwiseEqual(r.l, ref.l)) << "factor, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.x, ref.x)) << "solve, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.lowerSolve, ref.lowerSolve)) << "triLower, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.upperSolve, ref.upperSolve)) << "triUpperT, team " << size;
    }
    // The size-1 twin and the in-place factor are other call shapes: the ULP contract.
    const CholRun<double> one = cholTeam<N>(a, b, 1), inPlace = cholTeam<N>(a, b, 64, true);
    EXPECT_TRUE(ulpClose(one.l, ref.l, tol)) << "factor, size 1";
    EXPECT_TRUE(ulpClose(one.x, ref.x, tol)) << "solve, size 1";
    EXPECT_TRUE(ulpClose(inPlace.l, ref.l, tol)) << "factor formed in place on the operand";
}

TEST(LinalgTeamCholesky, FactorMatchesTheOneThreadFactorToAFewUlp)
{
    // Entry for entry the same arithmetic as `linalg::cholesky`; the compiler may contract the two loops differently.
    constexpr std::size_t N = 32;
    const std::vector<double> a = spdMatrix<double>(N, 81), b = randomVector<double>(N, 82);
    const CholRun<double> r = cholTeam<N>(a, b, 4);
    const al::Cholesky<double, N> f = al::cholesky<N>(a.data());
    for (std::size_t i = 0; i < N * N; ++i)
        EXPECT_NEAR(r.l[i], f.l[i], 4 * eps<double>() * std::fabs(f.l[i])) << i;
}

TEST(LinalgTeamCholesky, NotPositiveDefiniteInfoMatchesAndSolvesAreZero)
{
    constexpr std::size_t N = 32;
    std::vector<double> a = spdMatrix<double>(N, 91);
    a[9 * N + 9] = -1.0;
    const std::vector<double> b = randomVector<double>(N, 92);
    for (bool inPlace : {false, true}) {
        const CholRun<double> r = cholTeam<N>(a, b, 4, inPlace);
        EXPECT_EQ(r.info, al::cholesky<N>(a.data()).info);
        EXPECT_EQ(r.info, 10);
        for (std::size_t i = 0; i < N; ++i) {
            EXPECT_EQ(r.x[i], 0.0);
            for (std::size_t j = 9; j < N; ++j)
                EXPECT_EQ(r.l[i * N + j], 0.0) << "column " << j << " from the failing one on is zero";
        }
    }
}

TEST(LinalgTeamCholesky, FromAFactorChecksTheDiagonal)
{
    constexpr std::size_t N = 12;
    const al::Cholesky<double, N> ref = al::cholesky<N>(spdMatrix<double>(N, 5).data());
    std::vector<double> l(ref.l, ref.l + N * N), out(N * N, 7.0);
    std::int32_t info = -1;
    runTeam(4, [&](const tl::Team& t) { tl::choleskyFrom<N>(t, {out.data(), &info}, l.data()); });
    EXPECT_EQ(info, 0);
    EXPECT_TRUE(bitwiseEqual(out, l));
    l[4 * N + 4] = 0.0;
    runTeam(4, [&](const tl::Team& t) { tl::choleskyFrom<N>(t, {out.data(), &info}, l.data()); });
    EXPECT_EQ(info, 5);
    for (double v : out)
        EXPECT_EQ(v, 0.0);
}

// ---------------------------------------------------------------- Eigh

template<class T>
Real eighResidual(const std::vector<T>& a, const EighRun<T>& r, std::size_t n, Real& ortho)
{
    Real worst = 0, scale = 0;
    ortho = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            Real av = 0, orth = 0;
            for (std::size_t k = 0; k < n; ++k) {
                av += static_cast<Real>(a[i * n + k]) * static_cast<Real>(r.v[k * n + j]);
                orth += static_cast<Real>(r.v[k * n + i]) * static_cast<Real>(r.v[k * n + j]);
            }
            worst = std::max(worst, std::fabs(av - static_cast<Real>(r.v[i * n + j]) * static_cast<Real>(r.w[j])));
            scale = std::max(scale, std::fabs(static_cast<Real>(a[i * n + j])));
            ortho = std::max(ortho, std::fabs(orth - (i == j ? Real(1) : Real(0))));
        }
    return worst / scale;
}

template<class T>
std::vector<T> symmetricMatrix(std::size_t n, std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<T> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            a[i * n + j] = a[j * n + i] = static_cast<T>(rng.next());
    return a;
}

template<class T, std::size_t N>
void checkEigh()
{
    const std::vector<T> a = symmetricMatrix<T>(N, 101 + N);
    const EighRun<T> r = eighTeam<N>(a, 1);
    const double tol = 50.0 * static_cast<double>(N) * eps<T>();
    Real ortho = 0;
    EXPECT_LE(static_cast<double>(eighResidual(a, r, N, ortho)), tol) << "A V = V W, N=" << N;
    EXPECT_LE(static_cast<double>(ortho), tol) << "V^T V = I, N=" << N;
    EXPECT_EQ(r.info & 1, 0) << "converged, N=" << N;
    for (std::size_t i = 0; i + 1 < N; ++i)
        EXPECT_LE(r.w[i], r.w[i + 1]) << "ascending at " << i;
    Real trace = 0, sum = 0;  // sum of the eigenvalues is the trace
    for (std::size_t i = 0; i < N; ++i) {
        trace += static_cast<Real>(a[i * N + i]);
        sum += static_cast<Real>(r.w[i]);
    }
    EXPECT_NEAR(static_cast<double>(trace), static_cast<double>(sum), tol * static_cast<double>(N));
}

TEST(LinalgTeamEigh, InvariantsAtLargeSizesF64)
{
    forSizes<33, 64, 71, 128>([](auto n) { checkEigh<double, decltype(n)::value>(); });
}

TEST(LinalgTeamEigh, InvariantsAtLargeSizesF32)
{
    forSizes<33, 64, 71>([](auto n) { checkEigh<float, decltype(n)::value>(); });
}

TEST(LinalgTeamEigh, UlpContractAgainstTheOneThreadFormAndAgreesAcrossTeamSizes)
{
    forSizes<5, 16, 32>([](auto n) {
        constexpr std::size_t N = decltype(n)::value;
        const std::vector<double> a = symmetricMatrix<double>(N, 111 + N);
        const al::Eigh<double, N> f = al::eigh<N>(a.data());
        const EighRun<double> ref = eighTeam<N>(a, 4);
        EXPECT_EQ(ref.info, f.info);
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_NEAR(ref.w[i], f.w[i], 50.0 * N * eps<double>() * std::max(1.0, std::fabs(f.w[i]))) << "w " << i;
        for (std::size_t i = 0; i < N * N; ++i)
            EXPECT_NEAR(ref.v[i], f.v[i], 50.0 * N * eps<double>()) << "v " << i;
        const EighRun<double> one = eighTeam<N>(a, 1);  // the size-1 twin is another call site: the ULP contract
        EXPECT_TRUE(ulpClose(one.w, ref.w, 50.0 * N * eps<double>())) << "w, size 1";
        EXPECT_TRUE(ulpClose(one.v, ref.v, 50.0 * N * eps<double>())) << "v, size 1";
        // Host teams of 4..256 threads replay one compiled path: bitwise, the determinism of that path.
        for (int size : {32, 256}) {
            if (N > 5 && size == 256)
                continue;  // a 256-thread host barrier per rotation is slow; the device rows cover 256
            const EighRun<double> r = eighTeam<N>(a, size);
            EXPECT_TRUE(bitwiseEqual(r.w, ref.w)) << "w, N=" << N << " team " << size;
            EXPECT_TRUE(bitwiseEqual(r.v, ref.v)) << "v, N=" << N << " team " << size;
            EXPECT_EQ(r.info, ref.info);
        }
    });
}

TEST(LinalgTeamEigh, CloseEigenvaluesSetInfoBitTwoAndNaNSetsBitOne)
{
    constexpr std::size_t N = 20;
    std::vector<double> a(N * N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        a[i * N + i] = (i < 2) ? 1.0 : 1.0 + static_cast<double>(i);  // a degenerate pair
    EXPECT_EQ(eighTeam<N>(a, 4).info & 2, 2);
    a[3 * N + 3] = std::nan("");
    EXPECT_EQ(eighTeam<N>(a, 4).info & 1, 1);
}

TEST(LinalgTeamEigh, SizeOneIsTheScalar)
{
    const std::vector<double> a{2.5};
    const EighRun<double> r = eighTeam<1>(a, 1);
    EXPECT_EQ(r.w[0], 2.5);
    EXPECT_EQ(r.v[0], 1.0);
    EXPECT_EQ(r.info, 0);
}

// ---------------------------------------------------------------- QR least squares

/** @brief Least squares in long double through the normal equations (well conditioned here). */
template<class T>
std::vector<Real> referenceLeastSquares(const std::vector<T>& a, const std::vector<T>& b, std::size_t m, std::size_t n)
{
    std::vector<Real> g(n * n, 0), rhs(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t r = 0; r < m; ++r)
                g[i * n + j] += static_cast<Real>(a[r * n + i]) * static_cast<Real>(a[r * n + j]);
        for (std::size_t r = 0; r < m; ++r)
            rhs[i] += static_cast<Real>(a[r * n + i]) * static_cast<Real>(b[r]);
    }
    // Gaussian elimination on the long double system
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = j + 1; i < n; ++i) {
            const Real f = g[i * n + j] / g[j * n + j];
            for (std::size_t c = j; c < n; ++c)
                g[i * n + c] -= f * g[j * n + c];
            rhs[i] -= f * rhs[j];
        }
    }
    for (std::size_t r = n; r-- > 0;) {
        for (std::size_t c = r + 1; c < n; ++c)
            rhs[r] -= g[r * n + c] * rhs[c];
        rhs[r] /= g[r * n + r];
    }
    return rhs;
}

template<class T, std::size_t N>
void checkQr(std::size_t m)
{
    const std::vector<T> a = randomVector<T>(m * N, 121 + N), b = randomVector<T>(m, 122 + N);
    const QrRun<T> r = qrTeam<N>(a, b, m, 1, 1);
    ASSERT_EQ(r.info, 0);
    const double tol = 50.0 * static_cast<double>(N) * eps<T>();
    EXPECT_LE(forwardError(r.x, referenceLeastSquares(a, b, m, N)), tol * 10) << "N=" << N << " m=" << m;
    // consistent system: b = A x0 recovers x0
    const std::vector<T> x0 = randomVector<T>(N, 123 + N);
    std::vector<T> bc(m);
    for (std::size_t i = 0; i < m; ++i) {
        Real s = 0;
        for (std::size_t j = 0; j < N; ++j)
            s += static_cast<Real>(a[i * N + j]) * static_cast<Real>(x0[j]);
        bc[i] = static_cast<T>(s);
    }
    const QrRun<T> rc = qrTeam<N>(a, bc, m, 1, 1);
    EXPECT_LE(forwardError(rc.x, std::vector<Real>(x0.begin(), x0.end())), tol * 10) << "consistent N=" << N;
}

TEST(LinalgTeamQr, LeastSquaresMatchesALongDoubleReferenceF64)
{
    checkQr<double, 33>(100);
    checkQr<double, 64>(200);
    checkQr<double, 71>(300);
    checkQr<double, 128>(300);
    checkQr<double, 512>(700);
}

TEST(LinalgTeamQr, LeastSquaresMatchesALongDoubleReferenceF32)
{
    checkQr<float, 33>(100);
    checkQr<float, 64>(200);
    checkQr<float, 71>(300);
    checkQr<float, 128>(300);
}

TEST(LinalgTeamQr, SquareSystemAndAFewRows)
{
    checkQr<double, 33>(33);
    checkQr<double, 4>(5);
    checkQr<double, 1>(9);
}

TEST(LinalgTeamQr, AgreesAcrossTeamSizesAndMultipleRightHandSides)
{
    constexpr std::size_t N = 40, M = 150, K = 4;  // 3 row tiles
    const std::vector<double> a = randomVector<double>(M * N, 131), b = randomVector<double>(M * K, 132);
    const double tol = 50.0 * N * eps<double>();
    // Host teams of 32..256 threads replay one compiled path: bitwise, the determinism of that path.
    const QrRun<double> ref = qrTeam<N>(a, b, M, K, 32);
    for (int size : {64, 128, 256}) {
        const QrRun<double> r = qrTeam<N>(a, b, M, K, size);
        EXPECT_TRUE(bitwiseEqual(r.factor, ref.factor)) << "factor, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.tau, ref.tau)) << "tau, team " << size;
        EXPECT_TRUE(bitwiseEqual(r.x, ref.x)) << "solution, team " << size;
    }
    const QrRun<double> twin = qrTeam<N>(a, b, M, K, 1);  // the size-1 twin is another call site: the ULP contract
    EXPECT_TRUE(ulpClose(twin.factor, ref.factor, tol)) << "factor, size 1";
    EXPECT_TRUE(ulpClose(twin.x, ref.x, tol)) << "solution, size 1";
    // each column of the multi-right-hand-side solve matches the single solve (another instantiation: the ULP contract)
    for (std::size_t c = 0; c < K; ++c) {
        std::vector<double> bc(M);
        for (std::size_t i = 0; i < M; ++i)
            bc[i] = b[i * K + c];
        const QrRun<double> one = qrTeam<N>(a, bc, M, 1, 1);
        std::vector<double> col(N);
        for (std::size_t i = 0; i < N; ++i)
            col[i] = ref.x[i * K + c];
        EXPECT_TRUE(ulpClose(one.x, col, tol)) << "column " << c;
    }
}

TEST(LinalgTeamQr, ExactlyRankDeficientReportsTheColumnAndSolvesToZero)
{
    constexpr std::size_t N = 8, M = 20;
    std::vector<double> a = randomVector<double>(M * N, 141);
    for (std::size_t i = 0; i < M; ++i)
        a[i * N + 3] = 0.0;  // a zero column
    const QrRun<double> r = qrTeam<N>(a, randomVector<double>(M, 142), M, 1, 4);
    EXPECT_EQ(r.info, 4);
    for (double v : r.x)
        EXPECT_EQ(v, 0.0);
}

template<std::size_t N>
void checkQrGramSolve(std::size_t m, std::size_t k, int size)
{
    const std::vector<double> a = randomVector<double>(m * N, 181 + N), b = randomVector<double>(N * k, 182 + N);
    std::vector<double> factor(m * N), tau(N), work(tl::qrWorkSize(m, N)), x(N * k, 0.0), xin = b;
    std::int32_t info = -1;
    const tl::TeamQr<double, N> f{factor.data(), tau.data(), work.data(), &info};
    runTeam(size, [&](const tl::Team& t) {
        tl::qr<N>(t, f, a.data(), m);
        tl::qrGramSolve<N>(t, f, b.data(), x.data(), k);
        tl::qrGramSolve<N>(t, f, xin.data(), xin.data(), k);  // in place
    });
    ASSERT_EQ(info, 0);
    EXPECT_TRUE(ulpClose(x, xin, 50.0 * N * eps<double>())) << "in place agrees with out of place";
    std::vector<Real> g(N * N, 0);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t r = 0; r < m; ++r)
                g[i * N + j] += static_cast<Real>(a[r * N + i]) * static_cast<Real>(a[r * N + j]);
        }
    for (std::size_t c = 0; c < k; ++c) {
        std::vector<Real> rhs(N), sol;
        for (std::size_t i = 0; i < N; ++i)
            rhs[i] = b[i * k + c];
        // long double Gaussian elimination on the exact-ish Gram matrix
        std::vector<Real> mat = g;
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = j + 1; i < N; ++i) {
                const Real fct = mat[i * N + j] / mat[j * N + j];
                for (std::size_t cc = j; cc < N; ++cc)
                    mat[i * N + cc] -= fct * mat[j * N + cc];
                rhs[i] -= fct * rhs[j];
            }
        for (std::size_t r = N; r-- > 0;) {
            for (std::size_t cc = r + 1; cc < N; ++cc)
                rhs[r] -= mat[r * N + cc] * rhs[cc];
            rhs[r] /= mat[r * N + r];
        }
        std::vector<double> xc(N);
        for (std::size_t i = 0; i < N; ++i)
            xc[i] = x[i * k + c];
        EXPECT_LE(forwardError(xc, rhs), 500.0 * N * eps<double>()) << "N=" << N << " k=" << k << " col " << c << " team " << size;
    }
}

TEST(LinalgTeamQr, GramSolveMatchesALongDoubleReference)
{
    for (int size : {1, 8}) {
        checkQrGramSolve<33>(100, 3, size);
        checkQrGramSolve<71>(210, 4, size);
    }
}

TEST(LinalgTeamQr, GramSolveOfAFailedFactorIsZero)
{
    constexpr std::size_t N = 6, M = 20;
    std::vector<double> a = randomVector<double>(M * N, 183);
    for (std::size_t i = 0; i < M; ++i)
        a[i * N + 2] = 0.0;
    std::vector<double> factor(M * N), tau(N), work(tl::qrWorkSize(M, N)), x(N * 2, 5.0);
    const std::vector<double> b = randomVector<double>(N * 2, 184);
    std::int32_t info = -1;
    const tl::TeamQr<double, N> f{factor.data(), tau.data(), work.data(), &info};
    runTeam(4, [&](const tl::Team& t) {
        tl::qr<N>(t, f, a.data(), M);
        tl::qrGramSolve<N>(t, f, b.data(), x.data(), 2);
    });
    EXPECT_EQ(info, 3);
    for (double v : x)
        EXPECT_EQ(v, 0.0);
}

// ---------------------------------------------------------------- BlockSum

template<std::size_t P, std::size_t Q>
std::vector<double> blockSumTeam(const std::vector<double>& u, const std::vector<double>& v, std::size_t first,
    std::size_t count, const std::int32_t* index, int size)
{
    std::vector<double> g(P * Q, -1.0);
    runTeam(size, [&](const tl::Team& t) { tl::blockSum<P, Q>(t, g.data(), u.data(), v.data(), first, count, P, Q, index); });
    return g;
}

TEST(LinalgTeamBlockSum, IsTheSerialLoopToTheUlpBoundAtAnyTeamSize)
{
    constexpr std::size_t P = 71, Q = 71, rows = 300;
    const std::vector<double> j = randomVector<double>(rows * P, 151);
    std::vector<double> want(P * P, 0.0);
    for (std::size_t i = 0; i < P; ++i)
        for (std::size_t c = 0; c < P; ++c) {
            double s = 0.0;
            for (std::size_t r = 40; r < 40 + 200; ++r)
                s += j[r * P + i] * j[r * P + c];
            want[i * P + c] = s;
        }
    // Bound from the sum length: 2 * count * eps times the largest sum of |products|.
    const double bound = 2.0 * 200 * eps<double>() * maxAbs(blockSumTeam<P, Q>(absOf(j), absOf(j), 40, 200, nullptr, 1));
    const std::vector<double> ref = blockSumTeam<P, Q>(j, j, 40, 200, nullptr, 32);
    for (int size : {1, 32, 64, 128, 256}) {
        const std::vector<double> g = blockSumTeam<P, Q>(j, j, 40, 200, nullptr, size);
        EXPECT_LE(maxAbsDiff(g, want), bound) << "vs the serial loop, team " << size;
        if (size > 32) {  // host teams of 32..256 threads replay one compiled path: bitwise
            EXPECT_TRUE(bitwiseEqual(g, ref)) << "team " << size << " vs team 32";
        }
    }
}

TEST(LinalgTeamBlockSum, StridedOutputHoldsTheContiguousValuesToTheUlpBound)
{
    constexpr std::size_t P = 13, Q = 5, rows = 70, keys = 4;
    const std::vector<double> u = randomVector<double>(rows * P, 171), v = randomVector<double>(rows * Q, 172);
    const std::vector<double> want = blockSumTeam<P, Q>(u, v, 0, rows, nullptr, 1);
    const double bound = 2.0 * rows * eps<double>() * maxAbs(blockSumTeam<P, Q>(absOf(u), absOf(v), 0, rows, nullptr, 1));
    for (int size : {1, 8, 64}) {
        std::vector<double> plane(P * Q * keys, -1.0);
        constexpr std::size_t key = 2;  // payload-first plane: entry-major, the key fastest
        runTeam(size, [&](const tl::Team& t) {
            tl::blockSum<P, Q>(t, plane.data() + key, u.data(), v.data(), 0, rows, P, Q, nullptr, Q * keys, keys);
        });
        for (std::size_t e = 0; e < P * Q; ++e) {
            EXPECT_NEAR(plane[e * keys + key], want[e], bound) << "entry " << e << " team " << size;
            for (std::size_t other = 0; other < keys; ++other) {
                if (other != key) {
                    EXPECT_EQ(plane[e * keys + other], -1.0) << "only this key's slots are written";
                }
            }
        }
    }
}

TEST(LinalgTeamBlockSum, VectorPayloadAndAnActiveSetIndexMap)
{
    constexpr std::size_t P = 33, Q = 1, rows = 120;
    const std::vector<double> j = randomVector<double>(rows * P, 161), r = randomVector<double>(rows, 162);
    std::vector<std::int32_t> index;
    for (std::int32_t i = static_cast<std::int32_t>(rows) - 1; i >= 0; i -= 3)
        index.push_back(i);  // descending subset: the map's order is the sum's order
    std::vector<double> want(P, 0.0);
    for (std::size_t i = 0; i < P; ++i)
        for (std::size_t k = 0; k < index.size(); ++k)
            want[i] += j[static_cast<std::size_t>(index[k]) * P + i] * r[static_cast<std::size_t>(index[k])];
    const std::vector<double> g1 = blockSumTeam<P, Q>(j, r, 0, index.size(), index.data(), 1);
    const std::vector<double> g2 = blockSumTeam<P, Q>(j, r, 0, index.size(), index.data(), 64);
    const double bound = 2.0 * static_cast<double>(index.size()) * eps<double>() * maxAbs(blockSumTeam<P, Q>(absOf(j), absOf(r), 0, index.size(), index.data(), 1));
    EXPECT_LE(maxAbsDiff(g1, g2), bound) << "size 1 vs 64";
    EXPECT_LE(maxAbsDiff(g1, want), bound);
}

TEST(LinalgTeamBlockSum, RowsPerSampleMatchesTheExpandedRowListToTheUlpBound)
{
    constexpr std::size_t P = 9, M = 4, samples = 30;
    const std::vector<double> j = randomVector<double>(samples * M * P, 191);
    std::vector<std::int32_t> index{27, 3, 14, 15, 0, 22, 9};  // an active set, in a non-ascending order
    std::vector<std::int32_t> expanded;
    for (std::int32_t s : index)
        for (std::int32_t q = 0; q < static_cast<std::int32_t>(M); ++q)
            expanded.push_back(s * static_cast<std::int32_t>(M) + q);
    const std::vector<double> want = blockSumTeam<P, P>(j, j, 0, expanded.size(), expanded.data(), 1);
    // Bound from the sum length (rows added per entry): 2 * rows * eps times the largest sum of |products|.
    const double bound = 2.0 * static_cast<double>(expanded.size()) * eps<double>() * maxAbs(blockSumTeam<P, P>(absOf(j), absOf(j), 0, expanded.size(), expanded.data(), 1));
    for (int size : {1, 8, 64}) {
        std::vector<double> g(P * P, -1.0), plain(P * P, -1.0);
        runTeam(size, [&](const tl::Team& t) {
            tl::blockSum<P, P>(t, g.data(), j.data(), j.data(), 0, index.size(), P, P, index.data(), 0, 0, M);
        });
        EXPECT_LE(maxAbsDiff(g, want), bound) << "indexed, team " << size;
        // contiguous samples, no map: first = 5, 6 samples
        runTeam(size, [&](const tl::Team& t) {
            tl::blockSum<P, P>(t, plain.data(), j.data(), j.data(), 5, 6, P, P, nullptr, 0, 0, M);
        });
        const std::vector<double> flat = blockSumTeam<P, P>(j, j, 5 * M, 6 * M, nullptr, 1);
        EXPECT_LE(maxAbsDiff(plain, flat), 2.0 * 6 * M * eps<double>() * maxAbs(blockSumTeam<P, P>(absOf(j), absOf(j), 5 * M, 6 * M, nullptr, 1))) << "contiguous, team " << size;
    }
}

TEST(LinalgTeamBlockSum, EmptyRangeIsZero)
{
    const std::vector<double> j(10 * 8, 1.0);
    for (double v : blockSumTeam<8, 8>(j, j, 3, 0, nullptr, 4))
        EXPECT_EQ(v, 0.0);
}

TEST(LinalgTeamHandle, SizeOneTeamNeedsNoBarrierAndReportsItsPlace)
{
    const tl::Team t = tl::Team::here();
    EXPECT_EQ(t.rank, 0);
    EXPECT_EQ(t.size, 1);
    t.sync();
    EXPECT_EQ(tl::qrWorkSize(130, 5), 5u * 3u + 1u);
    EXPECT_EQ(tl::eighWorkSize(10), 104u);
}

} // namespace
} // namespace linalg_team
} // namespace aether_tests
