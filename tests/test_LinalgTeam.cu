// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_LinalgTeam.cu
 * @brief `aether::linalg::team` (CUDA mode): the shared host battery, plus the same
 *        factorisations run one thread block per matrix and compared with the host twin.
 *
 * The device may contract products into FMAs where the host build does not, so a device
 * result is compared with the host twin by a ULP contract (well-conditioned inputs); the
 * device is compared with itself bitwise across team sizes. See `test_LinalgTeam_common.h`.
 */

#include "test_LinalgTeam_common.h"

namespace aether_tests {
namespace linalg_team {
namespace {

constexpr std::size_t kLanes = 3;

template<class U>
struct Dev {
    U* p = nullptr;
    explicit Dev(std::size_t n) { cudaMalloc(&p, n * sizeof(U)); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    void put(const std::vector<U>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(U), cudaMemcpyHostToDevice); }
    std::vector<U> get(std::size_t n) const
    {
        std::vector<U> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(U), cudaMemcpyDeviceToHost);
        return h;
    }
};

template<std::size_t N, class T>
AETHER_KERNEL()
void luBlock(const T* a, const T* b, T* factor, std::int32_t* piv, std::int32_t* info, T* x, T* xt)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamLu<T, N> f{factor + m * N * N, piv + m * N, info + m};
    tl::lu<N>(t, f, a + m * N * N);
    tl::luSolve<N>(t, f, b + m * N, x + m * N);
    tl::luSolveT<N>(t, f, b + m * N, xt + m * N);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void cholBlock(const T* a, const T* b, T* l, std::int32_t* info, T* x)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamCholesky<T, N> f{l + m * N * N, info + m};
    tl::cholesky<N>(t, f, a + m * N * N);
    tl::choSolve<N>(t, f, b + m * N, x + m * N);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void eighBlock(const T* a, T* w, T* v, T* work, std::int32_t* info)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamEigh<T, N> f{w + m * N, v + m * N * N, work + m * tl::eighWorkSize(N), info + m};
    tl::eigh<N>(t, f, a + m * N * N);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void qrBlock(const T* a, T* b, std::size_t rows, T* factor, T* tau, T* work, std::int32_t* info, T* x)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamQr<T, N> f{factor + m * rows * N, tau + m * N, work + m * tl::qrWorkSize(rows, N), info + m};
    tl::qr<N>(t, f, a + m * rows * N, rows);
    tl::qrSolve<N>(t, f, rows, b + m * rows, x + m * N);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void qrMultiBlock(const T* a, T* b, std::size_t rows, std::size_t k, T* factor, T* tau, T* work, std::int32_t* info, T* x)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamQr<T, N> f{factor + m * rows * N, tau + m * N, work + m * tl::qrWorkSize(rows, N), info + m};
    tl::qr<N>(t, f, a + m * rows * N, rows);
    tl::qrSolve<N>(t, f, rows, b + m * rows * k, x + m * N * k, k);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void multiRhsBlock(const T* a, const T* spd, const T* b, std::size_t k, T* lu, std::int32_t* piv, std::int32_t* info,
                   T* l, std::int32_t* cinfo, T* out)
{
    // out per lane: luSolve, luSolveT, luInverse (N*N), choSolve, triLower, triUpperT, each N*k except the inverse
    const std::size_t m = blockIdx.x;
    const std::size_t per = 5 * N * k + N * N;
    const tl::Team t = tl::Team::here();
    const tl::TeamLu<T, N> f{lu + m * N * N, piv + m * N, info + m};
    tl::lu<N>(t, f, a + m * N * N);
    T* o = out + m * per;
    tl::luSolve<N>(t, f, b + m * N * k, o, k);
    tl::luSolveT<N>(t, f, b + m * N * k, o + N * k, k);
    tl::luInverse<N>(t, f, o + 2 * N * k);
    const tl::TeamCholesky<T, N> g{l + m * N * N, cinfo + m};
    tl::cholesky<N>(t, g, spd + m * N * N);
    T* oc = o + 2 * N * k + N * N;
    tl::choSolve<N>(t, g, b + m * N * k, oc, k);
    tl::triLower<N>(t, g, b + m * N * k, oc + N * k, k);
    tl::triUpperT<N>(t, g, b + m * N * k, oc + 2 * N * k, k);
}

template<std::size_t P, class T>
AETHER_KERNEL()
void gramStridedBlock(const T* j, std::size_t rows, T* plane, std::size_t keys)
{
    // lane m is key m of a payload-first plane (entry-major, key fastest)
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    tl::blockSum<P, P>(t, plane + m, j + m * rows * P, j + m * rows * P, 0, rows, P, P, nullptr, P * keys, keys);
}

template<std::size_t N, class T>
AETHER_KERNEL()
void qrGramBlock(const T* a, const T* b, std::size_t rows, std::size_t k, T* factor, T* tau, T* work, std::int32_t* info, T* x)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    const tl::TeamQr<T, N> f{factor + m * rows * N, tau + m * N, work + m * tl::qrWorkSize(rows, N), info + m};
    tl::qr<N>(t, f, a + m * rows * N, rows);
    tl::qrGramSolve<N>(t, f, b + m * N * k, x + m * N * k, k);
}

template<std::size_t P, class T>
AETHER_KERNEL()
void gramRowsPerBlock(const T* j, const std::int32_t* index, std::size_t samples, std::size_t rowsPer, T* gram)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    tl::blockSum<P, P>(t, gram + m * P * P, j, j, 0, samples, P, P, index, 0, 0, rowsPer);
}

template<std::size_t P, class T>
AETHER_KERNEL()
void gramBlock(const T* j, const T* r, std::size_t rows, T* gram, T* grad)
{
    const std::size_t m = blockIdx.x;
    const tl::Team t = tl::Team::here();
    tl::blockSum<P, P>(t, gram + m * P * P, j + m * rows * P, j + m * rows * P, 0, rows, P, P);
    tl::blockSum<P, 1>(t, grad + m * P, j + m * rows * P, r + m * rows, 0, rows, P, 1);
}

/** @brief Per-lane inputs: lane `m` is the matrix seeded `seed + m`, concatenated. */
template<class T, class Make>
std::vector<T> lanes(Make make)
{
    std::vector<T> all;
    for (std::size_t m = 0; m < kLanes; ++m) {
        const std::vector<T> one = make(m);
        all.insert(all.end(), one.begin(), one.end());
    }
    return all;
}

template<class T>
std::vector<T> slice(const std::vector<T>& all, std::size_t m, std::size_t n)
{
    return std::vector<T>(all.begin() + static_cast<std::ptrdiff_t>(m * n), all.begin() + static_cast<std::ptrdiff_t>((m + 1) * n));
}

TEST(LinalgTeamDevice, LuSolveMatchesTheHostTwinAndIsBitwiseAcrossTeamSizes)
{
    constexpr std::size_t N = 71;
    const std::vector<double> a = lanes<double>([](std::size_t m) { return generalMatrix<double>(N, 200 + m); });
    const std::vector<double> b = lanes<double>([](std::size_t m) { return randomVector<double>(N, 210 + m); });
    Dev<double> da(a.size()), db(b.size()), df(a.size()), dx(b.size()), dxt(b.size());
    Dev<std::int32_t> dp(kLanes * N), di(kLanes);
    da.put(a);
    db.put(b);
    std::vector<double> refFactor, refX, refXt;
    for (int size : {64, 128, 256}) {
        luBlock<N, double><<<kLanes, size>>>(da.p, db.p, df.p, dp.p, di.p, dx.p, dxt.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> factor = df.get(a.size()), x = dx.get(b.size()), xt = dxt.get(b.size());
        if (refFactor.empty()) {
            refFactor = factor;
            refX = x;
            refXt = xt;
            const std::vector<std::int32_t> info = di.get(kLanes);
            for (std::size_t m = 0; m < kLanes; ++m) {
                const LuRun<double> host = luTeam<N>(slice(a, m, N * N), slice(b, m, N), 1, 1);
                EXPECT_EQ(info[m], host.info);
                const double tol = 50.0 * static_cast<double>(N) * eps<double>();
                const std::vector<double> hx = host.x, dxm = slice(x, m, N), dxtm = slice(xt, m, N);
                EXPECT_LE(forwardError(dxm, std::vector<Real>(hx.begin(), hx.end())), tol) << "lane " << m;
                EXPECT_LE(forwardError(dxtm, std::vector<Real>(host.xt.begin(), host.xt.end())), tol) << "lane " << m;
                EXPECT_LE(forwardError(slice(factor, m, N * N), std::vector<Real>(host.factor.begin(), host.factor.end())), tol) << "lane " << m;
            }
        } else {
            EXPECT_TRUE(bitwiseEqual(factor, refFactor)) << "factor, team " << size;
            EXPECT_TRUE(bitwiseEqual(x, refX)) << "solve, team " << size;
            EXPECT_TRUE(bitwiseEqual(xt, refXt)) << "solveT, team " << size;
        }
    }
}

TEST(LinalgTeamDevice, SingularLaneReportsInfoAndZeroFills)
{
    constexpr std::size_t N = 33;
    std::vector<double> a = lanes<double>([](std::size_t m) { return generalMatrix<double>(N, 220 + m); });
    for (std::size_t j = 0; j < N; ++j)
        a[1 * N * N + 7 * N + j] = 0.0;  // lane 1: a zero row
    const std::vector<double> b = lanes<double>([](std::size_t m) { return randomVector<double>(N, 230 + m); });
    Dev<double> da(a.size()), db(b.size()), df(a.size()), dx(b.size()), dxt(b.size());
    Dev<std::int32_t> dp(kLanes * N), di(kLanes);
    da.put(a);
    db.put(b);
    luBlock<N, double><<<kLanes, 64>>>(da.p, db.p, df.p, dp.p, di.p, dx.p, dxt.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<std::int32_t> info = di.get(kLanes);
    const std::vector<double> x = dx.get(b.size());
    EXPECT_EQ(info[0], 0);
    EXPECT_GT(info[1], 0);
    EXPECT_EQ(info[2], 0);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(x[N + i], 0.0);
}

TEST(LinalgTeamDevice, CholeskySolveMatchesTheHostTwinAndIsBitwiseAcrossTeamSizes)
{
    constexpr std::size_t N = 71;
    const std::vector<double> a = lanes<double>([](std::size_t m) { return spdMatrix<double>(N, 240 + m); });
    const std::vector<double> b = lanes<double>([](std::size_t m) { return randomVector<double>(N, 250 + m); });
    Dev<double> da(a.size()), db(b.size()), dl(a.size()), dx(b.size());
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    db.put(b);
    std::vector<double> refL, refX;
    for (int size : {64, 128, 256}) {
        cholBlock<N, double><<<kLanes, size>>>(da.p, db.p, dl.p, di.p, dx.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> l = dl.get(a.size()), x = dx.get(b.size());
        if (refL.empty()) {
            refL = l;
            refX = x;
            for (std::size_t m = 0; m < kLanes; ++m) {
                const CholRun<double> host = cholTeam<N>(slice(a, m, N * N), slice(b, m, N), 1);
                const double tol = 50.0 * static_cast<double>(N) * eps<double>();
                EXPECT_EQ(di.get(kLanes)[m], host.info);
                EXPECT_LE(forwardError(slice(x, m, N), std::vector<Real>(host.x.begin(), host.x.end())), tol) << "lane " << m;
                EXPECT_LE(forwardError(slice(l, m, N * N), std::vector<Real>(host.l.begin(), host.l.end())), tol) << "lane " << m;
            }
        } else {
            EXPECT_TRUE(bitwiseEqual(l, refL)) << "factor, team " << size;
            EXPECT_TRUE(bitwiseEqual(x, refX)) << "solve, team " << size;
        }
    }
}

TEST(LinalgTeamDevice, EighMatchesTheHostTwinAndIsBitwiseAcrossTeamSizes)
{
    constexpr std::size_t N = 33;
    const std::vector<double> a = lanes<double>([](std::size_t m) { return symmetricMatrix<double>(N, 260 + m); });
    Dev<double> da(a.size()), dw(kLanes * N), dv(a.size()), dwork(kLanes * tl::eighWorkSize(N));
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    std::vector<double> refW, refV;
    for (int size : {32, 64, 128}) {
        eighBlock<N, double><<<kLanes, size>>>(da.p, dw.p, dv.p, dwork.p, di.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> w = dw.get(kLanes * N), v = dv.get(a.size());
        if (refW.empty()) {
            refW = w;
            refV = v;
            for (std::size_t m = 0; m < kLanes; ++m) {
                const EighRun<double> host = eighTeam<N>(slice(a, m, N * N), 1);
                EXPECT_EQ(di.get(kLanes)[m], host.info);
                for (std::size_t i = 0; i < N; ++i)
                    EXPECT_NEAR(w[m * N + i], host.w[i], 64 * eps<double>() * N) << "lane " << m << " w " << i;
            }
        } else {
            EXPECT_TRUE(bitwiseEqual(w, refW)) << "w, team " << size;
            EXPECT_TRUE(bitwiseEqual(v, refV)) << "v, team " << size;
        }
    }
}

TEST(LinalgTeamDevice, QrLeastSquaresMatchesTheHostTwinAndIsBitwiseAcrossTeamSizes)
{
    constexpr std::size_t N = 40, M = 150;
    const std::vector<double> a = lanes<double>([](std::size_t m) { return randomVector<double>(M * N, 270 + m); });
    const std::vector<double> b = lanes<double>([](std::size_t m) { return randomVector<double>(M, 280 + m); });
    Dev<double> da(a.size()), db(b.size()), df(a.size()), dtau(kLanes * N), dwork(kLanes * tl::qrWorkSize(M, N)), dx(kLanes * N);
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    std::vector<double> refF, refX;
    for (int size : {32, 64, 256}) {
        db.put(b);  // qrSolve overwrites its right-hand side
        qrBlock<N, double><<<kLanes, size>>>(da.p, db.p, M, df.p, dtau.p, dwork.p, di.p, dx.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> f = df.get(a.size()), x = dx.get(kLanes * N);
        if (refF.empty()) {
            refF = f;
            refX = x;
            for (std::size_t m = 0; m < kLanes; ++m) {
                const QrRun<double> host = qrTeam<N>(slice(a, m, M * N), slice(b, m, M), M, 1, 1);
                EXPECT_EQ(di.get(kLanes)[m], host.info);
                EXPECT_LE(forwardError(slice(x, m, N), std::vector<Real>(host.x.begin(), host.x.end())), 50.0 * N * eps<double>()) << "lane " << m;
            }
        } else {
            EXPECT_TRUE(bitwiseEqual(f, refF)) << "factor, team " << size;
            EXPECT_TRUE(bitwiseEqual(x, refX)) << "solution, team " << size;
        }
    }
}

TEST(LinalgTeamDevice, BlockSumMatchesTheHostTwinAndIsBitwiseAcrossTeamSizes)
{
    constexpr std::size_t P = 40, R = 500;
    const std::vector<double> j = lanes<double>([](std::size_t m) { return randomVector<double>(R * P, 290 + m); });
    const std::vector<double> r = lanes<double>([](std::size_t m) { return randomVector<double>(R, 300 + m); });
    Dev<double> dj(j.size()), dr(r.size()), dg(kLanes * P * P), dv(kLanes * P);
    dj.put(j);
    dr.put(r);
    std::vector<double> refG, refV;
    for (int size : {32, 64, 256}) {
        gramBlock<P, double><<<kLanes, size>>>(dj.p, dr.p, R, dg.p, dv.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> g = dg.get(kLanes * P * P), v = dv.get(kLanes * P);
        if (refG.empty()) {
            refG = g;
            refV = v;
            for (std::size_t m = 0; m < kLanes; ++m) {
                const std::vector<double> jm = slice(j, m, R * P), rm = slice(r, m, R);
                const std::vector<double> want = blockSumTeam<P, P>(jm, jm, 0, R, nullptr, 1);
                const std::vector<double> wantV = blockSumTeam<P, 1>(jm, rm, 0, R, nullptr, 1);
                for (std::size_t e = 0; e < P * P; ++e)
                    EXPECT_NEAR(g[m * P * P + e], want[e], 4 * eps<double>() * R) << "lane " << m << " gram " << e;
                for (std::size_t e = 0; e < P; ++e)
                    EXPECT_NEAR(v[m * P + e], wantV[e], 4 * eps<double>() * R) << "lane " << m << " grad " << e;
            }
        } else {
            EXPECT_TRUE(bitwiseEqual(g, refG)) << "gram, team " << size;
            EXPECT_TRUE(bitwiseEqual(v, refV)) << "gradient, team " << size;
        }
    }
}

/** @brief Every multi-right-hand-side solve path on the device (`N` not a multiple of the team size) against the host twin. */
template<std::size_t N>
void checkMultiRhs(std::size_t k, int size)
{
    const std::vector<double> a = lanes<double>([](std::size_t m) {  // rows reversed: every column pivots
        const std::vector<double> g = generalMatrix<double>(N, 310 + m);
        std::vector<double> r(N * N);
        for (std::size_t i = 0; i < N; ++i)
            std::copy(g.begin() + static_cast<std::ptrdiff_t>((N - 1 - i) * N), g.begin() + static_cast<std::ptrdiff_t>((N - i) * N), r.begin() + static_cast<std::ptrdiff_t>(i * N));
        return r;
    });
    const std::vector<double> spd = lanes<double>([](std::size_t m) { return spdMatrix<double>(N, 320 + m); });
    const std::vector<double> b = lanes<double>([k](std::size_t m) { return randomVector<double>(N * k, 330 + m); });
    const std::size_t per = 5 * N * k + N * N;
    Dev<double> da(a.size()), ds(spd.size()), db(b.size()), dlu(a.size()), dl(a.size()), dout(kLanes * per);
    Dev<std::int32_t> dp(kLanes * N), di(kLanes), dci(kLanes);
    da.put(a);
    ds.put(spd);
    db.put(b);
    multiRhsBlock<N, double><<<kLanes, size>>>(da.p, ds.p, db.p, k, dlu.p, dp.p, di.p, dl.p, dci.p, dout.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<double> out = dout.get(kLanes * per);
    const double tol = 50.0 * static_cast<double>(N) * eps<double>();
    for (std::size_t m = 0; m < kLanes; ++m) {
        const std::vector<double> am = slice(a, m, N * N), sm = slice(spd, m, N * N), bm = slice(b, m, N * k);
        const LuRun<double> hl = luTeam<N>(am, bm, k, 1);
        const CholRun<double> hc = cholTeam<N>(sm, slice(bm, 0, N), 1);  // host reference for the factor only
        const auto lane = [&](std::size_t off, std::size_t n) { return std::vector<double>(out.begin() + static_cast<std::ptrdiff_t>(m * per + off), out.begin() + static_cast<std::ptrdiff_t>(m * per + off + n)); };
        const auto ref = [](const std::vector<double>& v) { return std::vector<Real>(v.begin(), v.end()); };
        EXPECT_LE(forwardError(lane(0, N * k), ref(hl.x)), tol) << "luSolve N=" << N << " k=" << k << " team " << size << " lane " << m;
        EXPECT_LE(forwardError(lane(N * k, N * k), ref(hl.xt)), tol) << "luSolveT N=" << N << " k=" << k << " team " << size;
        EXPECT_LE(forwardError(lane(2 * N * k, N * N), ref(hl.inverse)), tol) << "luInverse N=" << N << " team " << size;
        // Cholesky paths against the long-double solve of each column (S X = B, L Y = B, L^T Y = B)
        const std::size_t oc = 2 * N * k + N * N;
        for (std::size_t c = 0; c < k; ++c) {
            std::vector<double> bc(N);
            for (std::size_t i = 0; i < N; ++i)
                bc[i] = bm[i * k + c];
            std::vector<double> xs(N), yl(N), yu(N);
            const auto x = lane(oc, N * k), ylo = lane(oc + N * k, N * k), yup = lane(oc + 2 * N * k, N * k);
            for (std::size_t i = 0; i < N; ++i) {
                xs[i] = x[i * k + c];
                yl[i] = ylo[i * k + c];
                yu[i] = yup[i * k + c];
            }
            EXPECT_LE(forwardError(xs, referenceSolve(sm, bc, N)), tol) << "choSolve N=" << N << " k=" << k << " col " << c << " team " << size;
            // L y = b and L^T y = b: check the residual against the host factor
            Real rl = 0, ru = 0, scale = 0;
            for (std::size_t i = 0; i < N; ++i) {
                Real sl = 0, su = 0;
                for (std::size_t j = 0; j < N; ++j) {
                    if (j <= i)
                        sl += static_cast<Real>(hc.l[i * N + j]) * static_cast<Real>(yl[j]);
                    if (j >= i)
                        su += static_cast<Real>(hc.l[j * N + i]) * static_cast<Real>(yu[j]);
                }
                rl = std::max(rl, std::fabs(sl - static_cast<Real>(bc[i])));
                ru = std::max(ru, std::fabs(su - static_cast<Real>(bc[i])));
                scale = std::max(scale, std::fabs(static_cast<Real>(bc[i])));
            }
            EXPECT_LE(static_cast<double>(rl / scale), tol * 10) << "triLower N=" << N << " k=" << k << " col " << c << " team " << size;
            EXPECT_LE(static_cast<double>(ru / scale), tol * 10) << "triUpperT N=" << N << " k=" << k << " col " << c << " team " << size;
        }
    }
}

TEST(LinalgTeamDevice, MultiRightHandSideSolvesAtSizesThatAreNotATeamMultiple)
{
    for (int size : {32, 64, 128}) {
        checkMultiRhs<33>(2, size);
        checkMultiRhs<33>(5, size);
        checkMultiRhs<71>(3, size);
    }
}

TEST(LinalgTeamDevice, QrMultiRightHandSideSolveMatchesTheHostTwin)
{
    constexpr std::size_t N = 33, M = 100, K = 3;
    const std::vector<double> a = lanes<double>([](std::size_t m) { return randomVector<double>(M * N, 340 + m); });
    const std::vector<double> b = lanes<double>([](std::size_t m) { return randomVector<double>(M * K, 350 + m); });
    for (int size : {32, 64}) {
        Dev<double> da(a.size()), db(b.size()), df(a.size()), dtau(kLanes * N), dwork(kLanes * tl::qrWorkSize(M, N)), dx(kLanes * N * K);
        Dev<std::int32_t> di(kLanes);
        da.put(a);
        db.put(b);
        qrMultiBlock<N, double><<<kLanes, size>>>(da.p, db.p, M, K, df.p, dtau.p, dwork.p, di.p, dx.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> x = dx.get(kLanes * N * K);
        for (std::size_t m = 0; m < kLanes; ++m) {
            const QrRun<double> host = qrTeam<N>(slice(a, m, M * N), slice(b, m, M * K), M, K, 1);
            EXPECT_LE(forwardError(slice(x, m, N * K), std::vector<Real>(host.x.begin(), host.x.end())), 50.0 * N * eps<double>()) << "lane " << m << " team " << size;
        }
    }
}

TEST(LinalgTeamDevice, BlockSumStridedOutputHoldsTheContiguousValuesBitForBit)
{
    constexpr std::size_t P = 40, R = 500;
    const std::vector<double> j = lanes<double>([](std::size_t m) { return randomVector<double>(R * P, 290 + m); });
    const std::vector<double> r = lanes<double>([](std::size_t m) { return randomVector<double>(R, 300 + m); });
    Dev<double> dj(j.size()), dr(r.size()), dg(kLanes * P * P), dv(kLanes * P), dplane(kLanes * P * P);
    dj.put(j);
    dr.put(r);
    gramBlock<P, double><<<kLanes, 64>>>(dj.p, dr.p, R, dg.p, dv.p);
    gramStridedBlock<P, double><<<kLanes, 128>>>(dj.p, R, dplane.p, kLanes);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<double> g = dg.get(kLanes * P * P), plane = dplane.get(kLanes * P * P);
    for (std::size_t m = 0; m < kLanes; ++m)
        for (std::size_t e = 0; e < P * P; ++e)
            ASSERT_EQ(plane[e * kLanes + m], g[m * P * P + e]) << "lane " << m << " entry " << e;
}

template<std::size_t N>
void checkQrGramDevice(std::size_t m, std::size_t k, int size)
{
    const std::vector<double> a = lanes<double>([m](std::size_t l) { return randomVector<double>(m * N, 360 + l); });
    const std::vector<double> b = lanes<double>([k](std::size_t l) { return randomVector<double>(N * k, 370 + l); });
    Dev<double> da(a.size()), db(b.size()), df(a.size()), dtau(kLanes * N), dwork(kLanes * tl::qrWorkSize(m, N)), dx(kLanes * N * k);
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    db.put(b);
    qrGramBlock<N, double><<<kLanes, size>>>(da.p, db.p, m, k, df.p, dtau.p, dwork.p, di.p, dx.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<double> x = dx.get(kLanes * N * k);
    for (std::size_t l = 0; l < kLanes; ++l) {
        const std::vector<double> al_ = slice(a, l, m * N), bl = slice(b, l, N * k);
        std::vector<Real> g(N * N, 0);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t r = 0; r < m; ++r)
                    g[i * N + j] += static_cast<Real>(al_[r * N + i]) * static_cast<Real>(al_[r * N + j]);
        for (std::size_t c = 0; c < k; ++c) {
            std::vector<Real> rhs(N);
            for (std::size_t i = 0; i < N; ++i)
                rhs[i] = bl[i * k + c];
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
                xc[i] = x[l * N * k + i * k + c];
            EXPECT_LE(forwardError(xc, rhs), 500.0 * N * eps<double>()) << "N=" << N << " k=" << k << " lane " << l << " col " << c << " team " << size;
        }
    }
}

TEST(LinalgTeamDevice, QrGramSolveMatchesALongDoubleReference)
{
    for (int size : {32, 64, 128}) {
        checkQrGramDevice<33>(100, 3, size);
        checkQrGramDevice<71>(210, 4, size);
    }
}

TEST(LinalgTeamDevice, BlockSumRowsPerSampleEqualsTheExpandedRowListBitForBit)
{
    constexpr std::size_t P = 9, M = 4, samples = 30;
    const std::vector<double> j = randomVector<double>(samples * M * P, 191);
    const std::vector<std::int32_t> index{27, 3, 14, 15, 0, 22, 9};
    std::vector<std::int32_t> expanded;
    for (std::int32_t s : index)
        for (std::int32_t q = 0; q < static_cast<std::int32_t>(M); ++q)
            expanded.push_back(s * static_cast<std::int32_t>(M) + q);
    Dev<double> dj(j.size()), dg(kLanes * P * P), de(kLanes * P * P);
    Dev<std::int32_t> di(index.size()), dx(expanded.size());
    dj.put(j);
    di.put(index);
    dx.put(expanded);
    std::vector<double> ref;
    for (int size : {32, 64, 256}) {
        gramRowsPerBlock<P, double><<<kLanes, size>>>(dj.p, di.p, index.size(), M, dg.p);
        gramRowsPerBlock<P, double><<<kLanes, size>>>(dj.p, dx.p, expanded.size(), 1, de.p);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const std::vector<double> g = dg.get(kLanes * P * P), e = de.get(kLanes * P * P);
        EXPECT_TRUE(bitwiseEqual(g, e)) << "rowsPer vs expanded, team " << size;
        if (ref.empty())
            ref = g;
        else
            EXPECT_TRUE(bitwiseEqual(g, ref)) << "across team sizes, team " << size;
        EXPECT_TRUE(bitwiseEqual(slice(g, 1, P * P), slice(g, 0, P * P))) << "lanes see the same rows";
    }
}

} // namespace
} // namespace linalg_team
} // namespace aether_tests
