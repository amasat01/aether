// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Linalg.cu
 * @brief `aether::linalg` (CUDA mode): the shared host battery, plus the same
 *        factorisations run one lane per thread and compared with the host.
 *
 * The device may contract products into FMAs where the host build does not, so
 * values agree to a few ulps; `info` and the zero fill of a failed lane agree
 * exactly. See `test_Linalg_common.h` for the host battery.
 */

#include "test_Linalg_common.h"

namespace aether_tests {
namespace linalg {
namespace {

constexpr std::size_t kLanes = 96;
constexpr std::size_t kN = 5;

template<class T>
AETHER_KERNEL()
void luLanes(const T* a, const T* b, T* factor, std::int32_t* info, T* x, T* det)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kLanes) {
        const al::Lu<T, kN> f = al::lu<kN>(a + i * kN * kN);
        for (std::size_t k = 0; k < kN * kN; ++k)
            factor[i * kN * kN + k] = f.a[k];
        info[i] = f.info;
        al::luSolve(f, b + i * kN, x + i * kN);
        det[i] = al::luDet(f);
    }
}

template<class T>
AETHER_KERNEL()
void cholLanes(const T* a, const T* b, T* factor, std::int32_t* info, T* x)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kLanes) {
        const al::Cholesky<T, kN> f = al::cholesky<kN>(a + i * kN * kN);
        al::choleskyFactor(f, factor + i * kN * kN);
        info[i] = f.info;
        al::choSolve(f, b + i * kN, x + i * kN);
    }
}

template<class T>
AETHER_KERNEL()
void eighLanes(const T* a, T* w, T* v, std::int32_t* info)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kLanes) {
        const al::Eigh<T, kN> r = al::eigh<kN>(a + i * kN * kN);
        for (std::size_t k = 0; k < kN; ++k)
            w[i * kN + k] = r.w[k];
        for (std::size_t k = 0; k < kN * kN; ++k)
            v[i * kN * kN + k] = r.v[k];
        info[i] = r.info;
    }
}

template<class T>
struct Dev {
    T* p = nullptr;
    explicit Dev(std::size_t n) { cudaMalloc(&p, n * sizeof(T)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template<class U>
    void put(const std::vector<U>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(U), cudaMemcpyHostToDevice); }
    template<class U>
    std::vector<U> get(std::size_t n) const
    {
        std::vector<U> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(U), cudaMemcpyDeviceToHost);
        return h;
    }
};

/** @brief Lanes cycle through well-posed, singular (a zero row) and not-positive-definite (a negative diagonal) inputs. */
template<class T>
std::vector<T> lanesOf(bool spd, std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<T> a;
    a.reserve(kLanes * kN * kN);
    for (std::size_t i = 0; i < kLanes; ++i) {
        Mat m = spd ? randomSpd(kN, rng) : randomMatrix(kN, rng);
        if (i % 7 == 3) {
            if (spd)
                m[2 * kN + 2] = -1.0;
            else
                for (std::size_t j = 0; j < kN; ++j)
                    m[1 * kN + j] = 0.0;
        }
        for (double v : m)
            a.push_back(static_cast<T>(v));
    }
    return a;
}

template<class T>
std::vector<T> rhsOf(std::uint64_t seed)
{
    Stream rng(seed);
    std::vector<T> b(kLanes * kN);
    for (T& v : b)
        v = static_cast<T>(rng.next());
    return b;
}

/** @brief `||A||_1 * ||A^-1||_1` of one lane's `kN x kN` matrix, in double on the host; infinite when singular. */
template<class T>
double conditionOf(const T* a)
{
    double m[kN * kN], mi[kN * kN];
    for (std::size_t i = 0; i < kN * kN; ++i)
        m[i] = static_cast<double>(a[i]);
    if (al::inv<kN>(m, mi) != 0)
        return std::numeric_limits<double>::infinity();
    const auto norm1 = [](const double* x) {
        double best = 0.0;
        for (std::size_t j = 0; j < kN; ++j) {
            double col = 0.0;
            for (std::size_t i = 0; i < kN; ++i)
                col += std::fabs(x[i * kN + j]);
            best = std::max(best, col);
        }
        return best;
    };
    return norm1(m) * norm1(mi);
}

/**
 * @brief Device against host, element by element. The bound is `max(64, kCondK * cond(lane)) * eps`
 *        (times `max(1, |host|)`), with `cond` the lane's 1-norm condition number when `lane` is given.
 *
 * The device contracts `s -= a * x` into an FMA and the host does not, so the two round
 * differently at every step, and what a solve does with a rounding difference grows with the
 * lane's condition number. The `-fmad=false` build of this test matches the host within the
 * plain 64 eps bound on every lane, which pins the cause: contraction, not the algorithm.
 * `kCondK` is the share of `cond * eps` one contraction difference per step can reach for `kN = 5`.
 */
constexpr double kCondK = 4.0;

template<class T>
void expectClose(const std::vector<T>& dev, const std::vector<T>& host, const char* what,
    const std::vector<double>& lane = {}, std::size_t width = 1)
{
    ASSERT_EQ(dev.size(), host.size());
    for (std::size_t i = 0; i < dev.size(); ++i) {
        const double scale = lane.empty() ? 0.0 : kCondK * lane[i / width];
        const double tol   = std::max(64.0, scale) * eps<T>();
        ASSERT_NEAR(static_cast<double>(dev[i]), static_cast<double>(host[i]), tol * std::max(1.0, std::fabs(static_cast<double>(host[i]))))
            << what << " " << i;
    }
}

template<class T>
std::vector<double> conditions(const std::vector<T>& a)
{
    std::vector<double> c(kLanes);
    for (std::size_t i = 0; i < kLanes; ++i)
        c[i] = conditionOf(a.data() + i * kN * kN);
    return c;
}

template<class T>
void checkLuLanes()
{
    const std::vector<T> a = lanesOf<T>(false, 101), b = rhsOf<T>(102);
    Dev<T> da(a.size()), db(b.size()), df(a.size()), dx(b.size()), dd(kLanes);
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    db.put(b);
    luLanes<T><<<2, 64>>>(da.p, db.p, df.p, di.p, dx.p, dd.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<std::int32_t> info = di.template get<std::int32_t>(kLanes);
    std::vector<T> hf(a.size()), hx(b.size()), hd(kLanes);
    for (std::size_t i = 0; i < kLanes; ++i) {
        const al::Lu<T, kN> f = al::lu<kN>(a.data() + i * kN * kN);
        EXPECT_EQ(info[i], f.info) << "lane " << i;
        std::copy(f.a, f.a + kN * kN, hf.begin() + static_cast<std::ptrdiff_t>(i * kN * kN));
        al::luSolve(f, b.data() + i * kN, hx.data() + i * kN);
        hd[i] = al::luDet(f);
    }
    const std::vector<double> cond = conditions(a);
    expectClose(df.template get<T>(a.size()), hf, "lu", cond, kN * kN);
    expectClose(dd.template get<T>(kLanes), hd, "det", cond);
    const std::vector<T> dxs = dx.template get<T>(b.size());
    expectClose(dxs, hx, "x", cond, kN);
    for (std::size_t i = 3; i < kLanes; i += 7)
        EXPECT_TRUE(allPlusZero(dxs.data() + i * kN, kN)) << "singular lane " << i << " is zero-filled on the device";
}

template<class T>
void checkCholLanes()
{
    const std::vector<T> a = lanesOf<T>(true, 111), b = rhsOf<T>(112);
    Dev<T> da(a.size()), db(b.size()), df(a.size()), dx(b.size());
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    db.put(b);
    cholLanes<T><<<2, 64>>>(da.p, db.p, df.p, di.p, dx.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<std::int32_t> info = di.template get<std::int32_t>(kLanes);
    std::vector<T> hf(a.size()), hx(b.size());
    for (std::size_t i = 0; i < kLanes; ++i) {
        const al::Cholesky<T, kN> f = al::cholesky<kN>(a.data() + i * kN * kN);
        EXPECT_EQ(info[i], f.info) << "lane " << i;
        al::choleskyFactor(f, hf.data() + i * kN * kN);
        al::choSolve(f, b.data() + i * kN, hx.data() + i * kN);
    }
    const std::vector<double> cond = conditions(a);
    expectClose(df.template get<T>(a.size()), hf, "factor", cond, kN * kN);
    const std::vector<T> dxs = dx.template get<T>(b.size());
    expectClose(dxs, hx, "x", cond, kN);
    for (std::size_t i = 3; i < kLanes; i += 7) {
        EXPECT_EQ(info[i], 3) << "lane " << i;
        EXPECT_TRUE(allPlusZero(dxs.data() + i * kN, kN)) << "failed lane " << i << " is zero-filled on the device";
    }
}

template<class T>
void checkEighLanes()
{
    Stream rng(121);
    std::vector<T> a;
    for (std::size_t i = 0; i < kLanes; ++i)
        for (double v : randomSymmetric(kN, rng))
            a.push_back(static_cast<T>(v));
    Dev<T> da(a.size()), dw(kLanes * kN), dv(a.size());
    Dev<std::int32_t> di(kLanes);
    da.put(a);
    eighLanes<T><<<2, 64>>>(da.p, dw.p, dv.p, di.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const std::vector<std::int32_t> info = di.template get<std::int32_t>(kLanes);
    std::vector<T> hw(kLanes * kN), hv(a.size());
    for (std::size_t i = 0; i < kLanes; ++i) {
        const al::Eigh<T, kN> r = al::eigh<kN>(a.data() + i * kN * kN);
        EXPECT_EQ(info[i], r.info) << "lane " << i;
        std::copy(r.w, r.w + kN, hw.begin() + static_cast<std::ptrdiff_t>(i * kN));
        std::copy(r.v, r.v + kN * kN, hv.begin() + static_cast<std::ptrdiff_t>(i * kN * kN));
    }
    expectClose(dw.template get<T>(hw.size()), hw, "w");
    expectClose(dv.template get<T>(hv.size()), hv, "v");
}

} // namespace

TEST(LinalgDevice, LuLanesMatchTheHostF64) { checkLuLanes<double>(); }
TEST(LinalgDevice, LuLanesMatchTheHostF32) { checkLuLanes<float>(); }
TEST(LinalgDevice, CholeskyLanesMatchTheHostF64) { checkCholLanes<double>(); }
TEST(LinalgDevice, CholeskyLanesMatchTheHostF32) { checkCholLanes<float>(); }
TEST(LinalgDevice, EighLanesMatchTheHostF64) { checkEighLanes<double>(); }
TEST(LinalgDevice, EighLanesMatchTheHostF32) { checkEighLanes<float>(); }

} // namespace linalg
} // namespace aether_tests
