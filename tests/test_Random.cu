// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// random/ tests (CUDA build). Paired with test_Random.cpp, which registers
// the same `RandomTest.*` names.
//
// The one place the two files deliberately diverge: every statistical /
// behavioural test below is byte-identical to test_Random.cpp's and runs
// on the host in both, since aether::random has no host/device algorithm
// split at all (one Philox, both arms), so re-running the same statistics
// over the same code would prove nothing the .cpp run has not already
// proven. What this file must prove instead is that the device compilation
// of that one algorithm produces the same numbers — so the two `Golden_*`
// tests below evaluate the stream inside a `__global__` kernel, download
// the results, and check them against the same committed golden rows the
// host tests use, plus against a host recomputation in the same binary.
// That is what makes cross-mode bit-identity a run rather than a claim,
// and it is why `Golden_BitsUniformExactBothModes` carries "BothModes" in
// its name.
//
// 20 named cases: known-answer statistics for uniform, normal, lognormal,
// exponential, bernoulli, uniformInt and multivariateNormal draws, four
// CPU-SIMD packet-equivalence checks, a free-function one-liner check, plus
// the two `Golden_*` device-vs-host bit-identity tests.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "test_Random_common.h"

namespace aether_tests {
namespace {

using aether::SampleIndex;
using aether::Vec3d;
using aether::random::Generator;
namespace rnd = aether::random::detail;

TEST_F(RandomTest, Determinism_SameSeedSameOutput)
{
    const Generator rng(SEED);
    auto fn = [rng](auto& v, SampleIndex i) { v[i] = rng.normal<double, 3>(0.0, 1.0); };

    aether::Array<double, 3> a(kN);
    aether::Array<double, 3> b(kN);
    fillSerial(a, kN, fn);
    fillSerial(b, kN, fn);

    expectIdentical(a.hostView(), b.hostView(), kN);
}

TEST_F(RandomTest, OrderIndependence)
{
    // A parallel fill and a plain serial fill agreeing BIT-FOR-BIT is the
    // proof that no draw depends on shared state or on visit order.
    const Generator rng(SEED);
    auto fn = [rng](auto& v, SampleIndex i) { v[i] = rng.normal<double, 3>(0.0, 1.0); };

    aether::Array<double, 3> par(kN);
    aether::Array<double, 3> ser(kN);
    fillParallel(par, kN, fn);
    fillSerial(ser, kN, fn);

    expectIdentical(par.hostView(), ser.hostView(), kN);
}

TEST_F(RandomTest, Uniform_MomentsInTolerance)
{
    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.uniform<double, 3>(-2.0, 5.0); });
    checkUniformMoments(a.hostView(), kN, -2.0, 5.0);
}

TEST_F(RandomTest, Normal_MomentsInTolerance)
{
    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.normal<double, 3>(2.0, 0.5); });
    checkNormalMoments(a.hostView(), kN, 2.0, 0.5);
}

TEST_F(RandomTest, Normal_PerComponentParams)
{
    // Per-component parameters: an Item<double,3> mean / sd drives each
    // component's own moments (scalar or Item, by value).
    const Vec3d mean(-1.0, 0.0, 3.0);
    const Vec3d sd(0.25, 1.0, 2.0);
    const Generator rng(SEED);

    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng, mean, sd](auto& v, SampleIndex i) { v[i] = rng.normal<double, 3>(mean, sd); });

    auto view = a.hostView();
    const double mv[3] = { -1.0, 0.0, 3.0 };
    const double sv[3] = { 0.25, 1.0, 2.0 };
    for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        double s2 = 0.0;
        for (std::size_t k = 0; k < kN; ++k) {
            const double x = comp(view, k, d);
            s += x;
            s2 += x * x;
        }
        const double m = s / static_cast<double>(kN);
        const double var = s2 / static_cast<double>(kN) - m * m;
        EXPECT_NEAR(m, mv[d], 6.0 * sv[d] / std::sqrt(static_cast<double>(kN)));
        EXPECT_NEAR(var, sv[d] * sv[d], 8.0 * sv[d] * sv[d] * std::sqrt(2.0 / static_cast<double>(kN)));
    }
}

TEST_F(RandomTest, Lognormal_MeanInTolerance)
{
    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.lognormal<double, 3>(0.0, 0.5); });
    checkLognormalMean(a.hostView(), kN, 0.0, 0.5);
}

TEST_F(RandomTest, Exponential_MeanInTolerance)
{
    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.exponential<double, 3>(2.0); });
    checkExponentialMean(a.hostView(), kN, 2.0);
}

TEST_F(RandomTest, Bernoulli_Rate)
{
    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.bernoulli<double, 3>(0.3); });
    checkBernoulliRate(a.hostView(), kN, 0.3);
}

TEST_F(RandomTest, UniformInt_Range)
{
    const Generator rng(SEED);
    aether::Array<int, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.uniformInt<int, 3>(0, 9); });
    checkUniformIntRange(a.hostView(), kN, 0, 9);
}

TEST_F(RandomTest, MultivariateNormal_CovarianceInTolerance)
{
    const double cov[3][3] = { { 4, 1, 0 }, { 1, 3, 1 }, { 0, 1, 2 } };
    const double meanv[3] = { 10.0, -5.0, 2.0 };
    const auto L = aether::random::cholesky<double, 3>(cov);
    const Vec3d mean(meanv[0], meanv[1], meanv[2]);

    const Generator rng(SEED);
    aether::Array<double, 3> a(kN);
    fillSerial(a, kN, [rng, mean, L](auto& v, SampleIndex i) { v[i] = rng.multivariateNormal<double, 3>(mean, L); });
    checkCovariance(a.hostView(), kN, meanv, cov);
}

TEST_F(RandomTest, DifferentSeedsDiffer)
{
    const Generator r1(SEED);
    const Generator r2(SEED + 1);

    aether::Array<double, 3> a(kN);
    aether::Array<double, 3> b(kN);
    fillSerial(a, kN, [r1](auto& v, SampleIndex i) { v[i] = r1.normal<double, 3>(0.0, 1.0); });
    fillSerial(b, kN, [r2](auto& v, SampleIndex i) { v[i] = r2.normal<double, 3>(0.0, 1.0); });

    auto va = a.hostView();
    auto vb = b.hostView();
    std::size_t diff = 0;
    for (std::size_t k = 0; k < kN; ++k)
        for (int d = 0; d < 3; ++d)
            if (comp(va, k, d) != comp(vb, k, d))
                ++diff;
    EXPECT_GT(diff, kN); // overwhelmingly different
}

TEST_F(RandomTest, LocalConstruction)
{
    const Generator rng(SEED);

    // Index-free construction from the same generator is deterministic.
    const Vec3d a = rng.normal<double, 3>(0.0, 1.0);
    const Vec3d a2 = rng.normal<double, 3>(0.0, 1.0);
    for (std::size_t d = 0; d < 3; ++d)
        EXPECT_EQ(a(d), a2(d));

    // substream(k) yields a distinct local draw.
    const Vec3d b = rng.substream(1).normal<double, 3>(0.0, 1.0);
    EXPECT_NE(a(std::size_t{ 0 }), b(std::size_t{ 0 }));

    // The three accessors. `get<I>(offset_t)` must agree with
    // `get<I>(SampleIndex)` at the same sample; the index-free `get<I>()`
    // lives in its own domain (INDEX_FREE_DOMAIN), so it must not collide
    // with the indexed draw at sample 0 — that separation is the whole
    // reason a local `Vec3d v = rng.normal(...)` is safe next to an array
    // fill.
    const auto leaf = rng.normal<double, 3>(0.0, 1.0);
    EXPECT_EQ(leaf.get<1>(SampleIndex::make(std::size_t{ 5 })), leaf.get<1>(aether::offset_t{ 5 }));
    EXPECT_EQ(leaf.get<1>(), leaf.get<1>());
    EXPECT_NE(leaf.get<1>(), leaf.get<1>(SampleIndex::make(std::size_t{ 0 })));

    // `eval` is the canonical spelling; `get` is the alias.
    EXPECT_EQ(leaf.eval<2>(SampleIndex::make(std::size_t{ 7 })), leaf.get<2>(SampleIndex::make(std::size_t{ 7 })));

    // Index-free multivariateNormal into a local Vec3d, reproducible.
    const double cov[3][3] = { { 4, 1, 0 }, { 1, 3, 1 }, { 0, 1, 2 } };
    const auto L = aether::random::cholesky<double, 3>(cov);
    const Vec3d mvnMean(1.0, 2.0, 3.0);
    const Vec3d z = rng.multivariateNormal<double, 3>(mvnMean, L);
    const Vec3d z2 = rng.multivariateNormal<double, 3>(mvnMean, L);
    for (std::size_t d = 0; d < 3; ++d)
        EXPECT_EQ(z(d), z2(d));
}

/* --- CPU-SIMD packet equivalence + ambient free functions ---- */

TEST_F(RandomTest, PacketEquivalence_Normal)
{
    const std::size_t pn = 4099; // not a multiple of any native packet width -> exercises the masked tail
    const Generator rng(SEED);

    aether::Array<double, 3> s(pn);
    fillSerial(s, pn, [rng](auto& v, SampleIndex i) { v[i] = rng.normal<double, 3>(1.0, 2.0); });

    aether::Array<double, 3> p(pn);
    auto pv = p.hostView();
    aether::packetEval(pv, rng.normal<double, 3>(1.0, 2.0));

    expectIdentical(s.hostView(), p.hostView(), pn);
}

TEST_F(RandomTest, PacketEquivalence_Uniform)
{
    const std::size_t pn = 4099;
    const Generator rng(SEED);

    aether::Array<double, 3> s(pn);
    fillSerial(s, pn, [rng](auto& v, SampleIndex i) { v[i] = rng.uniform<double, 3>(-3.0, 4.0); });

    aether::Array<double, 3> p(pn);
    auto pv = p.hostView();
    aether::packetEval(pv, rng.uniform<double, 3>(-3.0, 4.0));

    expectIdentical(s.hostView(), p.hostView(), pn);
}

TEST_F(RandomTest, PacketEquivalence_Composed)
{
    const std::size_t pn = 4099;
    aether::Array<double, 3> base(pn);
    {
        auto b = base.hostView();
        for (std::size_t k = 0; k < pn; ++k) {
            b(std::size_t{ 0 }, k) = static_cast<double>(k) * 1.0;
            b(std::size_t{ 1 }, k) = static_cast<double>(k) * 2.0;
            b(std::size_t{ 2 }, k) = static_cast<double>(k) * 3.0;
        }
    }
    const auto bview = base.hostView();
    const Generator rng(SEED);

    aether::Array<double, 3> s(pn);
    fillSerial(s, pn, [rng, bview](auto& v, SampleIndex i) { v[i] = bview + rng.normal<double, 3>(0.0, 1.0); });

    aether::Array<double, 3> p(pn);
    auto pv = p.hostView();
    aether::packetEval(pv, bview + rng.normal<double, 3>(0.0, 1.0));

    expectIdentical(s.hostView(), p.hostView(), pn);
}

TEST_F(RandomTest, PacketEquivalence_MultivariateNormal)
{
    const std::size_t pn = 4099;
    const double cov[3][3] = { { 4, 1, 0 }, { 1, 3, 1 }, { 0, 1, 2 } };
    const auto L = aether::random::cholesky<double, 3>(cov);
    const Vec3d mean(10.0, -5.0, 2.0);
    const Generator rng(SEED);

    aether::Array<double, 3> s(pn);
    fillSerial(s, pn, [rng, mean, L](auto& v, SampleIndex i) { v[i] = rng.multivariateNormal<double, 3>(mean, L); });

    aether::Array<double, 3> p(pn);
    auto pv = p.hostView();
    aether::packetEval(pv, rng.multivariateNormal<double, 3>(mean, L));

    expectIdentical(s.hostView(), p.hostView(), pn);
}

TEST_F(RandomTest, FreeFunctions)
{
    namespace fr = aether::random;
    const std::size_t pn = 512;

    aether::Array<double, 3> a(pn);
    fillSerial(a, pn, [](auto& v, SampleIndex i) { v[i] = fr::normal<double, 3>(0.5, 2.0); });
    aether::Array<double, 3> b(pn);
    fillSerial(b, pn, [](auto& v, SampleIndex i) { v[i] = fr::Generator(fr::DEFAULT_SEED).normal<double, 3>(0.5, 2.0); });
    expectIdentical(a.hostView(), b.hostView(), pn);

    aether::Array<double, 3> c(pn);
    fillSerial(c, pn, [](auto& v, SampleIndex i) { v[i] = fr::seed(777).normal<double, 3>(0.0, 1.0); });
    aether::Array<double, 3> d(pn);
    fillSerial(d, pn, [](auto& v, SampleIndex i) { v[i] = fr::Generator(777).normal<double, 3>(0.0, 1.0); });
    expectIdentical(c.hostView(), d.hostView(), pn);
}

/* --------------------------- FP32 (float Real) --------------------------- */

TEST_F(RandomTest, Float_DeterminismAndOrderIndependence)
{
    const Generator rng(SEED);
    auto fn = [rng](auto& v, SampleIndex i) { v[i] = rng.normal<float, 3>(0.0f, 1.0f); };

    aether::Array<float, 3> a(kN);
    aether::Array<float, 3> b(kN);
    fillSerial(a, kN, fn);
    fillParallel(b, kN, fn);

    expectIdentical(a.hostView(), b.hostView(), kN);
}

TEST_F(RandomTest, Float_Uniform_MomentsInTolerance)
{
    const Generator rng(SEED);
    aether::Array<float, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.uniform<float, 3>(-1.0f, 3.0f); });
    checkUniformMoments(a.hostView(), kN, -1.0, 3.0);
}

TEST_F(RandomTest, Float_Normal_MomentsInTolerance)
{
    const Generator rng(SEED);
    aether::Array<float, 3> a(kN);
    fillSerial(a, kN, [rng](auto& v, SampleIndex i) { v[i] = rng.normal<float, 3>(2.0f, 0.5f); });
    checkNormalMoments(a.hostView(), kN, 2.0, 0.5);
}

} // namespace

/* ------------------------- golden-stream anchors -------------------------- */

/** @brief Device evaluation of the three core primitives, one row per thread. */
__global__ void goldenCoreKernel(const std::uint64_t* seeds, const std::uint32_t* globals, const std::uint64_t* counters,
    std::size_t n, std::uint64_t* bits, double* uniformD, float* uniformF, double* normalD, float* normalF)
{
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (k >= n)
        return;
    const std::uint64_t seed   = seeds[k];
    const aether::offset_t glb = globals[k];
    const std::uint64_t ctr    = counters[k];
    bits[k]     = rnd::randomBits64(seed, glb, ctr);
    uniformD[k] = rnd::uniform01<double>(seed, glb, ctr);
    uniformF[k] = rnd::uniform01<float>(seed, glb, ctr);
    normalD[k]  = rnd::standardNormal<double>(seed, glb, ctr);
    normalF[k]  = rnd::standardNormal<float>(seed, glb, ctr);
}

/** @brief Device evaluation of the keying chain, one row per thread. */
__global__ void goldenKeyKernel(const std::uint64_t* seeds, const std::uint64_t* streamIds, const std::uint64_t* ks,
    std::size_t n, std::uint64_t* keys, std::uint64_t* subKeys)
{
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (k >= n)
        return;
    const Generator g(seeds[k], streamIds[k]);
    keys[k]    = g.key();
    subKeys[k] = g.substream(ks[k]).key();
}

namespace {

/** @brief `cudaMalloc` + `cudaMemcpy` H->D of `n` elements; null on failure. */
template<class T>
T* deviceCopyOf(const T* host, std::size_t n)
{
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
        return nullptr;
    if (cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(p);
        return nullptr;
    }
    return static_cast<T*>(p);
}

/** @brief `cudaMalloc` of `n` uninitialised elements; null on failure. */
template<class T>
T* deviceAlloc(std::size_t n)
{
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
        return nullptr;
    return static_cast<T*>(p);
}

/** @brief Everything the golden kernels need, allocated once per test body.
 *         Frees in the destructor so an ASSERT_* early-exit cannot leak. */
struct GoldenDeviceRun {
    std::size_t n = 0;
    std::uint64_t* seeds = nullptr;
    std::uint32_t* globals = nullptr;
    std::uint64_t* counters = nullptr;
    std::uint64_t* bits = nullptr;
    double* uniformD = nullptr;
    float* uniformF = nullptr;
    double* normalD = nullptr;
    float* normalF = nullptr;

    ~GoldenDeviceRun()
    {
        cudaFree(seeds);
        cudaFree(globals);
        cudaFree(counters);
        cudaFree(bits);
        cudaFree(uniformD);
        cudaFree(uniformF);
        cudaFree(normalD);
        cudaFree(normalF);
    }

    bool allocated() const
    {
        return seeds && globals && counters && bits && uniformD && uniformF && normalD && normalF;
    }
};

/**
 * @brief Run `goldenCoreKernel` over every golden row and copy the five
 *        result buffers back into the supplied host vectors.
 *
 * Returns false only when a CUDA allocation failed; a failed LAUNCH throws
 * through `aether::cuda::checkLastLaunch` (a kernel that never started
 * otherwise reports as a silent value divergence).
 */
bool runGoldenCoreOnDevice(std::vector<std::uint64_t>& bits, std::vector<double>& uniformD,
    std::vector<float>& uniformF, std::vector<double>& normalD, std::vector<float>& normalF)
{
    const std::size_t n = golden::kCoreRowCount;

    std::vector<std::uint64_t> hSeeds(n);
    std::vector<std::uint32_t> hGlobals(n);
    std::vector<std::uint64_t> hCounters(n);
    for (std::size_t r = 0; r < n; ++r) {
        hSeeds[r]    = golden::kCoreRows[r].seed;
        hGlobals[r]  = golden::kCoreRows[r].global;
        hCounters[r] = golden::kCoreRows[r].counter;
    }

    GoldenDeviceRun d;
    d.n        = n;
    d.seeds    = deviceCopyOf(hSeeds.data(), n);
    d.globals  = deviceCopyOf(hGlobals.data(), n);
    d.counters = deviceCopyOf(hCounters.data(), n);
    d.bits     = deviceAlloc<std::uint64_t>(n);
    d.uniformD = deviceAlloc<double>(n);
    d.uniformF = deviceAlloc<float>(n);
    d.normalD  = deviceAlloc<double>(n);
    d.normalF  = deviceAlloc<float>(n);
    if (!d.allocated())
        return false;

    constexpr unsigned int blockDimX = 128u;
    const unsigned int numBlocks     = static_cast<unsigned int>((n + blockDimX - 1u) / blockDimX);
    goldenCoreKernel<<<numBlocks, blockDimX>>>(
        d.seeds, d.globals, d.counters, n, d.bits, d.uniformD, d.uniformF, d.normalD, d.normalF);
    aether::cuda::checkLastLaunch("goldenCoreKernel");

    bits.resize(n);
    uniformD.resize(n);
    uniformF.resize(n);
    normalD.resize(n);
    normalF.resize(n);
    if (cudaMemcpy(bits.data(), d.bits, n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (cudaMemcpy(uniformD.data(), d.uniformD, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (cudaMemcpy(uniformF.data(), d.uniformF, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (cudaMemcpy(normalD.data(), d.normalD, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (cudaMemcpy(normalF.data(), d.normalF, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return true;
}

TEST_F(RandomTest, Golden_BitsUniformExactBothModes)
{
    // BOTH MODES, in one run: the DEVICE compilation of aether's Philox is
    // evaluated by a kernel and compared against (a) the committed golden rows
    // — the same rows test_Random.cpp checks the host arm against, minted by
    // tools/random/mint_golden.sh and certified against libcurand by
    // tools/random/curand_host_probe.cpp — and (b) a HOST recomputation in
    // this same binary. EXACT: the integer and uniform primitives are pure
    // integer work plus one contraction-proof scaling, so no optimisation
    // level, ISA or fp-contract setting may move them, on either arm.
    ASSERT_EQ(sizeof(golden::kCoreRows) / sizeof(golden::kCoreRows[0]), golden::kCoreRowCount);
    ASSERT_GT(golden::kCoreRowCount, 0u);

    std::vector<std::uint64_t> devBits;
    std::vector<double> devUniformD;
    std::vector<float> devUniformF;
    std::vector<double> devNormalD;
    std::vector<float> devNormalF;
    ASSERT_TRUE(runGoldenCoreOnDevice(devBits, devUniformD, devUniformF, devNormalD, devNormalF))
        << "CUDA allocation/copy failed for the golden device run";

    for (std::size_t r = 0; r < golden::kCoreRowCount; ++r) {
        const golden::CoreRow& row = golden::kCoreRows[r];
        SCOPED_TRACE(testing::Message() << "core row " << r << " seed=" << row.seed << " global=" << row.global
                                        << " counter=" << row.counter);
        // Device vs golden.
        EXPECT_EQ(devBits[r], row.bits64);
        EXPECT_EQ(bitsOfDouble(devUniformD[r]), row.uniformDoubleBits);
        EXPECT_EQ(bitsOfFloat(devUniformF[r]), row.uniformFloatBits);
        // Host vs golden (and hence host vs device).
        EXPECT_EQ(rnd::randomBits64(row.seed, row.global, row.counter), row.bits64);
        EXPECT_EQ(bitsOfDouble(rnd::uniform01<double>(row.seed, row.global, row.counter)), row.uniformDoubleBits);
        EXPECT_EQ(bitsOfFloat(rnd::uniform01<float>(row.seed, row.global, row.counter)), row.uniformFloatBits);
    }

    // The keying chain (SplitMix64 mixing + the substream salt) is part of the
    // stream's identity: a generator whose key moved draws a different stream
    // from the same seed. Checked on the device too — `Generator` is passed BY
    // VALUE into kernels, so its arithmetic must be device-legal.
    ASSERT_EQ(sizeof(golden::kKeyRows) / sizeof(golden::kKeyRows[0]), golden::kKeyRowCount);
    const std::size_t kn = golden::kKeyRowCount;
    std::vector<std::uint64_t> hSeeds(kn);
    std::vector<std::uint64_t> hStreamIds(kn);
    std::vector<std::uint64_t> hKs(kn);
    for (std::size_t r = 0; r < kn; ++r) {
        hSeeds[r]     = golden::kKeyRows[r].seed;
        hStreamIds[r] = golden::kKeyRows[r].streamId;
        hKs[r]        = golden::kKeyRows[r].substreamK;
    }
    std::uint64_t* dSeeds     = deviceCopyOf(hSeeds.data(), kn);
    std::uint64_t* dStreamIds = deviceCopyOf(hStreamIds.data(), kn);
    std::uint64_t* dKs        = deviceCopyOf(hKs.data(), kn);
    std::uint64_t* dKeys      = deviceAlloc<std::uint64_t>(kn);
    std::uint64_t* dSubKeys   = deviceAlloc<std::uint64_t>(kn);
    const bool keyAlloc       = dSeeds && dStreamIds && dKs && dKeys && dSubKeys;
    std::vector<std::uint64_t> keys(kn, 0ull);
    std::vector<std::uint64_t> subKeys(kn, 0ull);
    if (keyAlloc) {
        goldenKeyKernel<<<1, 64>>>(dSeeds, dStreamIds, dKs, kn, dKeys, dSubKeys);
        aether::cuda::checkLastLaunch("goldenKeyKernel");
        EXPECT_EQ(cudaMemcpy(keys.data(), dKeys, kn * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(subKeys.data(), dSubKeys, kn * sizeof(std::uint64_t), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    cudaFree(dSeeds);
    cudaFree(dStreamIds);
    cudaFree(dKs);
    cudaFree(dKeys);
    cudaFree(dSubKeys);
    ASSERT_TRUE(keyAlloc) << "CUDA allocation failed for the golden key run";

    for (std::size_t r = 0; r < kn; ++r) {
        const golden::KeyRow& row = golden::kKeyRows[r];
        SCOPED_TRACE(testing::Message() << "key row " << r);
        EXPECT_EQ(keys[r], row.key);
        EXPECT_EQ(subKeys[r], row.substreamKey);
        const Generator g(row.seed, row.streamId);
        EXPECT_EQ(g.key(), row.key);
        EXPECT_EQ(g.substream(row.substreamK).key(), row.substreamKey);
    }
}

TEST_F(RandomTest, Golden_NormalWithinTol)
{
    // TIER-TOL, not TIER-BIT, and deliberately so: Box-Muller here routes
    // sqrt/log/sincos through aether::math, whose host and device legs are
    // different implementations of the same functions (on the FP64 path, the
    // zero-stack detail::cwSinCos on device vs std::sin/std::cos on
    // host). The band is absolute AND relative (normalBand) so a tail value is
    // not held to an absolute-only bound it cannot meet.
    std::vector<std::uint64_t> devBits;
    std::vector<double> devUniformD;
    std::vector<float> devUniformF;
    std::vector<double> devNormalD;
    std::vector<float> devNormalF;
    ASSERT_TRUE(runGoldenCoreOnDevice(devBits, devUniformD, devUniformF, devNormalD, devNormalF))
        << "CUDA allocation/copy failed for the golden device run";

    for (std::size_t r = 0; r < golden::kCoreRowCount; ++r) {
        const golden::CoreRow& row = golden::kCoreRows[r];
        SCOPED_TRACE(testing::Message() << "core row " << r << " seed=" << row.seed << " global=" << row.global
                                        << " counter=" << row.counter);
        const double bandD = normalBand(row.normalDouble, golden::kNormalDoubleTol);
        const double bandF = normalBand(static_cast<double>(row.normalFloat), static_cast<double>(golden::kNormalFloatTol));
        // Device vs golden.
        EXPECT_NEAR(devNormalD[r], row.normalDouble, bandD);
        EXPECT_NEAR(static_cast<double>(devNormalF[r]), static_cast<double>(row.normalFloat), bandF);
        // Host vs golden.
        EXPECT_NEAR(rnd::standardNormal<double>(row.seed, row.global, row.counter), row.normalDouble, bandD);
        EXPECT_NEAR(static_cast<double>(rnd::standardNormal<float>(row.seed, row.global, row.counter)),
            static_cast<double>(row.normalFloat), bandF);
    }
}

} // namespace
} // namespace aether_tests
