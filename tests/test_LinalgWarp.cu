// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_LinalgWarp.cu
 * @brief `aether::linalg::warp::eighValues` against the one-thread `aether::linalg::eigh`: eigenvalues
 *        and `info` bit for bit over five kinds of 32 x 32 symmetric matrices, in float and double,
 *        and the float kernel's local-memory footprint (none).
 */

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace linalg_warp {
namespace {

namespace al = aether::linalg;

constexpr std::size_t N = 32;
constexpr std::size_t kMatrices = 1000;
constexpr int kWarpsPerBlock = 4;

template<class T>
AETHER_KERNEL()
void threadForm(const T* a, T* w, std::int32_t* info)
{
    const std::size_t m = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= kMatrices) return;
    al::Eigh<T, N> r;
    al::eigh<N>(r, a + m * N * N);
    for (std::size_t i = 0; i < N; ++i) w[m * N + i] = r.w[i];
    info[m] = r.info;
}

template<class T>
AETHER_KERNEL()
void warpForm(const T* a, T* w, std::int32_t* info)
{
    __shared__ al::warp::EighValuesScratch<T> scratch[kWarpsPerBlock];
    const std::size_t warp = threadIdx.x >> 5;
    const std::size_t m = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (m >= kMatrices) return;
    const al::warp::EighValues<T> r = al::warp::eighValues<N>(scratch[warp], a + m * N * N);
    w[m * N + (threadIdx.x & 31)] = r.w;
    if ((threadIdx.x & 31) == 0) info[m] = r.info;
}

/** @brief Five kinds: shifted SPD, indefinite, clustered spectrum, near-diagonal, badly scaled. */
std::vector<double> corpus()
{
    std::mt19937_64 gen(7);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud;
    std::vector<double> out(kMatrices * N * N);
    for (std::size_t m = 0; m < kMatrices; ++m) {
        double* a = out.data() + m * N * N;
        std::vector<double> g(N * N);
        for (auto& x : g) x = (m % 5 == 0) ? ud(gen) : nd(gen);
        std::vector<double> d(4);
        for (auto& x : d) x = nd(gen);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                const double s = 0.5 * (g[i * N + j] + g[j * N + i]);
                double v = s;
                switch (m % 5) {
                case 0: v = s + (i == j ? double(N) : 0.0); break;
                case 1: break;
                case 2: v = (i / 8 == j / 8) ? d[i / 8] + 1e-3 * s : 1e-3 * s; break;
                case 3: v = (i == j) ? d[0] + i : ((i == 3 && j == 7) || (i == 7 && j == 3) ? 1e-3 * s : 0.0); break;
                default: v = s * std::pow(10.0, -6.0 + 12.0 * i / 31.0) * std::pow(10.0, 0.5 * (-6.0 + 12.0 * j / 31.0)); break;
                }
                a[i * N + j] = v;
            }
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < i; ++j) a[i * N + j] = a[j * N + i];
    }
    return out;
}

template<class T>
void runBoth()
{
    const std::vector<double> src = corpus();
    std::vector<T> a(src.begin(), src.end());
    T *da, *dwT, *dwW;
    std::int32_t *diT, *diW;
    cudaMalloc(&da, a.size() * sizeof(T));
    cudaMalloc(&dwT, kMatrices * N * sizeof(T));
    cudaMalloc(&dwW, kMatrices * N * sizeof(T));
    cudaMalloc(&diT, kMatrices * sizeof(std::int32_t));
    cudaMalloc(&diW, kMatrices * sizeof(std::int32_t));
    cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
    threadForm<T><<<(kMatrices + 63) / 64, 64>>>(da, dwT, diT);
    warpForm<T><<<(kMatrices + kWarpsPerBlock - 1) / kWarpsPerBlock, 32 * kWarpsPerBlock>>>(da, dwW, diW);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<T> wT(kMatrices * N), wW(kMatrices * N);
    std::vector<std::int32_t> iT(kMatrices), iW(kMatrices);
    cudaMemcpy(wT.data(), dwT, wT.size() * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(wW.data(), dwW, wW.size() * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(iT.data(), diT, iT.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(iW.data(), diW, iW.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(dwT); cudaFree(dwW); cudaFree(diT); cudaFree(diW);
    EXPECT_EQ(0, std::memcmp(wT.data(), wW.data(), wT.size() * sizeof(T))) << "eigenvalues differ in bits";
    EXPECT_EQ(iT, iW);
}

} // namespace

TEST(LinalgWarp, EighValuesAreBitIdenticalToTheThreadFormInFloat)
{
    runBoth<float>();
}

TEST(LinalgWarp, EighValuesAreBitIdenticalToTheThreadFormInDouble)
{
    runBoth<double>();
}

TEST(LinalgWarp, FloatKernelKeepsNoLocalMemory)
{
    cudaFuncAttributes attr{};
    ASSERT_EQ(cudaFuncGetAttributes(&attr, warpForm<float>), cudaSuccess);
    EXPECT_EQ(attr.localSizeBytes, 0u);
}

} // namespace linalg_warp
} // namespace aether_tests
