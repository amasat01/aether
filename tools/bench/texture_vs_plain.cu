// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tools/bench/texture_vs_plain.cu
 * @brief Texture-cache bench — `TableHandle<double, Plain>` vs
 *        `TableHandle<double, Texture>`, scattered-index coefficient reads
 *        (mimicking a Chebyshev-coefficient fill kernel's own access
 *        pattern: a computed, non-sequential per-thread block of table
 *        indices, not `table[threadIdx]` — texture locality is the entire
 *        premise under test).
 *
 * Device only: unlike `tools/bandmath/bench_pair.cu`, there is no
 * meaningful host arm here — a texture object has no host-side fetch path
 * at all (`aether::TableHandle<T,Texture>`'s host-pass fallback is a
 * plain pointer read, which is not a distinct thing to bench against
 * itself; benching "plain vs plain" would measure nothing). Running it
 * and capturing its output (md5-fenced by the run) is a manual step,
 * separate from the regular build.
 *
 * This bench asserts nothing about speed — it ships the measurement, not
 * a claim.
 *
 * Reported per size `N`: the e/n slope (texture-carrier median time /
 * plain-carrier median time) and a linearity check (ns/element roughly
 * constant across sizes — a departure says the access pattern is hitting a
 * cache-capacity or launch-overhead knee, not a clean per-element cost).
 *
 * Usage:
 *   nvcc -arch=sm_61 -std=c++20 -O3 -DAETHER_HAS_CUDA=1 -I<repo root> \
 *        tools/bench/texture_vs_plain.cu -o texture_vs_plain
 *   CUDA_VISIBLE_DEVICES=<n> ./texture_vs_plain
 */
#include <aether/aether.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using aether::Chunk;
using aether::Device;
using aether::Plain;
using aether::TableHandle;
using aether::Texture;
using aether::TextureBinding;

namespace {

/** @brief Table size the scattered reads are drawn from — a "coefficient
 *  block" shape (the kind of per-body Chebyshev block a fill kernel
 *  reads), comfortably larger than a texture cache line so locality is a
 *  real question, not a triviality. */
constexpr int kTableSize = 4096;
/** @brief Reads per thread — mimics a FILL kernel touching several
 *  coefficients (one interval's worth) per sample, not just one. */
constexpr int kBlock = 8;

__global__ void plainScatterKernel(
    aether::TableHandle<double, Plain> table, const int* __restrict__ idx, double* __restrict__ out, int nThreads)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nThreads)
        return;
    double acc = 0.0;
    for (int k = 0; k < kBlock; ++k)
        acc += table[static_cast<aether::offset_t>(idx[t * kBlock + k])];
    out[t] = acc;
}

__global__ void texScatterKernel(
    aether::TableHandle<double, Texture> table, const int* __restrict__ idx, double* __restrict__ out, int nThreads)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nThreads)
        return;
    double acc = 0.0;
    for (int k = 0; k < kBlock; ++k)
        acc += table[static_cast<aether::offset_t>(idx[t * kBlock + k])];
    out[t] = acc;
}

/** @brief Median of `reps` timed launches of `launch()`, after `warmups`
 *  untimed ones — mirrors `tools/bandmath/bench_pair.cu`'s own
 *  `timeKernel` idiom. */
template<class Launch>
float medianKernelMs(Launch launch, int warmups, int reps)
{
    for (int w = 0; w < warmups; ++w)
        launch();
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    std::vector<float> times(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(t0);
        launch();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        times[static_cast<std::size_t>(r)] = ms;
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void runOneSize(int nThreads, std::mt19937_64& rng, const Chunk& table, const TableHandle<double, Plain>& plain,
    const TableHandle<double, Texture>& tex)
{
    std::uniform_int_distribution<int> idxDist(0, kTableSize - 1);
    std::vector<int> hIdx(static_cast<std::size_t>(nThreads) * kBlock);
    for (auto& v : hIdx)
        v = idxDist(rng);

    Chunk idxStage = Chunk::allocate(Device(kDLCUDAHost), hIdx.size() * sizeof(int));
    std::memcpy(idxStage.data(), hIdx.data(), hIdx.size() * sizeof(int));
    Chunk idxDevice = Chunk::allocate(Device(kDLCUDA), hIdx.size() * sizeof(int));
    aether::copy(idxDevice, idxStage);

    Chunk outDevice = Chunk::allocate(Device(kDLCUDA), static_cast<std::size_t>(nThreads) * sizeof(double));

    const int block = 256;
    const int grid = (nThreads + block - 1) / block;
    const int warmups = 3, reps = 9;

    const auto* idxPtr = reinterpret_cast<const int*>(idxDevice.data());
    auto* outPtr = reinterpret_cast<double*>(outDevice.data());

    const float plainMs = medianKernelMs(
        [&] { plainScatterKernel<<<grid, block>>>(plain, idxPtr, outPtr, nThreads); }, warmups, reps);
    const float texMs = medianKernelMs(
        [&] { texScatterKernel<<<grid, block>>>(tex, idxPtr, outPtr, nThreads); }, warmups, reps);

    const double plainNsPerElem = static_cast<double>(plainMs) * 1e6 / (static_cast<double>(nThreads) * kBlock);
    const double texNsPerElem = static_cast<double>(texMs) * 1e6 / (static_cast<double>(nThreads) * kBlock);
    const double eOverN = (plainMs > 0.0f) ? static_cast<double>(texMs) / static_cast<double>(plainMs) : 0.0;

    std::printf("nThreads=%-8d plain=%.4f ms (%.3f ns/elem)  texture=%.4f ms (%.3f ns/elem)  e/n=%.3fx\n",
        nThreads, plainMs, plainNsPerElem, texMs, texNsPerElem, eOverN);

    (void)table;
}

void benchDeviceArm()
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("=== texture_vs_plain bench — DEVICE ARM ===\n");
    std::printf("card: %s  major=%d minor=%d  SM count=%d  clock=%d kHz\n", prop.name, prop.major, prop.minor,
        prop.multiProcessorCount, prop.clockRate);
    std::printf("table size=%d doubles, block=%d reads/thread (scattered, Chebyshev coefficient-block shape)\n",
        kTableSize, kBlock);
    std::printf("NOTE: this bench asserts NOTHING about speed — it ships the "
                "measurement, not a claim.\n");

    std::mt19937_64 rng(0xA57EA5BE7A0B1EULL);
    std::vector<double> hTable(kTableSize);
    for (int i = 0; i < kTableSize; ++i)
        hTable[i] = static_cast<double>(i) * 0.001 - 2.0;

    Chunk tableStage = Chunk::allocate(Device(kDLCUDAHost), hTable.size() * sizeof(double));
    std::memcpy(tableStage.data(), hTable.data(), hTable.size() * sizeof(double));
    Chunk tableDevice = Chunk::allocate(Device(kDLCUDA), hTable.size() * sizeof(double));
    aether::copy(tableDevice, tableStage);

    const auto v = aether::make_view<double, aether::dyn>(tableDevice, static_cast<std::size_t>(kTableSize));
    TableHandle<double, Plain> plain(v);
    TextureBinding<double> binding = TextureBinding<double>::bind(tableDevice, static_cast<std::size_t>(kTableSize));
    if (!binding.valid()) {
        std::fprintf(stderr, "texture_vs_plain: TextureBinding::bind failed to produce a valid binding\n");
        return;
    }
    TableHandle<double, Texture> tex(v.data(), binding.handle(), 0);

    const int sizes[] = { 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20 };
    for (int n : sizes)
        runOneSize(n, rng, tableDevice, plain, tex);
}

} // namespace

int main()
{
    // This bench is CUDA-only by construction (see file docstring: there is
    // no meaningful host arm for a texture-object comparison) — build with
    // `-DAETHER_HAS_CUDA=1`, matching the usage line above. Unlike
    // `tools/bandmath/bench_pair.cu`, there is no `AETHER_HAS_CUDA`-gated
    // fallback here: every symbol in this file already requires CUDA to
    // exist at all.
    benchDeviceArm();
    return 0;
}
