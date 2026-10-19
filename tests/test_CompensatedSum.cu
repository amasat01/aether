// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Compensated-sum building blocks: host checks (test_CompensatedSum_common.h)
// plus the warp butterfly against its host twin, bitwise.

#include "test_CompensatedSum_common.h"

namespace aether_tests {
namespace comp_sum {
namespace {

namespace at = aether::accum::atomic::detail;

// Layout: hi[lane * N + k]. Each lane writes its owned columns.
template<int N, class T>
AETHER_KERNEL()
void butterflyKernel(const T* hi, const T* comp, T* outHi, T* outComp)
{
    const int lane = threadIdx.x;
    T h[N], c[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        h[k] = hi[lane * N + k];
        c[k] = comp[lane * N + k];
    }
    at::butterfly32_<N>(h, c, lane);
    int lo, m;
    at::butterflyOwned_(lane, N, lo, m);
    for (int j = 0; j < m; ++j) {
        outHi[lo + j]   = h[j];
        outComp[lo + j] = c[j];
    }
}

template<int N, class T>
void checkButterfly(int activeLanes)
{
    std::mt19937 rng(100 + N + activeLanes);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<T> hi(32 * N, T(0)), comp(32 * N, T(0));
    for (int l = 0; l < activeLanes; ++l)
        for (int k = 0; k < N; ++k) {
            const T v        = static_cast<T>(d(rng) * std::pow(2.0, (l + k) % 12 - 6));
            hi[l * N + k]    = v;
            comp[l * N + k]  = static_cast<T>(d(rng) * 1e-7) * v;
        }
    T *dHi, *dComp, *dOutHi, *dOutComp;
    ASSERT_EQ(cudaMalloc(&dHi, hi.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dComp, comp.size() * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOutHi, N * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOutComp, N * sizeof(T)), cudaSuccess);
    cudaMemcpy(dHi, hi.data(), hi.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dComp, comp.data(), comp.size() * sizeof(T), cudaMemcpyHostToDevice);
    butterflyKernel<N, T><<<1, 32>>>(dHi, dComp, dOutHi, dOutComp);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<T> oh(N), oc(N);
    cudaMemcpy(oh.data(), dOutHi, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(oc.data(), dOutComp, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(dHi);
    cudaFree(dComp);
    cudaFree(dOutHi);
    cudaFree(dOutComp);

    for (int k = 0; k < N; ++k) {
        T ch[32], cc[32], eh, ec;
        for (int l = 0; l < 32; ++l) {
            ch[l] = hi[l * N + k];
            cc[l] = comp[l * N + k];
        }
        at::reduce32Host_(ch, cc, eh, ec);
        EXPECT_TRUE(sameBits(oh[k], eh) && sameBits(oc[k], ec))
            << "N " << N << " active " << activeLanes << " column " << k;
    }
}

template<class T>
void checkAllWidths()
{
    for (int active : { 32, 19 }) {
        checkButterfly<1, T>(active);
        checkButterfly<7, T>(active);
        checkButterfly<32, T>(active);
        checkButterfly<80, T>(active);
    }
}

TEST(CompensatedSumButterfly, DeviceEqualsHostTwinFloat) { checkAllWidths<float>(); }
TEST(CompensatedSumButterfly, DeviceEqualsHostTwinDouble) { checkAllWidths<double>(); }

} // namespace
} // namespace comp_sum
} // namespace aether_tests
