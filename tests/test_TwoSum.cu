// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The shared two-sum rows plus the device arm: the device (Fast2Sum, integer
// compare) residual equals the host one bit for bit. See test_TwoSum_common.h.

#include "test_TwoSum_common.h"

namespace aether_tests {
namespace twosum {
namespace {

template<class E>
__global__ void residuals(const E* a, const E* b, E* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = aether::accum::twoSumErr(a[i], b[i], a[i] + b[i]);
}

template<class E>
__global__ void residualsAtomic(const E* a, const E* b, E* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = aether::accum::twoSumErrAtomic(a[i], b[i], a[i] + b[i]);
}

template<bool Atomic, class E>
void checkDevice()
{
    const auto c = corpus();
    const int n = static_cast<int>(c.size());
    std::vector<E> a(n), b(n), want(n), got(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<E>(c[i].a);
        b[i] = static_cast<E>(c[i].b);
        volatile E sv = a[i] + b[i];
        want[i] = aether::accum::twoSumErr(a[i], b[i], static_cast<E>(sv));
    }
    E *da, *db, *dout;
    ASSERT_EQ(cudaMalloc(&da, n * sizeof(E)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&db, n * sizeof(E)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, n * sizeof(E)), cudaSuccess);
    cudaMemcpy(da, a.data(), n * sizeof(E), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * sizeof(E), cudaMemcpyHostToDevice);
    if constexpr (Atomic)
        residualsAtomic<E><<<(n + 255) / 256, 256>>>(da, db, dout, n);
    else
        residuals<E><<<(n + 255) / 256, 256>>>(da, db, dout, n);
    cudaMemcpy(got.data(), dout, n * sizeof(E), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dout);
    for (int i = 0; i < n; ++i)
        ASSERT_TRUE(same(got[i], want[i])) << a[i] << " + " << b[i] << ": device " << got[i] << " host " << want[i];
}

TEST(TwoSum, DeviceResidualEqualsTheHostOneDouble) { checkDevice<false, double>(); }
TEST(TwoSum, DeviceResidualEqualsTheHostOneFloat) { checkDevice<false, float>(); }

TEST(TwoSum, DeviceAtomicFormEqualsTheHostResidualDouble) { checkDevice<true, double>(); }
TEST(TwoSum, DeviceAtomicFormEqualsTheHostResidualFloat) { checkDevice<true, float>(); }

template<class E>
__global__ void accumSeq(const E* v, E* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    E s = E(0), c = E(0);
    for (int k = 0; k < 8; ++k) {
        const E t = v[i * 8 + k], ns = s + t;
        c = c + aether::accum::twoSumErrAccum(s, t, ns);
        s = ns;
    }
    out[i] = c;
}

TEST(TwoSum, DeviceAccumulatedAccumFormEqualsTheHostPinnedForm)
{
    const auto c = corpus();
    const int n = static_cast<int>(c.size() / 8);
    std::vector<double> v(n * 8), want(n), got(n);
    for (int i = 0; i < n * 8; ++i)
        v[i] = (i & 1) ? c[i].a : c[i].b;
    for (int i = 0; i < n; ++i) {
        double s = 0, cc = 0;
        for (int k = 0; k < 8; ++k) {
            volatile double sv = s + v[i * 8 + k];
            const double ns = sv;
            cc += aether::accum::twoSumErr(s, v[i * 8 + k], ns);
            s = ns;
        }
        want[i] = cc;
    }
    double *dv, *dout;
    ASSERT_EQ(cudaMalloc(&dv, v.size() * 8), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, n * 8), cudaSuccess);
    cudaMemcpy(dv, v.data(), v.size() * 8, cudaMemcpyHostToDevice);
    accumSeq<double><<<(n + 255) / 256, 256>>>(dv, dout, n);
    cudaMemcpy(got.data(), dout, n * 8, cudaMemcpyDeviceToHost);
    cudaFree(dv); cudaFree(dout);
    for (int i = 0; i < n; ++i)
        ASSERT_TRUE(same(got[i], want[i])) << "sequence " << i << ": device " << got[i] << " host " << want[i];
}

} // namespace
} // namespace twosum
} // namespace aether_tests
