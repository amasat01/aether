// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Ray.cu
 * @brief `aether::geometry` ray primitives (CUDA mode): the shared host
 *        battery, plus the whole lane corpus run one lane per thread and
 *        compared with the host.
 *
 * The device may contract products into FMAs where the host build does not, so
 * values agree to a few ulps of the scene's scale; the hit flag agrees exactly.
 * See `test_Ray_common.h` for the host battery.
 */

#include "test_Ray_common.h"

namespace aether_tests {
namespace ray {
namespace {

template<class T>
AETHER_KERNEL()
void rayLanes(const Lane<T>* lanes, std::size_t n, ag::Vec4<T>* out)
{
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Lane<T>& l = lanes[i];
        out[3 * i + 0] = ag::rayTriangle<T>(l.o, l.d, l.a, l.b, l.c, l.tmax);
        out[3 * i + 1] = ag::rayAabb<T>(l.o, l.d, l.a, l.b, l.tmax);
        out[3 * i + 2] = ag::raySphere<T>(l.o, l.d, l.a, l.r, l.tmax);
    }
}

template<class T>
void checkTwin()
{
    const std::vector<Lane<T>> c = corpus<T>();
    Lane<T>* dl = nullptr;
    ag::Vec4<T>* dr = nullptr;
    ASSERT_EQ(cudaMalloc(&dl, c.size() * sizeof(Lane<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dr, 3 * c.size() * sizeof(ag::Vec4<T>)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dl, c.data(), c.size() * sizeof(Lane<T>), cudaMemcpyHostToDevice), cudaSuccess);
    rayLanes<T><<<static_cast<unsigned>((c.size() + 63) / 64), 64>>>(dl, c.size(), dr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<ag::Vec4<T>> got(3 * c.size());
    ASSERT_EQ(cudaMemcpy(got.data(), dr, got.size() * sizeof(ag::Vec4<T>), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dl);
    cudaFree(dr);
    const double tol = 256.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
    for (std::size_t i = 0; i < c.size(); ++i) {
        const ag::Vec4<T> want[3] = { tri(c[i]), box(c[i]), sph(c[i]) };
        for (int k = 0; k < 3; ++k) {
            const ag::Vec4<T>& g = got[3 * i + static_cast<std::size_t>(k)];
            EXPECT_EQ(g.v[0], want[k].v[0]) << "hit, primitive " << k << " lane " << i;
            for (int j = 1; j < 4; ++j) {
                const double w = static_cast<double>(want[k].v[j]), d = static_cast<double>(g.v[j]);
                EXPECT_NEAR(d, w, tol * std::max(1.0, std::fabs(w))) << "value " << j << ", primitive " << k << " lane " << i;
            }
        }
    }
}

} // namespace

TEST(RayDevice, LanesMatchTheHostF64) { checkTwin<double>(); }
TEST(RayDevice, LanesMatchTheHostF32) { checkTwin<float>(); }

} // namespace ray
} // namespace aether_tests
