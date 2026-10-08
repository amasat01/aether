// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_AccumOrderPermutation.cu
 * @brief `AccumOrderPermutationTest.BitwiseUnderPermutedSchedules`: the
 *        kernel-injection admissibility bar ("the accumulate must be
 *        ORDER-INDEPENDENT in the bits") expressed as its own aether
 *        test, rather than left
 *        implicit in `AccumPlane.CompensatedAtomicIsOrderIndependent`
 *        (which only exercises 2 orders — ascending/descending body order).
 *        This test drives >= 3 DISTINCT launch-order permutations (identity,
 *        reverse, and pseudo-random shuffles) of the SAME body
 *        contributions through separate per-body kernel launches and
 *        requires every permutation's delivered plane to be bit-identical.
 *
 * DELIBERATELY SELF-CONTAINED (does NOT `#include "AccumPlaneCommon.h"`):
 * that header's `TEST_F(AccumPlane, ...)` bodies are non-inline and already
 * live in `test_AccumPlane.cu` — a second `.cu` translation unit in the
 * SAME `aether_tests` binary including the same header would duplicate
 * those symbols and fail the link. This file's own value profiles (a
 * magnitude-spread family and a cancellation family, both with a
 * deterministic per-element jitter/sign) are small, local copies of the
 * same idea, not a re-use of that header.
 *
 * CUDA-mode only: the permutation axis is about DEVICE launch/thread
 * scheduling — the host arm has no launch-order
 * axis to permute (a serial loop), and its own exactness is already
 * covered by `AccumPlane.HostCompensatedAtomicIsExactOnEveryProfile`.
 * Absolute correctness against the exact-sum reference is likewise already
 * covered by `AccumPlane.DeviceCompensatedAtomicIsExactOnEveryProfile`;
 * this test's own claim is narrower and sharper: SELF-CONSISTENCY across
 * permutations, non-vacuously (the value families are cancellation-prone
 * enough that an order-dependent implementation would not survive by luck).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::SampleIndex;
using aether::Vec3d;
namespace acc = aether::accum;

using CmpPlane      = acc::Plane<double, 3, acc::CompensatedAtomic>;
using CmpView        = CmpPlane::ViewT;
using Vec3dArrView  = aether::Array<double, 3>::ViewT;

constexpr int kBlock   = 256;
constexpr int kSamples = 128;
constexpr int kBodies  = 8;

constexpr int grid(int n) { return (n + kBlock - 1) / kBlock; }
constexpr int termIndex(int k, int d, int i) { return (k * 3 + d) * kSamples + i; }

__global__ void permAddBodyKernel(CmpView p, Vec3dArrView terms, int k, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const SampleIndex ti = SampleIndex::make(static_cast<std::size_t>(k * n + i));
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    Vec3d t;
    t.template get<0>() = terms(0, ti.global());
    t.template get<1>() = terms(1, ti.global());
    t.template get<2>() = terms(2, ti.global());
    p[si] += t;
}

__global__ void permDeliverKernel(CmpView p, Vec3dArrView out)
{
    const aether::offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.size())
        return;
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    p.deliver(out, si);
}

/// @brief A magnitude-spread ("force-model-like") profile (`cancel=false`)
/// or a cancellation-heavy profile (`cancel=true`) — small, local, and
/// deterministic (reproducible on failure).
inline std::vector<double> buildTerms(bool cancel)
{
    std::vector<double> v(static_cast<std::size_t>(kBodies * 3 * kSamples), 0.0);
    for (int k = 0; k < kBodies; ++k) {
        for (int d = 0; d < 3; ++d) {
            for (int i = 0; i < kSamples; ++i) {
                const unsigned h = static_cast<unsigned>(i) * 2654435761u + static_cast<unsigned>(d) * 40503u
                    + static_cast<unsigned>(k) * 2246822519u;
                const double jit = 1.0 + 0.4375 * static_cast<double>((h >> 8) & 0xFFFFu) / 65536.0;
                double val         = 0.0;
                if (cancel) {
                    const double A          = 1.0e-2 * jit;
                    static const double dl[] = { 0.0, 2.22e-16, 9.1e-13, 9.3e-10, 9.5e-7, 9.8e-4 };
                    val                       = (k % 2 == 0) ? A : -A * (1.0 - dl[i % 6]);
                } else {
                    static const double rel[] = { 1.0, 1.1e-3, 3.1e-6, 6.0e-7, 5.0e-8, 1.0e-10, 4.0e-12, 8.0e-13 };
                    const double sgn           = ((h >> 3) & 1u) ? 1.0 : -1.0;
                    val                          = 8.0e-3 * rel[k % 8] * jit * (k == 0 ? 1.0 : sgn);
                }
                v[static_cast<std::size_t>(termIndex(k, d, i))] = val;
            }
        }
    }
    return v;
}

inline void uploadTerms(const std::vector<double>& v, aether::Array<double, 3>& out)
{
    auto h = out.hostView();
    for (int k = 0; k < kBodies; ++k) {
        for (int i = 0; i < kSamples; ++i) {
            const std::size_t s = static_cast<std::size_t>(k * kSamples + i);
            h(0, s) = v[static_cast<std::size_t>(termIndex(k, 0, i))];
            h(1, s) = v[static_cast<std::size_t>(termIndex(k, 1, i))];
            h(2, s) = v[static_cast<std::size_t>(termIndex(k, 2, i))];
        }
    }
    out.upload();
}

inline std::vector<double> downloadOut(aether::Array<double, 3>& out, int n)
{
    out.download();
    cudaDeviceSynchronize();
    auto h = out.hostView();
    std::vector<double> r(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        r[static_cast<std::size_t>(0 * n + i)] = h(0, static_cast<std::size_t>(i));
        r[static_cast<std::size_t>(1 * n + i)] = h(1, static_cast<std::size_t>(i));
        r[static_cast<std::size_t>(2 * n + i)] = h(2, static_cast<std::size_t>(i));
    }
    return r;
}

/// @brief `numPermutations` distinct body-order permutations: identity,
/// full reverse, then deterministic (seeded) shuffles.
inline std::vector<std::vector<int>> makePermutations(int numPermutations)
{
    std::vector<std::vector<int>> perms;
    std::vector<int> identity(static_cast<std::size_t>(kBodies));
    for (int k = 0; k < kBodies; ++k)
        identity[static_cast<std::size_t>(k)] = k;
    perms.push_back(identity);
    perms.emplace_back(identity.rbegin(), identity.rend());

    std::mt19937 rng(0xACC0u);
    while (static_cast<int>(perms.size()) < numPermutations) {
        std::vector<int> p = identity;
        std::shuffle(p.begin(), p.end(), rng);
        perms.push_back(p);
    }
    return perms;
}

inline std::vector<double> runPermutation(aether::Array<double, 3>& terms, const std::vector<int>& order)
{
    CmpPlane plane{ static_cast<std::size_t>(kSamples) };
    aether::Array<double, 3> out{ static_cast<std::size_t>(kSamples) };
    out.upload();
    plane.upload();
    plane.zeroDeviceAsync();
    for (int k : order)
        permAddBodyKernel<<<grid(kSamples), kBlock>>>(plane.deviceView(), terms.deviceView(), k, kSamples);
    permDeliverKernel<<<grid(kSamples), kBlock>>>(plane.deviceView(), out.deviceView());
    return downloadOut(out, kSamples);
}

} // namespace

class AccumOrderPermutationTest : public ::testing::Test { };

TEST_F(AccumOrderPermutationTest, BitwiseUnderPermutedSchedules)
{
    constexpr int kNumPermutations = 5; // >= 3, so a fixed pair of orders can't accidentally agree
    const std::vector<std::vector<int>> perms = makePermutations(kNumPermutations);
    ASSERT_GE(perms.size(), 3u);

    for (bool cancel : { false, true }) {
        const std::vector<double> profile = buildTerms(cancel);
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadTerms(profile, terms);

        std::vector<std::vector<double>> results;
        for (const auto& order : perms)
            results.push_back(runPermutation(terms, order));

        // Non-vacuity: the results must not be trivially all-zero (which
        // would make "bit-identical across permutations" a vacuous claim).
        bool anyNonZero = false;
        for (double v : results[0])
            anyNonZero = anyNonZero || (v != 0.0);
        ASSERT_TRUE(anyNonZero) << "cancel=" << cancel << ": degenerate profile, did the kernel run?";

        for (std::size_t r = 1; r < results.size(); ++r) {
            for (std::size_t s = 0; s < results[0].size(); ++s) {
                ASSERT_EQ(results[r][s], results[0][s])
                    << "cancel=" << cancel << " permutation " << r << " vs 0, slot " << s
                    << " -- the admissibility bar requires the compensated-atomic sink to be bit-identical under ANY launch order";
            }
        }
    }
}

} // namespace aether_tests
