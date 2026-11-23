// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Edge rules (CUDA build; test_Edge.cpp is the host twin with the same case names): the helpers run on the device with
// the extent a compile-time constant, as the emitted units call them, over negative and positive indices. The oracle is
// the rule stated in words, applied index by index on the host.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/view/Edge.h>

namespace aether_tests {
namespace {

// One thread per index; `E` is a template constant, so the compiler sees a literal extent in every helper.
template<std::size_t E>
__global__ void edgeKernel(const std::int64_t* in, int n, std::size_t* clampOut, std::size_t* wrapOut, int* insideOut,
    std::size_t* uncheckedOut)
{
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n) return;
    const std::int64_t index = in[k];
    clampOut[k]     = aether::edge::clamp(index, E);
    wrapOut[k]      = aether::edge::wrap(index, E);
    insideOut[k]    = aether::edge::inside(index, E) ? 1 : 0;
    uncheckedOut[k] = (index >= 0 && index < static_cast<std::int64_t>(E)) ? aether::edge::unchecked(index, E) : 0;
}

// The four results of the helpers on the device for the indices in `in`.
struct DeviceResult {
    std::vector<std::size_t> clamp, wrap, unchecked;
    std::vector<int> inside;
};

template<std::size_t E>
DeviceResult runOnDevice(const std::vector<std::int64_t>& in)
{
    const int n = static_cast<int>(in.size());
    std::int64_t* dIn;
    std::size_t *dClamp, *dWrap, *dUnchecked;
    int* dInside;
    cudaMalloc(&dIn, n * sizeof(std::int64_t));
    cudaMalloc(&dClamp, n * sizeof(std::size_t));
    cudaMalloc(&dWrap, n * sizeof(std::size_t));
    cudaMalloc(&dUnchecked, n * sizeof(std::size_t));
    cudaMalloc(&dInside, n * sizeof(int));
    cudaMemcpy(dIn, in.data(), n * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    edgeKernel<E><<<(n + 63) / 64, 64>>>(dIn, n, dClamp, dWrap, dInside, dUnchecked);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    DeviceResult r;
    r.clamp.resize(n);
    r.wrap.resize(n);
    r.unchecked.resize(n);
    r.inside.resize(n);
    cudaMemcpy(r.clamp.data(), dClamp, n * sizeof(std::size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.wrap.data(), dWrap, n * sizeof(std::size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.unchecked.data(), dUnchecked, n * sizeof(std::size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.inside.data(), dInside, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dClamp);
    cudaFree(dWrap);
    cudaFree(dUnchecked);
    cudaFree(dInside);
    return r;
}

// `body(std::integral_constant<std::size_t, E>{})` for every extent 1..7.
template<class Body>
void forEveryExtentUpToSeven(Body&& body)
{
    body(std::integral_constant<std::size_t, 1>{});
    body(std::integral_constant<std::size_t, 2>{});
    body(std::integral_constant<std::size_t, 3>{});
    body(std::integral_constant<std::size_t, 4>{});
    body(std::integral_constant<std::size_t, 5>{});
    body(std::integral_constant<std::size_t, 6>{});
    body(std::integral_constant<std::size_t, 7>{});
}

// Indices from three extents below zero to three above the extent, as the host twin walks them.
std::vector<std::int64_t> indicesAround(std::int64_t extent)
{
    std::vector<std::int64_t> in;
    for (std::int64_t index = -3 * extent - 2; index <= 3 * extent + 2; ++index) in.push_back(index);
    return in;
}

class EdgeTest : public ::testing::Test { };

TEST_F(EdgeTest, ClampHoldsAnIndexAtTheNearestValidOne)
{
    forEveryExtentUpToSeven([](auto extent) {
        constexpr std::size_t E = decltype(extent)::value;
        const auto in           = indicesAround(E);
        const auto got          = runOnDevice<E>(in);
        for (std::size_t k = 0; k < in.size(); ++k) {
            const std::int64_t last = static_cast<std::int64_t>(E) - 1;
            const std::int64_t want = in[k] < 0 ? 0 : (in[k] > last ? last : in[k]);
            EXPECT_EQ(got.clamp[k], static_cast<std::size_t>(want)) << "extent " << E << " index " << in[k];
        }
    });
}

TEST_F(EdgeTest, WrapCountsAnOutOfRangeIndexAroundTheExtent)
{
    const std::vector<std::int64_t> in = { -5, -4, -1, 0, 3, 4, 9 };
    const std::size_t out[]            = { 3, 0, 3, 0, 3, 0, 1 };
    const auto got                     = runOnDevice<4>(in);
    for (std::size_t k = 0; k < in.size(); ++k) EXPECT_EQ(got.wrap[k], out[k]) << "index " << in[k];
}

TEST_F(EdgeTest, WrapEqualsTheMathematicalRemainderForEveryExtentUpToSeven)
{
    forEveryExtentUpToSeven([](auto extent) {
        constexpr std::size_t E = decltype(extent)::value;
        const auto in           = indicesAround(E);
        const auto got          = runOnDevice<E>(in);
        for (std::size_t k = 0; k < in.size(); ++k) {
            std::int64_t expected = in[k];  // the reference: step by the extent into [0, extent)
            while (expected < 0) expected += static_cast<std::int64_t>(E);
            while (expected >= static_cast<std::int64_t>(E)) expected -= static_cast<std::int64_t>(E);
            EXPECT_EQ(got.wrap[k], static_cast<std::size_t>(expected)) << "extent " << E << " index " << in[k];
        }
    });
}

TEST_F(EdgeTest, InsideIsTrueExactlyForIndicesBelowTheExtent)
{
    forEveryExtentUpToSeven([](auto extent) {
        constexpr std::size_t E = decltype(extent)::value;
        const auto in           = indicesAround(E);
        const auto got          = runOnDevice<E>(in);
        for (std::size_t k = 0; k < in.size(); ++k) {
            const bool want = in[k] >= 0 && in[k] < static_cast<std::int64_t>(E);
            EXPECT_EQ(got.inside[k] != 0, want) << "extent " << E << " index " << in[k];
        }
    });
}

TEST_F(EdgeTest, UncheckedPassesAValidIndexThrough)
{
    forEveryExtentUpToSeven([](auto extent) {
        constexpr std::size_t E = decltype(extent)::value;
        const auto in           = indicesAround(E);
        const auto got          = runOnDevice<E>(in);
        for (std::size_t k = 0; k < in.size(); ++k)
            if (in[k] >= 0 && in[k] < static_cast<std::int64_t>(E))
                EXPECT_EQ(got.unchecked[k], static_cast<std::size_t>(in[k])) << "extent " << E << " index " << in[k];
    });
}

} // namespace
} // namespace aether_tests
