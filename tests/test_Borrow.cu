// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk::borrow() tests (CUDA build; test_Borrow.cpp is the host-build twin
// over the same fixture). Borrows a raw cudaMalloc'd pointer (bypassing
// Chunk::allocate entirely) as a non-owning kDLCUDA Chunk, builds a View
// over it, and runs a device kernel that computes in place through that
// view — proving the wrap is usable as a genuine device-side compute
// target, not just a host bookkeeping shim.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::SampleIndex;
using aether::Vec3dView;

class BorrowTest : public ::testing::Test { };

AETHER_KERNEL()
void scaleInPlaceKernel(Vec3dView v, double s)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= v.samples())
        return;
    v(0, i.global()) *= s;
    v(1, i.global()) *= s;
    v(2, i.global()) *= s;
}

TEST_F(BorrowTest, BorrowedRawCudaPointerComputesInPlaceOnDevice)
{
    constexpr std::size_t C = 3, N = 64;
    const std::size_t bytes = C * N * sizeof(double);

    void* raw = nullptr;
    ASSERT_EQ(cudaMalloc(&raw, bytes), cudaSuccess);

    {
        auto chunk = aether::Chunk::borrow(raw, bytes, aether::Device(kDLCUDA));
        EXPECT_FALSE(chunk.owns());
        auto dv = aether::make_view<double, C, aether::dyn>(chunk, N);

        // Stage known host values through a pinned chunk, upload through
        // the BORROWED device chunk, compute in place on the device, then
        // read back through that SAME borrowed chunk.
        auto hostChunk = aether::Chunk::allocate(aether::Device(kDLCUDAHost), bytes);
        auto hv         = aether::make_view<double, C, aether::dyn>(hostChunk, N);
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t i = 0; i < N; ++i)
                hv(c, i) = static_cast<double>(c * 10 + i);

        aether::copy(chunk, hostChunk); // H -> D, through the borrowed chunk

        const auto cfg = aether::cuda::launchConfig(N);
        scaleInPlaceKernel<<<cfg.blocks, cfg.threads>>>(dv, 2.0);
        aether::cuda::checkLastLaunch("scaleInPlaceKernel");

        aether::copy(hostChunk, chunk); // D -> H, through the same borrowed chunk
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t i = 0; i < N; ++i)
                EXPECT_EQ(hv(c, i), static_cast<double>(c * 10 + i) * 2.0);
        // `chunk` (non-owning) goes out of scope here — `raw` must survive.
    }

    // If the borrowed chunk had actually freed `raw` above, this is a
    // double-free/use-after-free.
    ASSERT_EQ(cudaFree(raw), cudaSuccess);
}

} // namespace
} // namespace aether_tests
