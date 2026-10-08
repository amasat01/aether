// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Chunk copy tests (CUDA build; test_Copy.cpp is the host-build twin).
// Adds the H<->CUDAHost<->D<->D round-trip (bit-exact, blocking and async
// on the default stream) that needs a real CUDA backend; "illegal path" is
// exercised here with real allocated Chunks (kDLCUDA is available) rather
// than test_Copy.cpp's 0-byte trick.

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/chunk/Copy.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>

namespace aether_tests {
namespace {

class CopyTest : public ::testing::Test { };

TEST_F(CopyTest, SizeMismatchThrows)
{
    auto src = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    auto dst = aether::Chunk::allocate(aether::Device(kDLCPU), 32);
    EXPECT_THROW(aether::copy(dst, src), aether::Error);
}

TEST_F(CopyTest, IllegalCpuCudaDirectPathThrows)
{
    auto cpu = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    auto dev = aether::Chunk::allocate(aether::Device(kDLCUDA), 64);
    EXPECT_THROW(aether::copy(dev, cpu), aether::Error);
    EXPECT_THROW(aether::copy(cpu, dev), aether::Error);
}

TEST_F(CopyTest, IllegalCpuCudaDirectAsyncPathThrows)
{
    // copyAsync's legal-path set is narrower than copy()'s: no async
    // specialisation exists for any pair touching plain (pageable) CPU
    // memory — only CUDAHost<->CUDAHost/CUDA and CUDA<->CUDA are truly
    // async-capable.
    auto cpu = aether::Chunk::allocate(aether::Device(kDLCPU), 64);
    auto host = aether::Chunk::allocate(aether::Device(kDLCUDAHost), 64);
    aether::Stream stream = 0; // default stream
    EXPECT_THROW(aether::copyAsync(host, cpu, stream), aether::Error);
}

TEST_F(CopyTest, HostDeviceDeviceRoundTripBlockingIsBitExact)
{
    const std::size_t n = 256;
    auto h0 = aether::Chunk::allocate(aether::Device(kDLCUDAHost), n);
    for (std::size_t i = 0; i < n; ++i)
        h0.data()[i] = static_cast<std::byte>((i * 131 + 7) & 0xFF);

    auto d0 = aether::Chunk::allocate(aether::Device(kDLCUDA), n);
    auto d1 = aether::Chunk::allocate(aether::Device(kDLCUDA), n);
    auto h1 = aether::Chunk::allocate(aether::Device(kDLCUDAHost), n);

    aether::copy(d0, h0); // H -> D
    aether::copy(d1, d0); // D -> D
    aether::copy(h1, d1); // D -> H

    EXPECT_EQ(std::memcmp(h0.data(), h1.data(), n), 0);
}

TEST_F(CopyTest, HostDeviceDeviceRoundTripAsyncIsBitExact)
{
    const std::size_t n = 256;
    auto h0 = aether::Chunk::allocate(aether::Device(kDLCUDAHost), n);
    for (std::size_t i = 0; i < n; ++i)
        h0.data()[i] = static_cast<std::byte>((i * 197 + 3) & 0xFF);

    auto d0 = aether::Chunk::allocate(aether::Device(kDLCUDA), n);
    auto d1 = aether::Chunk::allocate(aether::Device(kDLCUDA), n);
    auto h1 = aether::Chunk::allocate(aether::Device(kDLCUDAHost), n);

    aether::Stream stream = 0; // default stream

    aether::copyAsync(d0, h0, stream); // H -> D
    aether::copyAsync(d1, d0, stream); // D -> D
    aether::copyAsync(h1, d1, stream); // D -> H
    cudaStreamSynchronize(stream);

    EXPECT_EQ(std::memcmp(h0.data(), h1.data(), n), 0);
}

} // namespace
} // namespace aether_tests
