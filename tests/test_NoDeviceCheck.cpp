// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::Error` on GPU-less use of CUDA device kinds, in an
// `AETHER_CPP_MODE` build.
//
// aether's "no device" condition is structural rather than runtime-detected:
// in an `AETHER_CPP_MODE` build, `AETHER_HAS_CUDA` is undefined, so there is
// no CUDA backend to allocate through at all — not a device a CUDA-capable
// binary discovers is missing. `Chunk::allocate()` already throws
// `aether::Error` for `kDLCUDA`/`kDLCUDAHost` in that build
// (`aether/chunk/Chunk.h`'s `#else` arm) — this exercises that real entry
// point (every `aether::Array<T>` construction goes through
// `Chunk::allocate`), never a GTEST_SKIP: this build genuinely has no CUDA
// backend to skip around, so both cases below run for real on every
// cpp-mode gate.
//
// CPP_MODE-only by construction (no `.cu` pair): the CUDA-mode counterpart
// of "allocate a kDLCUDA chunk" is a normal, successful allocation, not an
// error case — there is nothing analogous to test on that side.
// tests/nodevice/check_nodevice_throws.sh re-runs exactly these two cases
// filtered and checks ran==passed==2 from the manifest (never a literal).

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/err/Error.h>

namespace aether_tests {
namespace {

class NoDeviceCheckTest : public ::testing::Test { };

TEST_F(NoDeviceCheckTest, CudaDeviceChunkAllocateThrowsTypedError)
{
    EXPECT_THROW({ aether::Chunk::allocate(aether::Device(kDLCUDA), 64); }, aether::Error);
}

TEST_F(NoDeviceCheckTest, CudaHostChunkAllocateThrowsTypedError)
{
    EXPECT_THROW({ aether::Chunk::allocate(aether::Device(kDLCUDAHost), 64); }, aether::Error);
}

} // namespace
} // namespace aether_tests
