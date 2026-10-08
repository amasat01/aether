// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Bootstrap smoke test (CUDA build).
//
// Paired with test_Smoke.cpp (same fixture, same case names) — only one of
// the two compiles into any given aether_tests binary: tests/CMakeLists.txt
// globs test_*.cu here and test_*.cpp in AETHER_CPP_MODE.

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/macros.h>

namespace aether_tests {
namespace {

AETHER_DEVICEHOST() int answer()
{
    return 42;
}

AETHER_KERNEL()
void writeAnswerKernel(int* out)
{
    *out = 42;
}

class SmokeTest : public ::testing::Test { };

TEST_F(SmokeTest, MacrosCompile)
{
    EXPECT_EQ(answer(), 42);

    int* d_out = nullptr;
    (void)cudaMalloc(&d_out, sizeof(int));
    (void)cudaMemset(d_out, 0, sizeof(int));
    writeAnswerKernel<<<1, 1>>>(d_out);
    aether::cuda::checkLastLaunch("writeAnswerKernel");
    (void)cudaDeviceSynchronize();

    int h_out = 0;
    (void)cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    (void)cudaFree(d_out);

    EXPECT_EQ(h_out, 42);
}

TEST_F(SmokeTest, VersionVisible)
{
    EXPECT_EQ(aether::version_major, 0);
}

} // namespace
} // namespace aether_tests
