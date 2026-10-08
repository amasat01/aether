// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Host/device layout-identity gate (the nvcc side).
//
// The central claim: a CUDA-mode aether build may contain host-compiler
// (`g++`) translation units alongside nvcc ones, and both must see the same
// types with the same layout. A claim like that is worth nothing as prose:
// this file makes it a gate that a real disagreement would fail.
//
// The partner TU is `tests/hosttu/host_tu_probe.cpp`, compiled by g++ with
// `LANGUAGE CXX` forced inside this same CUDA build and linked into this
// same binary. Neither side defines `AETHER_CPP_MODE` — the two axes stay
// separate (build mode chooses the member set, compiler chooses only the
// legal syntax).
//
// What would make this red: anything that changes a member set, alignment
// or padding between the two compilers — an `#if defined(__CUDACC__)`
// around a data member or a member declaration, a type whose specialisation
// exists on one side only (the `Fetch<double>` / `int2` trap fixed in
// `aether/dtype/Fetch.h`), or a header that quietly falls back to a
// different definition when `__CUDACC__` is absent.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "tests/hosttu/host_tu_probe.h"

namespace aether_tests {
namespace {

using aether_r8::kLayoutNames;
using aether_r8::kLayoutRowCount;
using aether_r8::LayoutRow;

class HostTuLayoutTest : public ::testing::Test { };

/** @brief The nvcc-side expansion of the SAME list — a separate, uniquely
 *         named, non-inline function so the linker cannot fold it onto the
 *         host TU's copy and turn the comparison into a tautology. */
void deviceTuFillLayoutTable(LayoutRow* out)
{
    std::size_t i = 0;
#define AETHER_R8_MEASURE_ONE(...)                                                                                     \
    out[i].name  = #__VA_ARGS__;                                                                                       \
    out[i].size  = static_cast<std::uint64_t>(sizeof(__VA_ARGS__));                                                    \
    out[i].align = static_cast<std::uint64_t>(alignof(__VA_ARGS__));                                                   \
    ++i;
    AETHER_R8_LAYOUT_TYPES(AETHER_R8_MEASURE_ONE)
#undef AETHER_R8_MEASURE_ONE
    static_cast<void>(i);
}

// NON-VACUITY, at COMPILE time as well as at run time: a gate that compares
// two EMPTY tables passes every time, and a runtime EXPECT cannot fire in a
// binary that was never run. Pin the list size here too.
static_assert(kLayoutRowCount == 23,
    "AETHER_R8_LAYOUT_TYPES changed size — update the pin (and say so in the "
    "commit); an X-macro that silently expands to nothing would otherwise make "
    "the layout comparison below vacuously green.");

// NON-VACUITY: a gate that compares two EMPTY tables passes every time.
// Pin the list's size explicitly so silently emptying `AETHER_R8_LAYOUT_TYPES`
// (or an X-macro that expands to nothing) fails here before the comparison
// below can report a vacuous green.
TEST_F(HostTuLayoutTest, LayoutTableIsNotEmpty)
{
    EXPECT_EQ(kLayoutRowCount, 23u);
    ASSERT_GE(kLayoutRowCount, 20u);
    for (std::size_t i = 0; i < kLayoutRowCount; ++i)
        EXPECT_NE(kLayoutNames[i], nullptr);
}

TEST_F(HostTuLayoutTest, SizeAndAlignAgreeBetweenHostCompilerAndNvcc)
{
    std::vector<LayoutRow> host(kLayoutRowCount);
    std::vector<LayoutRow> device(kLayoutRowCount);
    aether_r8::hostFillLayoutTable(host.data());
    deviceTuFillLayoutTable(device.data());

    for (std::size_t i = 0; i < kLayoutRowCount; ++i) {
        SCOPED_TRACE(kLayoutNames[i]);
        EXPECT_STREQ(host[i].name, device[i].name);
        EXPECT_EQ(host[i].size, device[i].size) << "sizeof disagrees between g++ and nvcc";
        EXPECT_EQ(host[i].align, device[i].align) << "alignof disagrees between g++ and nvcc";
        EXPECT_GT(host[i].size, 0u);
    }
}

TEST_F(HostTuLayoutTest, HostBuiltArraysReadBackIdenticallyInADeviceTu)
{
    constexpr std::size_t N = 37;
    aether_r8::HostArraySummary sum{};
    auto* owned = aether_r8::hostMakeArrays(N, &sum);
    ASSERT_NE(owned, nullptr);

    // Same objects, nvcc's view of the same headers.
    auto hv = aether_r8::hostVectorHostView(owned);
    auto dv = aether_r8::hostVectorDeviceView(owned);

    EXPECT_EQ(hv.data(), sum.vectorHostData);
    EXPECT_EQ(dv.data(), sum.vectorDeviceData);
    EXPECT_EQ(static_cast<std::uint64_t>(hv.samples()), sum.samples);
    EXPECT_EQ(static_cast<std::uint64_t>(hv.rank()), sum.vectorRank);
    EXPECT_EQ(static_cast<std::uint64_t>(hv.extent(0)), sum.vectorExtent0);
    EXPECT_EQ(static_cast<std::uint64_t>(hv.extent(1)), sum.vectorExtent1);
    EXPECT_EQ(static_cast<std::int64_t>(hv.mapping().stride(0)), sum.vectorStride0);
    EXPECT_EQ(static_cast<std::int64_t>(hv.mapping().stride(1)), sum.vectorStride1);
    EXPECT_EQ(static_cast<std::uint64_t>(sizeof(aether::Array<float>)), sum.arrayFloatSize);
    EXPECT_EQ(static_cast<std::uint64_t>(sizeof(aether::Array<float, 3>)), sum.arrayVec3Size);

    // The VALUES the host TU wrote, addressed through nvcc's own mapping.
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_FLOAT_EQ(hv(c, i), static_cast<float>(1000 * c + i));

    aether_r8::hostDestroyArrays(owned);
}

TEST_F(HostTuLayoutTest, HostBuiltDeviceAndStreamCrossTheTuBoundary)
{
    constexpr std::size_t N = 8;
    aether_r8::HostArraySummary sum{};
    auto* owned = aether_r8::hostMakeArrays(N, &sum);
    ASSERT_NE(owned, nullptr);

    const aether::Device dev = aether_r8::hostMakeCudaDevice();
    EXPECT_TRUE(dev.is_cuda());
    EXPECT_EQ(dev.id(), 0);

    const aether::Stream s = aether_r8::hostStream(owned);
    EXPECT_NE(s, nullptr);
    EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);

    // Host and device chunks really are distinct allocations (CUDA mode),
    // as the host TU reported them.
    EXPECT_NE(sum.vectorHostData, sum.vectorDeviceData);
    EXPECT_NE(sum.scalarHostData, sum.scalarDeviceData);

    aether_r8::hostDestroyArrays(owned);
}

/** @brief The kernel this file — and only this file — is allowed to define:
 *         `AETHER_KERNEL()` is not even defined in the host TU, so a
 *         kernel there is a loud compile error rather than a silent host
 *         function. */
AETHER_KERNEL() void addOneToVectorKernel(aether::Array<float, 3>::ViewT v)
{
    const auto i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= static_cast<std::size_t>(v.samples()))
        return;
    for (std::size_t c = 0; c < 3; ++c)
        v(c, i.global()) += 1.0F;
}

TEST_F(HostTuLayoutTest, KernelLaunchedHereMutatesDataAHostTuUploaded)
{
    // THE TWO AXES, end to end: g++ allocated, filled and uploaded; nvcc
    // launches over the very same device view; g++ downloads; the values
    // agree. Any layout disagreement would make the kernel index the wrong
    // elements and this comparison fail.
    constexpr std::size_t N = 64;
    aether_r8::HostArraySummary sum{};
    auto* owned = aether_r8::hostMakeArrays(N, &sum);
    ASSERT_NE(owned, nullptr);

    auto dv = aether_r8::hostVectorDeviceView(owned);
    ASSERT_EQ(dv.data(), sum.vectorDeviceData);

    addOneToVectorKernel<<<1, static_cast<unsigned>(N)>>>(dv);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    aether_r8::hostDownloadVector(owned);
    auto hv = aether_r8::hostVectorHostView(owned);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_FLOAT_EQ(hv(c, i), static_cast<float>(1000 * c + i) + 1.0F);

    aether_r8::hostDestroyArrays(owned);
}

} // namespace
} // namespace aether_tests
