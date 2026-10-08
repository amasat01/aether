// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// residency/PartitionedArray.h tests (CUDA build; test_PartitionedArray.cpp
// covers a different scope — see that file's own docstring): this file
// covers the genuine multi-GPU path — scatter() a pair of operand arrays
// across `devices.size()` CUDA devices, run an axpy kernel independently
// per device over its own (padded-pitch) block, gather() the results back,
// and compare bit-exact against a single-device reference that runs the
// identical kernel over the whole array on one GPU (the host-computed
// expected values use the same `a + s*b` sequence the kernel itself
// evaluates).
//
// N = 777 is deliberately not a multiple of the 2-device padded pitch, so
// the round trip exercises a genuine short trailing block (Partition.h's
// own worked-example shape), not just the every-block-full case.
//
// The 2-device case self-skips (GTEST_SKIP) when
// `cudaGetDeviceCount() < 2`.

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::Device;
using aether::PartitionedArray;
using aether::SampleIndex;

constexpr int kBlockSize = 128;

class PartitionedArrayTest : public ::testing::Test { };

/** @brief `out = a + s*b`, rank-1 — the SAME formula, SAME op order, run
 *         both as the single-device reference (`ScalarView<double>`,
 *         `layout_right`, `Array::deviceView()`) and per-device on the
 *         partitioned path (`PartitionedArray::DeviceViewT`,
 *         `layout_stride` — see that method's own docstring for why), so
 *         templating on the view types is what lets ONE kernel body serve
 *         both call sites and stay bit-comparable. */
template<class ViewA, class ViewB, class ViewOut>
AETHER_KERNEL()
void partitionedAxpyKernel(ViewA a, ViewB b, double s, ViewOut out)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= a.samples())
        return;
    out(i.global()) = a(i.global()) + s * b(i.global());
}

TEST_F(PartitionedArrayTest, ScatterAxpyGatherRoundTripMatchesSingleDeviceReferenceBitExact)
{
    int deviceCount = 0;
    ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
        GTEST_SKIP() << "needs >= 2 CUDA devices (found " << deviceCount
                      << "); skipped because a second GPU is not present here";
    }

    constexpr std::size_t N = 777; // not a multiple of the 2-device padded pitch -> a genuine SHORT last block.
    const double s           = 1.5;

    std::vector<double> hostA(N), hostB(N), refOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        hostA[i]  = static_cast<double>(i) * 0.37 - 12.0;
        hostB[i]  = static_cast<double>(i) * -0.11 + 3.0;
        refOut[i] = hostA[i] + s * hostB[i]; // SAME op order the kernel computes, to avoid a false mismatch from reassociation.
    }

    // --- Single-device reference: the IDENTICAL kernel over the WHOLE N on device 0. ---
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    Array<double> refA(N), refB(N), refDeviceOut(N);
    {
        auto refAH = refA.hostView();
        auto refBH = refB.hostView();
        for (std::size_t i = 0; i < N; ++i) {
            refAH(i) = hostA[i];
            refBH(i) = hostB[i];
        }
    }
    refA.upload();
    refB.upload();
    const auto refCfg = aether::cuda::launchConfig(N, kBlockSize);
    partitionedAxpyKernel<<<refCfg.blocks, refCfg.threads>>>(refA.deviceView(), refB.deviceView(), s, refDeviceOut.deviceView());
    aether::cuda::checkLastLaunch("partitionedAxpyKernel(single-device reference)");
    refDeviceOut.download();

    // --- Partitioned, 2-device round trip. ---
    std::vector<Device> devices{ Device(kDLCUDA, 0), Device(kDLCUDA, 1) };
    PartitionedArray<double, aether::dyn> aPart(devices, N);
    PartitionedArray<double, aether::dyn> bPart(devices, N);
    PartitionedArray<double, aether::dyn> outPart(devices, N);

    {
        auto aH = aPart.hostView();
        auto bH = bPart.hostView();
        for (std::size_t i = 0; i < N; ++i) {
            aH(i) = hostA[i];
            bH(i) = hostB[i];
        }
    }

    aPart.scatter();
    bPart.scatter();

    for (std::size_t r = 0; r < aPart.parts(); ++r) {
        const std::size_t real = aPart.realCount(r);
        if (real == 0)
            continue;
        ASSERT_EQ(cudaSetDevice(aPart.device(r).id()), cudaSuccess);
        const auto cfg = aether::cuda::launchConfig(real, kBlockSize);
        partitionedAxpyKernel<<<cfg.blocks, cfg.threads>>>(aPart.deviceView(r), bPart.deviceView(r), s, outPart.deviceView(r));
        aether::cuda::checkLastLaunch("partitionedAxpyKernel(partitioned)");
    }

    outPart.gather();

    auto outHost = outPart.hostView();
    auto refHost = refDeviceOut.hostView();
    EXPECT_EQ(0, std::memcmp(outHost.data(), refHost.data(), N * sizeof(double)))
        << "partitioned 2-device scatter/axpy/gather diverges from the single-device reference (bit-identity required)";
}

} // namespace
} // namespace aether_tests
