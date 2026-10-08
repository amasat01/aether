// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// view/WorkView.h test (CUDA-only — shared memory has no meaningful
// AETHER_CPP_MODE counterpart, so this file carries no test_WorkView.cpp
// pair; matches the existing test_BoundedTrig.cu precedent, a CUDA-only
// test file with no `.cpp` sibling). A shared-staging kernel
// (global->shared->compute->global) vs a direct-global kernel, memcmp
// exact, launch-check + blockingDownload conventions.
//
// `stagedAxpy3Kernel` stages both operands into shared memory via
// `make_work_view` (`view/WorkView.h`'s shared-memory idiom) — `i.work()`
// addresses the work-local slot, exactly matching that header's own
// docstring idiom — computes the axpy3 formula from the shared copies,
// and writes the result back to global `out`. `directAxpy3Kernel` computes
// the identical formula straight from global memory (the scalar path, no
// staging). Same input data, same launch grid -> the two must produce
// bit-identical output (the staging round-trip through shared memory must
// not perturb a single bit).

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;

// `Array::deviceView()` returns `layout_stride`, not the compact
// `Vec3dView` (`layout_right`) alias — same convention as
// `test_Partition.cu`'s own `Block3dView`.
using Vec3dStridedView = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;

class WorkViewStagingTest : public ::testing::Test { };

constexpr int kBlockSize = 128;
// n = 1000: several full blocks plus a genuine partial tail block, so the
// staging kernel's shared-memory carve is exercised at BOTH block shapes.
constexpr std::size_t kN = 1000;

AETHER_KERNEL()
void stagedAxpy3Kernel(Vec3dStridedView a, Vec3dStridedView b, double s, Vec3dStridedView out)
{
    AETHER_SHARED() double bufA[3 * kBlockSize];
    AETHER_SHARED() double bufB[3 * kBlockSize];
    auto workA = aether::make_work_view<double, 3>(bufA, kBlockSize);
    auto workB = aether::make_work_view<double, 3>(bufB, kBlockSize);

    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() < a.samples()) {
        workA(0, i.work()) = a(0, i.global());
        workA(1, i.work()) = a(1, i.global());
        workA(2, i.work()) = a(2, i.global());
        workB(0, i.work()) = b(0, i.global());
        workB(1, i.work()) = b(1, i.global());
        workB(2, i.work()) = b(2, i.global());
    }
    __syncthreads();
    if (i.global() < a.samples()) {
        out(0, i.global()) = workA(0, i.work()) + s * workB(0, i.work());
        out(1, i.global()) = workA(1, i.work()) + s * workB(1, i.work());
        out(2, i.global()) = workA(2, i.work()) + s * workB(2, i.work());
    }
}

AETHER_KERNEL()
void directAxpy3Kernel(Vec3dStridedView a, Vec3dStridedView b, double s, Vec3dStridedView out)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= a.samples())
        return;
    out[i] = a[i].get() + s * b[i].get();
}

TEST_F(WorkViewStagingTest, SharedStagedAxpy3MatchesDirectGlobalExactly)
{
    Array<double, 3> a(kN), b(kN), outStaged(kN), outDirect(kN);
    auto ah = a.hostView();
    auto bh = b.hostView();
    for (std::size_t idx = 0; idx < kN; ++idx) {
        for (int d = 0; d < 3; ++d) {
            ah(d, idx) = static_cast<double>(idx) * 0.13 + d * 0.7 - 5.0;
            bh(d, idx) = static_cast<double>(idx) * -0.09 + d * 0.3 + 2.0;
        }
    }
    a.upload();
    b.upload();

    const double s = 1.75;
    const auto cfg = aether::cuda::launchConfig(kN, kBlockSize);

    stagedAxpy3Kernel<<<cfg.blocks, cfg.threads>>>(a.deviceView(), b.deviceView(), s, outStaged.deviceView());
    aether::cuda::checkLastLaunch("stagedAxpy3Kernel");

    directAxpy3Kernel<<<cfg.blocks, cfg.threads>>>(a.deviceView(), b.deviceView(), s, outDirect.deviceView());
    aether::cuda::checkLastLaunch("directAxpy3Kernel");

    // Array::download() is BLOCKING (no Stream argument) — the
    // "blockingDownload" convention calls out.
    outStaged.download();
    outDirect.download();

    auto stagedHost = outStaged.hostView();
    auto directHost = outDirect.hostView();
    // `hostView()`'s component pitch is `capacity()`, not `kN` —
    // a flat memcmp over `size()` elements would walk past component 0's
    // real data into padding (and read the WRONG offset for components
    // 1/2). Compare each component's `kN` REAL, contiguous elements
    // (the sample-mode stride is always 1) separately instead.
    for (int d = 0; d < 3; ++d) {
        EXPECT_EQ(0, std::memcmp(&stagedHost(d, 0), &directHost(d, 0), kN * sizeof(double)))
            << "shared-staged axpy3 diverges from the direct-global reference (bit-identity required), component " << d;
    }
}

} // namespace
} // namespace aether_tests
