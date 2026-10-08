// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// CUDA graph capture probe (CUDA-only — stream capture has no meaningful
// AETHER_CPP_MODE counterpart, so this file carries no test_GraphCapture.cpp
// pair; matches the existing test_BoundedTrig.cu / test_WorkView.cu
// precedent, a CUDA-only test file with no `.cpp` sibling).
//
// Question: does an aether kernel launch survive
// `cudaStream{Begin,End}Capture` stream capture, the mechanism CUDA graph
// replay relies on? Capture an axpy kernel launch plus an
// `Array::download(stream)` copyAsync into one graph, instantiate it once,
// replay it three times, and every replay's downloaded values must be
// exact (memcmp) against an uncaptured run of the identical kernel.
//
// Capture-safety note: every synchronizing CUDA call
// (`cudaDeviceSynchronize`, `cudaStreamSynchronize` on the capturing
// stream) is illegal between `cudaStreamBeginCapture` and
// `cudaStreamEndCapture` and would abort capture with an error. This is why
// the captured kernel launch below is not followed by
// `aether::cuda::checkLastLaunch()` (that helper calls
// `cudaDeviceSynchronize()` internally, `aether/backend/cuda/Launch.h`) —
// only the non-synchronizing `cudaPeekAtLastError()` is used inside the
// capture bracket; the full launch-check convention applies to the
// uncaptured reference kernel, which runs outside any capture.

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;
using aether::ScalarView;

constexpr int kBlockSize = 128;
constexpr std::size_t kN = 500;

// `Array::deviceView()` returns `layout_stride`, not the compact
// `ScalarView<double>` (`layout_right`) alias — same convention as
// `test_Partition.cu`'s own `Block3dView`.
using ScalarDStridedView = aether::View<double, aether::extents<aether::dyn>, aether::layout_stride>;

/** @brief `out = a + s*b`, rank-1 — deliberately the simplest possible
 *         kernel; D-b probes CAPTURE mechanics, not kernel complexity. */
AETHER_KERNEL()
void graphAxpyKernel(ScalarDStridedView a, ScalarDStridedView b, double s, ScalarDStridedView out)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= a.samples())
        return;
    out(i.global()) = a(i.global()) + s * b(i.global());
}

class GraphCaptureTest : public ::testing::Test { };

TEST_F(GraphCaptureTest, CapturedKernelPlusCopyAsyncReplaysExactlyThreeTimesVsUncaptured)
{
    Array<double> a(kN), b(kN), outUncaptured(kN), outCaptured(kN);
    {
        auto ah = a.hostView();
        auto bh = b.hostView();
        for (std::size_t i = 0; i < kN; ++i) {
            ah(i) = static_cast<double>(i) * 0.21 - 4.0;
            bh(i) = static_cast<double>(i) * 0.05 + 1.0;
        }
    }
    a.upload();
    b.upload();

    const double s  = 2.25;
    const auto cfg  = aether::cuda::launchConfig(kN, kBlockSize);

    // --- Uncaptured reference run (ordinary launch-check conventions apply). ---
    graphAxpyKernel<<<cfg.blocks, cfg.threads>>>(a.deviceView(), b.deviceView(), s, outUncaptured.deviceView());
    aether::cuda::checkLastLaunch("graphAxpyKernel(uncaptured)");
    outUncaptured.download();

    // --- Captured run: kernel launch + copyAsync, on a dedicated stream. ---
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);

    graphAxpyKernel<<<cfg.blocks, cfg.threads, 0, stream>>>(a.deviceView(), b.deviceView(), s, outCaptured.deviceView());
    // Non-synchronizing launch check ONLY (see this file's own header note —
    // cudaDeviceSynchronize/cudaStreamSynchronize on the capturing stream
    // are illegal mid-capture).
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    outCaptured.download(stream); // Array::download(Stream) — copyAsync, captured into the SAME graph.

    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);

    cudaGraphExec_t graphExec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, 0), cudaSuccess);

    auto capturedHost = outCaptured.hostView();
    auto refHost      = outUncaptured.hostView();
    for (int rep = 0; rep < 3; ++rep) {
        ASSERT_EQ(cudaGraphLaunch(graphExec, stream), cudaSuccess) << "replay " << rep;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess) << "replay " << rep;
        EXPECT_EQ(0, std::memcmp(capturedHost.data(), refHost.data(), kN * sizeof(double)))
            << "replay " << rep << " diverges from the uncaptured reference (bit-identity required)";
    }

    ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
} // namespace aether_tests
