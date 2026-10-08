// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/bench_bandwidth.cu — axpy3-shaped bandwidth microbench: stream
// (`out = a + s*b`, `Vec3dView`, N = 2^24), three arms —
//   - scalar:    the plain scalar path (`View::operator[](SampleIndex)`).
//   - bundled:   `BundleIndex<2>`/`DeviceBundle` (`View::
//                operator[](BundleIndex<W>)`), vectorized `double2`
//                LDG.128/STG.128 when aligned.
//   - readonly:  bundled plus `View::as_readonly()` on the two input
//                operands (`__ldg` routed through the same vectorized
//                bundle load path — `bundleGet`'s View overload already
//                excludes ReadOnly from its own 128-bit fast path, see
//                `backend/cuda/bundle/LoadStore.h`'s docstring, so this
//                arm measures `__ldg` scalar reads at bundle-loop cadence,
//                not a combined .128+.CI instruction — an honest, single-
//                variable-changed arm against the bundled baseline).
//
// Build only: no GPU execution at compile/link time — `tools/bench_bandwidth.sh`
// builds and runs this. This file's own `main()` performs 3 warmup
// launches + 5 timed reps per arm and prints the median elapsed time,
// achieved GB/s, and the achieved fraction of this workspace's pinned
// Quadro P2000 peak bandwidth (~140 GB/s) — an evidence card, not a
// pass/fail gate.
//
// Byte accounting (per axpy3 sample): read `a` (3 doubles) + read `b` (3
// doubles) + write `out` (3 doubles) = 9 doubles = 72 bytes/sample; total
// moved = 72 * N bytes, identical across all three arms (same algorithm,
// same data volume — the one variable changed between arms is the access
// mechanism, never the amount of data moved).

#include <aether/aether.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kN = std::size_t{ 1 } << 24; // 2^24
constexpr int kWarmups = 3;
constexpr int kTimedReps = 5;
constexpr double kPeakGBs = 140.0; // Quadro P2000 pinned peak bandwidth
constexpr double kBytesPerSample = 9.0 * sizeof(double); // 3 (a) + 3 (b) + 3 (out)

AETHER_KERNEL()
void scalarAxpy3Kernel(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out)
{
    const aether::SampleIndex i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= a.samples())
        return;
    out[i] = a[i].get() + s * b[i].get();
}

AETHER_KERNEL()
void bundledAxpy3Kernel(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out)
{
    constexpr std::size_t W = 2;
    const aether::BundleIndex<W> bi = aether::BundleIndex<W>::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (bi.base() >= a.samples())
        return;
    auto expr = a + s * b;
    if (bi.base() + W <= a.samples()) {
        out[bi] = expr;
    } else {
        aether::bundleAssign(out, expr, bi, bi.mask(a.samples()));
    }
}

AETHER_KERNEL()
void bundledReadOnlyAxpy3Kernel(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out)
{
    constexpr std::size_t W = 2;
    const aether::BundleIndex<W> bi = aether::BundleIndex<W>::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (bi.base() >= a.samples())
        return;
    auto aro = a.as_readonly();
    auto bro = b.as_readonly();
    auto expr = aro + s * bro;
    if (bi.base() + W <= a.samples()) {
        out[bi] = expr;
    } else {
        aether::bundleAssign(out, expr, bi, bi.mask(a.samples()));
    }
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void launchScalar(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out, int blocks, int threads)
{
    scalarAxpy3Kernel<<<blocks, threads>>>(a, b, s, out);
}
void launchBundled(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out, int blocks, int threads)
{
    bundledAxpy3Kernel<<<blocks, threads>>>(a, b, s, out);
}
void launchReadOnly(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out, int blocks, int threads)
{
    bundledReadOnlyAxpy3Kernel<<<blocks, threads>>>(a, b, s, out);
}

} // namespace

int main()
{
    aether::Array<double, 3> a(kN), b(kN), out(kN);
    auto ah = a.hostView();
    auto bh = b.hostView();
    for (std::size_t idx = 0; idx < kN; ++idx) {
        for (int d = 0; d < 3; ++d) {
            ah(d, idx) = static_cast<double>(idx % 997) * 0.001 + d * 0.1;
            bh(d, idx) = static_cast<double>((idx * 7) % 991) * 0.001 - d * 0.05;
        }
    }
    a.upload();
    b.upload();

    const double s = 2.5;
    const int threadsPerBlock = 256;

    std::printf("bench_bandwidth: N=%zu samples, axpy3 (Vec3dView), %d warmups + %d timed reps per arm\n", kN,
        kWarmups, kTimedReps);
    std::printf("%-24s %12s %12s %10s\n", "arm", "median_ms", "GB/s", "frac_peak");

    for (int arm = 0; arm < 3; ++arm) {
        const char* name = arm == 0 ? "scalar" : (arm == 1 ? "bundled_w2" : "bundled_readonly_w2");
        const int blocks = arm == 0 ? aether::cuda::launchConfig(kN, threadsPerBlock).blocks
                                     : static_cast<int>(aether::cuda::bundleLaunchConfig(kN, 2, threadsPerBlock).blocks);

        auto launch = [&]() {
            if (arm == 0)
                launchScalar(a.deviceView(), b.deviceView(), s, out.deviceView(), blocks, threadsPerBlock);
            else if (arm == 1)
                launchBundled(a.deviceView(), b.deviceView(), s, out.deviceView(), blocks, threadsPerBlock);
            else
                launchReadOnly(a.deviceView(), b.deviceView(), s, out.deviceView(), blocks, threadsPerBlock);
        };

        for (int w = 0; w < kWarmups; ++w) {
            launch();
            aether::cuda::checkLastLaunch(name);
        }

        std::vector<double> msReps;
        msReps.reserve(kTimedReps);
        for (int r = 0; r < kTimedReps; ++r) {
            cudaEvent_t start, stop;
            AETHER_CHECK_CUDA(cudaEventCreate(&start));
            AETHER_CHECK_CUDA(cudaEventCreate(&stop));
            AETHER_CHECK_CUDA(cudaEventRecord(start));
            launch();
            AETHER_CHECK_CUDA(cudaEventRecord(stop));
            AETHER_CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            AETHER_CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            msReps.push_back(static_cast<double>(ms));
            AETHER_CHECK_CUDA(cudaEventDestroy(start));
            AETHER_CHECK_CUDA(cudaEventDestroy(stop));
        }
        aether::cuda::checkLastLaunch(name);

        const double medMs = median(msReps);
        const double bytes = kBytesPerSample * static_cast<double>(kN);
        const double gbs = (bytes / 1.0e9) / (medMs / 1.0e3);
        const double frac = gbs / kPeakGBs;
        std::printf("%-24s %12.4f %12.2f %10.3f\n", name, medMs, gbs, frac);
    }

    out.download();
    std::printf("bench_bandwidth: done (evidence card only — no pass/fail).\n");
    return 0;
}
