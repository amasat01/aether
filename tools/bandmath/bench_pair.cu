// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tools/bandmath/bench_pair.cu
 * @brief Paired native-double vs `Band` bench.
 *
 * Dual arm, one source file:
 *   - host arm (`benchHostArm`, plain C++ loops, `std::chrono`): runs
 *     directly, no GPU execution (the device arm below is compiled and
 *     verified but never launched from here).
 *   - device arm (`benchDeviceArm`, CUDA kernels + `cudaEvent_t` timing,
 *     `<cudaDeviceProp>` card identification): built here (compile is a
 *     genuine check on its own — see `test_BandMathSass.cu`'s identical
 *     note), run separately on the target GPU (`bench_bandwidth.sh`'s own
 *     governance pattern: predeclared arms, warm-ups, median of N, raw
 *     timings retained, no pass/fail — exploratory, not confirmatory).
 *
 * `fma` is excluded — see `tests/test_BandMathCert_common.h`'s file
 * docstring: Band-level `fma` is not ported in aether.
 *
 * Usage:
 *   nvcc -arch=sm_61 -std=c++20 -O3 -DAETHER_HAS_CUDA=1 -I<repo root> \
 *        tools/bandmath/bench_pair.cu -o bench_pair
 *   ./bench_pair --host
 *   ./bench_pair --device
 */
#include "aether/banded/banded.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

using aether::banded::Band;
namespace bd = aether::banded::detail;

namespace {

// =========================================================================
//  Deterministic data generation — operands inside the SAME certified
//  admission window BandMathCert.h's in_domain rows use
//  (bd::kBandCarrierFloorExp + bd::kBandAdmissionMargin = -88 .. 115-8=107),
//  so the bench measures the CERTIFIED path, not the degraded/rejected
//  edges (those are a correctness question, BandMathCert.h's job, not a
//  throughput one).
// =========================================================================
std::vector<double> genOperands(int n, std::uint64_t seed, int expLo, int expHi)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> mant(1.0, 2.0);
    std::uniform_int_distribution<int> expd(expLo, expHi);
    std::uniform_int_distribution<int> signd(0, 1);
    std::vector<double> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) {
        const double m = mant(rng);
        const int e    = expd(rng);
        const double s = signd(rng) ? -1.0 : 1.0;
        out[static_cast<std::size_t>(i)] = std::ldexp(m, e) * s;
    }
    return out;
}

std::vector<double> genOperands(int n, std::uint64_t seed)
{
    return genOperands(
        n, seed, bd::kBandCarrierFloorExp + bd::kBandAdmissionMargin, bd::kBandNominalCeilingExp - bd::kBandAdmissionMargin);
}

double medianOf(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// =========================================================================
//  HOST ARM
// =========================================================================
struct BenchResult {
    double nativeMedianNs = 0.0;
    double emuMedianNs    = 0.0;
    double dispersionNative = 0.0; // (max-min)/median over the timed reps
    double dispersionEmu    = 0.0;
};

template<typename BandOp, typename NativeOp>
BenchResult benchPairHost(const char* name, const std::vector<double>& av,
    const std::vector<double>& bv, int warmups, int reps, BandOp bandOp, NativeOp nativeOp)
{
    const std::size_t n = av.size();
    std::vector<Band> abnd(n), bbnd(n);
    for (std::size_t i = 0; i < n; i++) {
        std::uint64_t ba = 0, bb = 0;
        std::memcpy(&ba, &av[i], 8);
        std::memcpy(&bb, &bv[i], 8);
        abnd[i] = bd::bandFromIEEE(static_cast<std::uint32_t>(ba), static_cast<std::uint32_t>(ba >> 32));
        bbnd[i] = bd::bandFromIEEE(static_cast<std::uint32_t>(bb), static_cast<std::uint32_t>(bb >> 32));
    }
    volatile double sinkD = 0.0;
    volatile float sinkF  = 0.0f;

    for (int w = 0; w < warmups; w++) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; i++)
            s += nativeOp(av[i], bv[i]);
        sinkD = s;
        float sf = 0.0f;
        for (std::size_t i = 0; i < n; i++)
            sf += bandOp(abnd[i], bbnd[i]).hi;
        sinkF = sf;
    }

    std::vector<double> nativeTimes(static_cast<std::size_t>(reps)), emuTimes(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        double s      = 0.0;
        for (std::size_t i = 0; i < n; i++)
            s += nativeOp(av[i], bv[i]);
        sinkD          = s;
        const auto t1  = std::chrono::high_resolution_clock::now();
        nativeTimes[static_cast<std::size_t>(r)]
            = std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(n);
    }
    for (int r = 0; r < reps; r++) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        float sf      = 0.0f;
        for (std::size_t i = 0; i < n; i++)
            sf += bandOp(abnd[i], bbnd[i]).hi;
        sinkF          = sf;
        const auto t1  = std::chrono::high_resolution_clock::now();
        emuTimes[static_cast<std::size_t>(r)]
            = std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(n);
    }

    BenchResult res;
    res.nativeMedianNs = medianOf(nativeTimes);
    res.emuMedianNs     = medianOf(emuTimes);
    const double nmin = *std::min_element(nativeTimes.begin(), nativeTimes.end());
    const double nmax = *std::max_element(nativeTimes.begin(), nativeTimes.end());
    const double emin = *std::min_element(emuTimes.begin(), emuTimes.end());
    const double emax = *std::max_element(emuTimes.begin(), emuTimes.end());
    res.dispersionNative = (res.nativeMedianNs > 0.0) ? (nmax - nmin) / res.nativeMedianNs : 0.0;
    res.dispersionEmu    = (res.emuMedianNs > 0.0) ? (emax - emin) / res.emuMedianNs : 0.0;

    // Values gate vs a host `long double` chain — PER-ELEMENT relative
    // error, maxed over the sample (not a SUMMED comparison: operands span
    // ~195 binades (K_BAND_ADMITTED_LO..HI), so summing widely-varying-
    // magnitude terms in a plain accumulator is itself catastrophically
    // imprecise regardless of arm — a methodology trap this bench's first
    // draft fell into (mul's summed gate read FAIL from THIS artifact, not
    // from any Band defect; BandMathCert.h's per-row ULP certification is
    // the real correctness claim, this is a coarse bench-time sanity check).
    long double maxRelErrNative = 0.0L, maxRelErrEmu = 0.0L;
    for (std::size_t i = 0; i < n; i++) {
        const long double ld = static_cast<long double>(nativeOp(av[i], bv[i]));
        const long double nd = static_cast<long double>(av[i]);
        (void)nd;
        const long double denom = (ld == 0.0L) ? 1.0L : std::fabs(ld);
        const long double nativeRel
            = std::fabs(static_cast<long double>(nativeOp(av[i], bv[i])) - ld) / denom;
        const Band bres = bandOp(abnd[i], bbnd[i]);
        const long double bandVal
            = static_cast<long double>(bres.hi) + static_cast<long double>(bres.lo);
        const long double emuRel = std::fabs(bandVal - ld) / denom;
        maxRelErrNative           = std::max(maxRelErrNative, nativeRel);
        maxRelErrEmu              = std::max(maxRelErrEmu, emuRel);
    }
    const bool valuesGateOk = (maxRelErrNative < 1e-9L) && (maxRelErrEmu < 1e-5L);

    const double eOverN = (res.nativeMedianNs > 0.0) ? res.emuMedianNs / res.nativeMedianNs : 0.0;
    std::printf(
        "%-10s native=%.3f ns/elem  band=%.3f ns/elem  e/n=%.2fx  disp(native)=%.1f%%  "
        "disp(band)=%.1f%%  values_gate=%s\n",
        name, res.nativeMedianNs, res.emuMedianNs, eOverN, res.dispersionNative * 100.0,
        res.dispersionEmu * 100.0, valuesGateOk ? "PASS" : "FAIL");
    (void)sinkD;
    (void)sinkF;
    return res;
}

void benchHostArm()
{
    const int n       = 100000;
    const int warmups = 3;
    const int reps    = 7;
    const auto av      = genOperands(n, 0xBA0DA7B00001ULL);
    const auto bv      = genOperands(n, 0xBA0DA7B00002ULL);

    std::printf("=== bandmath bench_pair — HOST ARM (this package's own run) ===\n");
    std::printf("n=%d warmups=%d reps=%d (median printed; raw arrays retained in-process)\n", n,
        warmups, reps);
    std::printf(
        "bottleneck reading: the HOST arm has no FP64:FP32 hardware ratio to report (a CPU core\n"
        "issues FP64 and FP32 at roughly the same rate) — the number that matters here is the\n"
        "FP32 INSTRUCTION COUNT itself (the card rows this package mints), which is\n"
        "hardware-independent and is what actually differs between the native-double and\n"
        "Band-emulated arms; the DEVICE arm is where FP64:FP32 throughput\n"
        "ratio changes the e/n slope.\n");

    benchPairHost(
        "copysign", av, bv, warmups, reps, [](Band a, Band b) { return bd::copysign(a, b); },
        [](double a, double b) { return std::copysign(a, b); });
    benchPairHost(
        "fmax", av, bv, warmups, reps, [](Band a, Band b) { return bd::fmax(a, b); },
        [](double a, double b) { return std::fmax(a, b); });
    benchPairHost(
        "fmin", av, bv, warmups, reps, [](Band a, Band b) { return bd::fmin(a, b); },
        [](double a, double b) { return std::fmin(a, b); });
    benchPairHost(
        "add", av, bv, warmups, reps, [](Band a, Band b) { return bd::add(a, b); },
        [](double a, double b) { return a + b; });
    benchPairHost(
        "sub", av, bv, warmups, reps, [](Band a, Band b) { return bd::sub(a, b); },
        [](double a, double b) { return a - b; });
    // mul's admission is on the PRODUCT's exponent (~e(a)+e(b)), not each
    // factor individually (@see gen_corpus.py's `target_operand_exp` — the
    // SAME halving this bench's first draft omitted, which is why its
    // first run read a spurious mul values_gate FAIL: many pairs drawn
    // from the full [-88,107] range on EACH factor land a product past
    // FP32's hard ceiling, which is the documented ceiling escape, not a
    // Band defect). Halve the per-factor range so the PRODUCT stays
    // in-domain.
    const int mulLo = (bd::kBandCarrierFloorExp + bd::kBandAdmissionMargin) / 2;
    const int mulHi = (bd::kBandNominalCeilingExp - bd::kBandAdmissionMargin) / 2;
    const auto avMul = genOperands(n, 0xBA0DA7B00003ULL, mulLo, mulHi);
    const auto bvMul = genOperands(n, 0xBA0DA7B00004ULL, mulLo, mulHi);
    benchPairHost(
        "mul", avMul, bvMul, warmups, reps, [](Band a, Band b) { return bd::mul(a, b); },
        [](double a, double b) { return a * b; });
    // abs is unary; reuse the binary harness with `b` ignored by both arms.
    benchPairHost(
        "abs", av, bv, warmups, reps, [](Band a, Band) { return bd::abs(a); },
        [](double a, double) { return std::fabs(a); });

    std::printf("fma: NOT BENCHED — aether::banded::detail::fma does not exist in aether.\n");
}

#ifdef AETHER_HAS_CUDA
// =========================================================================
//  DEVICE ARM — built here, run separately on the target GPU.
// =========================================================================
__global__ void bpAddKernel(float* out, const float* a, const float* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Band x = Band{ a[3 * i], a[3 * i + 1], a[3 * i + 2] };
    const Band y = Band{ b[3 * i], b[3 * i + 1], b[3 * i + 2] };
    const Band r = bd::add(x, y);
    out[3 * i]   = r.hi;
}
__global__ void bpMulKernel(float* out, const float* a, const float* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Band x = Band{ a[3 * i], a[3 * i + 1], a[3 * i + 2] };
    const Band y = Band{ b[3 * i], b[3 * i + 1], b[3 * i + 2] };
    const Band r = bd::mul(x, y);
    out[3 * i]   = r.hi;
}
__global__ void bpNativeAddKernel(double* out, const double* a, const double* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = a[i] + b[i];
}
__global__ void bpNativeMulKernel(double* out, const double* a, const double* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = a[i] * b[i];
}

void benchDeviceArm()
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("=== bandmath bench_pair — DEVICE ARM ===\n");
    std::printf("card: %s  major=%d minor=%d\n", prop.name, prop.major, prop.minor);
    // FP64:FP32 throughput ratio is not exposed directly by cudaDeviceProp;
    // print the SM count/clock instead — the actual ratio is a per-arch
    // constant looked up from the compute-capability table elsewhere in
    // the codebase, not re-derived here.
    std::printf("SM count=%d, clock=%d kHz (FP64:FP32 ratio is a per-arch constant, not read "
                "from cudaDeviceProp directly)\n",
        prop.multiProcessorCount, prop.clockRate);

    const int n = 1 << 16;
    std::vector<float> ha(3 * n), hb(3 * n);
    std::vector<double> da(n), db(n);
    for (int i = 0; i < n; i++) {
        ha[3 * i] = 1.0f + 0.001f * static_cast<float>(i % 1000);
        hb[3 * i] = 2.0f - 0.0007f * static_cast<float>(i % 1000);
        da[i]     = static_cast<double>(ha[3 * i]);
        db[i]     = static_cast<double>(hb[3 * i]);
    }
    float *dA = nullptr, *dB = nullptr, *dOutF = nullptr;
    double *dDA = nullptr, *dDB = nullptr, *dOutD = nullptr;
    cudaMalloc(&dA, sizeof(float) * 3 * n);
    cudaMalloc(&dB, sizeof(float) * 3 * n);
    cudaMalloc(&dOutF, sizeof(float) * 3 * n);
    cudaMalloc(&dDA, sizeof(double) * n);
    cudaMalloc(&dDB, sizeof(double) * n);
    cudaMalloc(&dOutD, sizeof(double) * n);
    cudaMemcpy(dA, ha.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hb.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dDA, da.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dDB, db.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    const int block = 256, grid = (n + block - 1) / block;
    const int warmups = 3, reps = 7;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    auto timeKernel = [&](auto launch) {
        for (int w = 0; w < warmups; w++)
            launch();
        std::vector<float> times(static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; r++) {
            cudaEventRecord(t0);
            launch();
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, t0, t1);
            times[static_cast<std::size_t>(r)] = ms;
        }
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    };

    const float addBandMs = timeKernel([&] { bpAddKernel<<<grid, block>>>(dOutF, dA, dB, n); });
    const float addNatMs  = timeKernel([&] { bpNativeAddKernel<<<grid, block>>>(dOutD, dDA, dDB, n); });
    const float mulBandMs = timeKernel([&] { bpMulKernel<<<grid, block>>>(dOutF, dA, dB, n); });
    const float mulNatMs  = timeKernel([&] { bpNativeMulKernel<<<grid, block>>>(dOutD, dDA, dDB, n); });
    std::printf("add       native=%.4f ms  band=%.4f ms  e/n=%.2fx\n", addNatMs, addBandMs,
        addBandMs / addNatMs);
    std::printf("mul       native=%.4f ms  band=%.4f ms  e/n=%.2fx\n", mulNatMs, mulBandMs,
        mulBandMs / mulNatMs);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOutF);
    cudaFree(dDA);
    cudaFree(dDB);
    cudaFree(dOutD);
}
#endif // AETHER_HAS_CUDA

} // namespace

int main(int argc, char** argv)
{
    std::string mode = "host";
    if (argc > 1)
        mode = argv[1];
    if (mode == "--device") {
#ifdef AETHER_HAS_CUDA
        benchDeviceArm();
#else
        std::fprintf(stderr, "bench_pair: built without AETHER_HAS_CUDA; --device unavailable\n");
        return 2;
#endif
    } else {
        benchHostArm();
    }
    return 0;
}
