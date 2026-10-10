// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Race-correctness tests for `aether::accum::atomic` primitives (CUDA
// mode).
//
// Each primitive is exercised by a CUDA kernel in which many threads race
// on a single (or paired) memory location. Adjacent-bool writes are also
// tested to confirm the byte-aligned 32-bit-word atomicOr trick does not
// stomp neighbouring slots.
//
// aether's atomic API is VIEW-based (no separate handle type — `View`
// already collapsed that indirection, see `aether/accum/atomic.h`'s
// docstring), so the kernels here construct a plain scalar
// `aether::View<T, extents<dyn>>` (`aether::make_view`) around the test
// buffer and pass it by value — same shape any real call site uses.

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::SampleIndex;

/* ------------------------------------------------------------------ */
/* File-scope race kernels                                             */
/* ------------------------------------------------------------------ */

template<class T>
using RaceView = aether::View<T, aether::extents<aether::dyn>>;

AETHER_KERNEL()
void minAbsRealRaceKernel(RaceView<double> tgt, const double* cand, std::size_t n)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (static_cast<std::size_t>(i.global()) >= n)
        return;
    /* Single shared slot: every thread targets sample 0. */
    aether::accum::atomic::minAbsReal(tgt, SampleIndex::make(std::size_t{ 0 }), cand[i.global()]);
}

AETHER_KERNEL()
void orBoolRaceSingleKernel(RaceView<bool> tgt, bool val, std::size_t n)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (static_cast<std::size_t>(i.global()) >= n)
        return;
    aether::accum::atomic::orBool(tgt, SampleIndex::make(std::size_t{ 0 }), val);
}

AETHER_KERNEL()
void orBoolRaceAdjacentKernel(RaceView<bool> slots, std::size_t nSlots, std::size_t totalThreads)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (static_cast<std::size_t>(i.global()) >= totalThreads)
        return;
    const std::size_t slot = static_cast<std::size_t>(i.global()) % nSlots;
    const bool writeTrue    = (slot % 2 == 0);
    aether::accum::atomic::orBool(slots, SampleIndex::make(slot), writeTrue);
}

AETHER_KERNEL()
void maxIntRaceKernel(RaceView<int> tgt, const int* cand, std::size_t n)
{
    const auto i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (static_cast<std::size_t>(i.global()) >= n)
        return;
    aether::accum::atomic::maxInt(tgt, SampleIndex::make(std::size_t{ 0 }), cand[i.global()]);
}

template<class T>
RaceView<T> makeRaceView(T* devPtr, std::size_t n)
{
    return aether::make_view<T, aether::dyn>(devPtr, aether::Device(kDLCUDA), n);
}

/* ------------------------------------------------------------------ */
/* minAbsReal                                                          */
/* ------------------------------------------------------------------ */

class AtomicMinAbsRealTest : public ::testing::Test {
protected:
    static constexpr std::size_t N = 4096;
};

TEST_F(AtomicMinAbsRealTest, ConcurrentMinPreservesSign)
{
    /* Initial value: positive, magnitude 100. Candidates are signed with
     * various magnitudes; the smallest (in magnitude) is +1/N. After the
     * race, the stored value must have magnitude 1/N and the SIGN of the
     * initial value (positive). */
    double host = 100.0;
    double* devT;
    cudaMalloc(&devT, sizeof(double));
    cudaMemcpy(devT, &host, sizeof(double), cudaMemcpyHostToDevice);

    auto* hostC = new double[N];
    for (std::size_t k = 0; k < N; ++k) {
        const double mag = 1.0 / static_cast<double>(k + 1);
        hostC[k]          = (k % 2 == 0) ? mag : -mag;
    }
    /* Inject a small-magnitude negative — final stored value should still
     * be positive (sign of initial preserved). */
    hostC[0] = -0.5;

    double* devC;
    cudaMalloc(&devC, N * sizeof(double));
    cudaMemcpy(devC, hostC, N * sizeof(double), cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    const int nBlocks       = static_cast<int>((N + blockSize - 1) / blockSize);
    minAbsRealRaceKernel<<<nBlocks, blockSize>>>(makeRaceView(devT, 1), devC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&host, devT, sizeof(double), cudaMemcpyDeviceToHost);

    const double expectedMag = 1.0 / static_cast<double>(N);
    EXPECT_DOUBLE_EQ(std::abs(host), expectedMag);
    EXPECT_GT(host, 0.0) << "Sign of initial value must be preserved";

    cudaFree(devT);
    cudaFree(devC);
    delete[] hostC;
}

TEST_F(AtomicMinAbsRealTest, NoUpdateWhenLargerMagnitude)
{
    /* Initial magnitude already smaller than every candidate — value must
     * be unchanged after the race. */
    double host = -1e-12;
    double* devT;
    cudaMalloc(&devT, sizeof(double));
    cudaMemcpy(devT, &host, sizeof(double), cudaMemcpyHostToDevice);

    auto* hostC = new double[N];
    for (std::size_t k = 0; k < N; ++k)
        hostC[k] = (k % 2 == 0) ? 1.0 : -2.0;
    double* devC;
    cudaMalloc(&devC, N * sizeof(double));
    cudaMemcpy(devC, hostC, N * sizeof(double), cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    const int nBlocks       = static_cast<int>((N + blockSize - 1) / blockSize);
    minAbsRealRaceKernel<<<nBlocks, blockSize>>>(makeRaceView(devT, 1), devC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&host, devT, sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_DOUBLE_EQ(host, -1e-12);

    cudaFree(devT);
    cudaFree(devC);
    delete[] hostC;
}

/* ------------------------------------------------------------------ */
/* orBool                                                              */
/* ------------------------------------------------------------------ */

class AtomicOrBoolTest : public ::testing::Test { };

TEST_F(AtomicOrBoolTest, SingleSlot_AllWritersTrue_ResultTrue)
{
    constexpr std::size_t N = 1024;
    bool host                = false;
    bool* devT;
    cudaMalloc(&devT, sizeof(bool));
    cudaMemcpy(devT, &host, sizeof(bool), cudaMemcpyHostToDevice);

    orBoolRaceSingleKernel<<<4, 256>>>(makeRaceView(devT, 1), true, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&host, devT, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(host);

    cudaFree(devT);
}

TEST_F(AtomicOrBoolTest, OneByteAllocation_StaysInBounds)
{
    /* A 1-byte allocation holds a single slot: the store must not widen to
     * the surrounding 32-bit word (out of bounds under compute-sanitizer). */
    unsigned char host = 0;
    unsigned char* devT;
    ASSERT_EQ(cudaMalloc(&devT, 1), cudaSuccess);
    cudaMemcpy(devT, &host, 1, cudaMemcpyHostToDevice);

    orBoolRaceSingleKernel<<<4, 256>>>(
        makeRaceView(reinterpret_cast<bool*>(devT), 1), true, 1024);
    cudaDeviceSynchronize();

    cudaMemcpy(&host, devT, 1, cudaMemcpyDeviceToHost);
    EXPECT_EQ(host, 1);

    cudaFree(devT);
}

TEST_F(AtomicOrBoolTest, SingleSlot_AllWritersFalse_ResultFalse)
{
    /* val=false should be a no-op — slot stays at its initial value. */
    bool host = false;
    bool* devT;
    cudaMalloc(&devT, sizeof(bool));
    cudaMemcpy(devT, &host, sizeof(bool), cudaMemcpyHostToDevice);

    orBoolRaceSingleKernel<<<4, 256>>>(makeRaceView(devT, 1), false, 1024);
    cudaDeviceSynchronize();

    bool out = true;
    cudaMemcpy(&out, devT, sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_FALSE(out);

    cudaFree(devT);
}

TEST_F(AtomicOrBoolTest, AdjacentSlots_NoCrossCorruption)
{
    /* 16-slot bool array. Many threads OR true into even slots and false
     * into odd slots concurrently. After the race, even slots must be true
     * and odd slots must remain false — which would fail if the byte-OR
     * widening trick stomped neighbouring slots. */
    constexpr std::size_t SLOTS            = 16;
    constexpr std::size_t WRITERS_PER_SLOT = 64;
    bool host[SLOTS] = { false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false };
    bool* dev;
    cudaMalloc(&dev, SLOTS * sizeof(bool));
    cudaMemcpy(dev, host, SLOTS * sizeof(bool), cudaMemcpyHostToDevice);

    const std::size_t total = SLOTS * WRITERS_PER_SLOT;
    orBoolRaceAdjacentKernel<<<4, 256>>>(makeRaceView(dev, SLOTS), SLOTS, total);
    cudaDeviceSynchronize();

    cudaMemcpy(host, dev, SLOTS * sizeof(bool), cudaMemcpyDeviceToHost);
    for (std::size_t k = 0; k < SLOTS; ++k) {
        if (k % 2 == 0)
            EXPECT_TRUE(host[k]) << "even slot " << k << " should be true";
        else
            EXPECT_FALSE(host[k]) << "odd slot " << k << " was corrupted by neighbour OR";
    }

    cudaFree(dev);
}

/* ------------------------------------------------------------------ */
/* maxInt                                                              */
/* ------------------------------------------------------------------ */

class AtomicMaxIntTest : public ::testing::Test { };

TEST_F(AtomicMaxIntTest, ConcurrentMaxIsCorrect)
{
    constexpr std::size_t N = 4096;
    int host                 = -1;
    int* devT;
    cudaMalloc(&devT, sizeof(int));
    cudaMemcpy(devT, &host, sizeof(int), cudaMemcpyHostToDevice);

    int* hostC = new int[N];
    for (std::size_t k = 0; k < N; ++k)
        hostC[k] = static_cast<int>(k * 3 - 100); /* spans neg + pos */
    hostC[N / 3] = 999999;                        /* the maximum */
    int* devC;
    cudaMalloc(&devC, N * sizeof(int));
    cudaMemcpy(devC, hostC, N * sizeof(int), cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    const int nBlocks       = static_cast<int>((N + blockSize - 1) / blockSize);
    maxIntRaceKernel<<<nBlocks, blockSize>>>(makeRaceView(devT, 1), devC, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&host, devT, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(host, 999999);

    cudaFree(devT);
    cudaFree(devC);
    delete[] hostC;
}

} // namespace
} // namespace aether_tests
