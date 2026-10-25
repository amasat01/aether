// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::accum::run` on the device: one lane per sample, blocks of 256, the
// commits counted through the header's own debug counters. The cases and
// checks are test_Run_common.h's; test_Run.cpp is the host arm.

#define AETHER_RUN_COUNT_COMMITS 1

#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/accum/run.h>

namespace aether_tests {
namespace {

template<class T>
using RunView = aether::View<T, aether::extents<aether::dyn>>;

struct Case;

constexpr bool kDevice = true;

template<class T, bool Compensated>
AETHER_KERNEL()
void runKernel(RunView<T> target, RunView<T> comp, const long long* key, const T* term, int form,
               long long own, std::size_t n)
{
    namespace run = aether::accum::run;
    const auto i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    const std::size_t l = static_cast<std::size_t>(i.global());
    if (l >= n)
        return;
    const auto self = aether::SampleIndex::make(static_cast<std::size_t>(static_cast<long long>(l) + own));
    const auto at   = aether::SampleIndex::make(static_cast<std::size_t>(key[l]));
    auto open       = run::open(target);
    if constexpr (Compensated) {
        if (form == 0)
            run::addCompensated<32>(target, comp, at, term[l], self, open);
        else if (form == 1)
            run::addCompensated<16>(target, comp, at, term[l], self, open);
        else if (form == 2)
            run::addTestedCompensated(target, comp, at, term[l], open);
        else
            run::addSortedCompensated(target, comp, at, term[l], open);
    } else {
        (void)comp;
        if (form == 0)
            run::add<32>(target, at, term[l], self, open);
        else if (form == 1)
            run::add<16>(target, at, term[l], self, open);
        else if (form == 2)
            run::addTested(target, at, term[l], open);
        else
            run::addSorted(target, at, term[l], open);
    }
    run::close(open);
}

template<class T>
void runArm(const Case& c, bool compensated, std::vector<long double>& out, unsigned long long (&commits)[2]);

} // namespace
} // namespace aether_tests

#include "test_Run_common.h"

namespace aether_tests {
namespace {

template<class T>
void runArm(const Case& c, bool compensated, std::vector<long double>& out, unsigned long long (&commits)[2])
{
    const std::size_t n = c.key.size();
    std::vector<T> term(n);
    for (std::size_t l = 0; l < n; ++l)
        term[l] = static_cast<T>(termOf(l));
    long long* dKey = nullptr;
    T *dTerm = nullptr, *dTarget = nullptr, *dComp = nullptr;
    ASSERT_EQ(cudaMalloc(&dKey, n * sizeof(long long)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dTerm, n * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dTarget, c.slots * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dComp, c.slots * sizeof(T)), cudaSuccess);
    cudaMemcpy(dKey, c.key.data(), n * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(dTerm, term.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(dTarget, 0, c.slots * sizeof(T));
    cudaMemset(dComp, 0, c.slots * sizeof(T));
    const unsigned long long zero[2] = { 0, 0 };
    ASSERT_EQ(cudaMemcpyToSymbol(aether_run_commits, zero, sizeof(zero)), cudaSuccess);

    const auto target = aether::make_view<T, aether::dyn>(dTarget, aether::Device(kDLCUDA), c.slots);
    const auto comp   = aether::make_view<T, aether::dyn>(dComp, aether::Device(kDLCUDA), c.slots);
    const int form    = static_cast<int>(c.form);
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    if (compensated)
        runKernel<T, true><<<blocks, 256>>>(target, comp, dKey, dTerm, form, c.own, n);
    else
        runKernel<T, false><<<blocks, 256>>>(target, comp, dKey, dTerm, form, c.own, n);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<T> hi(c.slots), lo(c.slots);
    cudaMemcpy(hi.data(), dTarget, c.slots * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(lo.data(), dComp, c.slots * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaMemcpyFromSymbol(commits, aether_run_commits, sizeof(commits)), cudaSuccess);
    out.assign(c.slots, 0.0L);
    for (std::size_t k = 0; k < c.slots; ++k)
        out[k] = static_cast<long double>(hi[k]) + static_cast<long double>(lo[k]);
    cudaFree(dKey);
    cudaFree(dTerm);
    cudaFree(dTarget);
    cudaFree(dComp);
}

} // namespace
} // namespace aether_tests
