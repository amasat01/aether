// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::accum::run` on the host (AETHER_CPP_MODE): OpenMP over serial
// ranges of kHostRange lanes, one open run per range, the commits counted
// through the header's own debug counters. The cases and checks are
// test_Run_common.h's; test_Run.cu is the device arm.

#define AETHER_RUN_COUNT_COMMITS 1

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/accum/run.h>

namespace aether_tests {
namespace {

struct Case;

constexpr bool kDevice = false;

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
    namespace run = aether::accum::run;
    const std::size_t n = c.key.size();
    std::vector<T> hi(c.slots, T(0)), lo(c.slots, T(0));
    auto target = aether::make_view<T, aether::dyn>(hi.data(), aether::Device(kDLCPU), c.slots);
    auto comp   = aether::make_view<T, aether::dyn>(lo.data(), aether::Device(kDLCPU), c.slots);
    aether_run_commits[0] = aether_run_commits[1] = 0;
    const long long ranges = static_cast<long long>((n + kHostRange - 1) / kHostRange);
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < ranges; ++r) {
        auto open = run::open(target);
        const std::size_t end = std::min(n, static_cast<std::size_t>(r + 1) * kHostRange);
        for (std::size_t l = static_cast<std::size_t>(r) * kHostRange; l < end; ++l) {
            const auto self = aether::SampleIndex::make(static_cast<std::size_t>(static_cast<long long>(l) + c.own));
            const auto at   = aether::SampleIndex::make(static_cast<std::size_t>(c.key[l]));
            const T term    = static_cast<T>(termOf(l));
            if (compensated) {
                switch (c.form) {
                case Form::Aligned32: run::addCompensated<32>(target, comp, at, term, self, open); break;
                case Form::Aligned16: run::addCompensated<16>(target, comp, at, term, self, open); break;
                case Form::Tested: run::addTestedCompensated(target, comp, at, term, open); break;
                case Form::Sorted: run::addSortedCompensated(target, comp, at, term, open); break;
                }
            } else {
                switch (c.form) {
                case Form::Aligned32: run::add<32>(target, at, term, self, open); break;
                case Form::Aligned16: run::add<16>(target, at, term, self, open); break;
                case Form::Tested: run::addTested(target, at, term, open); break;
                case Form::Sorted: run::addSorted(target, at, term, open); break;
                }
            }
        }
        run::close(open);
    }
    commits[0] = aether_run_commits[0];
    commits[1] = aether_run_commits[1];
    out.assign(c.slots, 0.0L);
    for (std::size_t k = 0; k < c.slots; ++k)
        out[k] = static_cast<long double>(hi[k]) + static_cast<long double>(lo[k]);
}

} // namespace
} // namespace aether_tests
