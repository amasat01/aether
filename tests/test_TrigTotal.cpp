// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Totality of FP64 `aether::math::sin`/`cos`/`sincos` on the host route
// (`std::`): the same corpus and gate as the device twin
// test_TrigTotal.cu, so a host build proves the host leg total too. The
// corpus and scoring live in test_TrigTotal_common.h.

#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

#include "test_TrigTotal_common.h"

namespace aether_tests {
namespace TrigTotalTest {

constexpr bool kUsePinned = false;

static Columns evaluate()
{
    Columns r;
    r.xs = makeCorpus();
    for (const double x : r.xs) {
        r.sin.push_back(aether::math::sin(x));
        r.cos.push_back(aether::math::cos(x));
        double s, c;
        aether::math::sincos(x, &s, &c);
        r.sc.push_back(s);
        r.cc.push_back(c);
    }
    return r;
}

TEST(TrigTotalTest, SpecialsHaveCStandardResults)
{
    expectCStandardSpecials(evaluate());
}

TEST(TrigTotalTest, SinIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.sin, kSin, kUsePinned, "sin");
}

TEST(TrigTotalTest, CosIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.cos, kCos, kUsePinned, "cos");
}

TEST(TrigTotalTest, SinCosIsTotalAgainstLibm)
{
    const Columns r = evaluate();
    score(r.xs, r.sc, kSin, kUsePinned, "sincos.sin");
    score(r.xs, r.cc, kCos, kUsePinned, "sincos.cos");
}

} // namespace TrigTotalTest
} // namespace aether_tests
