// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A rank-0 `Item<T>` must be readable. `Item::offset_()` folded the offset
// unconditionally over `Rank`, and `Carray<T,0>` deliberately has no
// `operator[]`, so `Item<T>()()` / `ScalarView[i].get()` failed to compile
// before the `if constexpr (Rank == 0)` guard in `aether/view/Item.h`.
#pragma once
#include "aether/aether.h"
#include <gtest/gtest.h>
#include <vector>

namespace aether_tests {

inline void itemRank0ReadsAndWrites()
{
    aether::Item<double> s{};              // rank-0 item: one scalar
    s() = 2.5;
    EXPECT_EQ(s(), 2.5);
    EXPECT_EQ(s.get<>(), 2.5);
}

inline void scalarViewGetMaterializesARank0Item()
{
    std::vector<double> buf{ 1.0, 2.0, 3.0 };
    const aether::Device dev(kDLCPU);
    auto v = aether::make_view<double, aether::dyn>(buf.data(), dev, buf.size());
    const auto item = v[aether::SampleIndex::make(1)].get();   // rank-0 materialization
    EXPECT_EQ(item(), 2.0);
}

} // namespace aether_tests
