// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// view/TableHandle.h tests (host / AETHER_CPP_MODE build;
// test_TableHandle.cu covers identical host-logic content plus a
// device-compiled `__ldg` read path via a real kernel launch).

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/index/Offset.h>
#include <aether/layout/Extents.h>
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/TableHandle.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class TableHandleTest : public ::testing::Test { };

TEST_F(TableHandleTest, FlatReadMatchesUnderlyingValues)
{
    constexpr std::size_t N = 8;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), N * sizeof(double));
    auto v     = aether::make_view<double, aether::dyn>(chunk, N);
    for (std::size_t i = 0; i < N; ++i)
        v(static_cast<aether::offset_t>(i)) = static_cast<double>(i) * 1.5;

    aether::TableHandle<double> h(v);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(h[static_cast<aether::offset_t>(i)], static_cast<double>(i) * 1.5);
}

TEST_F(TableHandleTest, DataPointerMatchesUnderlyingView)
{
    constexpr std::size_t N = 4;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), N * sizeof(double));
    auto v     = aether::make_view<double, aether::dyn>(chunk, N);

    aether::TableHandle<double> h(v);
    EXPECT_EQ(h.data(), v.data());
}

TEST_F(TableHandleTest, DefaultConstructedHandleHasNullData)
{
    aether::TableHandle<double> h;
    EXPECT_EQ(h.data(), nullptr);
}

} // namespace
} // namespace aether_tests
