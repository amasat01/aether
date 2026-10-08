// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// view/TableHandle.h tests (CUDA build; test_TableHandle.cpp covers the
// host-only correctness). This file additionally exercises the
// device-compiled `__ldg` read path via a real kernel launch.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::SampleIndex;
using aether::TableHandle;

class TableHandleTest : public ::testing::Test { };

// `TableHandle`'s ctor (`aether/view/TableHandle.h`) is fixed to
// `View<T, extents<dyn>, layout_right>` — a `view/` internals change to
// widen it is out of scope here. For a RANK-1
// (scalar) view this is not a real restriction: `layout_right` and
// `layout_stride` address IDENTICAL offsets when there is no outer/
// component mode to pitch against (`offset(i) == i` either way — see
// test_ArrayCapacity.cpp's ScalarArrayPitchStrideIsAlwaysOneRegardlessOfCapacity),
// so re-deriving a compact view over `Array::hostView()`/`deviceView()`'s
// OWN memory via the raw-pointer `make_view` overload is exact, not an
// approximation.
template<class T>
static aether::View<T, aether::extents<aether::dyn>> asPacked(
    const aether::View<T, aether::extents<aether::dyn>, aether::layout_stride>& v)
{
    return aether::make_view<T, aether::dyn>(v.data(), v.device(), v.samples());
}

TEST_F(TableHandleTest, FlatReadMatchesUnderlyingValues)
{
    constexpr std::size_t N = 8;
    Array<double> table(N);
    auto v = table.hostView();
    for (std::size_t i = 0; i < N; ++i)
        v(static_cast<aether::offset_t>(i)) = static_cast<double>(i) * 1.5;

    TableHandle<double> h(asPacked(v));
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(h[static_cast<aether::offset_t>(i)], static_cast<double>(i) * 1.5);
}

TEST_F(TableHandleTest, DataPointerMatchesUnderlyingView)
{
    constexpr std::size_t N = 4;
    Array<double> table(N);
    auto v = table.hostView();

    TableHandle<double> h(asPacked(v));
    EXPECT_EQ(h.data(), v.data());
}

TEST_F(TableHandleTest, DefaultConstructedHandleHasNullData)
{
    TableHandle<double> h;
    EXPECT_EQ(h.data(), nullptr);
}

// `Array::deviceView()` for a rank-1 Array returns
// `View<double, extents<dyn>, layout_stride>` (see `asPacked()` above for
// why this is offset-identical to `ScalarView<double>` for rank 1).
using ScalarDStridedView = aether::View<double, aether::extents<aether::dyn>, aether::layout_stride>;

AETHER_KERNEL()
void tableReadKernel(TableHandle<double> table, ScalarDStridedView out)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= out.samples())
        return;
    out(i.global()) = table[i.global()];
}

TEST_F(TableHandleTest, DeviceLdgReadMatchesHostValues)
{
    constexpr std::size_t N = 257; // several full blocks + a genuine partial tail
    Array<double> table(N), out(N);
    auto th = table.hostView();
    for (std::size_t i = 0; i < N; ++i)
        th(static_cast<aether::offset_t>(i)) = static_cast<double>(i) * 2.0 - 3.0;
    table.upload();

    TableHandle<double> handle(asPacked(table.deviceView()));
    const auto cfg = aether::cuda::launchConfig(N, 128);
    tableReadKernel<<<cfg.blocks, cfg.threads>>>(handle, out.deviceView());
    aether::cuda::checkLastLaunch("tableReadKernel");

    out.download();
    auto oh = out.hostView();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(oh(static_cast<aether::offset_t>(i)), th(static_cast<aether::offset_t>(i)));
}

} // namespace
} // namespace aether_tests
