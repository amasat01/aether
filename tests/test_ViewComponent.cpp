// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `View::component<I>()` tests (host / AETHER_CPP_MODE build;
// test_ViewComponent.cu covers the same fixture): AETHER_CPP_MODE has a
// single kDLCPU chunk, so hostView() and deviceView() alias the same
// memory -- the .cu twin's device __global__ read/write-through is here a
// plain host read/write through deviceView() (see test_Array.cpp's own
// note for the identical situation with the plain View path).
//
// `View::component<I>()` returns a rank-1 strided view over component `I`
// of a rank-2 (component, sample) view, with the pitch read off the
// source mapping's stride() -- never recomputed from samples() or any
// padding policy -- so it stays correct even when
// Array::capacity() != Array::samples() (see
// ReadsBackCorrectlyOnceCapacityOutgrowsSamples below).

#include <cstddef>
#include <type_traits>

#include <gtest/gtest.h>

#include <aether/array/Array.h>
#include <aether/layout/Extents.h>
#include <aether/layout/Layout.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class ViewComponentTest : public ::testing::Test { };

TEST_F(ViewComponentTest, WriteThroughComponentLandsInTheParent)
{
    constexpr std::size_t C = 3, N = 10;
    aether::Array<double, C> arr(N);

    auto x = arr.hostView().component<0>();
    auto y = arr.hostView().component<1>();
    auto z = arr.hostView().component<2>();
    for (std::size_t i = 0; i < N; ++i) {
        x(i) = static_cast<double>(i) + 0.5;
        y(i) = static_cast<double>(i) + 100.5;
        z(i) = static_cast<double>(i) + 200.5;
    }

    auto hv = arr.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(hv(0, i), static_cast<double>(i) + 0.5) << "i=" << i;
        EXPECT_EQ(hv(1, i), static_cast<double>(i) + 100.5) << "i=" << i;
        EXPECT_EQ(hv(2, i), static_cast<double>(i) + 200.5) << "i=" << i;
    }
}

TEST_F(ViewComponentTest, ReadsBackWhatTheParentWrote)
{
    constexpr std::size_t C = 3, N = 8;
    aether::Array<double, C> arr(N);
    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 1000 + i);

    auto y = arr.hostView().component<1>();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(y(i), static_cast<double>(1000 + i)) << "i=" << i;
}

TEST_F(ViewComponentTest, ShapeAndPitchMatchTheSourceMapping)
{
    constexpr std::size_t C = 3, N = 12;
    aether::Array<double, C> arr(N);
    auto hv = arr.hostView();
    auto x = hv.component<0>();

    EXPECT_EQ(x.rank(), 1u);
    EXPECT_EQ(x.samples(), hv.samples());
    EXPECT_EQ(x.mapping().stride(0), hv.mapping().stride(1));
    EXPECT_EQ(x.data(), hv.data());
    EXPECT_EQ(x.device(), hv.device());
}

TEST_F(ViewComponentTest, ReadsBackCorrectlyOnceCapacityOutgrowsSamples)
{
    // reserve() past samples() makes hostView()'s pitch (capacity())
    // strictly larger than samples() -- the exact case component<I>()'s
    // docstring warns a samples()-derived pitch would silently mis-stride.
    constexpr std::size_t C = 3, N = 5;
    aether::Array<double, C> arr(N);
    arr.reserve(64);
    ASSERT_GT(arr.capacity(), arr.samples());

    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 10 + i);

    auto comp2 = arr.hostView().component<2>();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(comp2(i), static_cast<double>(20 + i)) << "i=" << i;
}

TEST_F(ViewComponentTest, ConstComponentIsReadOnlyAndMatchesTheParent)
{
    constexpr std::size_t C = 3, N = 6;
    aether::Array<double, C> arr(N);
    auto hv = arr.hostView();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            hv(c, i) = static_cast<double>(c * 10 + i);

    const aether::Array<double, C>& carr = arr;
    auto z = carr.hostView().component<2>();
    static_assert(
        std::is_same_v<decltype(z), aether::View<const double, aether::extents<aether::dyn>, aether::layout_stride>>,
        "component<I>() const must return a read-only View<const T, ...>");
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(z(i), static_cast<double>(20 + i)) << "i=" << i;
}

TEST_F(ViewComponentTest, WriteThroughComponentUploadThenReadThroughDeviceViewLandsInTheParent)
{
    // Acceptance test, host-backend half:
    // "write via the component view on host, upload(), read back through
    // the parent's device view" -- AETHER_CPP_MODE's upload() is a no-op
    // over a single shared chunk (see this file's own header note), so
    // deviceView() reads the SAME memory hostView() just wrote.
    constexpr std::size_t C = 3, N = 9;
    aether::Array<double, C> arr(N);

    auto y = arr.hostView().component<1>();
    for (std::size_t i = 0; i < N; ++i)
        y(i) = static_cast<double>(i) * 3.0 + 1.0;

    arr.upload();

    auto dv = arr.deviceView();
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_EQ(dv(1, i), static_cast<double>(i) * 3.0 + 1.0) << "i=" << i;
}

} // namespace
} // namespace aether_tests
