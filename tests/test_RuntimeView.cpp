// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// RuntimeView tests (host / AETHER_CPP_MODE build; test_RuntimeView.cu
// covers the same fixture and case names), but the covered device-kind set
// differs by construction: this build has only kDLCPU available, so
// test_RuntimeView.cu additionally covers a kDLCUDA promotion
// (pointer-arithmetic only — no dereference of device memory, matching
// test_View.cu's own pattern).

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/dtype/DType.h>
#include <aether/err/Error.h>
#include <aether/layout/Extents.h>
#include <aether/view/RuntimeView.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class RuntimeViewTest : public ::testing::Test { };

TEST_F(RuntimeViewTest, FromViewCapturesDataDtypeDeviceRankExtentsStrides)
{
    constexpr std::size_t C = 3, N = 7;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto v     = aether::make_view<double, C, aether::dyn>(chunk, N);

    const aether::RuntimeView rt = aether::fromView(v);
    EXPECT_EQ(rt.data, static_cast<void*>(v.data()));
    EXPECT_TRUE(rt.dtype == aether::dtype_of<double>());
    EXPECT_EQ(rt.device, aether::Device(kDLCPU));
    EXPECT_EQ(rt.rank, 2u);
    EXPECT_EQ(rt.extents[0], C);
    EXPECT_EQ(rt.extents[1], N);
    // layout_right / SoA: stride(0) = N (the batch mode's extent), stride(1) = 1.
    EXPECT_EQ(rt.strides[0], static_cast<aether::offset_t>(N));
    EXPECT_EQ(rt.strides[1], 1u);
    EXPECT_EQ(rt.size(), static_cast<aether::offset_t>(C * N));
}

TEST_F(RuntimeViewTest, FromViewOverAFullyStaticExtentsShapeHasNoBatchMode)
{
    constexpr std::size_t R = 3, C = 3;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), R * C * sizeof(double));
    auto v     = aether::make_view<double, R, C>(chunk);

    const aether::RuntimeView rt = aether::fromView(v);
    EXPECT_EQ(rt.rank, 2u);
    EXPECT_EQ(rt.extents[0], R);
    EXPECT_EQ(rt.extents[1], C);
    EXPECT_EQ(rt.strides[0], C);
    EXPECT_EQ(rt.strides[1], 1u);
}

TEST_F(RuntimeViewTest, PromotionRoundTripsPointerIdentityAndValues)
{
    constexpr std::size_t N = 5;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * N * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, N);
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < N; ++i)
            v(c, i) = static_cast<double>(c * 10 + i);

    const aether::RuntimeView rt = aether::fromView(v);
    const aether::Vec3dView promoted = rt.as<aether::Vec3dView>();

    EXPECT_EQ(promoted.data(), v.data()); // zero-copy: SAME pointer
    for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(promoted(c, i), v(c, i));
}

TEST_F(RuntimeViewTest, PromotionAcceptsAFullyStaticExtentsShape)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 9 * sizeof(double));
    auto v     = aether::make_view<double, 3, 3>(chunk);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            v(r, c) = static_cast<double>(r * 3 + c);

    const aether::RuntimeView rt = aether::fromView(v);
    using StaticT               = aether::View<double, aether::extents<3, 3>>;
    const StaticT promoted      = rt.as<StaticT>();
    EXPECT_EQ(promoted.data(), v.data());
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_EQ(promoted(r, c), v(r, c));
}

TEST_F(RuntimeViewTest, PromotionRejectsDtypeMismatch)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5);
    const aether::RuntimeView rt = aether::fromView(v);
    EXPECT_THROW((rt.as<aether::VecView<float, 3>>()), aether::Error);
}

TEST_F(RuntimeViewTest, PromotionRejectsRankMismatch)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5); // rank 2
    const aether::RuntimeView rt = aether::fromView(v);
    EXPECT_THROW((rt.as<aether::MatView<double, 3, 3>>()), aether::Error); // rank 3
}

TEST_F(RuntimeViewTest, PromotionRejectsExtentMismatch)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5);
    const aether::RuntimeView rt = aether::fromView(v);
    EXPECT_THROW((rt.as<aether::VecView<double, 4>>()), aether::Error);
}

TEST_F(RuntimeViewTest, PromotionRejectsStridePatternMismatch)
{
    // Hand-built descriptor: extents {3,5} but TRANSPOSED (column-major)
    // strides {1,3} — layout_right's row-major expectation is {5,1}.
    std::vector<double> buf(15, 0.0);
    aether::RuntimeView rt;
    rt.data     = buf.data();
    rt.dtype    = aether::dtype_of<double>();
    rt.device   = aether::Device(kDLCPU);
    rt.rank     = 2;
    rt.extents[0] = 3;
    rt.extents[1] = 5;
    rt.strides[0] = 1;
    rt.strides[1] = 3;

    using StaticT = aether::View<double, aether::extents<3, 5>>;
    EXPECT_THROW(rt.as<StaticT>(), aether::Error);
}

TEST_F(RuntimeViewTest, PromotionRejectsUnknownDeviceKind)
{
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), 3 * 5 * sizeof(double));
    auto v     = aether::make_view<double, 3, aether::dyn>(chunk, 5);
    aether::RuntimeView rt = aether::fromView(v);
    rt.device               = aether::Device(kDLOpenCL); // never supported, any build
    EXPECT_THROW((rt.as<aether::VecView<double, 3>>()), aether::Error);
}

} // namespace
} // namespace aether_tests
