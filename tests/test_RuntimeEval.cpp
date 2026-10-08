// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Runtime evaluator tests (host / AETHER_CPP_MODE build; test_RuntimeEval.cu
// covers the same fixture and case names), but test_RuntimeEval.cu
// additionally carries a device-vs-static-kernel validation check (runtime
// evaluator kernel vs a static kernel over the same chunk, memcmp exact for
// `a + s*b`) — a genuinely CUDA-only comparison this build cannot exercise
// (there is no separate "kernel" to diverge from on the host arm).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/device/Device.h>
#include <aether/dtype/DType.h>
#include <aether/err/Error.h>
#include <aether/eval/Runtime.h>
#include <aether/layout/Extents.h>
#include <aether/view/MakeRuntimeView.h> // fromView() lives here, split out of view/RuntimeView.h
#include <aether/view/MakeView.h> // make_view() lives here, split out of view/View.h
#include <aether/view/RuntimeView.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class RuntimeEvalTest : public ::testing::Test { };

// Deterministic fill matching the twin-harness convention elsewhere in this
// suite (tests/twin/test_TwinArithmetic.*): varies with both component and
// sample so a transposed/shape bug would not cancel.
void fillDeterministic(std::size_t dim, std::size_t n, std::vector<double>& a, std::vector<double>& b)
{
    a.assign(dim * n, 0.0);
    b.assign(dim * n, 0.0);
    for (std::size_t idx = 0; idx < n; ++idx) {
        for (std::size_t c = 0; c < dim; ++c) {
            a[c * n + idx] = static_cast<double>(idx) * 0.75 + static_cast<double>(c) * 0.1 - 3.0;
            b[c * n + idx] = static_cast<double>(idx) * -0.5 + static_cast<double>(c) + 1.25;
        }
    }
}

TEST_F(RuntimeEvalTest, AssignCopiesSameDtype)
{
    constexpr std::size_t C = 3, N = 5;
    std::vector<double> aBuf, bBufUnused;
    fillDeterministic(C, N, aBuf, bBufUnused);
    std::vector<double> outBuf(C * N, -1.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);
    const aether::RuntimeView aRt = aether::fromView(av);
    const aether::RuntimeView oRt = aether::fromView(ov);

    aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aRt, aether::RuntimeView{}, oRt);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i));
}

TEST_F(RuntimeEvalTest, AssignCastsDoubleToFloat)
{
    constexpr std::size_t C = 3, N = 5;
    std::vector<double> aBuf, unused;
    fillDeterministic(C, N, aBuf, unused);
    std::vector<float> outBuf(C * N, -1.0f);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<float, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);
    const aether::RuntimeView aRt = aether::fromView(av);
    const aether::RuntimeView oRt = aether::fromView(ov);

    aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aRt, aether::RuntimeView{}, oRt);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_FLOAT_EQ(ov(c, i), static_cast<float>(av(c, i)));
}

TEST_F(RuntimeEvalTest, AssignCastsFloatToDouble)
{
    constexpr std::size_t C = 3, N = 5;
    std::vector<float> aBuf(C * N);
    for (std::size_t i = 0; i < C * N; ++i)
        aBuf[i] = static_cast<float>(i) * 0.5f - 1.25f;
    std::vector<double> outBuf(C * N, -1.0);

    auto av = aether::make_view<float, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);
    const aether::RuntimeView aRt = aether::fromView(av);
    const aether::RuntimeView oRt = aether::fromView(ov);

    aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aRt, aether::RuntimeView{}, oRt);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), static_cast<double>(av(c, i)));
}

TEST_F(RuntimeEvalTest, AddMatchesHandReference)
{
    constexpr std::size_t C = 3, N = 5;
    std::vector<double> aBuf, bBuf;
    fillDeterministic(C, N, aBuf, bBuf);
    std::vector<double> outBuf(C * N, 0.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto bv = aether::make_view<double, C, aether::dyn>(bBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);

    aether::eval::runtimeEval(
        aether::eval::RuntimeOp::Add, aether::fromView(av), aether::fromView(bv), aether::fromView(ov));

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) + bv(c, i));
}

TEST_F(RuntimeEvalTest, SubMatchesHandReference)
{
    constexpr std::size_t C = 3, N = 5;
    std::vector<double> aBuf, bBuf;
    fillDeterministic(C, N, aBuf, bBuf);
    std::vector<double> outBuf(C * N, 0.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto bv = aether::make_view<double, C, aether::dyn>(bBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);

    aether::eval::runtimeEval(
        aether::eval::RuntimeOp::Sub, aether::fromView(av), aether::fromView(bv), aether::fromView(ov));

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) - bv(c, i));
}

TEST_F(RuntimeEvalTest, ScaleMatchesHandReference)
{
    constexpr std::size_t C = 3, N = 5;
    constexpr double s = -2.5;
    std::vector<double> aBuf, unused;
    fillDeterministic(C, N, aBuf, unused);
    std::vector<double> outBuf(C * N, 0.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);

    aether::eval::runtimeEval(
        aether::eval::RuntimeOp::Scale, aether::fromView(av), aether::RuntimeView{}, aether::fromView(ov), s);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), s * av(c, i));
}

TEST_F(RuntimeEvalTest, AddScaledMatchesHandReference)
{
    constexpr std::size_t C = 3, N = 5;
    constexpr double s = 0.5;
    std::vector<double> aBuf, bBuf;
    fillDeterministic(C, N, aBuf, bBuf);
    std::vector<double> outBuf(C * N, 0.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto bv = aether::make_view<double, C, aether::dyn>(bBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);

    aether::eval::runtimeEval(
        aether::eval::RuntimeOp::AddScaled, aether::fromView(av), aether::fromView(bv), aether::fromView(ov), s);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) + s * bv(c, i));
}

TEST_F(RuntimeEvalTest, SubScaledMatchesHandReference)
{
    constexpr std::size_t C = 3, N = 5;
    constexpr double s = 1.75;
    std::vector<double> aBuf, bBuf;
    fillDeterministic(C, N, aBuf, bBuf);
    std::vector<double> outBuf(C * N, 0.0);

    auto av = aether::make_view<double, C, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), N);
    auto bv = aether::make_view<double, C, aether::dyn>(bBuf.data(), aether::Device(kDLCPU), N);
    auto ov = aether::make_view<double, C, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), N);

    aether::eval::runtimeEval(
        aether::eval::RuntimeOp::SubScaled, aether::fromView(av), aether::fromView(bv), aether::fromView(ov), s);

    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_DOUBLE_EQ(ov(c, i), av(c, i) - s * bv(c, i));
}

TEST_F(RuntimeEvalTest, StridedNonContiguousViewsAreAddressedCorrectly)
{
    // Hand-built non-contiguous 1-D descriptors: a strided VIEW of every
    // OTHER element in a flat buffer (extent 5, stride 2) — genuinely
    // exercises "strided access via the runtime strides",
    // not just the SoA contiguous case every `fromView()` call above uses.
    constexpr std::size_t extent = 5, stride = 2;
    std::vector<double> aBuf(extent * stride, 0.0), outBuf(extent * stride, -1.0);
    for (std::size_t i = 0; i < aBuf.size(); ++i)
        aBuf[i] = static_cast<double>(i) * 1.5 - 3.0;

    aether::RuntimeView aRt;
    aRt.data       = aBuf.data();
    aRt.dtype      = aether::dtype_of<double>();
    aRt.device     = aether::Device(kDLCPU);
    aRt.rank       = 1;
    aRt.extents[0] = extent;
    aRt.strides[0] = stride;

    aether::RuntimeView oRt;
    oRt.data       = outBuf.data();
    oRt.dtype      = aether::dtype_of<double>();
    oRt.device     = aether::Device(kDLCPU);
    oRt.rank       = 1;
    oRt.extents[0] = extent;
    oRt.strides[0] = stride;

    aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aRt, aether::RuntimeView{}, oRt);

    for (std::size_t k = 0; k < extent; ++k) {
        EXPECT_DOUBLE_EQ(outBuf[k * stride], aBuf[k * stride]);
        // Every UN-touched (odd) slot must remain at its sentinel value —
        // proves the evaluator did NOT walk the buffer contiguously.
        if (k * stride + 1 < outBuf.size()) {
            EXPECT_DOUBLE_EQ(outBuf[k * stride + 1], -1.0);
        }
    }
}

TEST_F(RuntimeEvalTest, ShapeMismatchThrows)
{
    std::vector<double> aBuf(3 * 5, 0.0), outBuf(3 * 4, 0.0);
    auto av = aether::make_view<double, 3, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), 5);
    auto ov = aether::make_view<double, 3, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), 4);
    EXPECT_THROW(aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aether::fromView(av),
                     aether::RuntimeView{}, aether::fromView(ov)),
        aether::Error);
}

TEST_F(RuntimeEvalTest, DtypeMismatchThrowsForNonAssignOp)
{
    std::vector<double> aBuf(3 * 5, 0.0), outBuf(3 * 5, 0.0);
    std::vector<float> bBuf(3 * 5, 0.0f);
    auto av = aether::make_view<double, 3, aether::dyn>(aBuf.data(), aether::Device(kDLCPU), 5);
    auto bv = aether::make_view<float, 3, aether::dyn>(bBuf.data(), aether::Device(kDLCPU), 5);
    auto ov = aether::make_view<double, 3, aether::dyn>(outBuf.data(), aether::Device(kDLCPU), 5);
    EXPECT_THROW(aether::eval::runtimeEval(
                     aether::eval::RuntimeOp::Add, aether::fromView(av), aether::fromView(bv), aether::fromView(ov)),
        aether::Error);
}

TEST_F(RuntimeEvalTest, UnsupportedDtypeThrows)
{
    std::vector<std::int32_t> aBuf(3 * 5, 0);
    std::vector<std::int32_t> outBuf(3 * 5, 0);
    aether::RuntimeView aRt;
    aRt.data       = aBuf.data();
    aRt.dtype      = aether::dtype_of<std::int32_t>(); // NOT double/float — v1 unsupported by the evaluator
    aRt.device     = aether::Device(kDLCPU);
    aRt.rank       = 2;
    aRt.extents[0] = 3;
    aRt.extents[1] = 5;
    aRt.strides[0] = 5;
    aRt.strides[1] = 1;
    aether::RuntimeView oRt = aRt;
    oRt.data                = outBuf.data();

    EXPECT_THROW(aether::eval::runtimeEval(aether::eval::RuntimeOp::Assign, aRt, aether::RuntimeView{}, oRt),
        aether::Error);
}

} // namespace
} // namespace aether_tests
