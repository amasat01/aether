// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Const-read-path tests for expr/view (CUDA build; test_ConstView.cpp is
// the identical host-build twin). `View::operator[](const SampleIndex&)
// const` -> `ConstSampleRef<const View>` (`.get()`, `.eval<Is...>()`, no
// `= += -=`) is DEVICEHOST-safe with no CUDA-specific behaviour a plain
// host build cannot already exercise. This build additionally covers a
// device-kernel leg (`ConstView.DeviceReadsThroughAConstViewKernelArg`): a
// device kernel taking a const-qualified `View` by value (so `v[i]`
// resolves to the const `operator[]` overload inside the kernel body) and
// reading through it, built host-side and passed in — the same
// `tests/test_TableView.cu` `tableViewPlainKernel` shape (build host-side,
// read device-side).
//
// A `const View` is a conforming expression leaf exactly like the mutable
// one — `aether_expression<const View<...>>` must hold (static_assert
// below, mirroring the namespace-scope check `aether/view/View.h` itself
// carries on one representative instantiation). A bool-readable-proxy
// consumer is served by `.get()` (and the finer-grained `.eval<Is...>()`)
// — no implicit conversion operator is added.
//
// "All ranks in the house set" = vector (`Vec3dView`, element_extents rank
// 1) and matrix (`Mat33dView`, rank 2) — the two shapes every other
// rank-generic battery in this tree actually exercises (`test_View.cpp`'s
// own `SoaProof*` rows: pointer-arithmetic for a vector view, then a matrix
// view; no test anywhere instantiates a scalar-rank `Item<T>` — see the
// known-bug note below for why this file follows suit rather than being
// the first). The `aether_expression<const ScalarView<double>>` check below
// is unaffected (it never touches `Item`) and is kept for completeness.
//
// Known bug (found while drafting this file, not fixed here): a rank-0
// `Item<T>` (e.g. `ItemFromExtents<double, extents<>>`, what `ScalarView<T>
// ::element_extents` materializes into) fails to compile any access —
// `operator()()`, `get<>()`, and therefore `SampleRef::get()`/
// `ConstSampleRef::get()` on a `ScalarView` — because `Item::offset_()`
// unconditionally compiles the loop body
// `extArr[i]`/`idxArr[i]` even though it never
// runs for `Rank == 0`, and `Carray<T,0>` (`aether/layout/detail/Carray.h`
// zero-size specialization, by-design) provides no `operator[]` at all
// ("nothing ever reads an element of a zero-size Carray" — true at runtime,
// not at compile time for a non-`if constexpr` loop). Pre-existing:
// `ConstSampleRef::get()` reuses the identical
// `detail::ItemFromExtents`/`detail::assign` machinery `SampleRef::get()`
// already used, unmodified here.
//
// The compile-fail complement to this file — `cv[i] = expr` must not
// compile — cannot live here by construction (a test that compiles is a
// test whose subject compiled): see
// `tests/compile_fail/check_constview_no_assign.sh`.

#include <cstddef>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/backend/cuda/Launch.h>

namespace aether_tests {
namespace {

using aether::ConstSampleRef;
using aether::Device;
using aether::extents;
using aether::Item;
using aether::Mat33d;
using aether::Mat33dView;
using aether::SampleIndex;
using aether::ScalarView;
using aether::Vec3d;
using aether::Vec3dView;
using aether::View;

// ---------------------------------------------------------------------------
// `const View` must satisfy `aether_expression`. Checked here
// again (redundant with `aether/view/View.h`'s own namespace-scope check) on
// EXACTLY the instantiations this file's fixtures use.
// ---------------------------------------------------------------------------
static_assert(aether::aether_expression<const ScalarView<double>>,
    "const ScalarView<double> must satisfy aether_expression");
static_assert(aether::aether_expression<const Vec3dView>, "const Vec3dView must satisfy aether_expression");
static_assert(aether::aether_expression<const Mat33dView>, "const Mat33dView must satisfy aether_expression");

class ConstViewTest : public ::testing::Test { };

TEST_F(ConstViewTest, ConstGetReadsAVectorRankView)
{
    constexpr std::size_t N = 5;
    std::vector<double> buf(3 * N);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), Device(kDLCPU), N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            v(c, i) = static_cast<double>(c) * 10.0 + static_cast<double>(i);

    const auto& cv = v;
    for (std::size_t i = 0; i < N; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        const Vec3d snap      = cv[si].get();
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(snap(c), v(c, i));
    }

    // The snapshot is a genuine copy, same claim `SampleRef::get()` makes
    // (tests/test_ExprAssign.cpp `SampleRefGetMaterializesAnItemSnapshot`).
    const Vec3d snap0 = cv[SampleIndex::make(0)].get();
    v(0, 0)            = 999.0;
    EXPECT_NE(snap0(0), v(0, 0));
}

TEST_F(ConstViewTest, ConstGetReadsAMatrixRankView)
{
    constexpr std::size_t N = 3;
    std::vector<double> buf(3 * 3 * N);
    auto v = aether::make_view<double, 3, 3, aether::dyn>(buf.data(), Device(kDLCPU), N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 3; ++c)
                v(r, c, i) = static_cast<double>(r * 3 + c) + static_cast<double>(i) * 100.0;

    const auto& cv        = v;
    const SampleIndex si   = SampleIndex::make(1);
    const Mat33d snap      = cv[si].get();
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(snap(r, c), v(r, c, 1));
}

TEST_F(ConstViewTest, ConstEvalForwardsDirectComponentReadsWithoutBuildingAnItem)
{
    constexpr std::size_t N = 4;
    std::vector<double> buf(3 * N);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), Device(kDLCPU), N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            v(c, i) = static_cast<double>(c) + static_cast<double>(i) * 0.5;

    const auto& cv       = v;
    const SampleIndex si = SampleIndex::make(2);
    const ConstSampleRef<std::remove_reference_t<decltype(cv)>> csr = cv[si];
    EXPECT_DOUBLE_EQ((csr.template eval<0>()), v(0, 2));
    EXPECT_DOUBLE_EQ((csr.template eval<1>()), v(1, 2));
    EXPECT_DOUBLE_EQ((csr.template eval<2>()), v(2, 2));
    // Same reads directly off the temporary proxy `cv[si]` returns.
    EXPECT_DOUBLE_EQ((cv[si].template eval<0>()), v(0, 2));
}

// ---------------------------------------------------------------------------
// Device leg
// ---------------------------------------------------------------------------

constexpr std::size_t kDeviceN = 5;

/** @brief Walk every sample of a vector-rank `View` through its CONST read
 *  path — the parameter is a CONST-qualified `View` BY VALUE, so `v[i]`
 *  inside this kernel body resolves to `View::operator[](const
 *  SampleIndex&) const` (never the mutable overload; a `const`-qualified
 *  local could not call it). Mirrors `tests/test_TableView.cu`'s
 *  `tableViewPlainKernel`: built host-side, read device-side, single
 *  thread does the whole walk. */
__global__ void constViewGetKernel(const View<double, extents<3, aether::dyn>> v, double* out)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < kDeviceN; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        const Vec3d snap      = v[si].get();
        for (std::size_t c = 0; c < 3; ++c)
            out[n++] = snap(c);
    }
}

bool deviceAvailable()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

} // namespace

TEST(ConstView, DeviceReadsThroughAConstViewKernelArg)
{
    if (!deviceAvailable())
        GTEST_SKIP() << "no CUDA device";

    std::vector<double> hin(3 * kDeviceN);
    for (std::size_t i = 0; i < kDeviceN; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            hin[c * kDeviceN + i] = static_cast<double>(c) * 10.0 + static_cast<double>(i);

    double* din = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&din, hin.size() * sizeof(double)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(din, hin.data(), hin.size() * sizeof(double), cudaMemcpyHostToDevice));

    const auto v = aether::make_view<double, 3, aether::dyn>(din, Device(kDLCUDA), kDeviceN);

    double* dout = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dout, 3 * kDeviceN * sizeof(double)));
    constViewGetKernel<<<1, 1>>>(v, dout);
    aether::cuda::checkLastLaunch("constViewGetKernel");
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<double> hout(3 * kDeviceN, -1.0);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(hout.data(), dout, hout.size() * sizeof(double), cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < kDeviceN; ++i)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(hout[i * 3 + c], hin[c * kDeviceN + i]);

    cudaFree(din);
    cudaFree(dout);
}

} // namespace aether_tests
