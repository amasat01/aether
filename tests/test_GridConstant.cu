// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `AETHER_GRID_CONSTANT()` over a batched vector `View`.
//
// CUDA-only by construction (no `.cpp` pair): `AETHER_GRID_CONSTANT()` is
// undefined in `AETHER_CPP_MODE` builds (`aether/macros.h` only defines it
// in the CUDA-mode branch, matching `AETHER_KERNEL`/`AETHER_SHARED` — all
// three are genuinely CUDA-only concepts).
//
// The finding this fixes in place: `AETHER_GRID_CONSTANT()` expands to
// plain `const` off-device, so a grid-constant vector parameter is a const
// `View` on the host pass. `View::operator[](const SampleIndex&)` (the
// `view[i].get()` read idiom) is not `const`-qualified (it binds a
// non-const `View&` into the `SampleRef` it returns) and is therefore not
// callable on it; `View::operator()` is `const`-qualified, so a
// grid-constant vector input must be read per-component through it —
// exactly what the N-scalar `Item` constructor exists to make ergonomic.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Array;
using aether::Item;
using aether::SampleIndex;

// `Array::deviceView()` returns `layout_stride`, not the compact
// `Vec3dView`/`ScalarView<double>` (`layout_right`) aliases — same
// convention as `test_Partition.cu`'s own `Block3dView`.
using Vec3dStridedView   = aether::View<double, aether::extents<3, aether::dyn>, aether::layout_stride>;
using ScalarDStridedView = aether::View<double, aether::extents<aether::dyn>, aether::layout_stride>;

class GridConstantTest : public ::testing::Test { };

AETHER_KERNEL()
void gridConstantReadKernel(AETHER_GRID_CONSTANT() Vec3dStridedView vec, ScalarDStridedView out)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= out.samples())
        return;
    // Per-component reads through operator() — see the file docstring: a
    // grid-constant `Vec3dView` is const here, and only operator() is
    // const-qualified.
    const Item<double, 3> v{ vec(0, i.global()), vec(1, i.global()), vec(2, i.global()) };
    out(i.global()) = v.get<0>() + v.get<1>() + v.get<2>();
}

TEST_F(GridConstantTest, ReadOnlyVectorViaOperatorParen)
{
    constexpr std::size_t N = 129; // several full blocks + a genuine partial tail
    Array<double, 3> a(N);
    Array<double> out(N);
    auto ah = a.hostView();
    for (std::size_t idx = 0; idx < N; ++idx) {
        ah(0, idx) = static_cast<double>(idx) + 1.0;
        ah(1, idx) = static_cast<double>(idx) * 2.0 + 0.5;
        ah(2, idx) = static_cast<double>(idx) * -1.0 + 0.25;
    }
    a.upload();

    const auto cfg = aether::cuda::launchConfig(N, 128);
    gridConstantReadKernel<<<cfg.blocks, cfg.threads>>>(a.deviceView(), out.deviceView());
    aether::cuda::checkLastLaunch("gridConstantReadKernel");

    out.download();
    auto oh = out.hostView();
    for (std::size_t idx = 0; idx < N; ++idx) {
        const double expected = ah(0, idx) + ah(1, idx) + ah(2, idx);
        EXPECT_EQ(oh(idx), expected);
    }
}

} // namespace
} // namespace aether_tests
