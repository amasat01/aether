// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Device-safe umbrella tests (CUDA build; test_DeviceUmbrella.cpp is the
// twin over the same fixture and case names) — only one of the two
// compiles into any given aether_tests binary: tests/CMakeLists.txt globs
// test_*.cu here and test_*.cpp in AETHER_CPP_MODE.
//
// Every case below runs entirely on the host (no kernel launch — the
// claims under test are about which headers a TU can reach, not about
// device execution). The actual device-side claim (does `aether/device.h`
// compile and materialize an assignment under NVRTC) is a separate,
// non-GTest gate: tools/nvrtc/audit_device_umbrella.sh.
//
// Covers two claims:
//  (a) `aether/device.h` alone compiles and materializes an ET assignment
//      (`Item c = a + b`, through a `View<double>` -> `SampleRef::get()` ->
//      `Item`'s out-of-line expression ctor, expr/Assign.h) — the exact
//      mechanism that used to drag in host-only `err/Error.h`
//      unconditionally before the umbrella split.
//  (b) `make_view()` is reachable through `aether/aether.h` and through
//      `aether/view/MakeView.h` directly — the split moved it out of
//      `aether/view/View.h`, not out of existence. The complementary
//      negative claim ("not through aether/device.h alone") is a
//      must-not-compile property, which cannot live in a binary that
//      compiled — see tests/compile_fail/check_device_umbrella_hostfree.sh.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/view/MakeView.h>

namespace aether_tests {
namespace {

class DeviceUmbrellaTest : public ::testing::Test { };

TEST_F(DeviceUmbrellaTest, DeviceHeaderMaterializesItemAssignmentOverView)
{
    // aether/device.h alone (reached here through aether/aether.h, which
    // includes it unconditionally as its device-safe half — the NVRTC audit
    // is what proves device.h compiles STANDALONE, see the file docstring)
    // -- View<double> -> SampleRef::get() -> Item, then a materializing
    // `Item c = a + b`.
    using Ext = aether::extents<3, aether::dyn>;
    Ext ext(1);
    aether::layout_right::mapping<Ext> map(ext);
    double abuf[3] = { 1.0, 2.0, 3.0 };
    double bbuf[3] = { 4.0, 5.0, 6.0 };
    aether::View<double, Ext> av(abuf, map, aether::Device(kDLCPU));
    aether::View<double, Ext> bv(bbuf, map, aether::Device(kDLCPU));
    const aether::SampleIndex i = aether::SampleIndex::make(0);

    aether::Item<double, 3> a = av[i].get();
    aether::Item<double, 3> b = bv[i].get();
    aether::Item<double, 3> c = a + b; // materializing ET assignment

    EXPECT_DOUBLE_EQ(c(0), 5.0);
    EXPECT_DOUBLE_EQ(c(1), 7.0);
    EXPECT_DOUBLE_EQ(c(2), 9.0);
}

TEST_F(DeviceUmbrellaTest, MakeViewReachableThroughAetherH)
{
    std::vector<double> buf(3 * 2, 0.0);
    auto v = aether::make_view<double, 3, aether::dyn>(buf.data(), aether::Device(kDLCPU), 2);
    EXPECT_EQ(v.samples(), static_cast<aether::offset_t>(2));
}

TEST_F(DeviceUmbrellaTest, MakeViewReachableThroughMakeViewHeaderDirectly)
{
    // The umbrella split moved make_view() out of view/View.h into
    // view/MakeView.h -- it did not move it out of existence. This TU
    // #includes view/MakeView.h explicitly (above), independent of
    // whatever else aether/aether.h happens to pull in.
    std::vector<double> buf(4, 0.0);
    auto v = aether::make_view<double, aether::dyn>(buf.data(), aether::Device(kDLCPU), 4);
    EXPECT_EQ(v.samples(), static_cast<aether::offset_t>(4));
}

} // namespace
} // namespace aether_tests
