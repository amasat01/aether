// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// TableView tests (host / AETHER_CPP_MODE build; test_TableView.cu covers
// the same fixed cases) over the thin `aether::TableView`/
// `aether::makeTableView` facade (`aether/view/TableView.h`) over
// `aether::View`/`aether::make_view` — see that header's file docstring for
// the semantics (offset folds into the pointer; no duck-typed carrier).
//
// The property under test: the view's `at(i,j,k)` fold (here:
// `operator()(i,j,k)`) reproduces, exactly, the row-major stride arithmetic
// a consumer would otherwise write by hand. The ramp trick: the carrier is
// filled so element `i` holds the value `i`, so a read through the view
// returns the flat index it resolved to.
//
// `.cu` twin: these same cases verbatim, plus a device-kernel leg (row 13,
// a plain foreign-pointer carrier) which cannot run here. Row 14
// (`DeviceTextureCarrierReadsMatchThePlainCarrier`) has no counterpart yet
// — aether has no texture-memory carrier family yet
// (`aether/view/TableHandle.h`'s own docstring).

#include <cstddef>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/device/Device.h>
#include <aether/layout/Extents.h>
#include <aether/layout/Layout.h>
#include <aether/view/MakeTableView.h> // makeTableView() lives here, split out of view/TableView.h
#include <aether/view/TableView.h>

namespace aether_tests {
namespace {

using aether::dyn;
using aether::extents;
using aether::layout_right;
using aether::makeTableView;
using aether::TableView;

/** @brief Carrier length, comfortably larger than every table below. */
constexpr std::size_t kCarrierSize = 256u;

/** @brief Extents shared by the static/dynamic/mixed cases, so the three
 *  spellings are compared against the SAME known answers. */
constexpr std::size_t kD0 = 2u;
constexpr std::size_t kD1 = 3u;
constexpr std::size_t kD2 = 4u;

/** @brief Fill a carrier with the ramp `a[i] == i`. */
void fillRamp(std::vector<double>& a)
{
    for (std::size_t i = 0; i < a.size(); i++)
        a[i] = static_cast<double>(i);
}

/** @brief Hand-written row-major stride math — the expression a consumer
 *  writes today, kept deliberately separate from the view's own fold. */
constexpr std::size_t handFlat3(const std::size_t i, const std::size_t j, const std::size_t k)
{
    return i * kD1 * kD2 + j * kD2 + k;
}

/** @brief RKF45's `a` coefficients — a compile-time-extent table that
 *  lives today as a raw 2D C array. */
constexpr std::size_t kTabN = 5u;
constexpr double kTableau[kTabN][kTabN] = {
    { 0.25, 0.0, 0.0, 0.0, 0.0 },
    { 3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0 },
    { 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0 },
    { 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0 },
    { -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0 },
};

/* Row 10 (ConstexprEvaluatesAtCompileTime): the constexpr half.
 * `makeTableView()` itself is not constexpr (it wraps `make_view`, which
 * throws `aether::Error` under a span guard — a host-only operation),
 * so this exercises `layout_right::mapping`'s fold and `View`'s own
 * `AETHER_DEVICEHOST() constexpr` constructor/accessors DIRECTLY, bypassing
 * `makeTableView`. Namespace scope (static storage duration), not a local
 * inside the TEST body: a `constexpr` object built from a POINTER to a
 * local (automatic-storage-duration) array is not reliably a valid
 * `constexpr` VARIABLE across further reads, confirmed by a direct probe. */
using Row10Ext = extents<kD0, kD1, kD2>;
constexpr double kRow10Ramp[kD0 * kD1 * kD2]
    = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };
constexpr Row10Ext kRow10Ext{};
constexpr layout_right::mapping<Row10Ext> kRow10Map(kRow10Ext);
constexpr TableView<double, Row10Ext> kRow10View(kRow10Ramp, kRow10Map, aether::Device(kDLCPU));
static_assert(kRow10Map(1u, 2u, 3u) == handFlat3(1u, 2u, 3u),
    "layout_right::mapping's fold must match the hand row-major math");
static_assert(kRow10View.rank() == 3u);
static_assert(kRow10View.extent(0) == kD0);
static_assert(kRow10View.extent(1) == kD1);
static_assert(kRow10View.extent(2) == kD2);
static_assert(kRow10View.size() == kD0 * kD1 * kD2);
static_assert(kRow10View(1u, 2u, 3u) == static_cast<double>(handFlat3(1u, 2u, 3u)),
    "a directly-constructed constexpr TableView must read at compile time");

// ---------------------------------------------------------------------------
// Fold known answers
// ---------------------------------------------------------------------------

TEST(TableView, Static2DFoldKnownAnswers)
{
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    const auto v = makeTableView<kD1, kD2>(a.data(), aether::Device(kDLCPU), 0u);
    for (std::size_t i = 0; i < kD1; i++)
        for (std::size_t j = 0; j < kD2; j++)
            EXPECT_EQ(v(i, j), static_cast<double>(i * kD2 + j)) << "i=" << i << " j=" << j;
}

TEST(TableView, Static3DFoldKnownAnswers)
{
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    const auto v = makeTableView<kD0, kD1, kD2>(a.data(), aether::Device(kDLCPU), 0u);
    for (std::size_t i = 0; i < kD0; i++)
        for (std::size_t j = 0; j < kD1; j++)
            for (std::size_t k = 0; k < kD2; k++)
                EXPECT_EQ(v(i, j, k), static_cast<double>(handFlat3(i, j, k)))
                    << "i=" << i << " j=" << j << " k=" << k;
}

TEST(TableView, DynamicFoldKnownAnswers)
{
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    const auto v = makeTableView<dyn, dyn, dyn>(a.data(), aether::Device(kDLCPU), 0u, kD0, kD1, kD2);
    for (std::size_t i = 0; i < kD0; i++)
        for (std::size_t j = 0; j < kD1; j++)
            for (std::size_t k = 0; k < kD2; k++)
                EXPECT_EQ(v(i, j, k), static_cast<double>(handFlat3(i, j, k)))
                    << "i=" << i << " j=" << j << " k=" << k;
}

TEST(TableView, MixedStaticDynamicFoldKnownAnswers)
{
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    /* Dynamic slot in the MIDDLE: the dynamic-slot bookkeeping is only
     * exercised when a static dimension precedes a dynamic one. */
    const auto mid = makeTableView<kD0, dyn, kD2>(a.data(), aether::Device(kDLCPU), 0u, kD1);
    /* Dynamic slot LEADING, static tail. */
    const auto lead = makeTableView<dyn, kD1, kD2>(a.data(), aether::Device(kDLCPU), 0u, kD0);
    /* Two dynamic slots split by a static one. */
    const auto split = makeTableView<dyn, kD1, dyn>(a.data(), aether::Device(kDLCPU), 0u, kD0, kD2);
    for (std::size_t i = 0; i < kD0; i++)
        for (std::size_t j = 0; j < kD1; j++)
            for (std::size_t k = 0; k < kD2; k++) {
                const double want = static_cast<double>(handFlat3(i, j, k));
                EXPECT_EQ(mid(i, j, k), want) << "mid i=" << i << " j=" << j << " k=" << k;
                EXPECT_EQ(lead(i, j, k), want) << "lead i=" << i << " j=" << j << " k=" << k;
                EXPECT_EQ(split(i, j, k), want) << "split i=" << i << " j=" << j << " k=" << k;
            }
}

// ---------------------------------------------------------------------------
// Offset and bijectivity
// ---------------------------------------------------------------------------

TEST(TableView, OffsetIsHonoured)
{
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    constexpr std::size_t kOffset = 37u;
    const auto v = makeTableView<kD0, kD1, kD2>(a.data(), aether::Device(kDLCPU), kOffset);
    for (std::size_t i = 0; i < kD0; i++)
        for (std::size_t j = 0; j < kD1; j++)
            for (std::size_t k = 0; k < kD2; k++)
                EXPECT_EQ(v(i, j, k), static_cast<double>(kOffset + handFlat3(i, j, k)))
                    << "i=" << i << " j=" << j << " k=" << k;
}

TEST(TableView, IndexTuplesAreBijectiveOntoTheSlice)
{
    /* A stride-derivation bug that merely SHIFTS or SKEWS the mapping can
     * still satisfy a handful of known answers; it cannot stay a bijection.
     * Every index tuple must land on a distinct element of
     * [offset, offset + size), and together they must cover it exactly. */
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    constexpr std::size_t kOffset = 11u;
    const auto v = makeTableView<kD0, dyn, kD2>(a.data(), aether::Device(kDLCPU), kOffset, kD1);
    ASSERT_EQ(v.size(), kD0 * kD1 * kD2);

    std::vector<int> hits(v.size(), 0);
    for (std::size_t i = 0; i < kD0; i++)
        for (std::size_t j = 0; j < kD1; j++)
            for (std::size_t k = 0; k < kD2; k++) {
                const double got = v(i, j, k);
                ASSERT_GE(got, static_cast<double>(kOffset))
                    << "read below the slice at i=" << i << " j=" << j << " k=" << k;
                ASSERT_LT(got, static_cast<double>(kOffset + v.size()))
                    << "read past the slice at i=" << i << " j=" << j << " k=" << k;
                hits[static_cast<std::size_t>(got) - kOffset]++;
            }
    for (std::size_t s = 0; s < hits.size(); s++)
        EXPECT_EQ(hits[s], 1) << "slot " << s << " hit " << hits[s] << " times, expected exactly once";
}

// ---------------------------------------------------------------------------
// Shape queries
// ---------------------------------------------------------------------------

TEST(TableView, RankExtentSizeQueriesStatic)
{
    using ViewT = TableView<double, extents<kD0, kD1, kD2>>;
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    const ViewT v = makeTableView<kD0, kD1, kD2>(a.data(), aether::Device(kDLCPU), 0u);
    EXPECT_EQ(v.rank(), 3u);
    EXPECT_EQ(ViewT::rank(), 3u);
    EXPECT_EQ(v.extent(0), kD0);
    EXPECT_EQ(v.extent(1), kD1);
    EXPECT_EQ(v.extent(2), kD2);
    EXPECT_EQ(v.size(), kD0 * kD1 * kD2);
    EXPECT_EQ(ViewT::extents_type::RankDynamic, 0u);
}

TEST(TableView, RankExtentSizeQueriesDynamic)
{
    using DynViewT = TableView<double, extents<dyn, dyn, dyn>>;
    using MixedViewT = TableView<double, extents<kD0, dyn, kD2>>;
    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    const DynViewT dynV = makeTableView<dyn, dyn, dyn>(a.data(), aether::Device(kDLCPU), 0u, kD0, kD1, kD2);
    EXPECT_EQ(dynV.rank(), 3u);
    EXPECT_EQ(dynV.extent(0), kD0);
    EXPECT_EQ(dynV.extent(1), kD1);
    EXPECT_EQ(dynV.extent(2), kD2);
    EXPECT_EQ(dynV.size(), kD0 * kD1 * kD2);
    EXPECT_EQ(DynViewT::extents_type::RankDynamic, 3u);

    const MixedViewT mixed = makeTableView<kD0, dyn, kD2>(a.data(), aether::Device(kDLCPU), 0u, kD1);
    EXPECT_EQ(mixed.extent(0), kD0);
    EXPECT_EQ(mixed.extent(1), kD1);
    EXPECT_EQ(mixed.extent(2), kD2);
    EXPECT_EQ(mixed.size(), kD0 * kD1 * kD2);
    EXPECT_EQ(MixedViewT::extents_type::RankDynamic, 1u);
}

TEST(TableView, FullyStaticViewCarriesNoRuntimeExtentStorage)
{
    // A compile-time-extent view is exactly the carrier plus a (now
    // pointer-folded, see OffsetIsHonoured) offset -- the extents are
    // types, not bytes. `aether::View` adds one more member than a bare
    // carrier+offset pair (`Device`, `aether/device/Device.h`), and the
    // empty `extents<Es...>` placeholder is not `[[no_unique_address]]`-
    // elided, so `sizeof(TableView) == sizeof(carrier) + sizeof(offset)`
    // does not hold verbatim. Compared instead against a reference struct
    // whose members have the same types and order as `View`'s own (pointer,
    // empty-extents placeholder, Device) so the two structs' padding
    // computations agree exactly -- confirmed by direct compilation (the
    // naive `sizeof(ptr) + sizeof(Device)` sum is 16 bytes; the actual
    // `View` and this matched reference struct both measure 24).
    using FullyStaticT = TableView<double, extents<kD0, kD1, kD2>>;
    struct CarrierPlusOffset {
        const double* c;
        FullyStaticT::extents_type ext;
        aether::Device dev;
    };
    static_assert(sizeof(FullyStaticT) == sizeof(CarrierPlusOffset),
        "a fully static TableView carries runtime-extent storage it must not");
    static_assert(sizeof(extents<kD0, kD1, kD2>) == 1, "a fully static extents must carry no runtime storage");
    /* And it stays a PODified handle, so it crosses a kernel boundary by
     * value like every other aether View. */
    static_assert(std::is_trivially_copyable_v<FullyStaticT>);
    static_assert(std::is_standard_layout_v<FullyStaticT>);
    SUCCEED();
}

TEST(TableView, ConstexprEvaluatesAtCompileTime)
{
    // The static_asserts anchoring this claim live at namespace scope above
    // (`kRow10View` etc.) -- see that block's comment for why. This body
    // re-observes the same claim at runtime (pairing static_asserts with
    // EXPECT_EQ reads through the same constexpr view). One thing this
    // suite does not cover: folding over an arbitrary duck-typed carrier
    // (any type exposing `operator[](idx_t) const`) -- `aether::View` is
    // hard-typed to a `T*` pointer, so there is no carrier-polymorphism
    // equivalent (aether views are pointer-backed by design).
    EXPECT_EQ(kRow10Map(1u, 2u, 3u), handFlat3(1u, 2u, 3u));
    EXPECT_EQ(kRow10View(1u, 2u, 3u), static_cast<double>(handFlat3(1u, 2u, 3u)));
}

// ---------------------------------------------------------------------------
// Consumer-shaped exemplars
// ---------------------------------------------------------------------------

TEST(TableView, CoefficientBlockMatchesHandStrideMath)
{
    /* An ephemeris coefficient block shape: the data starts at
     * `bodyUnitOffset + 1 + nIntervals` inside a bigger per-body buffer and
     * is (nIntervals x (pdeg + 1)) elements, both extents runtime values.
     * A caller with this shape writes that offset and stride out by hand at
     * the construction site and again at every tap; here the view is
     * compared against exactly that hand math.
     */
    constexpr std::size_t kBodyUnitOffset = 13u;
    constexpr std::size_t kNIntervals = 5u;
    constexpr std::size_t kPdeg = 7u;
    constexpr std::size_t kCoeffOffset = kBodyUnitOffset + 1u + kNIntervals;

    std::vector<double> a(kCarrierSize);
    fillRamp(a);
    ASSERT_GE(kCarrierSize, kCoeffOffset + kNIntervals * (kPdeg + 1u));

    const auto coeffs
        = makeTableView<dyn, dyn>(a.data(), aether::Device(kDLCPU), kCoeffOffset, kNIntervals, kPdeg + 1u);
    EXPECT_EQ(coeffs.size(), kNIntervals * (kPdeg + 1u));

    for (std::size_t interval = 0; interval < kNIntervals; interval++)
        for (std::size_t degree = 0; degree <= kPdeg; degree++) {
            const std::size_t hand = kCoeffOffset + interval * (kPdeg + 1u) + degree;
            EXPECT_EQ(coeffs(interval, degree), a[hand]) << "interval=" << interval << " degree=" << degree;
        }
}

TEST(TableView, TableauMatchesRawSubscript)
{
    /* A Butcher-tableau shape: a raw 2D `static constexpr` array reached
     * through `a<i,j>()`. The extents are compile-time, which is the shape
     * TableView prefers -- so the view must reproduce the raw subscript
     * exactly, element for element. */
    std::vector<double> a(kTabN * kTabN);
    for (std::size_t i = 0; i < kTabN; i++)
        for (std::size_t j = 0; j < kTabN; j++)
            a[i * kTabN + j] = kTableau[i][j];

    const auto tab = makeTableView<kTabN, kTabN>(a.data(), aether::Device(kDLCPU), 0u);
    EXPECT_EQ(tab.size(), kTabN * kTabN);
    for (std::size_t i = 0; i < kTabN; i++)
        for (std::size_t j = 0; j < kTabN; j++)
            EXPECT_EQ(tab(i, j), kTableau[i][j]) << "i=" << i << " j=" << j;
}

} // namespace
} // namespace aether_tests
