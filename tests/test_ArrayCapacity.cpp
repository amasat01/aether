// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Array capacity/reserve/resize/push_back/spare/commit/packed tests
// (host / AETHER_CPP_MODE build). Paired with test_ArrayCapacity.cu
// (device-side spare()/commit() round trip). Covers: the closed-form
// `capacity()` quantum table, reserve/push_back pointer+stride stability
// within capacity, resize-beyond-capacity realloc bit-exactness, spare()/
// commit() round trip, the formally-0-sized no-op case, packed()'s
// host-throw guard, and the DLPack "shape from size, strides from
// capacity" export.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/array/Array.h>
#include <aether/err/Error.h>
#include <aether/index/Offset.h>
#include <aether/interop/DLPack.h>
#include <aether/residency/Partition.h>
#include <aether/view/Item.h>

namespace aether_tests {
namespace {

class ArrayCapacityTest : public ::testing::Test { };

// ---------------------------------------------------------------------
// capacity() — closed-form quantum table: reuses
// `aether::padded_block_samples(n, 1, kCapacityAlignment)`'s own already-
// tested closed form (see test_Partition.cpp's PaddedBlockSamples* cases).
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, QuantumConstantIs32)
{
    EXPECT_EQ(aether::kCapacityAlignment, 32u);
}

TEST_F(ArrayCapacityTest, CapacityQuantisesUpToTheAlignmentQuantum)
{
    // A type alias sidesteps the preprocessor's naive top-level-comma
    // argument split inside `EXPECT_EQ(Array<double, 3>(...), ...)` — the
    // comma in `<double, 3>` is not inside any PARENS as far as the
    // macro preprocessor is concerned (it does not parse angle brackets).
    using Arr3 = aether::Array<double, 3>;
    EXPECT_EQ(Arr3(0).capacity(), 0u);
    EXPECT_EQ(Arr3(1).capacity(), 32u);
    EXPECT_EQ(Arr3(10).capacity(), 32u);
    EXPECT_EQ(Arr3(32).capacity(), 32u);
    EXPECT_EQ(Arr3(33).capacity(), 64u);
    EXPECT_EQ(Arr3(96).capacity(), 96u);
    EXPECT_EQ(Arr3(100).capacity(), 128u);
    // Cross-check against the reused closed form directly.
    for (std::size_t n : { std::size_t{ 0 }, std::size_t{ 1 }, std::size_t{ 31 }, std::size_t{ 32 },
             std::size_t{ 33 }, std::size_t{ 200 } }) {
        const std::size_t expected = aether::padded_block_samples(n, 1, aether::kCapacityAlignment);
        EXPECT_EQ(Arr3(n).capacity(), expected) << "n=" << n;
    }
}

TEST_F(ArrayCapacityTest, ScalarArrayPitchStrideIsAlwaysOneRegardlessOfCapacity)
{
    aether::Array<double> arr(10); // capacity 32, itemSize_ == 1: no separate component axis.
    auto v = arr.hostView();
    EXPECT_EQ(arr.capacity(), 32u);
    EXPECT_EQ(v.mapping().stride(0), 1u);
    EXPECT_EQ(v.samples(), 10u);
}

// ---------------------------------------------------------------------
// reserve()/push_back() — pointer and stride stability WITHIN capacity
// (growth within capacity never moves data or changes strides).
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, ReserveThenPushBackKeepsPointerAndStrideStableWithinCapacity)
{
    aether::Array<double, 3> arr(0);
    arr.reserve(50); // -> capacity 64
    ASSERT_EQ(arr.capacity(), 64u);

    auto v0                  = arr.hostView();
    const double* p0         = v0.data();
    const std::size_t stride0 = v0.mapping().stride(0);

    for (int i = 0; i < 40; ++i)
        arr.push_back(aether::Item<double, 3>{ static_cast<double>(i), static_cast<double>(i) * 2.0,
            static_cast<double>(i) * 3.0 });

    EXPECT_EQ(arr.capacity(), 64u); // still within capacity: no growth triggered
    EXPECT_EQ(arr.samples(), 40u);

    auto v1 = arr.hostView();
    EXPECT_EQ(v1.data(), p0);
    EXPECT_EQ(v1.mapping().stride(0), stride0);
    EXPECT_EQ(v1.samples(), 40u);

    for (std::size_t i = 0; i < 40; ++i) {
        EXPECT_DOUBLE_EQ(v1(0, i), static_cast<double>(i));
        EXPECT_DOUBLE_EQ(v1(1, i), static_cast<double>(i) * 2.0);
        EXPECT_DOUBLE_EQ(v1(2, i), static_cast<double>(i) * 3.0);
    }
}

TEST_F(ArrayCapacityTest, PushBackGrowsCapacityByAmortisedDoublingOncePastIt)
{
    aether::Array<double, 3> arr(0);
    for (int i = 0; i < 32; ++i)
        arr.push_back(aether::Item<double, 3>::Zeros());
    EXPECT_EQ(arr.capacity(), 32u);
    arr.push_back(aether::Item<double, 3>::Zeros()); // 33rd: past capacity -> doubles
    EXPECT_EQ(arr.capacity(), 64u);
    EXPECT_EQ(arr.samples(), 33u);
}

// ---------------------------------------------------------------------
// resize() beyond capacity — realloc + pitched copy, bit-exact.
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, ResizeBeyondCapacityReallocatesAndPreservesContentsBitExact)
{
    aether::Array<double, 3> arr(10); // capacity 32
    auto v = arr.hostView();
    for (std::size_t i = 0; i < 10; ++i) {
        v(0, i) = static_cast<double>(i);
        v(1, i) = static_cast<double>(i) + 100.0;
        v(2, i) = static_cast<double>(i) + 200.0;
    }
    ASSERT_EQ(arr.capacity(), 32u);

    arr.resize(50); // beyond 32 -> realloc to 64
    EXPECT_EQ(arr.capacity(), 64u);
    EXPECT_EQ(arr.samples(), 50u);

    auto v2 = arr.hostView();
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(v2(0, i), static_cast<double>(i));
        EXPECT_EQ(v2(1, i), static_cast<double>(i) + 100.0);
        EXPECT_EQ(v2(2, i), static_cast<double>(i) + 200.0);
    }
}

TEST_F(ArrayCapacityTest, ResizeWithinCapacityNeverReallocates)
{
    aether::Array<double, 3> arr(5); // capacity 32
    auto v0          = arr.hostView();
    const double* p0 = v0.data();
    arr.resize(20); // still <= 32
    EXPECT_EQ(arr.capacity(), 32u);
    EXPECT_EQ(arr.hostView().data(), p0);
}

TEST_F(ArrayCapacityTest, ClearDropsSamplesButKeepsCapacity)
{
    aether::Array<double, 3> arr(10);
    ASSERT_EQ(arr.capacity(), 32u);
    arr.clear();
    EXPECT_EQ(arr.samples(), 0u);
    EXPECT_EQ(arr.capacity(), 32u);
}

// ---------------------------------------------------------------------
// spare()/commit() round trip (host-loop twin of test_ArrayCapacity.cu's
// device kernel — see test_Array.cpp's own "host-loop twin" convention).
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, SpareCommitRoundTripHostLoop)
{
    aether::Array<double, 3> arr(0);
    arr.reserve(10); // -> capacity 32
    auto sp = arr.hostSpare();
    ASSERT_EQ(sp.samples(), 32u); // the WHOLE reserved capacity, since samples()==0

    for (std::size_t i = 0; i < 5; ++i) {
        sp(0, i) = static_cast<double>(i);
        sp(1, i) = static_cast<double>(i) * 2.0;
        sp(2, i) = static_cast<double>(i) * 3.0;
    }
    arr.commit(5);
    EXPECT_EQ(arr.samples(), 5u);

    auto v = arr.hostView();
    EXPECT_EQ(v.samples(), 5u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(v(0, i), static_cast<double>(i));
        EXPECT_EQ(v(1, i), static_cast<double>(i) * 2.0);
        EXPECT_EQ(v(2, i), static_cast<double>(i) * 3.0);
    }

    // spare() after a commit is the NEW tail [5, 32).
    auto sp2 = arr.hostSpare();
    EXPECT_EQ(sp2.samples(), 27u);
}

TEST_F(ArrayCapacityTest, CommitBeyondSpareCapacityThrows)
{
    aether::Array<double, 3> arr(0);
    arr.reserve(10); // -> capacity 32
    EXPECT_THROW(arr.commit(arr.capacity() + 1), aether::Error);
    EXPECT_EQ(arr.samples(), 0u); // rejected: samples() untouched
}

// ---------------------------------------------------------------------
// Formally 0-sized writable buffer: view() has sample extent 0;
// any per-sample assignment loop over it is a no-op BY CONSTRUCTION.
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, ZeroSizeViewHasZeroSampleExtentAndAssignmentIsANoOp)
{
    aether::Array<double, 3> arr(0);
    EXPECT_EQ(arr.samples(), 0u);
    auto v = arr.hostView();
    EXPECT_EQ(v.samples(), 0u);

    std::size_t iterations = 0;
    for (std::size_t i = 0; i < v.samples(); ++i) {
        v(0, i) = 1.0; // never executes: v.samples() == 0
        ++iterations;
    }
    EXPECT_EQ(iterations, 0u);
}

// ---------------------------------------------------------------------
// packed() — host-throw guard: only valid when capacity() == samples().
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, PackedThrowsWhenCapacityDiffersFromSamples)
{
    aether::Array<double, 3> arr(10); // capacity 32 != samples 10
    EXPECT_THROW(arr.hostPacked(), aether::Error);
    const aether::Array<double, 3>& carr = arr;
    EXPECT_THROW(carr.hostPacked(), aether::Error);
}

TEST_F(ArrayCapacityTest, PackedSucceedsAndIsCompactWhenCapacityEqualsSamples)
{
    aether::Array<double, 3> arr(32); // capacity exactly 32 == samples
    ASSERT_EQ(arr.capacity(), arr.samples());
    auto p = arr.hostPacked();
    EXPECT_EQ(p.samples(), 32u);
    EXPECT_EQ(p.mapping().extents().extent(1), 32u);
}

// ---------------------------------------------------------------------
// DLPack export: shape from size, strides from capacity — pins the SAME
// convention used elsewhere (one closed-form stride test).
// ---------------------------------------------------------------------

TEST_F(ArrayCapacityTest, DlpackExportShapeFromSizeStridesFromCapacity)
{
    aether::Array<double, 3> arr(10); // capacity 32, samples 10
    auto v = arr.hostView();

    DLManagedTensorVersioned* exported = aether::interop::toDLPack(v);
    ASSERT_NE(exported, nullptr);
    aether::interop::DLPackOwner owner = aether::interop::DLPackOwner::fromVersioned(exported);

    EXPECT_EQ(exported->dl_tensor.ndim, 2);
    EXPECT_EQ(exported->dl_tensor.shape[0], static_cast<std::int64_t>(3));
    EXPECT_EQ(exported->dl_tensor.shape[1], static_cast<std::int64_t>(10)); // shape FROM SIZE
    EXPECT_EQ(exported->dl_tensor.strides[0], static_cast<std::int64_t>(32)); // strides FROM CAPACITY
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);
}

} // namespace
} // namespace aether_tests
