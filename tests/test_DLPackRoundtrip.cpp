// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// DLPack round-trip tests (host / AETHER_CPP_MODE build; test_DLPackRoundtrip.cu
// is the twin over the same fixture and case names), but the covered
// device-kind set differs by construction: this build has only kDLCPU
// available.
//
// C-ABI level: every "producer"/"consumer" role below is a hand-built raw
// DLPack struct — no external framework (torch/cupy/numpy) is a build
// dependency here, exactly mirroring how a real producer/consumer would
// only ever see the C ABI.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>
#include <aether/chunk/Chunk.h>
#include <aether/device/Device.h>
#include <aether/dtype/DType.h>
#include <aether/dtype/dlpack.h>
#include <aether/err/Error.h>
#include <aether/interop/DLPack.h>
#include <aether/interop/DLPackLegacy.h>
#include <aether/layout/Extents.h>
#include <aether/view/RuntimeView.h>
#include <aether/view/View.h>

namespace aether_tests {
namespace {

class DLPackRoundtripTest : public ::testing::Test { };

// -----------------------------------------------------------------------
// Hand-built PRODUCER helpers: a counter-instrumented deleter, so a test
// can assert "called exactly once" from the OUTSIDE without depending on
// any aether internals.
// -----------------------------------------------------------------------

struct DeleteCounter {
    int legacyCalls = 0;
    int versionedCalls = 0;
};

void countingLegacyDeleter(DLManagedTensor* self)
{
    auto* counter = static_cast<DeleteCounter*>(self->manager_ctx);
    ++counter->legacyCalls;
}

void countingVersionedDeleter(DLManagedTensorVersioned* self)
{
    auto* counter = static_cast<DeleteCounter*>(self->manager_ctx);
    ++counter->versionedCalls;
}

TEST_F(DLPackRoundtripTest, ImportLegacyCapturesFieldsZeroCopy)
{
    constexpr std::size_t C = 3, N = 4;
    std::vector<double> buf(C * N);
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = static_cast<double>(i);
    std::vector<std::int64_t> shape{ static_cast<std::int64_t>(C), static_cast<std::int64_t>(N) };
    // strides == nullptr -> DLPack's own "compact, row-major" contract.

    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 2;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLFloat), 64, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    {
        aether::interop::DLPackImport imp = aether::interop::fromDLPack(&tensor);
        EXPECT_EQ(imp.view.data, static_cast<void*>(buf.data())); // zero-copy
        EXPECT_TRUE(imp.view.dtype == aether::dtype_of<double>());
        EXPECT_EQ(imp.view.device, aether::Device(kDLCPU));
        EXPECT_EQ(imp.view.rank, 2u);
        EXPECT_EQ(imp.view.extents[0], C);
        EXPECT_EQ(imp.view.extents[1], N);
        EXPECT_EQ(imp.view.strides[0], N); // compact row-major, derived from the nullptr strides contract
        EXPECT_EQ(imp.view.strides[1], 1u);
        EXPECT_EQ(counter.legacyCalls, 0); // owner alive — not deleted yet

        const aether::Vec3dView promoted = imp.view.as<aether::Vec3dView>();
        EXPECT_EQ(promoted.data(), buf.data());
        EXPECT_EQ(promoted(1, 2), buf[1 * N + 2]);
    } // imp (and its DLPackOwner) goes out of scope here

    EXPECT_EQ(counter.legacyCalls, 1); // deleter called EXACTLY once, on destruction
}

TEST_F(DLPackRoundtripTest, ImportVersionedCapturesFieldsZeroCopy)
{
    constexpr std::size_t C = 3, N = 4;
    std::vector<double> buf(C * N, 0.0);
    std::vector<std::int64_t> shape{ static_cast<std::int64_t>(C), static_cast<std::int64_t>(N) };
    std::vector<std::int64_t> strides{ static_cast<std::int64_t>(N), 1 }; // explicit, matches the compact contract

    DeleteCounter counter;
    DLManagedTensorVersioned tensor{};
    tensor.version                = DLPackVersion{ DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION };
    tensor.manager_ctx            = &counter;
    tensor.deleter                = &countingVersionedDeleter;
    tensor.flags                  = 0;
    tensor.dl_tensor.data         = buf.data();
    tensor.dl_tensor.device       = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim         = 2;
    tensor.dl_tensor.dtype        = DLDataType{ static_cast<std::uint8_t>(kDLFloat), 64, 1 };
    tensor.dl_tensor.shape        = shape.data();
    tensor.dl_tensor.strides      = strides.data();
    tensor.dl_tensor.byte_offset  = 0;

    aether::interop::DLPackImport imp = aether::interop::fromDLPack(&tensor);
    EXPECT_EQ(imp.view.data, static_cast<void*>(buf.data()));
    EXPECT_EQ(imp.view.rank, 2u);
    EXPECT_EQ(counter.versionedCalls, 0);

    imp.owner.release();
    EXPECT_EQ(counter.versionedCalls, 1);
    imp.owner.release(); // idempotent — must NOT call again
    EXPECT_EQ(counter.versionedCalls, 1);
}

TEST_F(DLPackRoundtripTest, ImportBoolDtypeMapsToKDlbool)
{
    std::vector<std::uint8_t> buf{ 1, 0, 1, 1, 0 };
    std::vector<std::int64_t> shape{ static_cast<std::int64_t>(buf.size()) };
    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 1;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLBool), 8, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    aether::interop::DLPackImport imp = aether::interop::fromDLPack(&tensor);
    EXPECT_TRUE(imp.view.dtype == aether::dtype_of<bool>());
    EXPECT_EQ(imp.view.dtype.code(), static_cast<std::uint8_t>(kDLBool));
    imp.owner.release();
    EXPECT_EQ(counter.legacyCalls, 1);
}

TEST_F(DLPackRoundtripTest, ExportRoundTripsThroughImportIdentity)
{
    constexpr std::size_t C = 3, N = 6;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto v     = aether::make_view<double, C, aether::dyn>(chunk, N);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            v(c, i) = static_cast<double>(c * 100 + i);

    DLManagedTensorVersioned* exported = aether::interop::toDLPack(v);
    ASSERT_NE(exported, nullptr);
    EXPECT_EQ(exported->dl_tensor.data, static_cast<void*>(v.data()));
    EXPECT_EQ(exported->dl_tensor.ndim, 2);
    EXPECT_EQ(exported->dl_tensor.shape[0], static_cast<std::int64_t>(C));
    EXPECT_EQ(exported->dl_tensor.shape[1], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[0], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);
    EXPECT_EQ(exported->dl_tensor.byte_offset, 0u);

    // Re-import our OWN export, as an external consumer would.
    aether::interop::DLPackImport reimported = aether::interop::fromDLPack(exported);
    EXPECT_EQ(reimported.view.data, static_cast<void*>(v.data()));
    EXPECT_EQ(reimported.view.rank, 2u);
    const aether::Vec3dView promoted = reimported.view.as<aether::Vec3dView>();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(promoted(c, i), v(c, i));
    // `reimported.owner`'s destructor calls `exported`'s deleter, which frees
    // `exported` itself (and its shape/strides storage) — no leak, no double
    // free (verified functionally: no crash/ASan would fire under `make test`'s
    // scrubbed environment; the exact-once property is directly counter-
    // instrumented in the hand-built-producer tests above).
}

TEST_F(DLPackRoundtripTest, RejectsUnsupportedDtype)
{
    std::vector<std::uint8_t> buf(32, 0); // scratch bytes — never actually read as a real complex value (rejected before any dereference)
    std::vector<std::int64_t> shape{ 4 };
    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 1;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLComplex), 64, 1 }; // never supported (v1: double/float/bool/int/uint only)
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    EXPECT_THROW(aether::interop::fromDLPack(&tensor), aether::Error);
    EXPECT_EQ(counter.legacyCalls, 1); // rejected import still frees exactly once
}

TEST_F(DLPackRoundtripTest, RejectsRankAboveCap)
{
    std::vector<double> buf(1, 0.0);
    std::vector<std::int64_t> shape(7, 1); // RuntimeRankCap == 6
    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 7;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLFloat), 64, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    EXPECT_THROW(aether::interop::fromDLPack(&tensor), aether::Error);
    EXPECT_EQ(counter.legacyCalls, 1);
}

TEST_F(DLPackRoundtripTest, RejectsNonzeroByteOffset)
{
    std::vector<double> buf(8, 0.0);
    std::vector<std::int64_t> shape{ 4 };
    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 1;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLFloat), 64, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 8; // != 0 -- rejected (v1)
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    EXPECT_THROW(aether::interop::fromDLPack(&tensor), aether::Error);
    EXPECT_EQ(counter.legacyCalls, 1);
}

TEST_F(DLPackRoundtripTest, RejectsUnknownDeviceKind)
{
    std::vector<double> buf(4, 0.0);
    std::vector<std::int64_t> shape{ 4 };
    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    // A code DLPack does not define; every DEFINED kind is a record in this
    // pure C++ build (see test_DLPackBuffer_common.h).
    tensor.dl_tensor.device      = DLDevice{ static_cast<DLDeviceType>(99), 0 };
    tensor.dl_tensor.ndim        = 1;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLFloat), 64, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    EXPECT_THROW(aether::interop::fromDLPack(&tensor), aether::Error);
    EXPECT_EQ(counter.legacyCalls, 1);
}

TEST_F(DLPackRoundtripTest, NullTensorPointerThrows)
{
    EXPECT_THROW(aether::interop::fromDLPack(static_cast<DLManagedTensor*>(nullptr)), aether::Error);
    EXPECT_THROW(aether::interop::fromDLPack(static_cast<DLManagedTensorVersioned*>(nullptr)), aether::Error);
}

// -----------------------------------------------------------------------
// aether/interop/DLPackLegacy.h's toDLPackLegacy — the export half (import
// of the legacy DLManagedTensor is covered above). Mirrors
// ExportRoundTripsThroughImportIdentity's own shape/values check, targeting
// the pre-1.0 struct instead.
// -----------------------------------------------------------------------

TEST_F(DLPackRoundtripTest, LegacyExportRoundTripsThroughImportIdentity)
{
    constexpr std::size_t C = 3, N = 5;
    auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
    auto v     = aether::make_view<double, C, aether::dyn>(chunk, N);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            v(c, i) = static_cast<double>(c * 10 + i);

    DLManagedTensor* exported = aether::interop::toDLPackLegacy(v);
    ASSERT_NE(exported, nullptr);
    EXPECT_EQ(exported->dl_tensor.data, static_cast<void*>(v.data()));
    EXPECT_EQ(exported->dl_tensor.ndim, 2);
    EXPECT_EQ(exported->dl_tensor.shape[0], static_cast<std::int64_t>(C));
    EXPECT_EQ(exported->dl_tensor.shape[1], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[0], static_cast<std::int64_t>(N));
    EXPECT_EQ(exported->dl_tensor.strides[1], 1);
    EXPECT_EQ(exported->dl_tensor.byte_offset, 0u);
    EXPECT_EQ(exported->dl_tensor.dtype.code, static_cast<std::uint8_t>(kDLFloat));
    ASSERT_NE(exported->deleter, nullptr);

    // Re-import our OWN legacy export, as an external (pre-1.0) consumer would
    // — fromDLPack(DLManagedTensor*) already exists (DLPack.h).
    aether::interop::DLPackImport reimported = aether::interop::fromDLPack(exported);
    EXPECT_EQ(reimported.view.data, static_cast<void*>(v.data()));
    EXPECT_EQ(reimported.view.rank, 2u);
    const aether::Vec3dView promoted = reimported.view.as<aether::Vec3dView>();
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t i = 0; i < N; ++i)
            EXPECT_EQ(promoted(c, i), v(c, i));
    // reimported.owner's destructor calls exported's deleter, freeing
    // `exported` itself and its shape/strides storage — no leak, no double
    // free (same "verified functionally" note as ExportRoundTripsThroughImportIdentity
    // above; the exact-once property is directly counter-instrumented in the
    // hand-built-producer tests earlier in this file).
}

TEST_F(DLPackRoundtripTest, ImportUint64PreservesRawBitsZeroCopy)
{
    // A (kDLUInt, 64) tensor carries an opaque 64-bit payload rather than an
    // arithmetic value (this is the tag BandedReal/BandCell8 storage words
    // use, aether/dtype/DType.h): import must hand every lane's bits back
    // unmodified, not reinterpret or renormalize them.
    constexpr std::size_t n = 4;
    constexpr std::uint64_t bits = 0x400921FB54442D18ULL; // an arbitrary finite double's raw bit pattern
    std::vector<std::uint64_t> buf(n, bits);
    std::vector<std::int64_t> shape{ static_cast<std::int64_t>(n) };

    DeleteCounter counter;
    DLManagedTensor tensor{};
    tensor.dl_tensor.data        = buf.data();
    tensor.dl_tensor.device      = DLDevice{ kDLCPU, 0 };
    tensor.dl_tensor.ndim        = 1;
    tensor.dl_tensor.dtype       = DLDataType{ static_cast<std::uint8_t>(kDLUInt), 64, 1 };
    tensor.dl_tensor.shape       = shape.data();
    tensor.dl_tensor.strides     = nullptr;
    tensor.dl_tensor.byte_offset = 0;
    tensor.manager_ctx           = &counter;
    tensor.deleter                = &countingLegacyDeleter;

    aether::interop::DLPackImport imp = aether::interop::fromDLPack(&tensor);
    EXPECT_EQ(imp.view.data, static_cast<void*>(buf.data())); // zero-copy
    EXPECT_TRUE(imp.view.dtype == aether::dtype_of<std::uint64_t>());
    ASSERT_EQ(imp.view.rank, 1u);
    ASSERT_EQ(imp.view.extents[0], n);

    const auto* words = static_cast<const std::uint64_t*>(imp.view.data);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(words[i], bits); // bit-for-bit, no reinterpretation

    imp.owner.release();
    EXPECT_EQ(counter.legacyCalls, 1);
}

TEST_F(DLPackRoundtripTest, LegacyExportBoolUsesKduintPolicyNotKdlbool)
{
    // Pre-1.0 DLDataTypeCode predates kDLBool (downstream vendored DLPack
    // subsets do not define it at all) — the legacy exporter maps bool to
    // kDLUInt/8 instead, documented policy.
    std::vector<std::uint8_t> buf{ 1, 0, 1 };
    aether::RuntimeView view;
    view.data = buf.data();
    view.dtype = aether::dtype_of<bool>();
    view.device = aether::Device(kDLCPU);
    view.rank = 1;
    view.extents[0] = static_cast<aether::offset_t>(buf.size());
    view.strides[0] = 1;

    ASSERT_TRUE(view.dtype.code() == static_cast<std::uint8_t>(kDLBool)); // sanity: the SOURCE dtype IS kDLBool

    DLManagedTensor* exported = aether::interop::toDLPackLegacy(view);
    ASSERT_NE(exported, nullptr);
    EXPECT_EQ(exported->dl_tensor.dtype.code, static_cast<std::uint8_t>(kDLUInt)); // NOT kDLBool
    EXPECT_EQ(exported->dl_tensor.dtype.bits, 8);
    ASSERT_NE(exported->deleter, nullptr);
    exported->deleter(exported);
}

} // namespace
} // namespace aether_tests
