// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_DLPackBuffer_common.h
 * @brief The `aether::interop::BufferView` battery shared by
 *        `test_DLPackBuffer.cpp` (AETHER_CPP_MODE) and
 *        `test_DLPackBuffer.cu` (CUDA mode): access reporting per producer
 *        form, the keep-alive chain, versioned and legacy export, the
 *        array-interface factory and every `Requirements` refusal.
 *
 * Every producer here is a hand-built raw DLPack struct with a counting
 * deleter, so "called exactly once, after the last view died" is asserted
 * from the outside. No memory is dereferenced, so the CUDA-typed rows need
 * no allocation.
 */

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <aether/dtype/dlpack.h>
#include <aether/err/Error.h>
#include <aether/interop/Buffer.h>

namespace aether_tests {
namespace dlpack_buffer {

using aether::interop::Access;
using aether::interop::BufferView;
using aether::interop::Requirements;

/** @brief A producer: storage, shape and a deleter counter. */
struct Producer {
    std::vector<double> storage = std::vector<double>(12, 0.0);
    std::int64_t shape[2] = { 3, 4 };
    std::int64_t strides[2] = { 4, 1 };
    int deletes = 0;
    DLManagedTensorVersioned versioned{};
    DLManagedTensor legacy{};

    DLManagedTensorVersioned* makeVersioned(bool readOnly, DLDeviceType dev = kDLCPU)
    {
        versioned = DLManagedTensorVersioned{};
        versioned.version = DLPackVersion{ 1, 0 };
        versioned.manager_ctx = this;
        versioned.deleter = [](DLManagedTensorVersioned* self) { ++static_cast<Producer*>(self->manager_ctx)->deletes; };
        versioned.flags = readOnly ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
        fill(versioned.dl_tensor, dev);
        return &versioned;
    }

    DLManagedTensor* makeLegacy(DLDeviceType dev = kDLCPU)
    {
        legacy = DLManagedTensor{};
        legacy.manager_ctx = this;
        legacy.deleter = [](DLManagedTensor* self) { ++static_cast<Producer*>(self->manager_ctx)->deletes; };
        fill(legacy.dl_tensor, dev);
        return &legacy;
    }

private:
    void fill(DLTensor& t, DLDeviceType dev)
    {
        t.data = storage.data();
        t.device = DLDevice{ dev, 0 };
        t.ndim = 2;
        t.dtype = DLDataType{ kDLFloat, 64, 1 };
        t.shape = shape;
        t.strides = strides;
        t.byte_offset = 0;
    }
};

/** @brief The message of the `aether::Error` `fn` throws ("" if none). */
template<class F>
std::string thrownMessage(F&& fn)
{
    try {
        fn();
    } catch (const aether::Error& e) {
        return e.what();
    }
    return "";
}

class DLPackBufferTest : public ::testing::Test { };

TEST_F(DLPackBufferTest, VersionedImportReportsReadWriteAndReadOnly)
{
    Producer p;
    {
        BufferView rw = aether::interop::importDLPack(p.makeVersioned(false), "test.Producer");
        EXPECT_EQ(rw.access, Access::ReadWrite);
        EXPECT_TRUE(rw.writable());
        EXPECT_EQ(rw.view.data, p.storage.data());
        EXPECT_EQ(rw.owner, "external");
        EXPECT_FALSE(rw.owned());
        EXPECT_EQ(rw.producer, "test.Producer");
    }
    EXPECT_EQ(p.deletes, 1);
    {
        BufferView ro = aether::interop::importDLPack(p.makeVersioned(true));
        EXPECT_EQ(ro.access, Access::ReadOnly);
        EXPECT_FALSE(ro.writable());
        aether::interop::assumeWritable(ro);
        EXPECT_FALSE(ro.writable()) << "a declared read-only buffer stays read-only";
    }
    EXPECT_EQ(p.deletes, 2);
}

TEST_F(DLPackBufferTest, LegacyImportReportsUnknownAndRecordsAssumption)
{
    Producer p;
    BufferView b = aether::interop::importDLPack(p.makeLegacy());
    EXPECT_EQ(b.access, Access::Unknown);
    EXPECT_FALSE(b.writable()) << "unknown is never upgraded by the layer itself";
    EXPECT_FALSE(b.assumedWritable);
    aether::interop::assumeWritable(b);
    EXPECT_TRUE(b.assumedWritable);
    EXPECT_TRUE(b.writable());
    EXPECT_EQ(b.access, Access::Unknown) << "the assertion is recorded, the report is unchanged";
}

#ifdef AETHER_HAS_CUDA
TEST_F(DLPackBufferTest, CudaImportReportsAccessForBothGenerations)
{
    Producer p;
    {
        BufferView v = aether::interop::importDLPack(p.makeVersioned(true, kDLCUDA));
        EXPECT_EQ(v.access, Access::ReadOnly);
        EXPECT_EQ(v.view.device.type(), kDLCUDA);
    }
    {
        BufferView v = aether::interop::importDLPack(p.makeVersioned(false, kDLCUDA));
        EXPECT_EQ(v.access, Access::ReadWrite);
    }
    {
        BufferView v = aether::interop::importDLPack(p.makeLegacy(kDLCUDA));
        EXPECT_EQ(v.access, Access::Unknown);
        EXPECT_EQ(v.view.device.type(), kDLCUDA);
    }
    EXPECT_EQ(p.deletes, 3);
}
#else
// A pure C++ build holds every DLPack device kind as a record: imported and
// re-exported unchanged, never dereferenced (the storage pointer below is only
// carried, so a real device address would be just as safe).
TEST_F(DLPackBufferTest, CppModeHoldsDeviceKindsAsRecords)
{
    for (const DLDeviceType dev : { kDLCUDA, kDLCUDAHost, kDLCUDAManaged, kDLROCM, kDLMetal }) {
        Producer p;
        {
            BufferView v = aether::interop::importDLPack(p.makeVersioned(false, dev));
            EXPECT_EQ(v.view.device.type(), dev);
            EXPECT_EQ(v.view.data, static_cast<void*>(p.storage.data()));
            DLManagedTensorVersioned* out = aether::interop::exportDLPack(v);
            EXPECT_EQ(out->dl_tensor.device.device_type, dev);
            EXPECT_EQ(out->dl_tensor.data, static_cast<void*>(p.storage.data()));
            out->deleter(out);
        }
        {
            BufferView v = aether::interop::importDLPack(p.makeLegacy(dev));
            DLManagedTensor* out = aether::interop::exportDLPackLegacy(v);
            EXPECT_EQ(out->dl_tensor.device.device_type, dev);
            out->deleter(out);
        }
        EXPECT_EQ(p.deletes, 2) << "device type " << static_cast<int>(dev);
    }
}
#endif

TEST_F(DLPackBufferTest, UndefinedDeviceCodeIsRefusedInEveryBuild)
{
    Producer p;
    const auto bogus = static_cast<DLDeviceType>(99);
    const std::string m = thrownMessage([&] { aether::interop::importDLPack(p.makeVersioned(false, bogus)); });
    EXPECT_NE(m.find("unknown/unsupported device kind (type=99)"), std::string::npos) << m;
    EXPECT_EQ(p.deletes, 1) << "a refused import still runs the deleter once";
}

TEST_F(DLPackBufferTest, OnlyHostAccessRefusesADeviceRecord)
{
    EXPECT_TRUE(aether::interop::hostAccessible(kDLCPU));
    EXPECT_TRUE(aether::interop::hostAccessible(kDLCUDAHost));
    EXPECT_FALSE(aether::interop::hostAccessible(kDLCUDA));
    EXPECT_FALSE(aether::interop::hostAccessible(kDLCUDAManaged));

    Producer p;
    BufferView host = aether::interop::importDLPack(p.makeVersioned(false));
    EXPECT_NO_THROW(aether::interop::requireHostAccess(host, "read"));
#ifndef AETHER_HAS_CUDA
    Producer q;
    BufferView dev = aether::interop::importDLPack(q.makeVersioned(false, kDLCUDA));
    EXPECT_EQ(thrownMessage([&] { aether::interop::requireHostAccess(dev, "read"); }),
        "read: device: host access needs host memory, got cuda (a record only on the host)");
#endif
}

TEST_F(DLPackBufferTest, AccessNamesAreTheSharedSpelling)
{
    EXPECT_STREQ(aether::interop::accessName(Access::ReadWrite), "read-write");
    EXPECT_STREQ(aether::interop::accessName(Access::ReadOnly), "read-only");
    EXPECT_STREQ(aether::interop::accessName(Access::Unknown), "unknown");
}

TEST_F(DLPackBufferTest, ByteOffsetIsFoldedIntoTheDataPointer)
{
    Producer p;
    DLManagedTensorVersioned* t = p.makeVersioned(false);
    t->dl_tensor.byte_offset = 8;
    p.shape[1] = 3;
    BufferView b = aether::interop::importDLPack(t);
    EXPECT_EQ(b.view.data, static_cast<void*>(p.storage.data() + 1));
}

TEST_F(DLPackBufferTest, RefusedImportStillCallsTheDeleterOnce)
{
    Producer p;
    DLManagedTensorVersioned* t = p.makeVersioned(false);
    t->version.major = 2;
    EXPECT_NE(thrownMessage([&] { (void)aether::interop::importDLPack(t); }).find("major version 2"), std::string::npos);
    EXPECT_EQ(p.deletes, 1);
}

TEST_F(DLPackBufferTest, VersionedExportSetsReadOnlyFlagIffReadOnly)
{
    Producer p;
    for (bool ro : { false, true }) {
        BufferView b = aether::interop::importDLPack(p.makeVersioned(ro));
        DLManagedTensorVersioned* out = aether::interop::exportDLPack(b);
        EXPECT_EQ((out->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0, ro);
        EXPECT_EQ(out->dl_tensor.data, p.storage.data());
        EXPECT_EQ(out->dl_tensor.shape[0], 3);
        EXPECT_EQ(out->dl_tensor.strides[0], 4);
        BufferView back = aether::interop::importDLPack(out, "again");
        EXPECT_EQ(back.access, ro ? Access::ReadOnly : Access::ReadWrite);
        EXPECT_EQ(back.view.data, p.storage.data());
    }
    EXPECT_EQ(p.deletes, 2);
}

TEST_F(DLPackBufferTest, UnknownExportsReadOnlyUnlessAsserted)
{
    Producer p;
    BufferView b = aether::interop::importDLPack(p.makeLegacy());
    DLManagedTensorVersioned* out = aether::interop::exportDLPack(b);
    EXPECT_NE(out->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);
    out->deleter(out);
    aether::interop::assumeWritable(b);
    out = aether::interop::exportDLPack(b);
    EXPECT_EQ(out->flags & DLPACK_FLAG_BITMASK_READ_ONLY, 0u);
    out->deleter(out);
}

TEST_F(DLPackBufferTest, LegacyExportRoundTripsAndRefusesReadOnly)
{
    Producer p;
    {
        BufferView b = aether::interop::importDLPack(p.makeVersioned(false));
        DLManagedTensor* out = aether::interop::exportDLPackLegacy(b);
        BufferView back = aether::interop::importDLPack(out);
        EXPECT_EQ(back.access, Access::Unknown);
        EXPECT_EQ(back.view.data, p.storage.data());
        EXPECT_EQ(static_cast<long long>(back.view.extents[1]), 4LL);
    }
    EXPECT_EQ(p.deletes, 1);
    BufferView ro = aether::interop::importDLPack(p.makeVersioned(true));
    const std::string msg = thrownMessage([&] { (void)aether::interop::exportDLPackLegacy(ro); });
    EXPECT_NE(msg.find("access:"), std::string::npos) << msg;
    EXPECT_NE(msg.find("read-only"), std::string::npos) << msg;
}

TEST_F(DLPackBufferTest, ReexportChainKeepsTheFirstProducerAliveUntilTheLastView)
{
    Producer p;
    BufferView last;
    {
        BufferView first = aether::interop::importDLPack(p.makeVersioned(false));
        BufferView second = aether::interop::importDLPack(aether::interop::exportDLPack(first));
        BufferView third = aether::interop::importDLPack(aether::interop::exportDLPackLegacy(second));
        last = aether::interop::importDLPack(aether::interop::exportDLPack(third));
    }
    EXPECT_EQ(p.deletes, 0) << "the first producer must outlive every view of the chain";
    EXPECT_EQ(last.view.data, p.storage.data());
    last = BufferView{};
    EXPECT_EQ(p.deletes, 1) << "exactly once, after the last view died";
}

TEST_F(DLPackBufferTest, OwnedBufferReportsTheOwnerAndNoProducer)
{
    Producer p;
    BufferView src = aether::interop::importDLPack(p.makeVersioned(false));
    BufferView own = aether::interop::ownedBuffer("eagle", src.view, Access::ReadWrite);
    EXPECT_TRUE(own.owned());
    EXPECT_EQ(own.owner, "eagle");
    EXPECT_TRUE(own.producer.empty());
}

TEST_F(DLPackBufferTest, ArrayInterfaceFactoryBuildsTheSameRecord)
{
    std::vector<double> storage(12, 0.0);
    aether::interop::ArrayInterface ai;
    ai.data = storage.data();
    ai.readOnly = true;
    ai.shape = { 3, 4 };
    ai.strides = { 32, 8 };
    ai.typestr = "<f8";
    ai.device = DLDevice{ kDLCPU, 0 };
    ai.producer = "numpy.ndarray";
    BufferView b = aether::interop::fromArrayInterface(ai);
    EXPECT_EQ(b.view.data, storage.data());
    EXPECT_EQ(b.access, Access::ReadOnly);
    EXPECT_EQ(b.producer, "numpy.ndarray");
    EXPECT_EQ(static_cast<long long>(b.view.strides[0]), 4LL);
    EXPECT_EQ(static_cast<long long>(b.view.strides[1]), 1LL);
    EXPECT_EQ(static_cast<int>(b.view.dtype.code()), static_cast<int>(kDLFloat));
    EXPECT_EQ(b.view.dtype.bits(), 64);

    ai.readOnly = false;
    ai.strides.clear();
    ai.typestr = "|u1";
    BufferView c = aether::interop::fromArrayInterface(ai);
    EXPECT_EQ(c.access, Access::ReadWrite);
    EXPECT_EQ(static_cast<long long>(c.view.strides[0]), 4LL);
    EXPECT_EQ(c.view.dtype.bits(), 8);

    ai.typestr = "<f8";
    ai.strides = { 32, 4 };
    EXPECT_NE(thrownMessage([&] { (void)aether::interop::fromArrayInterface(ai); }).find("not a multiple"),
        std::string::npos);
    ai.typestr = ">f8";
    ai.strides.clear();
    EXPECT_NE(thrownMessage([&] { (void)aether::interop::fromArrayInterface(ai); }).find("little-endian"),
        std::string::npos);
}

TEST_F(DLPackBufferTest, SatisfiedRequirementsRaiseNothing)
{
    Producer p;
    BufferView b = aether::interop::importDLPack(p.makeVersioned(false));
    Requirements r;
    r.dtype = aether::DType(DLDataType{ kDLFloat, 64, 1 });
    r.count = 12;
    r.shape = std::vector<std::int64_t>{ 3, 4 };
    r.contiguous = true;
    r.alignment = 8;
    r.deviceType = kDLCPU;
    r.deviceId = 0;
    r.writable = true;
    EXPECT_TRUE(aether::interop::refusals(b, r).empty());
    EXPECT_NO_THROW(aether::interop::require(b, r));
}

/** @brief The single refusal `r` raises against `b`. */
inline std::string onlyRefusal(const BufferView& b, const Requirements& r)
{
    const auto out = aether::interop::refusals(b, r);
    EXPECT_EQ(out.size(), 1u);
    return out.empty() ? std::string() : out.front();
}

TEST_F(DLPackBufferTest, EveryRequirementRefusalFiresWithItsNamedMessage)
{
    Producer p;
    BufferView b = aether::interop::importDLPack(p.makeVersioned(true));

    Requirements dtype;
    dtype.dtype = aether::DType(DLDataType{ kDLFloat, 32, 1 });
    EXPECT_EQ(onlyRefusal(b, dtype), "dtype: required float32, got float64");

    Requirements count;
    count.count = 10;
    EXPECT_EQ(onlyRefusal(b, count), "count: required 10 elements, got 12");

    Requirements shape;
    shape.shape = std::vector<std::int64_t>{ 4, 3 };
    EXPECT_EQ(onlyRefusal(b, shape), "shape: required (4, 3), got (3, 4)");

    Requirements device;
    device.deviceType = kDLCUDA;
    EXPECT_EQ(onlyRefusal(b, device), "device: required cuda, got cpu");

    Requirements deviceId;
    deviceId.deviceId = 1;
    EXPECT_EQ(onlyRefusal(b, deviceId), "device id: required 1, got 0");

    Requirements access;
    access.writable = true;
    EXPECT_EQ(onlyRefusal(b, access), "access: required read-write, got read-only");

    Requirements stride;
    stride.contiguous = true;
    BufferView strided = b;
    strided.view.strides[1] = 2;
    EXPECT_EQ(onlyRefusal(strided, stride), "stride: required C-contiguous with unit stride, got strides (4, 2) for shape (3, 4)");

    Requirements align;
    align.alignment = 16;
    BufferView misaligned = b;
    misaligned.view.data = static_cast<void*>(reinterpret_cast<char*>(p.storage.data()) + 4);
    const std::string a = onlyRefusal(misaligned, align);
    EXPECT_EQ(a.rfind("alignment: required 16-byte aligned, got address ", 0), 0u) << a;

    Producer q;
    BufferView unknown = aether::interop::importDLPack(q.makeLegacy());
    EXPECT_EQ(onlyRefusal(unknown, access),
        "access: required read-write, got unknown (assert writability explicitly to accept it)");

    Requirements all = dtype;
    all.count = 10;
    all.writable = true;
    const std::string joined = thrownMessage([&] { aether::interop::require(b, all); });
    EXPECT_NE(joined.find("dtype: required float32"), std::string::npos) << joined;
    EXPECT_NE(joined.find("; count: required 10"), std::string::npos) << joined;
    EXPECT_NE(joined.find("; access: required read-write"), std::string::npos) << joined;
}

} // namespace dlpack_buffer
} // namespace aether_tests
