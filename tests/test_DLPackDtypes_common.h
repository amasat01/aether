// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Shared body of test_DLPackDtypes.{cpp,cu}: the dtypes DLPack carries beyond float64/float32/int64/int32/
// uint64/uint32/uint8/bool round trip through both tensor generations, and the ones still refused say so.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <aether/device/Device.h>
#include <aether/dtype/DType.h>
#include <aether/err/Error.h>
#include <aether/interop/DLPack.h>
#include <aether/interop/DLPackLegacy.h>
#include <aether/view/RuntimeView.h>

namespace aether_tests {

struct DtypeCase {
    DLDataTypeCode code;
    std::uint8_t bits;
};

inline constexpr DtypeCase kAddedDtypes[] = {
    { kDLInt, 8 }, { kDLInt, 16 }, { kDLUInt, 16 }, { kDLFloat, 16 }, { kDLBfloat, 16 },
};

class DLPackDtypesTest : public ::testing::Test { };

inline aether::RuntimeView hostView(std::vector<std::byte>& storage, DtypeCase c)
{
    aether::RuntimeView v;
    v.data   = storage.data();
    v.dtype  = aether::DType(static_cast<std::uint8_t>(c.code), c.bits, 1);
    v.device = aether::Device(kDLCPU);
    v.rank   = 2;
    v.extents[0] = 3;
    v.extents[1] = 4;
    v.strides[0] = 4;
    v.strides[1] = 1;
    return v;
}

TEST_F(DLPackDtypesTest, AddedDtypesAreSupported)
{
    for (const DtypeCase& c : kAddedDtypes) {
        DLDataType dt{ static_cast<std::uint8_t>(c.code), c.bits, 1 };
        EXPECT_TRUE(aether::interop::detail::dtypeIsSupported(dt)) << int(c.code) << "/" << int(c.bits);
        dt.lanes = 2;
        EXPECT_FALSE(aether::interop::detail::dtypeIsSupported(dt));
    }
}

TEST_F(DLPackDtypesTest, VersionedRoundTripKeepsTheDtype)
{
    for (const DtypeCase& c : kAddedDtypes) {
        std::vector<std::byte> storage(12 * c.bits / 8);
        aether::RuntimeView v = hostView(storage, c);
        DLManagedTensorVersioned* t = aether::interop::toDLPack(v);
        ASSERT_NE(t, nullptr);
        EXPECT_EQ(t->dl_tensor.dtype.code, static_cast<std::uint8_t>(c.code));
        EXPECT_EQ(t->dl_tensor.dtype.bits, c.bits);
        auto imp = aether::interop::fromDLPack(t);
        EXPECT_EQ(imp.view.dtype, v.dtype);
        EXPECT_EQ(imp.view.data, storage.data());
        EXPECT_EQ(imp.view.extents[1], 4);
    }
}

TEST_F(DLPackDtypesTest, LegacyRoundTripKeepsTheDtype)
{
    for (const DtypeCase& c : kAddedDtypes) {
        std::vector<std::byte> storage(12 * c.bits / 8);
        aether::RuntimeView v = hostView(storage, c);
        DLManagedTensor* t = aether::interop::toDLPackLegacy(v);
        ASSERT_NE(t, nullptr);
        EXPECT_EQ(t->dl_tensor.dtype.code, static_cast<std::uint8_t>(c.code));
        auto imp = aether::interop::fromDLPack(t);
        EXPECT_EQ(imp.view.dtype, v.dtype);
        EXPECT_EQ(imp.view.data, storage.data());
    }
}

TEST_F(DLPackDtypesTest, ComplexAndWiderFloatsAreStillRefused)
{
    for (DLDataType dt : { DLDataType{ static_cast<std::uint8_t>(kDLComplex), 64, 1 },
             DLDataType{ static_cast<std::uint8_t>(kDLComplex), 128, 1 },
             DLDataType{ static_cast<std::uint8_t>(kDLFloat), 128, 1 },
             DLDataType{ static_cast<std::uint8_t>(kDLBfloat), 32, 1 } }) {
        EXPECT_FALSE(aether::interop::detail::dtypeIsSupported(dt));
    }
    std::vector<std::byte> storage(64);
    aether::RuntimeView v = hostView(storage, { kDLComplex, 64 });
    EXPECT_THROW(aether::interop::toDLPack(v), aether::Error);
    EXPECT_THROW(aether::interop::toDLPackLegacy(v), aether::Error);
}

} // namespace aether_tests
