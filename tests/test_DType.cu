// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// DType tests (CUDA build; test_DType.cpp is the host-build twin, same
// fixture and case names): DType carries no CUDA-specific behaviour, so the
// two files are intentionally near-identical. Purely host-logic tests — no
// GPU execution needed here.

#include <cstdint>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <aether/dtype/DType.h>
#include <aether/dtype/Format.h>
#include <aether/dtype/dlpack.h>
#include <aether/index/Offset.h>

namespace aether_tests {
namespace {

class DTypeTest : public ::testing::Test { };

TEST_F(DTypeTest, DtypeOfDouble)
{
    auto dt = aether::dtype_of<double>();
    EXPECT_EQ(dt.code(), static_cast<std::uint8_t>(kDLFloat));
    EXPECT_EQ(dt.bits(), 64);
    EXPECT_EQ(dt.lanes(), 1);
    EXPECT_EQ(dt.size_bytes(), 8u);
}

TEST_F(DTypeTest, DtypeOfFloat)
{
    auto dt = aether::dtype_of<float>();
    EXPECT_EQ(dt.code(), static_cast<std::uint8_t>(kDLFloat));
    EXPECT_EQ(dt.bits(), 32);
    EXPECT_EQ(dt.size_bytes(), 4u);
}

TEST_F(DTypeTest, DtypeOfBoolIsKDLBool)
{
    auto dt = aether::dtype_of<bool>();
    EXPECT_EQ(dt.code(), static_cast<std::uint8_t>(kDLBool));
    EXPECT_EQ(dt.bits(), 8);
    EXPECT_EQ(dt.size_bytes(), 1u);
}

TEST_F(DTypeTest, DtypeOfSignedIntegers)
{
    auto i64 = aether::dtype_of<std::int64_t>();
    EXPECT_EQ(i64.code(), static_cast<std::uint8_t>(kDLInt));
    EXPECT_EQ(i64.bits(), 64);
    EXPECT_EQ(i64.size_bytes(), 8u);

    auto i32 = aether::dtype_of<std::int32_t>();
    EXPECT_EQ(i32.code(), static_cast<std::uint8_t>(kDLInt));
    EXPECT_EQ(i32.bits(), 32);
    EXPECT_EQ(i32.size_bytes(), 4u);
}

// aether::int_t — the codegen Int value type, distinct from
// aether::offset_t (32-bit address arithmetic).
TEST_F(DTypeTest, IntTIsInt64AndDistinctFromOffsetT)
{
    static_assert(std::is_same_v<aether::int_t, std::int64_t>,
        "aether::int_t must be std::int64_t");
    static_assert(!std::is_same_v<aether::int_t, aether::offset_t>,
        "aether::int_t (64-bit value type) must stay distinct from "
        "aether::offset_t (32-bit address-arithmetic width)");

    auto dt = aether::dtype_of<aether::int_t>();
    EXPECT_EQ(dt.code(), static_cast<std::uint8_t>(kDLInt));
    EXPECT_EQ(dt.bits(), 64);
    EXPECT_EQ(dt.size_bytes(), 8u);
}

TEST_F(DTypeTest, DtypeOfUnsignedIntegers)
{
    auto u64 = aether::dtype_of<std::uint64_t>();
    EXPECT_EQ(u64.code(), static_cast<std::uint8_t>(kDLUInt));
    EXPECT_EQ(u64.bits(), 64);

    auto u32 = aether::dtype_of<std::uint32_t>();
    EXPECT_EQ(u32.code(), static_cast<std::uint8_t>(kDLUInt));
    EXPECT_EQ(u32.bits(), 32);

    auto u8 = aether::dtype_of<std::uint8_t>();
    EXPECT_EQ(u8.code(), static_cast<std::uint8_t>(kDLUInt));
    EXPECT_EQ(u8.bits(), 8);
    EXPECT_EQ(u8.size_bytes(), 1u);
}

TEST_F(DTypeTest, SizeBytesAccountsForLanes)
{
    aether::DType dt(static_cast<std::uint8_t>(kDLFloat), 32, 4);
    EXPECT_EQ(dt.size_bytes(), 16u);
}

TEST_F(DTypeTest, EqualityComparesCodeBitsLanes)
{
    auto a = aether::dtype_of<double>();
    auto b = aether::dtype_of<double>();
    auto c = aether::dtype_of<float>();
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(DTypeTest, DefaultConstructedIsZero)
{
    aether::DType dt;
    EXPECT_EQ(dt.code(), 0);
    EXPECT_EQ(dt.bits(), 0);
    EXPECT_EQ(dt.lanes(), 0);
}

TEST_F(DTypeTest, ConstexprConstructible)
{
    constexpr aether::DType dt(static_cast<std::uint8_t>(kDLFloat), 64, 1);
    static_assert(dt.size_bytes() == 8, "DType must be constexpr-constructible (L2)");
    constexpr auto dof = aether::dtype_of<double>();
    static_assert(dof.bits() == 64, "dtype_of<double>() must be constexpr (L2)");
    EXPECT_EQ(dt.size_bytes(), 8u);
}

TEST_F(DTypeTest, FormatToStringNamesTheCodeAndBits)
{
    auto dt = aether::dtype_of<double>();
    std::string s = aether::to_string(dt);
    EXPECT_NE(s.find("float"), std::string::npos);
    EXPECT_NE(s.find("64"), std::string::npos);

    aether::DType vec(static_cast<std::uint8_t>(kDLFloat), 32, 4);
    std::string vs = aether::to_string(vec);
    EXPECT_NE(vs.find("x4"), std::string::npos);
}

} // namespace
} // namespace aether_tests
