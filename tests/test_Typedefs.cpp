// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `aether::idx_t`/`aether::dims_t` track `aether::offset_t` exactly (one
// switch, `AETHER_INDEX_T`, `aether/typedefs.h`). Paired with
// test_Typedefs.cu (identical content): host-only static_asserts, no
// device code.

#include <type_traits>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {

static_assert(std::is_same_v<aether::idx_t, aether::offset_t>,
    "idx_t must be exactly offset_t — one switch, aether/typedefs.h");
static_assert(std::is_same_v<aether::dims_t, aether::offset_t>,
    "dims_t must be exactly offset_t — one switch, aether/typedefs.h");
static_assert(sizeof(aether::idx_t) == sizeof(aether::offset_t), "idx_t width must track offset_t");
static_assert(sizeof(aether::dims_t) == sizeof(aether::offset_t), "dims_t width must track offset_t");

class TypedefsTest : public ::testing::Test { };

TEST_F(TypedefsTest, WidthMatchesOffsetT)
{
    EXPECT_EQ(sizeof(aether::idx_t), sizeof(aether::offset_t));
    EXPECT_EQ(sizeof(aether::dims_t), sizeof(aether::offset_t));
}

} // namespace aether_tests
