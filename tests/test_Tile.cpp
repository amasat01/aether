// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Tile tests (host / AETHER_CPP_MODE build; test_Tile.cu covers the same
// fixture and case names) over the thin `aether::Tile`/`aether::slot<s,D>()`
// facade (`aether/view/Tile.h`) over `aether::Item`/
// `Expression::segment<Off,Len>()`.
//
// Four rows are dropped rather than ported: writing through a slot
// (`tile.template slot<1>() = 2.0 * Item<double,3>::Ones();`, etc.) has no
// aether counterpart, because `Expression::segment<Off,Len>()` — the
// mechanism `aether::slot<s,D>()` forwards to (`aether/view/Tile.h`) —
// returns a lazy `detail::Segment` read node with no write-back path
// (`aether/expr/nodes/Structural.h`'s own docstring); confirmed by direct
// compilation (`Segment` carries no `operator=`, unlike `Item`/`SampleRef`).
// This is the same root cause already on record for the existing
// `Vec3dTest`/`QuaternionTest` `*SegmentAssign`/`SegmentAddSubAssign`/
// `SegmentMulDivAssign`/`HeadAssign`/`tailAssign` rows — those four
// `TileTest` rows get the same treatment here, for the same reason. Only
// `DefaultConstructZeroes` and `InitValueConstructor` — construction only,
// no `slot()` call — survive unmodified; see
// `tests/test_ExprStructural.cpp`'s own header comment for the precedent
// this mirrors.

#include <cstddef>

#include <gtest/gtest.h>

#include <aether/view/Tile.h>

namespace aether_tests {
namespace {

using aether::Tile;

class TileTest : public ::testing::Test { };

/** @brief Default-construct: every component zero. */
TEST_F(TileTest, DefaultConstructZeroes)
{
    Tile<double, 3, 4> tile;
    for (std::size_t i = 0; i < tile.size(); i++) {
        EXPECT_DOUBLE_EQ(tile.data()[i], 0);
    }
}

/** @brief Initial-value constructor (`Item`'s N-scalar ctor):
 *  every component matches. Unlike a single-value broadcast constructor,
 *  `aether::Item`'s N-scalar ctor takes exactly `Size` values — spelled
 *  out here rather than broadcast, since `Tile` adds no broadcast-from-one
 *  constructor of its own (a thin alias adds no members). */
TEST_F(TileTest, InitValueConstructor)
{
    Tile<double, 3, 4> tile{ 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5 };
    for (std::size_t i = 0; i < tile.size(); i++) {
        EXPECT_DOUBLE_EQ(tile.data()[i], 7.5);
    }
}

} // namespace
} // namespace aether_tests
