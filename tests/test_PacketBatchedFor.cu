// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `optimalTileSize`/`packetBatchedFor` (aether/backend/cpu/{L2Cache.h,
// Tiled.h}). Three checks:
//   (a) known-answer schedule: exact tile+packet boundaries for 3 explicit
//       (N, tileOverride) pairs, one where N is not a multiple of the tile
//       size and one where the TILE SIZE ITSELF is not a multiple of the
//       SIMD packet width (so a tile boundary forces a mid-stream tail
//       packet a flat traversal at the same width would never produce).
//   (b) result bit-identity vs packetFlatFor on a 3-op expression tree
//       (`a + k*b - c`: Scale, Sum, Sub).
//   (c) optimalTileSize monotone (non-decreasing) in the L2 parameter,
//       spanning both the MIN- and MAX-clamped ends so the assertion is
//       not a vacuous plateau.
//
// Host-only bodies compiled in both modes; paired with
// test_PacketBatchedFor.cpp (identical content): no device code here
// (packetBatchedFor/optimalTileSize are host-SIMD-only), so this file
// simply runs the same host path under nvcc's host frontend.

#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {

namespace {

struct PacketRecord {
    std::size_t base;
    std::size_t width;
};

/** @brief Independent reference enumeration: for each explicit tile
 *  `[boundaries[t], boundaries[t+1])`, the full-then-tail packet sequence
 *  at packet width `W` — the SAME grammar `packetFlatFor` uses WITHIN one
 *  range, applied per DECLARED tile boundary (given explicitly here,
 *  matching the `tileOverride` passed to the real call below) rather than
 *  re-deriving which boundaries `packetBatchedFor` itself should choose. */
std::vector<PacketRecord> expectedTrace(const std::vector<std::size_t>& boundaries, std::size_t W)
{
    std::vector<PacketRecord> out;
    for (std::size_t t = 0; t + 1 < boundaries.size(); ++t) {
        std::size_t i = boundaries[t];
        const std::size_t tileEnd = boundaries[t + 1];
        for (; i + W <= tileEnd; i += W)
            out.push_back({ i, W });
        if (i < tileEnd)
            out.push_back({ i, tileEnd - i });
    }
    return out;
}

/** @brief Run `aether::packetBatchedFor<double>` SERIALLY (parallel=false,
 *  outside any omp region) with an explicit `tileOverride`, recording the
 *  visited `(base, width)` sequence in traversal order. Serial + explicit
 *  tile ⇒ the thread-count clamp never fires and the tile boundaries are
 *  EXACTLY `{begin, begin+tile, begin+2*tile, ..., end}` (clipped) — a
 *  deterministic, machine-independent schedule. */
std::vector<PacketRecord> recordSchedule(std::size_t begin, std::size_t end, std::size_t tile)
{
    std::vector<PacketRecord> trace;
    aether::packetBatchedFor<double>(
        begin, end, [&](const auto& pi) { trace.push_back({ pi.base_, pi.active_ }); },
        /*bytesPerSample=*/64, /*parallel=*/false, /*tileOverride=*/tile);
    return trace;
}

} // namespace

class PacketBatchedForTest : public ::testing::Test { };

TEST_F(PacketBatchedForTest, KnownAnswerScheduleThreeSizes)
{
    constexpr std::size_t W = aether::simd::PreferredWidth<double>;

    // Size 1: N=512, tile=128 — exact multiple (4 equal tiles; 128 is a
    // multiple of any realistic SIMD width).
    {
        const std::vector<std::size_t> boundaries{ 0, 128, 256, 384, 512 };
        const auto expected = expectedTrace(boundaries, W);
        const auto actual = recordSchedule(0, 512, 128);
        ASSERT_EQ(expected.size(), actual.size()) << "Size1: packet count mismatch";
        for (std::size_t k = 0; k < expected.size(); ++k) {
            EXPECT_EQ(expected[k].base, actual[k].base) << "Size1: packet " << k << " base";
            EXPECT_EQ(expected[k].width, actual[k].width) << "Size1: packet " << k << " width";
        }
    }

    // Size 2: N=530, tile=128 — N is NOT a multiple of the tile size (the
    // required "non-multiple" case): a ragged final tile [512,530).
    {
        const std::vector<std::size_t> boundaries{ 0, 128, 256, 384, 512, 530 };
        const auto expected = expectedTrace(boundaries, W);
        const auto actual = recordSchedule(0, 530, 128);
        ASSERT_EQ(expected.size(), actual.size()) << "Size2: packet count mismatch";
        for (std::size_t k = 0; k < expected.size(); ++k) {
            EXPECT_EQ(expected[k].base, actual[k].base) << "Size2: packet " << k << " base";
            EXPECT_EQ(expected[k].width, actual[k].width) << "Size2: packet " << k << " width";
        }
    }

    // Size 3: N=100, tile=15 — the tile size itself is NOT a multiple of W
    // (15 is odd; W is a power of two), so EVERY tile restarts the packet
    // stream with a mid-tile tail — a schedule a flat [0,100) traversal at
    // the same W would never produce (100 % W == 0 for every realistic W
    // here, so a flat run would have NO tail at all). This is the case
    // that actually pins the tile-boundary-driven restart, distinct from
    // the aligned coincidence Size1/Size2 happen to share with a flat run.
    {
        const std::vector<std::size_t> boundaries{ 0, 15, 30, 45, 60, 75, 90, 100 };
        const auto expected = expectedTrace(boundaries, W);
        const auto actual = recordSchedule(0, 100, 15);
        ASSERT_EQ(expected.size(), actual.size()) << "Size3: packet count mismatch";
        for (std::size_t k = 0; k < expected.size(); ++k) {
            EXPECT_EQ(expected[k].base, actual[k].base) << "Size3: packet " << k << " base";
            EXPECT_EQ(expected[k].width, actual[k].width) << "Size3: packet " << k << " width";
        }
        // Sanity: prove the tile-restart claim isn't vacuous — at least one
        // packet in this trace is a mid-stream tail packet (strictly
        // narrower than W), not merely the one at the very end of [0,100).
        bool sawInteriorTail = false;
        for (std::size_t k = 0; k + 1 < expected.size(); ++k)
            if (expected[k].width < W)
                sawInteriorTail = true;
        EXPECT_TRUE(sawInteriorTail) << "Size3 must exercise a mid-stream tile-restart tail packet";
    }
}

TEST_F(PacketBatchedForTest, MatchesPacketFlatForOnThreeOpET)
{
    // 3-op ET: a + k*b - c (Scale, Sum, Sub). N deliberately odd, and the
    // tileOverride below deliberately not a multiple of N — the batched
    // path visits a ragged final tile the flat path never sees at all; the
    // two MUST still agree bit-for-bit.
    constexpr std::size_t n = 101;
    constexpr double k = 2.5;

    aether::Array<double, 3> a(n), b(n), c(n), destFlat(n), destBatched(n);
    auto av = a.hostView();
    auto bv = b.hostView();
    auto cv = c.hostView();
    for (std::size_t idx = 0; idx < n; ++idx) {
        for (std::size_t comp = 0; comp < 3; ++comp) {
            av(comp, idx) = static_cast<double>(idx) * 0.75 + static_cast<double>(comp) * 0.1 - 3.0;
            bv(comp, idx) = static_cast<double>(idx) * -0.5 + static_cast<double>(comp) + 1.25;
            cv(comp, idx) = static_cast<double>(idx) * 0.2 - static_cast<double>(comp) * 0.3 + 0.4;
        }
    }

    auto flatOut = destFlat.hostView();
    aether::packetEval(flatOut, av + k * bv - cv);

    auto batchedOut = destBatched.hostView();
    const auto expr = av + k * bv - cv;
    // tileOverride=17: neither a multiple of n nor of the SIMD width —
    // forces a genuinely ragged tiled traversal distinct from the flat one.
    aether::packetBatchedFor<double>(
        0, batchedOut.samples(), [&](const auto& pi) { aether::packetAssign(batchedOut, expr, pi); },
        /*bytesPerSample=*/64, /*parallel=*/false, /*tileOverride=*/17);

    ASSERT_EQ(destFlat.samples(), destBatched.samples());
    for (std::size_t idx = 0; idx < n; ++idx) {
        for (std::size_t comp = 0; comp < 3; ++comp) {
            const double fv = flatOut(comp, idx);
            const double bv2 = batchedOut(comp, idx);
            EXPECT_EQ(std::memcmp(&fv, &bv2, sizeof(double)), 0)
                << "packetBatchedFor diverges from packetFlatFor at comp=" << comp << " idx=" << idx;
        }
    }
}

class OptimalTileSizeTest : public ::testing::Test { };

TEST_F(OptimalTileSizeTest, MonotoneInL2Parameter)
{
    constexpr std::size_t bytesPerSample = 64;

    // Spans BOTH clamps: l2Tiny drives K to MIN_TILE_SAMPLES, l2Huge drives
    // K to MAX_TILE_SAMPLES — so the monotonicity assertion below is not
    // vacuously true (all-equal); genuine growth happens between the ends.
    const std::size_t kTiny = aether::optimalTileSize<double>(bytesPerSample, /*l2Override=*/4096);
    const std::size_t kMid = aether::optimalTileSize<double>(bytesPerSample, /*l2Override=*/262144);
    const std::size_t kHuge = aether::optimalTileSize<double>(bytesPerSample, /*l2Override=*/1048576);

    EXPECT_GE(kTiny, aether::MIN_TILE_SAMPLES);
    EXPECT_LE(kHuge, aether::MAX_TILE_SAMPLES);

    EXPECT_LE(kTiny, kMid) << "optimalTileSize must be non-decreasing in the L2 parameter";
    EXPECT_LE(kMid, kHuge) << "optimalTileSize must be non-decreasing in the L2 parameter";
    EXPECT_LT(kTiny, kHuge) << "the L2 range chosen must produce a GENUINE increase, not a vacuous plateau";

    // A finer sweep, purely non-decreasing (no strictness requirement — a
    // plateau AT either clamp is legitimate there).
    std::size_t previous = 0;
    bool first = true;
    for (std::size_t l2 : { std::size_t{ 4096 }, std::size_t{ 16384 }, std::size_t{ 65536 }, std::size_t{ 262144 },
             std::size_t{ 524288 }, std::size_t{ 1048576 }, std::size_t{ 4194304 } }) {
        const std::size_t kk = aether::optimalTileSize<double>(bytesPerSample, l2);
        if (!first) {
            EXPECT_GE(kk, previous) << "optimalTileSize regressed at l2=" << l2;
        }
        previous = kk;
        first = false;
    }
}

} // namespace aether_tests
