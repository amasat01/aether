// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// backend/cpu/simd/ `aether::simd::select` tests (host / AETHER_CPP_MODE
// build). No .cu twin: `select` is host-only SIMD machinery (Packet.h
// carries no AETHER_DEVICEHOST-qualified code at all; see that header's
// own docstring).
//
// `aether::simd::select(mask, a, b)` -- lane-wise `mask ? a : b` -- for
// every Packet<T,W>/PacketMask pair Packet.h already instantiates.
// Exact-bit vs a scalar reference, ISA arms mirroring `fmadd`'s own
// pattern (Packet.h itself).
//
// The compile-fail row this file pins: an unqualified `select(mask, a, b)`
// call, with <sys/select.h> in scope, resolved to POSIX `::select` (5
// parameters) before `aether::simd::select` existed -- a hidden friend is
// invisible to ordinary unqualified lookup (only ADL finds it), and ADL
// candidates merge with, never replace, whatever ordinary lookup already
// found in an enclosing namespace. Captured directly from this repo,
// before `select` was added to Packet.h
// (`g++ -std=c++23 -DAETHER_CPP_MODE=1 -march=x86-64-v3`, this exact
// <sys/select.h> + Packet.h combination):
//
//   error: cannot convert 'aether::simd::PacketMask<double, 4>' to 'int'
//      11 |     auto c = select(m, a, b);
//         |                     ^
//         |                     aether::simd::PacketMask<double, 4>
//   note:   initializing argument 1 of
//     'int select(int, fd_set*, fd_set*, fd_set*, timeval*)'
//
// `UnqualifiedCallResolvesToOursNotPosixSelect` below is the green
// control: the identical unqualified call, <sys/select.h> equally in
// scope, now compiles and returns the right per-lane values via ADL on the
// aether::simd::Packet/PacketMask argument types.

#include <sys/select.h>

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <aether/backend/cpu/simd/simd.h>

namespace aether_tests {
namespace {

using aether::simd::Packet;
using aether::simd::PacketMask;
using aether::simd::PreferredWidth;

class SimdSelectTest : public ::testing::Test { };

TEST_F(SimdSelectTest, UnqualifiedCallResolvesToOursNotPosixSelect)
{
    // The RED-first evidence this test exists to keep GREEN -- see the
    // file header comment for the exact diagnostic <sys/select.h> being in
    // scope produced before `aether::simd::select` existed.
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) + 1.0;
        bIn[k] = -static_cast<double>(k) - 100.0;
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);
    auto m = PacketMask<double, W>::firstN(W / 2 + (W % 2));

    using namespace aether::simd; // Packet/PacketMask in scope; select is still found by ADL regardless of this
    auto c = select(m, a, b); // UNQUALIFIED -- must resolve to aether::simd::select, not POSIX ::select

    double out[W];
    Packet<double, W>::store(out, c);
    for (std::size_t k = 0; k < W; ++k) {
        const double expect = m.lane(k) ? aIn[k] : bIn[k];
        EXPECT_EQ(out[k], expect) << "lane " << k;
    }
}

/** @brief Exact-bit `select` vs a per-lane scalar reference, swept over
 *  every `firstN(n)` prefix mask (n = 0..W: all-false, all-true, and every
 *  partial pattern in between) for one `DataT`. Uses only PacketMask's
 *  PUBLIC surface (`firstN`), so this compiles identically whether
 *  `PreferredWidth<DataT>` resolves to a real SIMD width or the scalar
 *  (1-lane) fallback. */
template<typename DataT>
void checkSelectVsScalarReference()
{
    constexpr std::size_t W = PreferredWidth<DataT>;
    using PacketT = Packet<DataT, W>;
    using MaskT   = PacketMask<DataT, W>;

    DataT aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<DataT>(k) * DataT(2) + DataT(1);
        bIn[k] = -static_cast<DataT>(k) - DataT(50);
    }
    auto a = PacketT::load(aIn);
    auto b = PacketT::load(bIn);

    for (std::size_t n = 0; n <= W; ++n) {
        auto m = MaskT::firstN(n);
        // UNQUALIFIED, found via ADL on `m`/`a`/`b`'s aether::simd types --
        // `select` is a hidden friend (like `fmadd`/`cmpGe`), so
        // `aether::simd::select(...)` (explicit-namespace QUALIFIED
        // lookup) does NOT find it at all; only an unqualified call does.
        auto c = select(m, a, b);
        DataT out[W];
        PacketT::store(out, c);
        for (std::size_t k = 0; k < W; ++k) {
            const DataT expect = (k < n) ? aIn[k] : bIn[k];
            EXPECT_EQ(out[k], expect) << "W=" << W << " n=" << n << " lane=" << k;
        }
    }
}

TEST_F(SimdSelectTest, ExactBitVsScalarReferenceDouble)
{
    checkSelectVsScalarReference<double>();
}

TEST_F(SimdSelectTest, ExactBitVsScalarReferenceFloat)
{
    checkSelectVsScalarReference<float>();
}

TEST_F(SimdSelectTest, ExactBitVsScalarReferenceNonFloatingDataTAtScalarWidth)
{
    // "every Packet<T,W>/PacketMask pair the header already instantiates"
    // includes the Width==1 scalar fallback for ANY DataT (PacketMask.h's
    // own `PacketMask<DataT,1>` specialization is unconditional) --
    // PreferredWidth<int> resolves to 1 (PacketTraits.h: only double/float
    // get a real SIMD width), so this drives select's scalar (W==1) arm
    // with a genuinely different DataT than the float/double cases above.
    checkSelectVsScalarReference<int>();
}

TEST_F(SimdSelectTest, AllTrueMaskAlwaysPicksA)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) + 1.0;
        bIn[k] = -1.0;
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);
    auto c = select(PacketMask<double, W>::allTrue(), a, b); // unqualified, ADL-found
    double out[W];
    Packet<double, W>::store(out, c);
    EXPECT_EQ(0, std::memcmp(out, aIn, sizeof(out)));
}

TEST_F(SimdSelectTest, AllFalseMaskAlwaysPicksB)
{
    constexpr std::size_t W = PreferredWidth<double>;
    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) + 1.0;
        bIn[k] = -1.0;
    }
    auto a = Packet<double, W>::load(aIn);
    auto b = Packet<double, W>::load(bIn);
    auto c = select(PacketMask<double, W>::allFalse(), a, b); // unqualified, ADL-found
    double out[W];
    Packet<double, W>::store(out, c);
    EXPECT_EQ(0, std::memcmp(out, bIn, sizeof(out)));
}

TEST_F(SimdSelectTest, AlternatingMaskPicksPerLane)
{
    // A genuine (0,1,0,1,...) pattern, built ONLY from PacketMask's public
    // firstN/&/~/| surface -- no reach into the private per-ISA StorageT,
    // so this stays portable across every width (including the W==1
    // scalar fallback, where the loop below trivially runs once).
    constexpr std::size_t W = PreferredWidth<double>;
    using PacketT = Packet<double, W>;
    using MaskT   = PacketMask<double, W>;

    double aIn[W], bIn[W];
    for (std::size_t k = 0; k < W; ++k) {
        aIn[k] = static_cast<double>(k) * 3.0 + 1.0;
        bIn[k] = -static_cast<double>(k) - 7.0;
    }
    auto a = PacketT::load(aIn);
    auto b = PacketT::load(bIn);

    MaskT m = MaskT::allFalse();
    for (std::size_t k = 0; k < W; k += 2) {
        MaskT laneKOnly = MaskT::firstN(k + 1);
        if (k > 0)
            laneKOnly = laneKOnly & ~MaskT::firstN(k);
        m = m | laneKOnly;
    }

    auto c = select(m, a, b); // unqualified, ADL-found
    double out[W];
    PacketT::store(out, c);
    for (std::size_t k = 0; k < W; ++k) {
        const double expect = (k % 2 == 0) ? aIn[k] : bIn[k];
        EXPECT_EQ(out[k], expect) << "lane " << k;
    }
}

} // namespace
} // namespace aether_tests
