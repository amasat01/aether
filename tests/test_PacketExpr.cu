// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// backend/cpu/{packet,Tiled}.h bit-identity gate (CUDA build;
// test_PacketExpr.cpp is the identical host-build twin). The same
// expression (axpy on Vec3/Vec6, plus a chained sum) evaluated via
// packetEval and via the scalar assignment over identical data must
// memcmp exact (the per-lane-delegation rule: if a packet path can't be
// made bit-identical, stop and report — do not loosen to tolerance).
//
// Why -ffp-contract=off (tests/CMakeLists.txt applies it to this file
// only): both the scalar reference (`RecursiveAssign`, expr/Assign.h) and
// the packet/SIMD side (`Packet<T,W>`'s explicit `_mm256_mul_pd`/
// `_mm256_add_pd` etc.) compute `a + k*b` from plain `+`/`*` operators, and
// GCC 14 at -O3 -march=x86-64-v3 -std=c++23 fuses those into a hardware
// FMA by default — confirmed empirically for both sides (objdump: without
// this flag, `vfmadd...` shows up in the compiled scalar path and in the
// compiled packet path; with it, neither does). On this toolchain the two
// sides happen to stay fused (or unfused) in step, so they never diverge
// either way here — but that symmetry is an unwritten, version/compiler-
// specific optimizer behavior, not a documented contract (a different
// compiler could contract scalar arithmetic while treating the packet
// side's target intrinsics as opaque, or vice versa). Bit-identity is
// achieved by construction — force both sides to the same, non-fused,
// single-rounding-per-op semantics — never by relying on today's compiler
// happening to keep them in step.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Device;
using aether::dyn;
using aether::make_view;
using aether::SampleIndex;

class PacketExprBitIdentityTest : public ::testing::Test { };

// n = 37: odd, not a multiple of any native packet width (2/4/8/16) this
// library targets, so every run below exercises a genuine masked TAIL
// packet in addition to full packets — matching the twin harness's own
// non-power-of-two convention (tests/twin/test_TwinArithmetic.*).
constexpr std::size_t kN = 37;

template<std::size_t Dim>
void fillDeterministic(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
    a.assign(Dim * kN, 0.0);
    b.assign(Dim * kN, 0.0);
    c.assign(Dim * kN, 0.0);
    for (std::size_t idx = 0; idx < kN; ++idx) {
        for (std::size_t d = 0; d < Dim; ++d) {
            a[d * kN + idx] = static_cast<double>(idx) * 0.75 + static_cast<double>(d) * 0.1 - 3.0;
            b[d * kN + idx] = static_cast<double>(idx) * -0.5 + static_cast<double>(d) + 1.25;
            c[d * kN + idx] = static_cast<double>(idx) * 0.2 - static_cast<double>(d) * 0.3 + 0.5;
        }
    }
}

// Reference: scalar per-sample assignment through the SampleRef path
// (`view[i] = expr`, backed by `expr/Assign.h`'s `RecursiveAssign`).
template<class ViewL, class Expr>
void scalarReferenceEval(ViewL& dest, const Expr& expr)
{
    for (std::size_t idx = 0; idx < dest.samples(); ++idx)
        dest[SampleIndex::make(idx)] = expr;
}

TEST_F(PacketExprBitIdentityTest, Vec3AxpyPacketEvalMatchesScalarAssignmentExactly)
{
    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<3>(aBuf, bBuf, cBuf);
    std::vector<double> outPacket(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovPacket = make_view<double, 3, dyn>(outPacket.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = 2.5;
    auto expr = av + k * bv; // out = a + k*b — the twin harness's own pinned axpy formula shape

    aether::packetEval(ovPacket, expr);
    scalarReferenceEval(ovScalar, expr);

    EXPECT_EQ(0, std::memcmp(outPacket.data(), outScalar.data(), outPacket.size() * sizeof(double)))
        << "Vec3 axpy: packetEval diverges from the scalar RecursiveAssign reference (bit-identity required)";
}

TEST_F(PacketExprBitIdentityTest, Vec6AxpyPacketEvalMatchesScalarAssignmentExactly)
{
    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<6>(aBuf, bBuf, cBuf);
    std::vector<double> outPacket(6 * kN, 0.0), outScalar(6 * kN, 0.0);

    auto av = make_view<double, 6, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 6, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovPacket = make_view<double, 6, dyn>(outPacket.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 6, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = -3.25;
    auto expr = av + k * bv;

    aether::packetEval(ovPacket, expr);
    scalarReferenceEval(ovScalar, expr);

    EXPECT_EQ(0, std::memcmp(outPacket.data(), outScalar.data(), outPacket.size() * sizeof(double)))
        << "Vec6 axpy: packetEval diverges from the scalar RecursiveAssign reference (bit-identity required)";
}

TEST_F(PacketExprBitIdentityTest, ChainedSumPacketEvalMatchesScalarAssignmentExactly)
{
    // (a + b) + (k*c - a) — Sum-of-Sum and CWiseScale-of-leaf together,
    // mirroring test_ExprArithmetic.cpp's ComposedExpressionMatches...
    // shape ("plus a chained sum").
    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<3>(aBuf, bBuf, cBuf);
    std::vector<double> outPacket(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto cv = make_view<double, 3, dyn>(cBuf.data(), Device(kDLCPU), kN);
    auto ovPacket = make_view<double, 3, dyn>(outPacket.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = 1.75;
    auto expr = (av + bv) + (k * cv - av);

    aether::packetEval(ovPacket, expr);
    scalarReferenceEval(ovScalar, expr);

    EXPECT_EQ(0, std::memcmp(outPacket.data(), outScalar.data(), outPacket.size() * sizeof(double)))
        << "chained sum: packetEval diverges from the scalar RecursiveAssign reference (bit-identity required)";
}

TEST_F(PacketExprBitIdentityTest, PacketEvalParallelMatchesScalarAssignmentExactly)
{
    // Same Vec3 axpy, driven through the OMP-parallel dispatch
    // (`packetEvalParallel` -> `packetFor(..., parallel=true)`) — the
    // per-lane arithmetic is identical to the serial path, so bit-identity
    // must hold here too.
    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<3>(aBuf, bBuf, cBuf);
    std::vector<double> outPacket(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovPacket = make_view<double, 3, dyn>(outPacket.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = 0.5;
    auto expr = av + k * bv;

    aether::packetEvalParallel(ovPacket, expr);
    scalarReferenceEval(ovScalar, expr);

    EXPECT_EQ(0, std::memcmp(outPacket.data(), outScalar.data(), outPacket.size() * sizeof(double)))
        << "packetEvalParallel: diverges from the scalar RecursiveAssign reference (bit-identity required)";
}

TEST_F(PacketExprBitIdentityTest, Vec3AxpyWithFmaSensitiveDataStillExactAcrossAllLanes)
{
    // Hand-picked (a,b,k) triple (brute-force search) where std::fma(k,b,a)
    // GENUINELY differs from `a + (k*b)` computed as two separate
    // roundings (the ASSERT_NE below proves this in isolation). Embedded
    // at sample 0, component 0 among otherwise ordinary axpy data as an
    // extra correctness check over large-magnitude, rounding-sensitive
    // values — on THIS toolchain (GCC 14) both the scalar reference and
    // the packet path get fused (or not) IN STEP regardless of
    // -ffp-contract=off (see the file docstring), so this test does not
    // itself demonstrate a RED without the flag; the flag stays applied as
    // portable, non-negotiable insurance rather than something re-derived
    // from what one compiler happens to do today.
    constexpr double kA = -585694.80845378607046;
    constexpr double kB = -992135.49521465529688;
    constexpr double kK = -973940.5769732396584;
    ASSERT_NE(std::fma(kK, kB, kA), kA + (kK * kB))
        << "sanity: this (a,b,k) triple must genuinely diverge under fma vs a"
           " separate mul+add, or the adversarial-data premise here is stale"
           " and needs re-deriving (brute-force search)";

    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<3>(aBuf, bBuf, cBuf);
    aBuf[0 * kN + 0] = kA;
    bBuf[0 * kN + 0] = kB;

    std::vector<double> outPacket(3 * kN, 0.0), outScalar(3 * kN, 0.0);
    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovPacket = make_view<double, 3, dyn>(outPacket.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    auto expr = av + kK * bv;
    aether::packetEval(ovPacket, expr);
    scalarReferenceEval(ovScalar, expr);

    EXPECT_EQ(0, std::memcmp(outPacket.data(), outScalar.data(), outPacket.size() * sizeof(double)))
        << "FMA-sensitive axpy: packetEval diverges from the scalar RecursiveAssign reference"
           " (exactly the divergence -ffp-contract=off exists to prevent)";
}

TEST_F(PacketExprBitIdentityTest, PacketCaptureThenAssignMatchesDirectPacketEvalExactly)
{
    // Single-pass fused pattern PacketItem.h's docstring shows:
    // materialize an intermediate into a register-resident PacketItem
    // (packetCapture), build a further expression over it, then
    // packetAssign — must match a direct packetEval of the ALGEBRAICALLY
    // equivalent expression exactly (both paths are pure Packet<T,W>
    // arithmetic — no scalar/SIMD asymmetry here, so this holds regardless
    // of -ffp-contract, but the flag is applied file-wide regardless).
    std::vector<double> aBuf, bBuf, cBuf;
    fillDeterministic<3>(aBuf, bBuf, cBuf);
    std::vector<double> outDirect(3 * kN, 0.0), outCaptured(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovDirect = make_view<double, 3, dyn>(outDirect.data(), Device(kDLCPU), kN);
    auto ovCaptured = make_view<double, 3, dyn>(outCaptured.data(), Device(kDLCPU), kN);

    const double k = 2.5;
    auto expr = av + k * bv;
    aether::packetEval(ovDirect, expr);

    aether::packetFlatFor<double>(0, kN, [&](const auto& pi) {
        auto captured = aether::packetCapture<double>(expr, pi);
        aether::packetAssign(ovCaptured, captured, pi);
    });

    EXPECT_EQ(0, std::memcmp(outDirect.data(), outCaptured.data(), outDirect.size() * sizeof(double)))
        << "packetCapture-then-packetAssign diverges from a direct packetEval of the same expression";
}

} // namespace
} // namespace aether_tests
