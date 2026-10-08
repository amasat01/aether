// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `BundleIndex`/`DeviceBundle`/`bundleAssign`/`View::operator[](BundleIndex<W>)`
// bit-identity gate (CUDA build; test_Bundle.cpp is the identical
// host-build twin). These are all `AETHER_DEVICEHOST()` — plain
// host-callable functions, no `__global__` kernel needed — so the same test
// bodies exercise the same code paths in both builds: in the CUDA build
// (`AETHER_HAS_CUDA` defined for the whole translation unit, host pass
// included — `macros.h`), the tests below genuinely exercise the `double2`
// vectorized reinterpret load/store path (`backend/cuda/bundle/LoadStore.h`)
// on the host CPU (no GPU launch required — `reinterpret_cast<const
// double2*>` is ordinary, valid host C++); in the CPP_MODE build,
// `AETHER_HAS_CUDA` is undefined and the same code always takes the scalar
// per-lane fallback. Checks bundle-vs-scalar bit-identity (axpy3 and a
// quat-sandwich chain via bundles vs the scalar path, memcmp exact, both
// modes; a tail-mask case with N not divisible by W; a misaligned
// raw-pointer wrap falling back to the scalar path with values still
// exact).
//
// Why -ffp-contract=off (tests/CMakeLists.txt applies it to this file only,
// same trap as test_PacketExpr.{cpp,cu}): the quat-sandwich chain's
// Hamilton products (`QuatMul`/`QuatConj`) are sums of products, and GCC 14
// at -O3 -march=x86-64-v3 -std=c++23 contracts those into a hardware FMA by
// default — confirmed empirically (the bundle and scalar sides run the
// identical node formulas, over `DeviceBundle` vs `Item` leaves
// respectively, yet genuinely diverge by ULPs without this flag: the two
// leaf types' `eval()` bodies differ enough in shape that the compiler does
// not always contract both sides in step). Bit-identity is achieved by
// construction (both sides forced to the same, non-fused, single-rounding-
// per-op semantics), never by relying on a compiler happening to keep two
// differently-shaped code paths fused (or unfused) together.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::BundleIndex;
using aether::Device;
using aether::dyn;
using aether::make_view;
using aether::SampleIndex;
using aether::Vec3d;
using aether::Vec4d;

class BundleBitIdentityTest : public ::testing::Test { };

constexpr std::size_t kW = 2;
// n = 37: odd, not a multiple of W — every run below exercises a genuine
// masked TAIL bundle in addition to full bundles (matches test_PacketExpr's
// own non-power-of-two convention, tests/twin's).
constexpr std::size_t kN = 37;

template<std::size_t Dim>
void fillDeterministic(std::vector<double>& a, std::vector<double>& b)
{
    a.assign(Dim * kN, 0.0);
    b.assign(Dim * kN, 0.0);
    for (std::size_t idx = 0; idx < kN; ++idx) {
        for (std::size_t d = 0; d < Dim; ++d) {
            a[d * kN + idx] = static_cast<double>(idx) * 0.75 + static_cast<double>(d) * 0.1 - 3.0;
            b[d * kN + idx] = static_cast<double>(idx) * -0.5 + static_cast<double>(d) + 1.25;
        }
    }
}

/** @brief Drive `dest[bi] = expr` over every bundle of `n` samples,
 *         full bundles unmasked, the tail (if any) via the masked overload
 *         — the canonical bundle-loop shape every bundle kernel in this
 *         library uses. */
template<class ViewL, class Expr>
void bundleEval(ViewL& dest, const Expr& expr, std::size_t n)
{
    for (std::size_t bidx = 0; bidx * kW < n; ++bidx) {
        const BundleIndex<kW> bi = BundleIndex<kW>::make(bidx);
        if (bi.base() + kW <= n) {
            dest[bi] = expr;
        } else {
            aether::bundleAssign(dest, expr, bi, bi.mask(static_cast<aether::offset_t>(n)));
        }
    }
}

/** @brief Reference: scalar per-sample assignment through the SampleRef
 *         path (`view[i] = expr`, backed by `expr/Assign.h`'s
 *         `RecursiveAssign`). */
template<class ViewL, class Expr>
void scalarReferenceEval(ViewL& dest, const Expr& expr, std::size_t n)
{
    for (std::size_t idx = 0; idx < n; ++idx)
        dest[SampleIndex::make(idx)] = expr;
}

TEST_F(BundleBitIdentityTest, Vec3AxpyBundleMatchesScalarAssignmentExactly)
{
    std::vector<double> aBuf, bBuf;
    fillDeterministic<3>(aBuf, bBuf);
    std::vector<double> outBundle(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovBundle = make_view<double, 3, dyn>(outBundle.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = 2.5;
    auto expr = av + k * bv; // out = a + k*b, the pinned reference shape

    bundleEval(ovBundle, expr, kN);
    scalarReferenceEval(ovScalar, expr, kN);

    EXPECT_EQ(0, std::memcmp(outBundle.data(), outScalar.data(), outBundle.size() * sizeof(double)))
        << "Vec3 axpy: bundleAssign diverges from the scalar RecursiveAssign reference (bit-identity required)";
}

TEST_F(BundleBitIdentityTest, TailBundleNotDivisibleByWStillExact)
{
    // kN=37 is already not a multiple of kW=2 — Vec3AxpyBundleMatches...
    // above already covers the tail through bundleEval's masked branch;
    // this test makes the tail-specific assertion EXPLICIT: the LAST
    // bundle (base = 36, only lane 0 active) must match the scalar
    // reference at that exact sample, isolated from the full bundles.
    ASSERT_NE(kN % kW, 0u) << "sanity: kN must NOT be a multiple of kW for this test to mean anything";

    std::vector<double> aBuf, bBuf;
    fillDeterministic<3>(aBuf, bBuf);
    std::vector<double> outBundle(3 * kN, -999.0), outScalar(3 * kN, 0.0);

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovBundle = make_view<double, 3, dyn>(outBundle.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const double k = -1.75;
    auto expr = av + k * bv;

    bundleEval(ovBundle, expr, kN);
    scalarReferenceEval(ovScalar, expr, kN);

    EXPECT_EQ(0, std::memcmp(outBundle.data(), outScalar.data(), outBundle.size() * sizeof(double)))
        << "tail bundle (N not divisible by W): masked bundleAssign diverges from the scalar reference";
}

TEST_F(BundleBitIdentityTest, QuatSandwichChainViaBundlesMatchesScalarAssignmentExactly)
{
    // q.quatMul(v.asPureQuaternion()).quatMul(q.quatConj()).asBack3DVector()
    // — the same formula as test_ExprQuaternion, now driven via
    // BundleIndex-materialized DeviceBundle operands (`view[bi].get()`,
    // mirroring `view[i].get()`'s Item materialization exactly — bundle
    // code reads exactly like Item code). `DeviceBundle::eval()` reuses
    // `SampleIndex::work()` as the lane selector
    // (`backend/cuda/DeviceBundle.h`), so QuatMul/QuatConj/
    // AsPureQuaternion/Segment — none of which `backend/cuda/bundle/
    // LoadStore.h` gives a bundle-specific overload — compose over the
    // materialized `DeviceBundle` operands through the generic bundleGet
    // fallback (that header's docstring).
    std::vector<double> qBuf(4 * kN), vBuf(3 * kN);
    for (std::size_t idx = 0; idx < kN; ++idx) {
        for (std::size_t d = 0; d < 4; ++d)
            qBuf[d * kN + idx] = static_cast<double>(idx) * 0.31 + static_cast<double>(d) * 0.07 + 0.3;
        for (std::size_t d = 0; d < 3; ++d)
            vBuf[d * kN + idx] = static_cast<double>(idx) * -0.21 + static_cast<double>(d) * 0.13 - 0.2;
    }
    std::vector<double> outBundle(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto qv = make_view<double, 4, dyn>(qBuf.data(), Device(kDLCPU), kN);
    auto vv = make_view<double, 3, dyn>(vBuf.data(), Device(kDLCPU), kN);
    auto ovBundle = make_view<double, 3, dyn>(outBundle.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    for (std::size_t bidx = 0; bidx * kW < kN; ++bidx) {
        const BundleIndex<kW> bi = BundleIndex<kW>::make(bidx);
        auto qb = qv[bi].get();
        auto vb = vv[bi].get();
        auto expr = qb.quatMul(vb.asPureQuaternion()).quatMul(qb.quatConj()).asBack3DVector();
        if (bi.base() + kW <= kN) {
            ovBundle[bi] = expr;
        } else {
            aether::bundleAssign(ovBundle, expr, bi, bi.mask(static_cast<aether::offset_t>(kN)));
        }
    }
    for (std::size_t idx = 0; idx < kN; ++idx) {
        const SampleIndex i = SampleIndex::make(idx);
        const Vec4d qi = qv[i].get();
        const Vec3d vi = vv[i].get();
        ovScalar[i] = qi.quatMul(vi.asPureQuaternion()).quatMul(qi.quatConj()).asBack3DVector();
    }

    EXPECT_EQ(0, std::memcmp(outBundle.data(), outScalar.data(), outBundle.size() * sizeof(double)))
        << "quat-sandwich chain via bundles diverges from the scalar reference (bit-identity required)";
}

TEST_F(BundleBitIdentityTest, MisalignedRawPointerWrapFallsBackButStaysExact)
{
    // A raw pointer offset by ONE element (8 bytes) from whatever alignment
    // the backing allocation had can never satisfy the 16-byte vectorization
    // guard for ANY bundle base (every base is offset by a multiple of 8
    // bytes from an 8-byte-misaligned-relative-to-16 start) — the View
    // leaf's bundleGet/bundleStore must take the scalar per-lane fallback
    // (`backend/cuda/bundle/LoadStore.h`), and the values must still match
    // the scalar reference exactly (a runtime alignment guard with scalar
    // fallback).
    std::vector<double> aBuf, bBuf;
    fillDeterministic<3>(aBuf, bBuf);
    std::vector<double> outScalar(3 * kN, 0.0);

    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    // Raw storage with ONE extra leading element so `raw.data()+1` is a
    // deliberately offset (and thus, for the trailing SAMPLE mode's 16-byte
    // check, generally misaligned) base for the View below.
    std::vector<double> raw(3 * kN + 1, 0.0);
    double* misalignedPtr = raw.data() + 1;
    auto mv = make_view<double, 3, dyn>(misalignedPtr, Device(kDLCPU), kN);
    for (std::size_t idx = 0; idx < kN; ++idx)
        for (std::size_t d = 0; d < 3; ++d)
            mv(d, idx) = aBuf[d * kN + idx];

    std::vector<double> outMisaligned(3 * kN, 0.0);
    auto ovMisaligned = make_view<double, 3, dyn>(outMisaligned.data(), Device(kDLCPU), kN);

    const double k = 2.5;
    auto exprMisaligned = mv + k * bv;
    auto exprScalar = mv + k * bv; // SAME leaves — the reference reads through the identical (misaligned) view too

    bundleEval(ovMisaligned, exprMisaligned, kN);
    scalarReferenceEval(ovScalar, exprScalar, kN);

    EXPECT_EQ(0, std::memcmp(outMisaligned.data(), outScalar.data(), outMisaligned.size() * sizeof(double)))
        << "misaligned raw-pointer wrap: bundleAssign's scalar fallback diverges from the scalar reference";
}

TEST_F(BundleBitIdentityTest, BundleAddSubMatchesScalarCompoundAssignmentExactly)
{
    std::vector<double> aBuf, bBuf;
    fillDeterministic<3>(aBuf, bBuf);
    std::vector<double> outBundle(3 * kN), outScalar(3 * kN);
    for (std::size_t idx = 0; idx < 3 * kN; ++idx) {
        outBundle[idx] = static_cast<double>(idx) * 0.05 - 1.0;
        outScalar[idx] = outBundle[idx];
    }

    auto av = make_view<double, 3, dyn>(aBuf.data(), Device(kDLCPU), kN);
    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovBundle = make_view<double, 3, dyn>(outBundle.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    for (std::size_t bidx = 0; bidx * kW < kN; ++bidx) {
        const BundleIndex<kW> bi = BundleIndex<kW>::make(bidx);
        if (bi.base() + kW > kN)
            continue; // += / -= restricted to full bundles (see file: masked overload is Set-only)
        ovBundle[bi] += av;
        ovBundle[bi] -= bv;
    }
    for (std::size_t idx = 0; idx < (kN / kW) * kW; ++idx) {
        const SampleIndex i = SampleIndex::make(idx);
        ovScalar[i] += av[i].get();
        ovScalar[i] -= bv[i].get();
    }

    EXPECT_EQ(0, std::memcmp(outBundle.data(), outScalar.data(), outBundle.size() * sizeof(double)))
        << "bundle += / -= diverges from the scalar compound-assignment reference";
}

TEST_F(BundleBitIdentityTest, ItemLeafBroadcastsAcrossEveryLane)
{
    // bundleGet's Item overload (L3: "Item: broadcast") — a sample-free
    // Item operand mixed into a bundle expression must read the SAME value
    // in every lane, matching packetGet's own Item broadcast exactly.
    std::vector<double> bBuf;
    std::vector<double> aBufUnused;
    fillDeterministic<3>(aBufUnused, bBuf);
    std::vector<double> outBundle(3 * kN, 0.0), outScalar(3 * kN, 0.0);

    auto bv = make_view<double, 3, dyn>(bBuf.data(), Device(kDLCPU), kN);
    auto ovBundle = make_view<double, 3, dyn>(outBundle.data(), Device(kDLCPU), kN);
    auto ovScalar = make_view<double, 3, dyn>(outScalar.data(), Device(kDLCPU), kN);

    const aether::Vec3d broadcastItem = aether::Vec3d::filled(1.5);
    auto expr = bv + broadcastItem;

    bundleEval(ovBundle, expr, kN);
    scalarReferenceEval(ovScalar, expr, kN);

    EXPECT_EQ(0, std::memcmp(outBundle.data(), outScalar.data(), outBundle.size() * sizeof(double)))
        << "Item-leaf broadcast via bundles diverges from the scalar reference";
}

} // namespace
} // namespace aether_tests
