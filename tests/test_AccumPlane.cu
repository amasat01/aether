// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_AccumPlane.cu
 * @brief Accumulation plane (CUDA mode): the shared host battery
 *        (AccumPlaneCommon.h) plus the DEVICE arms, covering the two
 *        `double` policies aether carries. Rows about an `InCarrier`/
 *        `SoftDouble` plane have no subject here (aether carries no such
 *        plane) and are not reproduced as vacuous passes; rows that would
 *        have mixed a `nat` and an `emu` leg keep their name and the `nat`
 *        leg, documented at each.
 *
 *  1. **The same arithmetic, through global memory and atomics.** The host
 *     mirror sums in one thread; the device sums through a hardware atomic
 *     whose arrival order it does not choose. Same profiles, same exact
 *     reference: the compensated-atomic plane must still deliver the
 *     correctly-rounded sum.
 *
 *  2. **Reproducibility under a schedule that actually races.** THE
 *     MANUFACTURED ADVERSARIAL SCHEDULE places one sample's contributions
 *     in adjacent (hence co-resident) blocks, so their arrival order is
 *     genuinely raced. The plain-atomic control is present and REQUIRED TO
 *     FAIL — a hammer whose control passes has certified nothing.
 */

#include "AccumPlaneCommon.h"

#include <cstdint>
#include <set>
#include <vector>

namespace aether_tests {
namespace AccumPlaneTest {

using CmpPlane = acc::Plane<double, 3, acc::CompensatedAtomic>;
using SerPlane = acc::Plane<double, 3, acc::Serialized>;
using CmpView  = CmpPlane::ViewT;
using SerView  = SerPlane::ViewT;
using Vec3dArrView = aether::Array<double, 3>::ViewT;

constexpr int kBlock = 256;

// =========================================================================
//  Device kernels
// =========================================================================

template<class RefT>
__global__ void planeZeroKernel(RefT p)
{
    const aether::offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.size())
        return;
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    p.zero(si);
}

/// @brief One body's native contribution, body-major (the natural layout).
/// `skipParity >= 0` makes the producer EARLY-RETURN on samples of that parity.
template<class RefT>
__global__ void planeAddNatKernel(RefT p, Vec3dArrView terms, int k, int n, int skipParity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (skipParity >= 0 && (i % 2 == skipParity))
        return;
    const SampleIndex ti = SampleIndex::make(static_cast<std::size_t>(k * n + i));
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    Vec3d t;
    t.template get<0>() = terms(0, ti.global());
    t.template get<1>() = terms(1, ti.global());
    t.template get<2>() = terms(2, ti.global());
    p[si] += t;
}

template<class RefT, class OutViewT>
__global__ void planeDeliverKernel(RefT p, OutViewT out)
{
    const aether::offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.size())
        return;
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    p.deliver(out, si);
}

/**
 * @brief THE MANUFACTURED ADVERSARIAL SCHEDULE. One thread per (body,
 * sample), fused: the `nb` contributions to one sample are placed in `nb`
 * ADJACENT blocks (co-resident), so their arrival order at the shared
 * address is genuinely raced.
 */
template<class RefT>
__global__ void planeFusedAdversarialKernel(RefT p, Vec3dArrView terms, int n, int nb)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nb)
        return;
    const int b    = t / kBlock;
    const int lane = t % kBlock;
    const int g    = b / nb;
    const int k    = b - g * nb;
    const int i    = g * kBlock + lane;
    if (i >= n)
        return;
    const SampleIndex ti = SampleIndex::make(static_cast<std::size_t>(k * n + i));
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    Vec3d v;
    v.template get<0>() = terms(0, ti.global());
    v.template get<1>() = terms(1, ti.global());
    v.template get<2>() = terms(2, ti.global());
    p[si] += v;
}

/**
 * @brief The KNOWN-NONDETERMINISTIC CONTROL: a plain `atomicAdd` plane
 * under the identical schedule. Not a candidate policy -- it exists so the
 * hammer can be shown to have failure power.
 */
__global__ void plainAtomicAdversarialKernel(Vec3dArrView accArr, Vec3dArrView terms, int n, int nb)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nb)
        return;
    const int b    = t / kBlock;
    const int lane = t % kBlock;
    const int g    = b / nb;
    const int k    = b - g * nb;
    const int i    = g * kBlock + lane;
    if (i >= n)
        return;
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    const SampleIndex ti = SampleIndex::make(static_cast<std::size_t>(k * n + i));
    atomicAdd(&accArr(0, si.global()), terms(0, ti.global()));
    atomicAdd(&accArr(1, si.global()), terms(1, ti.global()));
    atomicAdd(&accArr(2, si.global()), terms(2, ti.global()));
}

/// @brief Serial (per-body-launch) plain sum -- the order-dependence control's non-racing twin.
__global__ void plainSerialKernel(Vec3dArrView accArr, Vec3dArrView terms, int k, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
    const SampleIndex ti = SampleIndex::make(static_cast<std::size_t>(k * n + i));
    accArr(0, si.global()) = accArr(0, si.global()) + terms(0, ti.global());
    accArr(1, si.global()) = accArr(1, si.global()) + terms(1, ti.global());
    accArr(2, si.global()) = accArr(2, si.global()) + terms(2, ti.global());
}

// =========================================================================
//  Host-side scaffolding for the device arms
// =========================================================================

namespace {

constexpr int grid(int n) { return (n + kBlock - 1) / kBlock; }

/// @brief Upload one profile's native contributions, laid out (k, i).
inline void uploadNat(const TermTable& t, aether::Array<double, 3>& out)
{
    auto h = out.hostView();
    for (int k = 0; k < kBodies; ++k) {
        for (int i = 0; i < kSamples; ++i) {
            const std::size_t s = static_cast<std::size_t>(k * kSamples + i);
            h(0, s) = t.nat[static_cast<std::size_t>(termIndex(k, 0, i))];
            h(1, s) = t.nat[static_cast<std::size_t>(termIndex(k, 1, i))];
            h(2, s) = t.nat[static_cast<std::size_t>(termIndex(k, 2, i))];
        }
    }
    out.upload();
}

inline std::vector<double> downloadNat(aether::Array<double, 3>& out, int n)
{
    out.download();
    cudaDeviceSynchronize();
    auto h = out.hostView();
    std::vector<double> v(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(0 * n + i)] = h(0, static_cast<std::size_t>(i));
        v[static_cast<std::size_t>(1 * n + i)] = h(1, static_cast<std::size_t>(i));
        v[static_cast<std::size_t>(2 * n + i)] = h(2, static_cast<std::size_t>(i));
    }
    return v;
}

inline std::uint64_t fnv1a(const std::vector<double>& v)
{
    std::uint64_t h      = 1469598103934665603ull;
    const auto* p          = reinterpret_cast<const unsigned char*>(v.data());
    const std::size_t n = v.size() * sizeof(double);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// @brief Native device run of one profile: zero once, one launch per body
/// in ascending order, then the delivery terminal.
template<class PlaneT>
inline std::vector<double> runDeviceNat(aether::Array<double, 3>& terms)
{
    PlaneT plane{ static_cast<std::size_t>(kSamples) };
    aether::Array<double, 3> out{ static_cast<std::size_t>(kSamples) };
    out.upload();
    plane.upload();
    plane.zeroDeviceAsync();
    for (int k = 0; k < kBodies; ++k) {
        planeAddNatKernel<typename PlaneT::ViewT>
            <<<grid(kSamples), kBlock>>>(plane.deviceView(), terms.deviceView(), k, kSamples, -1);
    }
    planeDeliverKernel<typename PlaneT::ViewT, Vec3dArrView>
        <<<grid(kSamples), kBlock>>>(plane.deviceView(), out.deviceView());
    return downloadNat(out, kSamples);
}

} // namespace

// =========================================================================
//  Device rows: exactness on every enumerated profile
// =========================================================================

TEST_F(AccumPlane, DeviceCompensatedAtomicIsExactOnEveryProfile)
{
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadNat(t, terms);
        const auto got    = runDeviceNat<CmpPlane>(terms);
        const UlpReport r = scoreLanes(got, t.natOracle);
        EXPECT_EQ(r.worst, 0) << "profile " << profileName(p) << " lane " << r.worstLane << " sample "
                              << r.worstSample;
    }
}

TEST_F(AccumPlane, DeviceSerializedMatchesTheHostMirror)
{
    /* The serialized policy makes no exactness claim; what it DOES claim is
     * that the host mirror and the device arm perform the same additions
     * in the same order, so their answers are bit-identical. */
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadNat(t, terms);
        const auto dev = runDeviceNat<SerPlane>(terms);
        const auto hst = runHostPlane<acc::Serialized>(t, -1, natTerm);
        for (std::size_t s = 0; s < dev.size(); ++s)
            ASSERT_EQ(dev[s], hst[s]) << "profile " << profileName(p) << " slot " << s;
    }
}

TEST_F(AccumPlane, DeviceDroppingTheDominantTermIsFarFromTheOracle)
{
    /* Non-vacuity check on the device rows' nat leg (aether has no emu
     * plane). */
    for (int p = 0; p < P_COUNT; ++p) {
        TermTable t = buildProfile(p);
        TermTable m = t;
        for (int d = 0; d < kLanes; ++d) {
            for (int i = 0; i < kSamples; ++i) {
                const int k = m.dominantBody[static_cast<std::size_t>(d * kSamples + i)];
                m.nat[static_cast<std::size_t>(termIndex(k, d, i))] = 0.0;
            }
        }
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadNat(m, terms);
        const auto natGot = runDeviceNat<CmpPlane>(terms);
        EXPECT_GT(worstRelativeError(natGot, t.natOracle), 1e-3) << "nat, profile " << profileName(p);
    }
}

// =========================================================================
//  Device row: the zero-once-per-step ruling
// =========================================================================

TEST_F(AccumPlane, DevicePerSampleZeroClearsTheCompanionLane)
{
    /* The device twin of the host row of the same name: the in-kernel
     * `zero(i)` must clear the companion residue lane, not just the value
     * lane. */
    const TermTable t = buildProfile(P_FM);
    aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
    uploadNat(t, terms);

    CmpPlane plane{ static_cast<std::size_t>(kSamples) };
    aether::Array<double, 3> out{ static_cast<std::size_t>(kSamples) };
    out.upload();
    plane.upload();
    plane.zeroDeviceAsync();
    for (int k = 0; k < kBodies; ++k)
        planeAddNatKernel<CmpView><<<grid(kSamples), kBlock>>>(plane.deviceView(), terms.deviceView(), k, kSamples, -1);
    planeDeliverKernel<CmpView, Vec3dArrView><<<grid(kSamples), kBlock>>>(plane.deviceView(), out.deviceView());
    const auto loaded = downloadNat(out, kSamples);
    bool anyNonZero    = false;
    for (double v : loaded)
        anyNonZero = anyNonZero || (v != 0.0);
    ASSERT_TRUE(anyNonZero) << "the plane must be carrying a step, or this row cannot see the defect it exists for";
    {
        plane.compArray().download();
        cudaDeviceSynchronize();
        auto ch     = plane.compArray().hostView();
        bool compNz = false;
        for (int i = 0; i < kSamples && !compNz; ++i)
            compNz = ch(0, static_cast<std::size_t>(i)) != 0.0;
        ASSERT_TRUE(compNz) << "the residue lane is empty; this row would pass even with the companion zero deleted";
    }

    planeZeroKernel<CmpView><<<grid(kSamples), kBlock>>>(plane.deviceView());
    planeDeliverKernel<CmpView, Vec3dArrView><<<grid(kSamples), kBlock>>>(plane.deviceView(), out.deviceView());
    const auto cleared = downloadNat(out, kSamples);
    for (std::size_t s = 0; s < cleared.size(); ++s)
        ASSERT_EQ(cleared[s], 0.0) << "slot " << s;
    plane.compArray().download();
    cudaDeviceSynchronize();
    auto ch2 = plane.compArray().hostView();
    for (int i = 0; i < kSamples; ++i) {
        ASSERT_EQ(ch2(0, static_cast<std::size_t>(i)), 0.0) << "residue lane, sample " << i;
        ASSERT_EQ(ch2(1, static_cast<std::size_t>(i)), 0.0) << "residue lane, sample " << i;
        ASSERT_EQ(ch2(2, static_cast<std::size_t>(i)), 0.0) << "residue lane, sample " << i;
    }
}

TEST_F(AccumPlane, DeviceInactiveProducerCannotSeeThePreviousStep)
{
    /* The observable on the device's nat leg (aether has no emu plane). */
    const TermTable t = buildProfile(P_FM);
    aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
    uploadNat(t, terms);

    CmpPlane nat{ static_cast<std::size_t>(kSamples) };
    aether::Array<double, 3> natOut{ static_cast<std::size_t>(kSamples) };
    natOut.upload();
    nat.upload();

    /* --- step 1: every producer active --- */
    nat.zeroDeviceAsync();
    for (int k = 0; k < kBodies; ++k)
        planeAddNatKernel<CmpView><<<grid(kSamples), kBlock>>>(nat.deviceView(), terms.deviceView(), k, kSamples, -1);

    /* --- step 2: ONE unconditional zero, then a producer that skips odds --- */
    nat.zeroDeviceAsync();
    for (int k = 0; k < kBodies; ++k)
        planeAddNatKernel<CmpView><<<grid(kSamples), kBlock>>>(nat.deviceView(), terms.deviceView(), k, kSamples, 1);
    planeDeliverKernel<CmpView, Vec3dArrView><<<grid(kSamples), kBlock>>>(nat.deviceView(), natOut.deviceView());

    const auto natGot = downloadNat(natOut, kSamples);
    for (int d = 0; d < kLanes; ++d) {
        for (int i = 0; i < kSamples; ++i) {
            const std::size_t s = static_cast<std::size_t>(d * kSamples + i);
            if (i % 2 == 1) {
                ASSERT_EQ(natGot[s], 0.0) << "nat skipped sample " << i;
            } else {
                ASSERT_EQ(natGot[s], t.natOracle[s]) << "nat sample " << i;
            }
        }
    }
}

// =========================================================================
//  Device rows: the determinism hammer
// =========================================================================

namespace {

constexpr int kHammerSamples = 10240;
constexpr int kHammerRuns    = 16;

/// @brief The cancellation profile at hammer size.
inline void buildHammerTerms(aether::Array<double, 3>& terms, std::vector<double>& ora)
{
    auto h = terms.hostView();
    std::vector<std::vector<double>> raw(static_cast<std::size_t>(3 * kHammerSamples));
    for (int k = 0; k < kBodies; ++k) {
        for (int d = 0; d < kLanes; ++d) {
            for (int i = 0; i < kHammerSamples; ++i) {
                const unsigned hh = static_cast<unsigned>(i) * 2654435761u + static_cast<unsigned>(d) * 40503u
                    + static_cast<unsigned>(k) * 2246822519u;
                const double jit = 1.0 + 0.4375 * static_cast<double>((hh >> 8) & 0xFFFFu) / 65536.0;
                const double A   = 1.0e-2 * jit;
                const double dl  = cancResidue()[static_cast<std::size_t>(i % 6)];
                const double v   = (k % 2 == 0) ? A : -A * (1.0 - dl);
                const std::size_t s = static_cast<std::size_t>(k * kHammerSamples + i);
                if (d == 0)
                    h(0, s) = v;
                else if (d == 1)
                    h(1, s) = v;
                else
                    h(2, s) = v;
                raw[static_cast<std::size_t>(d * kHammerSamples + i)].push_back(v);
            }
        }
    }
    terms.upload();
    ora.assign(static_cast<std::size_t>(3 * kHammerSamples), 0.0);
    for (std::size_t s = 0; s < raw.size(); ++s) {
        ExactSum e;
        for (double v : raw[s])
            e.add(v);
        ora[s] = e.toDouble();
    }
}

} // namespace

TEST_F(AccumPlane, CompensatedAtomicIsBitStableUnderTheAdversarialSchedule)
{
    aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kHammerSamples) };
    std::vector<double> oracle;
    buildHammerTerms(terms, oracle);

    const int threads = kHammerSamples * kBodies;
    std::set<std::uint64_t> hashes;
    std::vector<double> last;
    for (int run = 0; run < kHammerRuns; ++run) {
        CmpPlane plane{ static_cast<std::size_t>(kHammerSamples) };
        aether::Array<double, 3> out{ static_cast<std::size_t>(kHammerSamples) };
        out.upload();
        plane.upload();
        plane.zeroDeviceAsync();
        planeFusedAdversarialKernel<CmpView>
            <<<grid(threads), kBlock>>>(plane.deviceView(), terms.deviceView(), kHammerSamples, kBodies);
        planeDeliverKernel<CmpView, Vec3dArrView>
            <<<grid(kHammerSamples), kBlock>>>(plane.deviceView(), out.deviceView());
        last = downloadNat(out, kHammerSamples);
        hashes.insert(fnv1a(last));
    }
    RecordProperty("compensated_distinct_hashes", static_cast<int>(hashes.size()));
    RecordProperty("hammer_runs", kHammerRuns);
    RecordProperty("hammer_samples", kHammerSamples);
    EXPECT_EQ(hashes.size(), 1u)
        << "the compensated atomic plane must be replay-stable under the manufactured adversarial schedule";
    const UlpReport r = scoreLanes(last, oracle);
    RecordProperty("compensated_worst_ulp", static_cast<int>(r.worst));
    EXPECT_EQ(r.worst, 0) << "lane " << r.worstLane << " sample " << r.worstSample;
}

TEST_F(AccumPlane, PlainAtomicControlFailsTheAdversarialSchedule)
{
    /* THE HAMMER'S OWN NON-VACUITY. This arm is NOT a candidate policy; it
     * is the known-bad control. It must come back nondeterministic AND
     * inaccurate under the manufactured schedule. */
    aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kHammerSamples) };
    std::vector<double> oracle;
    buildHammerTerms(terms, oracle);

    const int threads = kHammerSamples * kBodies;
    std::set<std::uint64_t> hashes;
    long long worstUlp = 0;
    for (int run = 0; run < kHammerRuns; ++run) {
        aether::Array<double, 3> accArr{ static_cast<std::size_t>(kHammerSamples) };
        accArr.upload();
        cudaMemset(accArr.deviceView().data(), 0,
            static_cast<std::size_t>(accArr.capacity()) * 3 * sizeof(double));
        plainAtomicAdversarialKernel<<<grid(threads), kBlock>>>(
            accArr.deviceView(), terms.deviceView(), kHammerSamples, kBodies);
        const auto got = downloadNat(accArr, kHammerSamples);
        hashes.insert(fnv1a(got));
        worstUlp = std::max(worstUlp, scoreLanes(got, oracle).worst);
    }
    RecordProperty("plain_atomic_distinct_hashes", static_cast<int>(hashes.size()));
    RecordProperty("plain_atomic_worst_ulp", static_cast<int>(std::min<long long>(worstUlp, 1 << 30)));
    EXPECT_GT(hashes.size(), 1u) << "the plain-atomic control came back replay-stable over " << kHammerRuns
                                 << " runs: the manufactured schedule is not racing on this device, so the "
                                    "compensated arm's stability certifies NOTHING";
    EXPECT_GT(worstUlp, 0) << "the plain-atomic control was exact: the value profile is not sensitive enough to "
                              "separate the policies";
}

TEST_F(AccumPlane, CompensatedAtomicIsOrderIndependent)
{
    /* The deterministic backbone of the hammer: the same contributions,
     * delivered in ASCENDING and then DESCENDING body order, must produce
     * bit-identical output on EVERY profile. */
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadNat(t, terms);

        std::vector<double> res[2];
        for (int dir = 0; dir < 2; ++dir) {
            CmpPlane plane{ static_cast<std::size_t>(kSamples) };
            aether::Array<double, 3> out{ static_cast<std::size_t>(kSamples) };
            out.upload();
            plane.upload();
            plane.zeroDeviceAsync();
            for (int j = 0; j < kBodies; ++j) {
                const int k = (dir == 0) ? j : (kBodies - 1 - j);
                planeAddNatKernel<CmpView>
                    <<<grid(kSamples), kBlock>>>(plane.deviceView(), terms.deviceView(), k, kSamples, -1);
            }
            planeDeliverKernel<CmpView, Vec3dArrView><<<grid(kSamples), kBlock>>>(plane.deviceView(), out.deviceView());
            res[dir] = downloadNat(out, kSamples);
        }
        for (std::size_t s = 0; s < res[0].size(); ++s)
            ASSERT_EQ(res[0][s], res[1][s]) << "profile " << profileName(p) << " slot " << s;
    }
}

TEST_F(AccumPlane, PlainSummationControlIsOrderDependent)
{
    /* The control for the row above: the IDENTICAL experiment on a plain
     * FP64 running sum must produce a DIFFERENT answer when the body order
     * is reversed. Required on the FORCE-MODEL profile, reported on the rest. */
    bool differsAnywhere = false;
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        aether::Array<double, 3> terms{ static_cast<std::size_t>(kBodies * kSamples) };
        uploadNat(t, terms);

        std::vector<double> res[2];
        for (int dir = 0; dir < 2; ++dir) {
            aether::Array<double, 3> accArr{ static_cast<std::size_t>(kSamples) };
            accArr.upload();
            cudaMemset(accArr.deviceView().data(), 0,
                static_cast<std::size_t>(accArr.capacity()) * 3 * sizeof(double));
            for (int j = 0; j < kBodies; ++j) {
                const int k = (dir == 0) ? j : (kBodies - 1 - j);
                plainSerialKernel<<<grid(kSamples), kBlock>>>(accArr.deviceView(), terms.deviceView(), k, kSamples);
            }
            res[dir] = downloadNat(accArr, kSamples);
        }
        bool differs = false;
        for (std::size_t s = 0; s < res[0].size() && !differs; ++s)
            differs = (res[0][s] != res[1][s]);
        differsAnywhere = differsAnywhere || differs;
        if (p == P_FM) {
            EXPECT_TRUE(differs) << "a plain running sum must be order-dependent on the force-model magnitude "
                                    "spread; if it is not, the order-independence row above is blind";
        }
        RecordProperty(std::string("plain_order_differs_") + profileName(p), differs ? 1 : 0);
    }
    EXPECT_TRUE(differsAnywhere);
}

} // namespace AccumPlaneTest
} // namespace aether_tests
