// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file AccumPlaneCommon.h
 * @brief Shared host battery for `aether/accum/AccumPlane.h`, covering the
 *        two `double` policies aether carries (`CompensatedAtomic`,
 *        `Serialized`).
 *
 * What is certified, and against what: the reference is a fixed-point
 * accumulator wide enough to hold every contribution these profiles
 * produce with no rounding at all (`ExactSum`, 768 bits), rounded once to
 * nearest-even at the end. It shares no arithmetic with the subject.
 *
 * Value profiles are enumerated, not sampled: four families (force-model
 * magnitude spread, cancellation, low/high admission corners).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace AccumPlaneTest {

using aether::SampleIndex;
using aether::Vec3d;
namespace acc = aether::accum;

// =========================================================================
//  The exact reference: a wide fixed-point accumulator, pure host arithmetic,
//  with no dependency on the aether types it certifies.
// =========================================================================

/**
 * @brief Exact sum of an arbitrary number of `double`s, rounded ONCE. 768
 *        bits of two's-complement fixed point, binary point at bit 256.
 */
class ExactSum {
public:
    static constexpr int kLimbs    = 12; ///< 12 x 64 = 768 bits
    static constexpr int kFracBits = 256; ///< binary point position

    void clear() { w_.assign(kLimbs, 0ull); }

    ExactSum() { clear(); }

    /// @brief Add `v` EXACTLY. Returns false if `v` falls outside the window.
    bool add(double v)
    {
        if (v == 0.0 || !std::isfinite(v))
            return v == 0.0;
        int e            = 0;
        double m         = std::frexp(v, &e); // v = m * 2^e, 0.5 <= |m| < 1
        double mi         = std::ldexp(m, 53); // exact: |mi| < 2^53, integral
        long long q       = static_cast<long long>(mi);
        const int shift   = e - 53 + kFracBits;
        if (shift < 0 || shift + 54 >= kLimbs * 64)
            return false;
        const bool neg = q < 0;
        unsigned long long mag = neg ? static_cast<unsigned long long>(-q) : static_cast<unsigned long long>(q);
        const int limb = shift / 64;
        const int off   = shift % 64;
        unsigned long long lowPart  = mag << off;
        unsigned long long highPart = off == 0 ? 0ull : (mag >> (64 - off));
        if (neg) {
            sub_(limb, lowPart);
            if (highPart)
                sub_(limb + 1, highPart);
        } else {
            add_(limb, lowPart);
            if (highPart)
                add_(limb + 1, highPart);
        }
        return true;
    }

    /// @brief The exact accumulated value, rounded to nearest-even `double`.
    [[nodiscard]] double toDouble() const
    {
        std::vector<unsigned long long> m = w_;
        const bool neg = (m[kLimbs - 1] >> 63) != 0ull;
        if (neg)
            negate_(m);
        int msb = -1;
        for (int i = kLimbs - 1; i >= 0; --i) {
            if (m[i] != 0ull) {
                for (int b = 63; b >= 0; --b) {
                    if ((m[i] >> b) & 1ull) {
                        msb = i * 64 + b;
                        break;
                    }
                }
                break;
            }
        }
        if (msb < 0)
            return 0.0;
        unsigned long long sig = 0ull;
        for (int k = 0; k < 53; ++k) {
            const int b = msb - 52 + k;
            if (b >= 0 && bit_(m, b))
                sig |= (1ull << k);
        }
        const int guardBit = msb - 53;
        const bool guard    = guardBit >= 0 && bit_(m, guardBit);
        bool sticky          = false;
        for (int b = 0; b < guardBit && !sticky; ++b)
            sticky = bit_(m, b);
        int exp = msb - 52 - kFracBits;
        if (guard && (sticky || (sig & 1ull))) {
            ++sig;
            if (sig == (1ull << 53)) {
                sig >>= 1;
                ++exp;
            }
        }
        const double out = std::ldexp(static_cast<double>(sig), exp);
        return neg ? -out : out;
    }

private:
    static bool bit_(const std::vector<unsigned long long>& m, int b)
    {
        return ((m[static_cast<std::size_t>(b) / 64] >> (static_cast<std::size_t>(b) % 64)) & 1ull) != 0ull;
    }
    static void negate_(std::vector<unsigned long long>& m)
    {
        unsigned long long carry = 1ull;
        for (int i = 0; i < kLimbs; ++i) {
            const unsigned long long inv = ~m[static_cast<std::size_t>(i)];
            const unsigned long long s   = inv + carry;
            carry                        = (s < inv) ? 1ull : 0ull;
            m[static_cast<std::size_t>(i)] = s;
        }
    }
    void add_(int limb, unsigned long long v)
    {
        for (int i = limb; i < kLimbs && v != 0ull; ++i) {
            const unsigned long long before = w_[static_cast<std::size_t>(i)];
            const unsigned long long after  = before + v;
            w_[static_cast<std::size_t>(i)] = after;
            v                                = (after < before) ? 1ull : 0ull;
        }
    }
    void sub_(int limb, unsigned long long v)
    {
        for (int i = limb; i < kLimbs && v != 0ull; ++i) {
            const unsigned long long before = w_[static_cast<std::size_t>(i)];
            const unsigned long long after  = before - v;
            w_[static_cast<std::size_t>(i)] = after;
            v                                = (before < v) ? 1ull : 0ull;
        }
    }

    std::vector<unsigned long long> w_;
};

/// @brief Monotone integer image of a `double`.
inline long long monotone(double v)
{
    long long b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b < 0 ? (static_cast<long long>(0x8000000000000000ull) - b) : b;
}

/// @brief ULP distance, saturating.
inline long long ulpDistance(double got, double want)
{
    if (std::isnan(got) || std::isnan(want))
        return std::numeric_limits<long long>::max();
    long long a = monotone(got), b = monotone(want);
    const unsigned long long ua = static_cast<unsigned long long>(a);
    const unsigned long long ub = static_cast<unsigned long long>(b);
    const unsigned long long d = a > b ? ua - ub : ub - ua;
    return d > static_cast<unsigned long long>(std::numeric_limits<long long>::max())
        ? std::numeric_limits<long long>::max()
        : static_cast<long long>(d);
}

// =========================================================================
//  Value profiles -- enumerated, not sampled
// =========================================================================

enum Profile { P_FM = 0, P_CANC, P_CORN_LO, P_CORN_HI, P_COUNT };

inline const char* profileName(int p)
{
    switch (p) {
    case P_FM: return "FM";
    case P_CANC: return "CANC";
    case P_CORN_LO: return "CORN_LO";
    default: return "CORN_HI";
    }
}

inline const double* fmRelative()
{
    static const double v[8] = { 1.0, 1.1e-3, 3.1e-6, 6.0e-7, 5.0e-8, 1.0e-10, 4.0e-12, 8.0e-13 };
    return v;
}
inline const double* cancResidue()
{
    static const double v[6] = { 0.0, 2.22e-16, 9.1e-13, 9.3e-10, 9.5e-7, 9.8e-4 };
    return v;
}
inline const int* cornLoExp()
{
    static const int v[10] = { -50, -53, -56, -57, -60, -70, -88, -96, -100, -110 };
    return v;
}
inline const int* cornHiExp()
{
    static const int v[5] = { 100, 107, 110, 115, 118 };
    return v;
}

inline constexpr int kSamples = 240;
inline constexpr int kBodies  = 8;
inline constexpr int kLanes   = 3;

inline int termIndex(int k, int d, int i) { return (k * kLanes + d) * kSamples + i; }

/** @brief One profile's native (`double`) contribution table. */
struct TermTable {
    std::vector<double> nat;
    std::vector<double> natOracle;
    std::vector<int> dominantBody;
};

inline TermTable buildProfile(int profile)
{
    TermTable t;
    const int n = kBodies * kLanes * kSamples;
    t.nat.assign(static_cast<std::size_t>(n), 0.0);
    t.natOracle.assign(static_cast<std::size_t>(kLanes * kSamples), 0.0);
    t.dominantBody.assign(static_cast<std::size_t>(kLanes * kSamples), 0);

    for (int k = 0; k < kBodies; ++k) {
        for (int d = 0; d < kLanes; ++d) {
            for (int i = 0; i < kSamples; ++i) {
                const unsigned h = static_cast<unsigned>(i) * 2654435761u + static_cast<unsigned>(d) * 40503u
                    + static_cast<unsigned>(k) * 2246822519u;
                const double jit = 1.0 + 0.4375 * static_cast<double>((h >> 8) & 0xFFFFu) / 65536.0;
                const double sgn = ((h >> 3) & 1u) ? 1.0 : -1.0;

                double vHi = 0.0;
                switch (profile) {
                case P_FM:
                    vHi = 8.0e-3 * fmRelative()[k % 8] * jit * (k == 0 ? 1.0 : sgn);
                    break;
                case P_CANC: {
                    const double A  = 1.0e-2 * jit;
                    const double dl = cancResidue()[i % 6];
                    vHi              = (k % 2 == 0) ? A : -A * (1.0 - dl);
                    break;
                }
                case P_CORN_LO:
                    vHi = (k == 0) ? jit : std::ldexp(jit, cornLoExp()[i % 10] - (k - 1)) * sgn;
                    break;
                default: {
                    const int E = cornHiExp()[i % 5];
                    vHi          = (k == 0) ? std::ldexp(jit, E) : std::ldexp(jit, E - 53 - 3 * (k - 1)) * sgn;
                    break;
                }
                }

                int ex = 0;
                std::frexp(vHi, &ex);
                double vLo = 0.0;
                if (vHi != 0.0 && (ex - 58) > -140)
                    vLo = std::ldexp(((h >> 17) & 1u) ? 1.0 : -1.0, ex - 58);

                const int idx                         = termIndex(k, d, i);
                t.nat[static_cast<std::size_t>(idx)] = vHi + vLo;
            }
        }
    }

    for (int d = 0; d < kLanes; ++d) {
        for (int i = 0; i < kSamples; ++i) {
            ExactSum en;
            double biggest = -1.0;
            int dom         = 0;
            for (int k = 0; k < kBodies; ++k) {
                const std::size_t idx = static_cast<std::size_t>(termIndex(k, d, i));
                en.add(t.nat[idx]);
                if (std::fabs(t.nat[idx]) > biggest) {
                    biggest = std::fabs(t.nat[idx]);
                    dom     = k;
                }
            }
            const std::size_t s   = static_cast<std::size_t>(d * kSamples + i);
            t.natOracle[s]        = en.toDouble();
            t.dominantBody[s]     = dom;
        }
    }
    return t;
}

// =========================================================================
//  Verdict helper
// =========================================================================

struct UlpReport {
    long long worst = 0;
    int worstLane   = -1;
    int worstSample = -1;
};

inline UlpReport scoreLanes(const std::vector<double>& got, const std::vector<double>& want)
{
    UlpReport r;
    for (int d = 0; d < kLanes; ++d) {
        for (int i = 0; i < kSamples; ++i) {
            const std::size_t s = static_cast<std::size_t>(d * kSamples + i);
            const long long u    = ulpDistance(got[s], want[s]);
            if (u > r.worst) {
                r.worst       = u;
                r.worstLane   = d;
                r.worstSample = i;
            }
        }
    }
    return r;
}

// =========================================================================
//  Host arms
// =========================================================================

/// @brief The native contribution for (k, i), as a register vec3.
inline Vec3d natTerm(const TermTable& t, int k, int i)
{
    Vec3d v;
    v.template get<0>() = t.nat[static_cast<std::size_t>(termIndex(k, 0, i))];
    v.template get<1>() = t.nat[static_cast<std::size_t>(termIndex(k, 1, i))];
    v.template get<2>() = t.nat[static_cast<std::size_t>(termIndex(k, 2, i))];
    return v;
}

/**
 * @brief Run one profile through a HOST plane, body by body in ascending
 * order, and return the delivered lanes. `skipBody >= 0` omits that body.
 */
template<class Policy, class TermFn>
inline std::vector<double> runHostPlane(const TermTable& tab, int skipBody, TermFn termOf)
{
    acc::Plane<double, 3, Policy> plane{ static_cast<std::size_t>(kSamples) };
    plane.zeroHost();
    auto ref = plane.hostView();

    for (int k = 0; k < kBodies; ++k) {
        if (k == skipBody)
            continue;
        for (int i = 0; i < kSamples; ++i) {
            const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
            ref[si] += termOf(tab, k, i);
        }
    }

    std::vector<double> out(static_cast<std::size_t>(kLanes * kSamples), 0.0);
    for (int i = 0; i < kSamples; ++i) {
        const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
        const auto v          = ref.delivered(si);
        out[static_cast<std::size_t>(0 * kSamples + i)] = v.template get<0>();
        out[static_cast<std::size_t>(1 * kSamples + i)] = v.template get<1>();
        out[static_cast<std::size_t>(2 * kSamples + i)] = v.template get<2>();
    }
    return out;
}

/// @brief Same, but the DOMINANT body of each (lane, sample) is dropped.
template<class Policy, class TermFn>
inline std::vector<double> runHostPlaneDropDominant(const TermTable& tab, TermFn termOf)
{
    acc::Plane<double, 3, Policy> plane{ static_cast<std::size_t>(kSamples) };
    plane.zeroHost();
    auto ref = plane.hostView();

    for (int k = 0; k < kBodies; ++k) {
        for (int i = 0; i < kSamples; ++i) {
            const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
            auto term              = termOf(tab, k, i);
            if (tab.dominantBody[static_cast<std::size_t>(0 * kSamples + i)] == k)
                term.template get<0>() = 0.0;
            if (tab.dominantBody[static_cast<std::size_t>(1 * kSamples + i)] == k)
                term.template get<1>() = 0.0;
            if (tab.dominantBody[static_cast<std::size_t>(2 * kSamples + i)] == k)
                term.template get<2>() = 0.0;
            ref[si] += term;
        }
    }

    std::vector<double> out(static_cast<std::size_t>(kLanes * kSamples), 0.0);
    for (int i = 0; i < kSamples; ++i) {
        const SampleIndex si = SampleIndex::make(static_cast<std::size_t>(i));
        const auto v          = ref.delivered(si);
        out[static_cast<std::size_t>(0 * kSamples + i)] = v.template get<0>();
        out[static_cast<std::size_t>(1 * kSamples + i)] = v.template get<1>();
        out[static_cast<std::size_t>(2 * kSamples + i)] = v.template get<2>();
    }
    return out;
}

/// @brief Worst RELATIVE magnitude error, for the non-vacuity arm.
inline double worstRelativeError(const std::vector<double>& got, const std::vector<double>& want)
{
    double worst = 0.0;
    for (std::size_t s = 0; s < want.size(); ++s) {
        const double den = std::fabs(want[s]);
        const double num = std::fabs(got[s] - want[s]);
        const double rel  = den > 0.0 ? num / den : (num > 0.0 ? 1.0 : 0.0);
        worst              = std::max(worst, rel);
    }
    return worst;
}

class AccumPlane : public ::testing::Test { };

// =========================================================================
//  Rows: the reference's own known answers
// =========================================================================

TEST_F(AccumPlane, ExactSumOracleKnownAnswers)
{
    {
        ExactSum e;
        EXPECT_DOUBLE_EQ(e.toDouble(), 0.0);
    }
    {
        ExactSum e;
        e.add(1.0);
        e.add(std::ldexp(1.0, -60));
        e.add(-std::ldexp(1.0, -60));
        EXPECT_EQ(e.toDouble(), 1.0) << "exact cancellation must be exact";
    }
    {
        ExactSum e;
        e.add(1.0);
        e.add(std::ldexp(1.0, -53));
        EXPECT_EQ(e.toDouble(), 1.0) << "tie must round to even";
    }
    {
        ExactSum e;
        e.add(1.0);
        e.add(std::ldexp(1.0, -53));
        e.add(std::ldexp(1.0, -200));
        EXPECT_EQ(e.toDouble(), std::nextafter(1.0, 2.0)) << "sticky bit must break the tie";
    }
    {
        ExactSum e;
        e.add(1.0);
        e.add(std::ldexp(1.0, -52));
        e.add(std::ldexp(1.0, -53));
        EXPECT_EQ(e.toDouble(), 1.0 + std::ldexp(1.0, -51)) << "tie must round to even, upward here";
    }
    {
        ExactSum e;
        e.add(std::ldexp(1.0, 118));
        e.add(std::ldexp(-1.0, 118));
        e.add(std::ldexp(1.0, -168));
        EXPECT_EQ(e.toDouble(), std::ldexp(1.0, -168)) << "170 binades of span must survive cancellation";
    }
    {
        ExactSum e;
        EXPECT_FALSE(e.add(std::ldexp(1.0, 600)));
        EXPECT_FALSE(e.add(std::ldexp(1.0, -300)));
    }
}

// =========================================================================
//  Rows: policy traits and defaults
//
//  Only the double-policy static_asserts apply here: aether has no
//  InCarrier/SoftDouble accumulation plane.
// =========================================================================

TEST_F(AccumPlane, TraitsDeclarePolicyConcurrency)
{
    static_assert(acc::PolicyTraits<acc::CompensatedAtomic>::concurrentSafe,
        "the compensated atomic policy is the one that keeps producer concurrency -- that is what it is for");
    static_assert(!acc::PolicyTraits<acc::Serialized>::concurrentSafe);

    static_assert(acc::PolicyTraits<acc::CompensatedAtomic>::hasCompanionLane);
    static_assert(!acc::PolicyTraits<acc::Serialized>::hasCompanionLane);

    static_assert(acc::Plane<double, 3>::concurrentSafe);
    static_assert(acc::PlaneView<double, 3>::concurrentSafe);
    static_assert(!acc::Plane<double, 3, acc::Serialized>::concurrentSafe);

    EXPECT_TRUE(acc::PolicyTraits<acc::CompensatedAtomic>::concurrentSafe);
    EXPECT_FALSE(acc::PolicyTraits<acc::Serialized>::concurrentSafe);
}

TEST_F(AccumPlane, DefaultPolicyPerScalar)
{
    static_assert(std::is_same_v<acc::DefaultPolicyT<double>, acc::CompensatedAtomic>,
        "the native default is the ACCURATE and REPRODUCIBLE one; a caller who wants the serialized trade must "
        "name it");
    static_assert(std::is_same_v<typename acc::Plane<double, 3>::StoreT, double>);
    SUCCEED();
}

TEST_F(AccumPlane, ZeroIsAllZeroBytesForEveryPolicy)
{
    EXPECT_TRUE((acc::Plane<double, 3, acc::CompensatedAtomic>::zeroIsAllZeroBytes()));
    EXPECT_TRUE((acc::Plane<double, 3, acc::Serialized>::zeroIsAllZeroBytes()));
}

// =========================================================================
//  Rows: host exactness on every enumerated profile
// =========================================================================

TEST_F(AccumPlane, HostCompensatedAtomicIsExactOnEveryProfile)
{
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        const auto got     = runHostPlane<acc::CompensatedAtomic>(t, -1, natTerm);
        const UlpReport r  = scoreLanes(got, t.natOracle);
        EXPECT_EQ(r.worst, 0) << "profile " << profileName(p) << " lane " << r.worstLane << " sample "
                              << r.worstSample;
    }
}

TEST_F(AccumPlane, HostSerializedIsThePlainRunningSum)
{
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        const auto got     = runHostPlane<acc::Serialized>(t, -1, natTerm);

        for (int d = 0; d < kLanes; ++d) {
            for (int i = 0; i < kSamples; ++i) {
                double naive = 0.0;
                for (int k = 0; k < kBodies; ++k)
                    naive += t.nat[static_cast<std::size_t>(termIndex(k, d, i))];
                ASSERT_EQ(got[static_cast<std::size_t>(d * kSamples + i)], naive)
                    << "profile " << profileName(p) << " lane " << d << " sample " << i;
            }
        }

        const UlpReport r = scoreLanes(got, t.natOracle);
        RecordProperty(
            std::string("serialized_worst_ulp_") + profileName(p), static_cast<int>(std::min<long long>(r.worst, 1 << 30)));
    }
}

TEST_F(AccumPlane, HostDroppingTheDominantTermIsFarFromTheOracle)
{
    /* Non-vacuity check on the CompensatedAtomic leg. */
    for (int p = 0; p < P_COUNT; ++p) {
        const TermTable t = buildProfile(p);
        const auto natGot  = runHostPlaneDropDominant<acc::CompensatedAtomic>(t, natTerm);
        EXPECT_GT(worstRelativeError(natGot, t.natOracle), 1e-3) << "nat, profile " << profileName(p);
        EXPECT_GT(scoreLanes(natGot, t.natOracle).worst, 1000) << "nat, profile " << profileName(p);
    }
}

// =========================================================================
//  Rows: the zero-once-per-step ruling and its observable
// =========================================================================

TEST_F(AccumPlane, HostZeroLeavesEveryLaneAtZero)
{
    /* Only the nat/ser (double) planes apply: aether has no emu plane. */
    acc::Plane<double, 3> nat{ 16 };
    acc::Plane<double, 3, acc::Serialized> ser{ 16 };
    nat.zeroHost();
    ser.zeroHost();

    auto nr = nat.hostView();
    auto sr = ser.hostView();
    for (std::size_t i = 0; i < 16; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        EXPECT_EQ(nr.delivered(si).template get<1>(), 0.0);
        EXPECT_EQ(sr.delivered(si).template get<2>(), 0.0);
    }
}

TEST_F(AccumPlane, HostPerSampleZeroClearsTheCompanionLane)
{
    /* `PlaneView::zero(i)` -- the in-kernel spelling -- must clear the
     * COMPANION lane too. */
    constexpr std::size_t n = 16;
    acc::Plane<double, 3, acc::CompensatedAtomic> nat{ n };
    nat.zeroHost();
    auto r = nat.hostView();

    Vec3d big, small;
    big.template get<0>()   = 1.0;
    big.template get<1>()   = -3.0;
    big.template get<2>()   = 7.0;
    small.template get<0>() = std::ldexp(1.0, -70);
    small.template get<1>() = std::ldexp(-1.0, -70);
    small.template get<2>() = std::ldexp(1.0, -70);
    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        r[si] += big;
        r[si] += small; // guarantees a NON-ZERO residue in the companion lane
    }
    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        ASSERT_NE(r.compView()(0, si.global()), 0.0) << "sample " << i;
    }

    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        r.zero(si);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        EXPECT_EQ(r.delivered(si).template get<0>(), 0.0) << "sample " << i;
        EXPECT_EQ(r.delivered(si).template get<1>(), 0.0) << "sample " << i;
        EXPECT_EQ(r.delivered(si).template get<2>(), 0.0) << "sample " << i;
        EXPECT_EQ(r.compView()(0, si.global()), 0.0) << "sample " << i;
        EXPECT_EQ(r.compView()(1, si.global()), 0.0) << "sample " << i;
        EXPECT_EQ(r.compView()(2, si.global()), 0.0) << "sample " << i;
    }
}

TEST_F(AccumPlane, HostInactiveProducerCannotSeeThePreviousStep)
{
    /* Only the nat (double) leg applies: aether has no emu plane. */
    constexpr std::size_t n = 32;
    acc::Plane<double, 3> nat{ n };
    nat.zeroHost();
    auto nr = nat.hostView();

    Vec3d dPrev;
    dPrev.template get<0>() = -7.25;
    dPrev.template get<1>() = 1e9;
    dPrev.template get<2>() = 3.5;

    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        nr[si] += dPrev;
    }

    /* --- the step boundary: ONE unconditional zero, no owner --- */
    nat.zeroHost();

    Vec3d dNow;
    dNow.template get<0>() = 2.0;
    dNow.template get<1>() = 4.0;
    dNow.template get<2>() = 8.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (i % 2 == 1)
            continue; // the producer early-returns on odd samples
        const SampleIndex si = SampleIndex::make(i);
        nr[si] += dNow;
    }

    for (std::size_t i = 0; i < n; ++i) {
        const SampleIndex si = SampleIndex::make(i);
        const double want     = (i % 2 == 0) ? 2.0 : 0.0;
        EXPECT_EQ(nr.delivered(si).template get<0>(), want) << "nat sample " << i;
    }
}

TEST_F(AccumPlane, HostAccumulateAddsRatherThanStores)
{
    /* `+=` must be an ADD, on every policy (nat/ser here: aether has no
     * emu plane). */
    acc::Plane<double, 3> nat{ 4 };
    acc::Plane<double, 3, acc::Serialized> ser{ 4 };
    nat.zeroHost();
    ser.zeroHost();
    auto nr = nat.hostView();
    auto sr = ser.hostView();

    Vec3d d;
    d.template get<0>() = 1.5;
    d.template get<1>() = -0.25;
    d.template get<2>() = 0.0;

    const SampleIndex si = SampleIndex::make(std::size_t{ 0 });
    for (int rep = 0; rep < 3; ++rep) {
        nr[si] += d;
        sr.accumulate(si, d); // the named spelling must be the same operation
    }
    EXPECT_EQ(nr.delivered(si).template get<0>(), 4.5);
    EXPECT_EQ(sr.delivered(si).template get<0>(), 4.5);
}

} // namespace AccumPlaneTest
} // namespace aether_tests
