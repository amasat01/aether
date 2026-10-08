// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Mini-RK smoke (host / AETHER_CPP_MODE build). Paired with test_MiniRK.cu
// (identical scalar-arm content; the CUDA file additionally adds a
// DeviceBundle<double,2,6> arm, gated bit-exact against the scalar arm —
// house convention, see tests/test_Bundle.cpp/.cu).
//
// A fixed-step RK4 propagator over extents<6,dyn> (a batched state: pos =
// elements 0..2, vel = elements 3..5, per sample), simple central-force
// RHS (d(pos)/dt = vel, d(vel)/dt = -mu*pos/|pos|^3), written PURELY in
// aether idiom: `View<double, extents<6,dyn>>` for the batched storage,
// `Item<double,6>`/`Item<double,3>` as the register-resident per-sample
// state (using the N-scalar constructor), the ET machinery (`+`, `s*e`,
// `.rCubedNorm()`) for the RK4 stage combination, `SampleIndex` for
// per-sample addressing. Gated against a PLAIN-DOUBLE hand-rolled RK4
// reference computed via a completely independent code path (no aether
// types anywhere in it) — abs+rel 1e-12, the value gate this smoke test
// exists for.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Chunk;
using aether::Device;
using aether::Item;
using aether::SampleIndex;

class MiniRKTest : public ::testing::Test { };

constexpr std::size_t kN     = 16;
constexpr std::size_t kSteps = 20;
constexpr double kH          = 0.01;
constexpr double kMu         = 1.0;

// --- aether-idiom RHS + RK4 step -------------------------------------------

AETHER_DEVICEHOST() inline Item<double, 6> centralForceRhs(const Item<double, 6>& s, double mu)
{
    const Item<double, 3> pos{ s.get<0>(), s.get<1>(), s.get<2>() };
    const Item<double, 3> vel{ s.get<3>(), s.get<4>(), s.get<5>() };
    const double invR3        = pos.rCubedNorm();
    const Item<double, 3> acc = (-mu * invR3) * pos;
    return Item<double, 6>{ vel.get<0>(), vel.get<1>(), vel.get<2>(), acc.get<0>(), acc.get<1>(), acc.get<2>() };
}

AETHER_DEVICEHOST() inline Item<double, 6> rk4Step(const Item<double, 6>& y, double h, double mu)
{
    const Item<double, 6> k1 = centralForceRhs(y, mu);
    const Item<double, 6> k2 = centralForceRhs(y + (h / 2.0) * k1, mu);
    const Item<double, 6> k3 = centralForceRhs(y + (h / 2.0) * k2, mu);
    const Item<double, 6> k4 = centralForceRhs(y + h * k3, mu);
    return Item<double, 6>(y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
}

// --- plain-double hand-rolled reference (genuinely independent — no aether
// type appears anywhere below). -----------------------------------------

struct RawState {
    double p[3];
    double v[3];
};

RawState rawCentralForceRhs(const RawState& s, double mu)
{
    RawState d;
    d.p[0]             = s.v[0];
    d.p[1]             = s.v[1];
    d.p[2]             = s.v[2];
    const double r2    = s.p[0] * s.p[0] + s.p[1] * s.p[1] + s.p[2] * s.p[2];
    const double r     = std::sqrt(r2);
    const double invR3 = 1.0 / (r2 * r);
    d.v[0]             = -mu * s.p[0] * invR3;
    d.v[1]             = -mu * s.p[1] * invR3;
    d.v[2]             = -mu * s.p[2] * invR3;
    return d;
}

RawState rawAdd(const RawState& a, const RawState& b, double sb)
{
    RawState r;
    for (int i = 0; i < 3; ++i) {
        r.p[i] = a.p[i] + sb * b.p[i];
        r.v[i] = a.v[i] + sb * b.v[i];
    }
    return r;
}

RawState rawRk4Step(const RawState& y, double h, double mu)
{
    const RawState k1 = rawCentralForceRhs(y, mu);
    const RawState k2 = rawCentralForceRhs(rawAdd(y, k1, h / 2.0), mu);
    const RawState k3 = rawCentralForceRhs(rawAdd(y, k2, h / 2.0), mu);
    const RawState k4 = rawCentralForceRhs(rawAdd(y, k3, h), mu);
    RawState out;
    for (int i = 0; i < 3; ++i) {
        out.p[i] = y.p[i] + (h / 6.0) * (k1.p[i] + 2.0 * k2.p[i] + 2.0 * k3.p[i] + k4.p[i]);
        out.v[i] = y.v[i] + (h / 6.0) * (k1.v[i] + 2.0 * k2.v[i] + 2.0 * k3.v[i] + k4.v[i]);
    }
    return out;
}

// Deterministic, non-degenerate initial condition per sample (r > 0 always).
RawState initialCondition(std::size_t idx)
{
    RawState s;
    s.p[0] = 1.0 + 0.1 * static_cast<double>(idx);
    s.p[1] = 0.2;
    s.p[2] = -0.3 + 0.05 * static_cast<double>(idx);
    s.v[0] = 0.0;
    s.v[1] = 1.0 - 0.02 * static_cast<double>(idx);
    s.v[2] = 0.05;
    return s;
}

// abs+rel 1e-12 (matches tests/test_ExprQuaternion.cpp's own convention).
void expectNear(double test, double truth, const char* what)
{
    EXPECT_NEAR(test, truth, 1e-12 + 1e-12 * std::abs(truth)) << what;
}

TEST_F(MiniRKTest, AetherRK4MatchesHandRolledDoubleReference)
{
    auto chunk = Chunk::allocate(Device(kDLCPU), 6 * kN * sizeof(double));
    auto state = aether::make_view<double, 6, aether::dyn>(chunk, kN);

    for (std::size_t idx = 0; idx < kN; ++idx) {
        const RawState s0             = initialCondition(idx);
        state[SampleIndex::make(idx)] = Item<double, 6>{ s0.p[0], s0.p[1], s0.p[2], s0.v[0], s0.v[1], s0.v[2] };
    }

    for (std::size_t step = 0; step < kSteps; ++step) {
        for (std::size_t idx = 0; idx < kN; ++idx) {
            const SampleIndex i     = SampleIndex::make(idx);
            const Item<double, 6> y = state[i].get();
            state[i]                = rk4Step(y, kH, kMu);
        }
    }

    for (std::size_t idx = 0; idx < kN; ++idx) {
        RawState ref = initialCondition(idx);
        for (std::size_t step = 0; step < kSteps; ++step)
            ref = rawRk4Step(ref, kH, kMu);

        const Item<double, 6> got = state[SampleIndex::make(idx)].get();
        const double expected[6]  = { ref.p[0], ref.p[1], ref.p[2], ref.v[0], ref.v[1], ref.v[2] };
        for (std::size_t e = 0; e < 6; ++e)
            expectNear(got(e), expected[e], "sample/component mismatch");
    }
}

} // namespace
} // namespace aether_tests
