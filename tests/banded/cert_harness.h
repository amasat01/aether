// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file cert_harness.h
 * @brief Carrier-generic DD-oracle certification harness, shared by every
 *        per-rung working-carrier cert TU (`test_Ff1Cert_common.h`,
 *        `test_Ff2Cert_common.h`, and any future rung's cert twin).
 *
 * The double-double reference oracle, the effective-bits metric, and the
 * carrier-generic df-vs-oracle bit measurement are ALL pure `double`
 * host-only arithmetic with zero SoftDouble/CompDD coupling.
 *
 * Each carrier's own harness (e.g. `test_Ff2Cert_common.h`) consumes this and
 * adds only the carrier-specific extraction (`ff2ExactDf`, the carrier's
 * chain shapes).
 *
 * ── The two measurement conventions (used deliberately across all cert TUs) ──
 * Oracle: a host-only double-double (Knuth twoSum + FMA twoProd, ~106-bit
 * effective, pure `double`) mirroring the carrier op SHAPES so per-op errors
 * are measured against the same algorithm at ~4x the precision.
 *   - PER-PRIMITIVE / ALGORITHM certs feed the oracle the SAME df values the
 *     carrier op consumes (read back from the carrier's own limbs, via the
 *     carrier's exact-df extractor) -- isolating the OPERATION's rounding
 *     error from the input-demote conditioning, so a correctly-rounded op
 *     reads at least its certified width even under cancellation. Use
 *     `dfBits`.
 *   - THE g-MEASUREMENT (chain terminal) feeds the oracle the EXACT double
 *     inputs (consumer-realistic: the carrier really does pay the demote at
 *     region entry) -- measuring the full carrier error the terminal will
 *     see. Use `effBits` on the stored terminal value.
 */

#include <cmath>

namespace aether_tests {
namespace cert {

// ─────────────────────────────────────────────────────────────────────────
//  Host DD-oracle (Knuth twoSum + FMA twoProd, ~106-bit, pure double).
//  Mirrors the carrier op SHAPES so per-op errors are measured against the
//  same algorithm at ~4x the precision.
// ─────────────────────────────────────────────────────────────────────────
namespace ddref {
struct DD {
    double hi, lo;
};
static inline void twoSum(double a, double b, double& s, double& e)
{
    s         = a + b;
    double bp = s - a;
    double ap = s - bp;
    e         = (a - ap) + (b - bp);
}
static inline void twoProd(double a, double b, double& p, double& e)
{
    p = a * b;
    e = std::fma(a, b, -p);
}
static inline DD dd(double x) { return DD{ x, 0.0 }; }
static inline DD add(DD a, DD b)
{
    double s, e;
    twoSum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    double h, l;
    twoSum(s, e, h, l);
    return DD{ h, l };
}
static inline DD neg(DD a) { return DD{ -a.hi, -a.lo }; }
static inline DD sub(DD a, DD b) { return add(a, neg(b)); }
static inline DD mul(DD a, DD b)
{
    double p, e;
    twoProd(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    double h, l;
    twoSum(p, e, h, l);
    return DD{ h, l };
}
static inline DD divv(DD a, DD b)
{
    double q1 = a.hi / b.hi;
    DD r      = sub(a, mul(dd(q1), b));
    double q2 = r.hi / b.hi;
    DD r2     = sub(r, mul(dd(q2), b));
    double q3 = r2.hi / b.hi;
    double h, l;
    twoSum(q1, q2, h, l);
    return add(DD{ h, l }, dd(q3));
}
static inline DD sqrtv(DD a)
{
    if (a.hi <= 0.0)
        return DD{ 0.0, 0.0 };
    double x = std::sqrt(a.hi);
    DD r     = sub(a, mul(dd(x), dd(x)));
    double c = r.hi / (2.0 * x);
    double h, l;
    twoSum(x, c, h, l);
    return DD{ h, l };
}
static inline double toDouble(DD a) { return a.hi + a.lo; }
} // namespace ddref

// ─────────────────────────────────────────────────────────────────────────
//  Metrics (carrier-generic)
// ─────────────────────────────────────────────────────────────────────────

/// @brief Effective bits = -log2(relative error). Clamped to [0, 60]; a zero
/// oracle with a zero result reads "perfect" (60).
static inline double effBits(double got, double oracle)
{
    if (oracle == 0.0)
        return got == 0.0 ? 60.0 : 0.0;
    double rel = std::fabs(got - oracle) / std::fabs(oracle);
    if (rel == 0.0)
        return 60.0;
    double b = -std::log2(rel);
    return b < 0.0 ? 0.0 : (b > 60.0 ? 60.0 : b);
}

/// @brief The carrier-generic core of the per-op "algorithm bits" convention:
/// effective bits of a carrier's EXACT df value (already extracted to a DD)
/// against a df-matched oracle. A carrier's harness supplies the extraction
/// (e.g. `algoBits(carrier, oracle) = dfBits(exactDf(carrier), oracle)`).
static inline double dfBits(ddref::DD gotExactDf, ddref::DD oracle)
{
    return effBits(ddref::toDouble(gotExactDf), ddref::toDouble(oracle));
}

} // namespace cert
} // namespace aether_tests
