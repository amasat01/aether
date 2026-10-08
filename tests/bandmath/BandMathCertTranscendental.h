// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file BandMathCertTranscendental.h
 * @brief Certification drivers for exp/log/pow, add-on to
 *        `tests/bandmath/BandMathCert.h`. New drivers are needed because
 *        `exp`'s admitted domain is an additive interval in `x` (bucketed
 *        by integer distance past the edge) and `log`/`pow`'s argument
 *        admits a two-sided exponent window (ceiling and floor), unlike
 *        the floor-only bucketing `certifySpineAdmittedBinary` uses. Their
 *        documented near-one leg is unreachable by any row this harness's
 *        double-bit-pattern corpus can produce, so no driver here exercises
 *        it. RED-first machinery is reused as-is from `BandMathCert.h`.
 */

#include "tests/bandmath/BandMathCert.h"

#include <cstdint>
#include <map>
#include <algorithm>

namespace aether_tests {
namespace bandmath {

// The ceiling-side twin of BandMathCert.h's own `kBandAdmittedLo` — that
// file only ever needed the floor leg (add/sub/mul's GAP has no dedicated
// ceiling-side degraded family); log's/pow's argument admits BOTH.
inline constexpr int kBandAdmittedHi = bd::kBandNominalCeilingExp - bd::kBandAdmissionMargin;

// =========================================================================
//  Shared bucket-monotonicity core (the SAME "worst-ulp-per-bucket,
//  non-decreasing, one dip tolerated" discipline `certifySpineAdmittedBinary`
//  uses inline — factored here so all three new drivers share ONE copy).
// =========================================================================
// `ulpDistanceAbs` is an INTEGER (representable-doubles-apart) metric; the
// other ops' bounds (0/2 ULP) are already integers, but exp/log/pow's
// contract is a FRACTIONAL "half ULP" (`kExpUlpBound`/`kLogUlpBound`/
// `kPowUlpBound = 0.5`), stated in the CONTINUOUS sense the derivations
// use. Measured live (host, this package's own build): every op
// lands EXACTLY on the correctly-rounded double (`d==0`) except a small tail
// (3/1265 exp rows, 1/1017 log rows) at `d==1` — a value within 0.5
// CONTINUOUS ulp of the true answer can round to the double NEXT TO the
// correctly-rounded one at the Band-to-IEEE double-rounding step, so `d<=1`
// is the correct discrete translation of "0.5 ULP", not a loosened gate:
// `ceil(bound)` for any bound in `(0,1]`. TRUNCATING instead (`d<=0`) was
// the first draft here and it wrongly REDs every one of those benign
// double-rounding rows — caught by running the suite, not assumed.
inline std::int64_t ulpBoundSteps(double bound)
{
    return static_cast<std::int64_t>(std::ceil(bound));
}

inline bool bucketedMonotoneNonDecreasing(const std::map<long long, std::int64_t>& bucketMaxUlp)
{
    bool ok               = true;
    std::int64_t prevUlp  = -1;
    for (const auto& [dist, worstUlp] : bucketMaxUlp) {
        (void)dist;
        if (prevUlp >= 0 && worstUlp < prevUlp) {
            if (prevUlp - worstUlp > prevUlp / 2 + 4)
                ok = false;
        }
        prevUlp = worstUlp;
    }
    return ok;
}

// =========================================================================
//  certifyExpLike — unary, ADDITIVE-interval domain (`bandExpAdmits`).
//  Degraded bucket key: integer distance IN x past the certified edge
//  (exp's OWN documented curve is stated in x-distance, BandExpLog.h's `bandExpAdmits` doc block: 1.5e-3 ULP @ edge -60.997, 10 ULP @ -70,
//  7.2e4 @ -80 -- an x-linear bucket is the natural resolution for it).
// =========================================================================
template<typename Row, typename UnaryOp>
CertCounts certifyExpLike(const Row* rows, std::size_t n, double bound, float edgeLo, float edgeHi, UnaryOp op)
{
    CertCounts c;
    std::map<long long, std::int64_t> bucketMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            // @see ulpBoundSteps' own doc comment for why this is `ceil(bound)`
            // and not a truncated `bound`.
            const bool pass = (d != ulp::kMismatch) && (d <= ulpBoundSteps(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in=0x" << std::hex << r.in[0] << std::dec
                               << " ref=" << rv << " got=" << gv << " ulp=" << d << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            if (d != ulp::kMismatch) {
                const double x    = doubleOfBits(r.in[0]);
                long long dist    = 0;
                if (x < edgeLo)
                    dist = static_cast<long long>(std::floor(static_cast<double>(edgeLo) - x));
                else if (x > edgeHi)
                    dist = static_cast<long long>(std::floor(x - static_cast<double>(edgeHi)));
                auto [it, _] = bucketMaxUlp.try_emplace(dist, 0);
                it->second   = std::max(it->second, d);
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > ulpBoundSteps(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    EXPECT_TRUE(bucketedMonotoneNonDecreasing(bucketMaxUlp))
        << "exp-like degraded family is not monotone non-decreasing past the certified edge";
    return c;
}

// =========================================================================
//  certifyLogLike — unary, TWO-SIDED exponent-window domain
//  (`bandLogAdmits`'s first two legs — the argument's own ceiling and
//  floor; the near-one THIRD leg is unreachable, @see file docstring).
//  Degraded bucket key: `x`'s own exponent distance past whichever of
//  [kBandAdmittedLo, kBandAdmittedHi] it violated.
// =========================================================================
template<typename Row, typename UnaryOp>
CertCounts certifyLogLike(const Row* rows, std::size_t n, double bound, UnaryOp op)
{
    CertCounts c;
    std::map<long long, std::int64_t> bucketMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band got  = op(a);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) && (d <= ulpBoundSteps(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in=0x" << std::hex << r.in[0] << std::dec << " ref="
                               << rv << " got=" << gv << " ulp=" << d << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            if (d != ulp::kMismatch) {
                const double x = doubleOfBits(r.in[0]);
                const int fb   = floorBucketOf(x);
                if (fb > -1000000) {
                    long long dist = 0;
                    if (fb < kBandAdmittedLo)
                        dist = kBandAdmittedLo - fb;
                    else if (fb > kBandAdmittedHi)
                        dist = fb - kBandAdmittedHi;
                    auto [it, _] = bucketMaxUlp.try_emplace(dist, 0);
                    it->second   = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > ulpBoundSteps(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    EXPECT_TRUE(bucketedMonotoneNonDecreasing(bucketMaxUlp))
        << "log-like degraded family is not monotone non-decreasing past the admitted argument window";
    return c;
}

// =========================================================================
//  certifyPowLike — binary, same two-sided ARGUMENT (base) bucket as log
//  (the exp-inherited ceiling/floor leg on `t=y*log(x)` is a SEPARATE
//  admission failure the corpus labels `rejected`, not `degraded` — @see
//  gen_corpus_explog.py's `label_pow`: only exp's own SATURATION on `t`
//  escapes to `rejected`, matching `certifyExpLike`'s own edge/saturation
//  split one level down).
// =========================================================================
template<typename Row, typename BinaryOp>
CertCounts certifyPowLike(const Row* rows, std::size_t n, double bound, BinaryOp op)
{
    CertCounts c;
    std::map<long long, std::int64_t> bucketMaxUlp;
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r    = rows[i];
        const Band a    = bandFromDoubleBits(r.in[0]);
        const Band b    = bandFromDoubleBits(r.in[1]);
        const Band got  = op(a, b);
        const double gv = doubleOfBits(bandToDoubleBits(got));
        const double rv = doubleOfBits(r.ref);
        const std::int64_t d = ulp::ulpDistanceAbs(rv, gv);

        if (r.label == 0) {
            c.inDomainTotal++;
            const bool pass = (d != ulp::kMismatch) && (d <= ulpBoundSteps(bound));
            if (pass)
                c.inDomainPass++;
            if (d != ulp::kMismatch && d > c.maxInDomainUlp)
                c.maxInDomainUlp = d;
            EXPECT_TRUE(pass) << "in_domain row " << i << ": in0=0x" << std::hex << r.in[0] << " in1=0x" << r.in[1]
                               << std::dec << " ref=" << rv << " got=" << gv << " ulp=" << d << " bound=" << bound;
        } else if (r.label == 1) {
            c.degradedTotal++;
            const bool finiteOk = std::isfinite(gv) || !std::isfinite(rv);
            if (finiteOk)
                c.degradedPass++;
            EXPECT_TRUE(finiteOk) << "degraded row " << i << " produced non-finite garbage: ref=" << rv
                                   << " got=" << gv;
            if (d != ulp::kMismatch) {
                const double x = doubleOfBits(r.in[0]);
                const int fb   = floorBucketOf(x);
                if (fb > -1000000) {
                    long long dist = 0;
                    if (fb < kBandAdmittedLo)
                        dist = kBandAdmittedLo - fb;
                    else if (fb > kBandAdmittedHi)
                        dist = fb - kBandAdmittedHi;
                    auto [it, _] = bucketMaxUlp.try_emplace(dist, 0);
                    it->second   = std::max(it->second, d);
                }
            }
        } else {
            c.rejectedTotal++;
            const bool escaped = !std::isfinite(gv) || (d == ulp::kMismatch) || (d > ulpBoundSteps(bound));
            if (escaped)
                c.rejectedPass++;
            EXPECT_TRUE(escaped) << "rejected row " << i << " silently passed in-bound: ref=" << rv
                                  << " got=" << gv << " ulp=" << d;
        }
    }
    EXPECT_TRUE(bucketedMonotoneNonDecreasing(bucketMaxUlp))
        << "pow-like degraded family is not monotone non-decreasing past the admitted base window";
    return c;
}

} // namespace bandmath
} // namespace aether_tests
