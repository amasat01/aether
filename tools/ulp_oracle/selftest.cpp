// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/ulp_oracle/selftest.cpp — the `selftest` CMake target.
//
// Two independent non-vacuity questions, both mechanically checked (never
// just claimed):
//   (1) INSTRUMENT SANITY — md5Hex() reproduces the two canonical RFC 1321
//       test vectors, and ulpDistance() actually reports a NONZERO distance
//       for a deliberate 1-bit reference perturbation (the "perturbation
//       arm" this self-test exists to catch). If either fails, nothing else in
//       this file — or in any header ulp_oracle.cpp mints — is trustworthy.
//   (2) ORACLE SANITY — mints exp/log/sin/atan2 corpora through the SAME
//       buildCorpus() path the CLI uses, evaluates glibc's double exp/log/
//       sin/atan2 against the minted MPFR references, and asserts every
//       result is within glibc's documented <= 1 ULP contract, printing the
//       max |ULP| actually observed per function.
//
// Host-only, no CUDA. See README.md.
#include "oracle_core.h"
#include "ulp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace aether_tools::oracle;
using aether_tools::ulp::Class;
using aether_tools::ulp::kMismatch;
using aether_tools::ulp::ulpDistance;

namespace {

double doubleOf(std::uint64_t b)
{
    double v;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

int g_fails = 0;

void require(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what);
    if (!ok) ++g_fails;
}

// ---------------------------------------------------------------------
//  (1a) md5Hex against the two canonical RFC 1321 test vectors.
// ---------------------------------------------------------------------
void checkMd5Vectors()
{
    std::printf("== md5Hex self-check ==\n");
    const std::string h1 = md5Hex("");
    const std::string h2 = md5Hex("abc");
    std::printf("  md5(\"\")    = %s\n", h1.c_str());
    std::printf("  md5(\"abc\") = %s\n", h2.c_str());
    require(h1 == "d41d8cd98f00b204e9800998ecf8427e", "md5(\"\") matches RFC 1321 vector");
    require(h2 == "900150983cd24fb0d6963f7d28e17f72", "md5(\"abc\") matches RFC 1321 vector");
}

// ---------------------------------------------------------------------
//  (2) one function's oracle-vs-glibc pass. `evalGot` calls the glibc
//  double routine on the same inputs the row carries.
// ---------------------------------------------------------------------
struct FuncResult {
    std::int64_t maxAbsUlp = 0;
    std::size_t mismatches = 0;
    std::size_t count = 0;
};

template<typename EvalGot>
FuncResult checkAgainstGlibc(const char* label, const std::vector<Row>& rows, bool binary, EvalGot evalGot)
{
    FuncResult res;
    res.count = rows.size();
    for (const Row& r : rows) {
        const double ref = doubleOf(r.ref);
        const double got = binary ? evalGot(doubleOf(r.in0), doubleOf(r.in1)) : evalGot(doubleOf(r.in0), 0.0);
        const std::int64_t d = ulpDistance(ref, got);
        if (d == kMismatch) {
            ++res.mismatches;
            continue;
        }
        const std::int64_t ad = d < 0 ? -d : d;
        if (ad > res.maxAbsUlp) res.maxAbsUlp = ad;
    }
    std::printf("  %-8s N=%-5zu maxAbsUlp=%-3lld mismatches=%zu\n", label, res.count,
        static_cast<long long>(res.maxAbsUlp), res.mismatches);
    return res;
}

// ---------------------------------------------------------------------
//  (1b) perturbation arm — reference bits XOR 1 must never come back as
//  0 ULP. Restricted to non-NaN reference rows: flipping a NaN's payload
//  bit is still a NaN, and this checker deliberately treats every NaN as
//  class-equal to every other NaN (see ulp.h) — a NaN row would report 0
//  here for a reason that has nothing to do with the vacuity this arm
//  exists to catch, so including it would just be testing the wrong thing.
// ---------------------------------------------------------------------
void perturbationArm(const char* label, const std::vector<Row>& rows)
{
    std::size_t tested = 0, zeroDistance = 0;
    for (const Row& r : rows) {
        if (r.refClass == static_cast<int>(Class::kNaN)) continue;
        const double ref       = doubleOf(r.ref);
        const double perturbed = doubleOf(r.ref ^ 1ULL);
        const std::int64_t d   = ulpDistance(ref, perturbed);
        ++tested;
        if (d == 0) ++zeroDistance;
    }
    std::printf("  %-8s perturbation arm: %zu/%zu rows nonzero-ULP (skipped %zu NaN refs)\n", label,
        tested - zeroDistance, tested, rows.size() - tested);
    require(zeroDistance == 0, (std::string(label) + ": every non-NaN reference-bit-xor-1 row is nonzero ULP")
                                    .c_str());
}

} // namespace

int main()
{
    checkMd5Vectors();
    std::printf("\n");

    if (g_fails != 0) {
        std::printf("*** SELFTEST ABORT: md5 self-check failed, oracle sanity is meaningless ***\n");
        return 3;
    }

    // -------------------------------------------------------------
    //  Corpora — moderate ranges plus a modest reduction-boundary
    //  sweep, evaluated through the exact same buildCorpus() path
    //  the ulp_oracle CLI uses (the always-on specials block included).
    // -------------------------------------------------------------
    std::vector<std::string> parseErrors;

    FuncId fExp, fLog, fSin, fAtan2;
    lookupFunc("exp", fExp);
    lookupFunc("log", fLog);
    lookupFunc("sin", fSin);
    lookupFunc("atan2", fAtan2);

    auto mustParse = [&](const char* label, const std::string& text, FuncId f, CorpusSpec& spec) {
        parseErrors.clear();
        const bool ok = parseCorpusSpec(text, f, spec, parseErrors);
        for (const auto& e : parseErrors) std::printf("  corpus-spec error (%s): %s\n", label, e.c_str());
        require(ok, (std::string(label) + ": literal selftest corpus-spec parses clean").c_str());
    };

    CorpusSpec specExp;
    mustParse("exp", "uniform -700 700 200\nloguniform 1e-300 700 200\nreduction ln2 8\n", fExp, specExp);

    CorpusSpec specLog;
    mustParse("log", "loguniform 1e-300 1e300 300\nreduction pow2 -100 100\n", fLog, specLog);

    CorpusSpec specSin;
    mustParse("sin", "uniform -1000 1000 300\nreduction pi_over_2 20\nreduction pi_over_4 20\n", fSin, specSin);

    CorpusSpec specAtan2;
    mustParse("atan2", "uniform2 -100 100 -100 100 300\nreduction2 pi_over_4 8\n", fAtan2, specAtan2);

    const std::vector<Row> rowsExp   = buildCorpus(fExp, specExp);
    const std::vector<Row> rowsLog   = buildCorpus(fLog, specLog);
    const std::vector<Row> rowsSin   = buildCorpus(fSin, specSin);
    const std::vector<Row> rowsAtan2 = buildCorpus(fAtan2, specAtan2);

    std::printf("== oracle-vs-glibc (glibc's <=1 ULP contract) ==\n");
    const FuncResult rExp = checkAgainstGlibc("exp", rowsExp, false, [](double x, double) { return std::exp(x); });
    const FuncResult rLog = checkAgainstGlibc("log", rowsLog, false, [](double x, double) { return std::log(x); });
    const FuncResult rSin = checkAgainstGlibc("sin", rowsSin, false, [](double x, double) { return std::sin(x); });
    const FuncResult rAtan2 = checkAgainstGlibc(
        "atan2", rowsAtan2, true, [](double y, double x) { return std::atan2(y, x); });

    require(rExp.maxAbsUlp <= 1 && rExp.mismatches == 0, "exp within 1 ULP, no class mismatches");
    require(rLog.maxAbsUlp <= 1 && rLog.mismatches == 0, "log within 1 ULP, no class mismatches");
    require(rSin.maxAbsUlp <= 1 && rSin.mismatches == 0, "sin within 1 ULP, no class mismatches");
    require(rAtan2.maxAbsUlp <= 1 && rAtan2.mismatches == 0, "atan2 within 1 ULP, no class mismatches");

    std::printf("\n== perturbation arm (non-vacuity: ref bits XOR 1 must be nonzero ULP) ==\n");
    perturbationArm("exp", rowsExp);
    perturbationArm("log", rowsLog);
    perturbationArm("sin", rowsSin);
    perturbationArm("atan2", rowsAtan2);

    std::printf("\nSELFTEST verdict: %s (%d failing checks)\n", g_fails ? "RED" : "GREEN", g_fails);
    return g_fails ? 1 : 0;
}
