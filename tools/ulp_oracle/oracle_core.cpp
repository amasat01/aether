// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/ulp_oracle/oracle_core.cpp — see oracle_core.h and README.md.
#include "oracle_core.h"
#include "ulp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

#include <mpfr.h>

namespace aether_tools {
namespace oracle {

// =====================================================================
//  Function table
// =====================================================================

namespace {
struct FuncEntry {
    const char* name;
    FuncId id;
    int arity;
    bool dual;
};

const FuncEntry kFuncs[] = {
    { "exp",    FuncId::EXP,    1, false },
    { "exp2",   FuncId::EXP2,   1, false },
    { "log",    FuncId::LOG,    1, false },
    { "log2",   FuncId::LOG2,   1, false },
    { "log1p",  FuncId::LOG1P,  1, false },
    { "expm1",  FuncId::EXPM1,  1, false },
    { "sqrt",   FuncId::SQRT,   1, false },
    { "rsqrt",  FuncId::RSQRT,  1, false },
    { "cbrt",   FuncId::CBRT,   1, false },
    { "sin",    FuncId::SIN,    1, false },
    { "cos",    FuncId::COS,    1, false },
    { "tan",    FuncId::TAN,    1, false },
    { "sincos", FuncId::SINCOS, 1, true  },
    { "atan",   FuncId::ATAN,   1, false },
    { "asin",   FuncId::ASIN,   1, false },
    { "acos",   FuncId::ACOS,   1, false },
    { "sinh",   FuncId::SINH,   1, false },
    { "cosh",   FuncId::COSH,   1, false },
    { "tanh",   FuncId::TANH,   1, false },
    { "pow",    FuncId::POW,    2, false },
    { "hypot",  FuncId::HYPOT,  2, false },
    { "atan2",  FuncId::ATAN2,  2, false },
    { "fmod",   FuncId::FMOD,   2, false },
};
constexpr int kNumFuncs = sizeof(kFuncs) / sizeof(kFuncs[0]);
} // namespace

bool lookupFunc(const std::string& name, FuncId& out)
{
    for (const auto& e : kFuncs) {
        if (name == e.name) { out = e.id; return true; }
    }
    return false;
}

const char* funcName(FuncId f)
{
    for (const auto& e : kFuncs) if (e.id == f) return e.name;
    return "?";
}

int funcArity(FuncId f)
{
    for (const auto& e : kFuncs) if (e.id == f) return e.arity;
    return 0;
}

bool funcIsDualOutput(FuncId f) { return f == FuncId::SINCOS; }

// =====================================================================
//  Bit helpers
// =====================================================================

namespace {

std::uint64_t bitsOf(double v)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

double doubleOf(std::uint64_t b)
{
    double v = 0.0;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

constexpr std::uint64_t kPosZero  = 0x0000000000000000ULL;
constexpr std::uint64_t kNegZero  = 0x8000000000000000ULL;
constexpr std::uint64_t kPosInf   = 0x7FF0000000000000ULL;
constexpr std::uint64_t kNegInf   = 0xFFF0000000000000ULL;
constexpr std::uint64_t kQNaN     = 0x7FF8000000000000ULL;

// -------- splitmix64: a standard fast, well-distributed PRNG. --------
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next()
    {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z                = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z                = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // uniform double in [0, 1) from the top 53 bits.
    double uniform01()
    {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

} // namespace

// =====================================================================
//  Always-on enumerated specials
// =====================================================================

namespace {

void addUnary(std::set<std::uint64_t>& pool, double v) { pool.insert(bitsOf(v)); }
void addUnaryBits(std::set<std::uint64_t>& pool, std::uint64_t b) { pool.insert(b); }

// The generic block every unary function gets, the same for every op:
// +-0, +-inf, canonical qNaN, +-min/max normal, +-min subnormal, +-1.
void addGenericUnarySpecials(std::set<std::uint64_t>& pool)
{
    addUnaryBits(pool, kPosZero);
    addUnaryBits(pool, kNegZero);
    addUnaryBits(pool, kPosInf);
    addUnaryBits(pool, kNegInf);
    addUnaryBits(pool, kQNaN);
    addUnary(pool, (std::numeric_limits<double>::min)());   // +min normal
    addUnary(pool, -(std::numeric_limits<double>::min)());  // -min normal
    addUnary(pool, (std::numeric_limits<double>::max)());   // +max normal
    addUnary(pool, -(std::numeric_limits<double>::max)());  // -max normal
    addUnary(pool, (std::numeric_limits<double>::denorm_min)());  // +min subnormal
    addUnary(pool, -(std::numeric_limits<double>::denorm_min)()); // -min subnormal
    addUnary(pool, 1.0);
    addUnary(pool, -1.0);
}

// Function-specific domain edges layered on top of the generic block.
void addDomainEdges(FuncId f, std::set<std::uint64_t>& pool)
{
    const double one = 1.0;
    switch (f) {
        case FuncId::LOG1P:
            // domain is (-1, inf): -1 itself -> -inf, just above -> finite
            // and very negative, just below -> NaN.
            addUnary(pool, -1.0);
            addUnary(pool, std::nextafter(-1.0, 0.0));
            addUnary(pool, std::nextafter(-1.0, -2.0));
            break;
        case FuncId::ASIN:
        case FuncId::ACOS:
            // domain is [-1, 1]: the edges themselves are already in the
            // generic +-1 block; add the first step OUTSIDE on each side.
            addUnary(pool, std::nextafter(one, 2.0));
            addUnary(pool, std::nextafter(-one, -2.0));
            break;
        case FuncId::SQRT:
        case FuncId::RSQRT:
            // domain is [0, inf): the smallest-magnitude negative value is
            // already in the generic block (-min subnormal) and is the
            // domain violation that matters; nothing extra to add.
            break;
        default:
            break;
    }
}

std::vector<std::uint64_t> unarySpecialsFor(FuncId f)
{
    std::set<std::uint64_t> pool;
    addGenericUnarySpecials(pool);
    addDomainEdges(f, pool);
    return std::vector<std::uint64_t>(pool.begin(), pool.end());
}

// Curated (NOT full cross-product) binary specials per function — the
// combinations whose special-value handling is actually load-bearing for
// that function's IEEE-754 contract.
std::vector<std::pair<std::uint64_t, std::uint64_t>> binarySpecialsFor(FuncId f)
{
    std::vector<std::pair<double, double>> p;
    const double z = 0.0, nz = -0.0, one = 1.0, none = -1.0, two = 2.0, ntwo = -2.0;
    const double inf = doubleOf(kPosInf), ninf = doubleOf(kNegInf), nan = doubleOf(kQNaN);
    const double dmin = (std::numeric_limits<double>::min)();
    const double dmax = (std::numeric_limits<double>::max)();
    const double dsub = (std::numeric_limits<double>::denorm_min)();

    switch (f) {
        case FuncId::POW:
            p = {
                { z, z }, { nz, z }, { z, nz }, { nz, nz },
                { z, one }, { z, none }, { z, two }, { z, ntwo },
                { one, inf }, { one, ninf }, { one, nan }, { one, z },
                { none, inf }, { none, ninf },
                { inf, z }, { inf, one }, { inf, none }, { inf, inf }, { inf, ninf },
                { ninf, 3.0 }, { ninf, two }, { ninf, none },
                { two, inf }, { two, ninf }, { 0.5, inf }, { 0.5, ninf },
                { ntwo, 3.0 }, { ntwo, two }, { ntwo, 2.5 },
                { nan, z }, { z, nan }, { nan, one }, { nan, nan },
            };
            break;
        case FuncId::HYPOT:
            p = {
                { z, z }, { nz, z }, { z, nz }, { nz, nz },
                { inf, z }, { z, inf }, { inf, nan }, { nan, inf },
                { inf, ninf }, { ninf, inf },
                { nan, one }, { one, nan }, { nan, nan },
                { dmax, dmax }, { dmin, dmin }, { dsub, dsub },
            };
            break;
        case FuncId::ATAN2: // (y, x)
            p = {
                { z, z }, { nz, z }, { z, nz }, { nz, nz },
                { z, one }, { z, none }, { nz, one }, { nz, none },
                { one, z }, { one, nz }, { none, z }, { none, nz },
                { inf, inf }, { inf, ninf }, { ninf, inf }, { ninf, ninf },
                { inf, one }, { one, inf }, { ninf, one }, { one, ninf },
                { nan, one }, { one, nan }, { nan, nan },
            };
            break;
        case FuncId::FMOD: // (x, y)
            p = {
                { 5.0, z }, { z, 5.0 }, { z, z },
                { inf, one }, { one, inf }, { ninf, one }, { one, ninf },
                { nan, one }, { one, nan }, { nan, nan },
                { -5.0, 3.0 }, { 5.0, -3.0 }, { -5.0, -3.0 },
                { dmax, 3.0 }, { one, dsub },
            };
            break;
        default:
            break;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    out.reserve(p.size());
    for (auto& xy : p) out.emplace_back(bitsOf(xy.first), bitsOf(xy.second));
    return out;
}

} // namespace

// =====================================================================
//  Corpus-spec parsing
// =====================================================================

namespace {

bool parseValueToken(const std::string& tok, double& out)
{
    std::string t = tok;
    std::string low = t;
    for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (low == "inf" || low == "+inf" || low == "infinity") { out = doubleOf(kPosInf); return true; }
    if (low == "-inf" || low == "-infinity") { out = doubleOf(kNegInf); return true; }
    if (low == "nan" || low == "qnan") { out = doubleOf(kQNaN); return true; }
    char* end = nullptr;
    out = std::strtod(t.c_str(), &end);
    return end != t.c_str() && *end == '\0';
}

bool parseIntToken(const std::string& tok, long long& out)
{
    char* end = nullptr;
    out       = std::strtoll(tok.c_str(), &end, 0);
    return end != tok.c_str() && *end == '\0';
}

// Three doubles bracketing v at 1-ULP spacing on either side — the
// "reduction boundary" idiom: a reduction-scheme cutover is only exercised
// meaningfully if BOTH sides of the cut are sampled, not just the exact
// cut value itself.
void addTriplet(std::set<std::uint64_t>& pool, double v)
{
    pool.insert(bitsOf(v));
    pool.insert(bitsOf(std::nextafter(v, -std::numeric_limits<double>::infinity())));
    pool.insert(bitsOf(std::nextafter(v, std::numeric_limits<double>::infinity())));
}

double mpfrMultipleOfPiOverDiv(long long k, int div)
{
    mpfr_t pi, r;
    mpfr_init2(pi, 300);
    mpfr_init2(r, 300);
    mpfr_const_pi(pi, MPFR_RNDN);
    mpfr_div_si(r, pi, div, MPFR_RNDN);
    mpfr_mul_si(r, r, k, MPFR_RNDN);
    const double out = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(pi);
    mpfr_clear(r);
    return out;
}

double mpfrMultipleOfLn2(long long k)
{
    mpfr_t l2, r;
    mpfr_init2(l2, 300);
    mpfr_init2(r, 300);
    mpfr_const_log2(l2, MPFR_RNDN);
    mpfr_mul_si(r, l2, k, MPFR_RNDN);
    const double out = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(l2);
    mpfr_clear(r);
    return out;
}

// Applies `kind`'s boundary set (shared by `reduction` and `reduction2`) to
// `emit`, which either inserts a single unary bit pattern or does whatever
// the caller wants with the raw double.
template<typename Emit>
bool applyReductionKind(const std::string& kind, const std::vector<std::string>& args,
    Emit emit, std::string& err)
{
    if (kind == "pi_over_2" || kind == "pi_over_4") {
        if (args.size() != 1) { err = "reduction " + kind + " expects one arg: K"; return false; }
        long long K = 0;
        if (!parseIntToken(args[0], K) || K < 0) { err = "reduction " + kind + ": bad K"; return false; }
        const int div = (kind == "pi_over_2") ? 2 : 4;
        for (long long k = -K; k <= K; ++k) emit(mpfrMultipleOfPiOverDiv(k, div));
        return true;
    }
    if (kind == "ln2") {
        if (args.size() != 1) { err = "reduction ln2 expects one arg: K"; return false; }
        long long K = 0;
        if (!parseIntToken(args[0], K) || K < 0) { err = "reduction ln2: bad K"; return false; }
        for (long long k = -K; k <= K; ++k) emit(mpfrMultipleOfLn2(k));
        return true;
    }
    if (kind == "pow2") {
        if (args.size() != 2) { err = "reduction pow2 expects two args: elo ehi"; return false; }
        long long elo = 0, ehi = 0;
        if (!parseIntToken(args[0], elo) || !parseIntToken(args[1], ehi) || elo > ehi) {
            err = "reduction pow2: bad elo/ehi";
            return false;
        }
        for (long long e = elo; e <= ehi; ++e) {
            const double v = std::ldexp(1.0, static_cast<int>(e));
            emit(v);
            emit(-v);
        }
        return true;
    }
    if (kind == "subnormal_edge") {
        if (!args.empty()) { err = "reduction subnormal_edge takes no args"; return false; }
        const double dsub = (std::numeric_limits<double>::denorm_min)();
        const double dmin = (std::numeric_limits<double>::min)();
        const double maxSub = std::nextafter(dmin, 0.0);
        for (double v : { dsub, 2.0 * dsub, dmin, maxSub }) {
            emit(v);
            emit(-v);
        }
        return true;
    }
    err = "unknown reduction kind: " + kind;
    return false;
}

std::vector<std::string> tokenize(const std::string& line)
{
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

} // namespace

// Internal directive-driven pools threaded from parse -> build; kept in the
// CorpusSpec's rawText only for provenance, re-derived here on demand so
// buildCorpus() does not need a second parser entry point.
struct ParsedDirectives {
    std::vector<std::string> lines; // raw, non-comment, non-blank directive lines
};

namespace {
ParsedDirectives splitDirectiveLines(const std::string& specText)
{
    ParsedDirectives pd;
    std::istringstream iss(specText);
    std::string line;
    while (std::getline(iss, line)) {
        // strip comment
        auto hashPos = line.find('#');
        if (hashPos != std::string::npos) line = line.substr(0, hashPos);
        // trim whitespace
        auto b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        auto e = line.find_last_not_of(" \t\r\n");
        pd.lines.push_back(line.substr(b, e - b + 1));
    }
    return pd;
}
} // namespace

bool parseCorpusSpec(const std::string& specText, FuncId f, CorpusSpec& outSpec,
    std::vector<std::string>& errors)
{
    outSpec.seed    = kDefaultSeed;
    outSpec.rawText = specText;
    const int arity = funcArity(f);

    const ParsedDirectives pd = splitDirectiveLines(specText);
    bool ok                   = true;
    for (const auto& line : pd.lines) {
        std::vector<std::string> tok = tokenize(line);
        if (tok.empty()) continue;
        const std::string& kw = tok[0];
        const bool isUnaryDirective =
            (kw == "uniform" || kw == "loguniform" || kw == "points" || kw == "reduction");
        const bool isBinaryDirective =
            (kw == "uniform2" || kw == "loguniform2" || kw == "points2" || kw == "reduction2");

        if (kw == "seed") {
            if (tok.size() != 2) { errors.push_back("seed: expects one value"); ok = false; continue; }
            long long v = 0;
            if (!parseIntToken(tok[1], v)) { errors.push_back("seed: not an integer: " + tok[1]); ok = false; continue; }
            outSpec.seed = static_cast<std::uint64_t>(v);
            continue;
        }
        if (isUnaryDirective && arity != 1) {
            errors.push_back(kw + ": unary directive on a binary function (" + std::string(funcName(f)) +
                "); use " + kw + "2");
            ok = false;
            continue;
        }
        if (isBinaryDirective && arity != 2) {
            errors.push_back(kw + ": binary directive on a unary function (" + std::string(funcName(f)) + ")");
            ok = false;
            continue;
        }
        if (!isUnaryDirective && !isBinaryDirective) {
            errors.push_back("unknown directive: " + kw);
            ok = false;
            continue;
        }

        // Validate shape only here (values are re-derived during
        // buildCorpus); this keeps parseCorpusSpec cheap (no MPFR calls)
        // while still catching malformed lines before any generation runs.
        if (kw == "uniform" || kw == "loguniform") {
            if (tok.size() < 4) { errors.push_back(kw + ": expects lo hi n [neg]"); ok = false; }
        } else if (kw == "uniform2" || kw == "loguniform2") {
            if (tok.size() < 6) { errors.push_back(kw + ": expects lo0 hi0 lo1 hi1 n [...]"); ok = false; }
        } else if (kw == "points" || kw == "points2") {
            if (tok.size() < 2) { errors.push_back(kw + ": expects at least one value"); ok = false; }
            if (kw == "points2" && (tok.size() - 1) % 2 != 0) {
                errors.push_back("points2: values must come in x,y pairs");
                ok = false;
            }
        } else if (kw == "reduction" || kw == "reduction2") {
            if (tok.size() < 2) { errors.push_back(kw + ": expects a kind"); ok = false; }
        }
    }
    return ok;
}

// =====================================================================
//  Directive-driven pool generation (re-walks the same lines buildCorpus
//  needs, now actually producing points — separated from parseCorpusSpec
//  so a syntax-only validation pass never touches MPFR).
// =====================================================================

namespace {

void generateUnaryPool(const std::string& specText, std::uint64_t seed, std::set<std::uint64_t>& pool)
{
    Rng rng(seed);
    const ParsedDirectives pd = splitDirectiveLines(specText);
    for (const auto& line : pd.lines) {
        std::vector<std::string> tok = tokenize(line);
        if (tok.empty()) continue;
        const std::string& kw = tok[0];
        if (kw == "uniform" && tok.size() >= 4) {
            double lo, hi;
            long long n;
            if (parseValueToken(tok[1], lo) && parseValueToken(tok[2], hi) && parseIntToken(tok[3], n)) {
                for (long long i = 0; i < n; ++i) pool.insert(bitsOf(lo + rng.uniform01() * (hi - lo)));
            }
        } else if (kw == "loguniform" && tok.size() >= 4) {
            double lo, hi;
            long long n;
            const bool neg = (tok.size() >= 5 && tok[4] == "neg");
            if (parseValueToken(tok[1], lo) && parseValueToken(tok[2], hi) && parseIntToken(tok[3], n) &&
                lo > 0.0 && hi > lo) {
                const double llo = std::log(lo), lhi = std::log(hi);
                for (long long i = 0; i < n; ++i) {
                    double v = std::exp(llo + rng.uniform01() * (lhi - llo));
                    if (neg) v = -v;
                    pool.insert(bitsOf(v));
                }
            }
        } else if (kw == "points") {
            for (std::size_t i = 1; i < tok.size(); ++i) {
                double v;
                if (parseValueToken(tok[i], v)) pool.insert(bitsOf(v));
            }
        } else if (kw == "reduction" && tok.size() >= 2) {
            std::vector<std::string> args(tok.begin() + 2, tok.end());
            std::string err;
            applyReductionKind(
                tok[1], args, [&](double v) { addTriplet(pool, v); }, err);
            // errors already surfaced by parseCorpusSpec's shape check;
            // a semantic error here (e.g. bad K) just yields no extra
            // points for that line rather than a second error channel.
        }
    }
}

void generateBinaryPool(
    const std::string& specText, std::uint64_t seed, std::set<std::pair<std::uint64_t, std::uint64_t>>& pool)
{
    Rng rng(seed);
    const ParsedDirectives pd = splitDirectiveLines(specText);
    for (const auto& line : pd.lines) {
        std::vector<std::string> tok = tokenize(line);
        if (tok.empty()) continue;
        const std::string& kw = tok[0];
        if (kw == "uniform2" && tok.size() >= 6) {
            double lo0, hi0, lo1, hi1;
            long long n;
            if (parseValueToken(tok[1], lo0) && parseValueToken(tok[2], hi0) && parseValueToken(tok[3], lo1) &&
                parseValueToken(tok[4], hi1) && parseIntToken(tok[5], n)) {
                for (long long i = 0; i < n; ++i) {
                    const double x = lo0 + rng.uniform01() * (hi0 - lo0);
                    const double y = lo1 + rng.uniform01() * (hi1 - lo1);
                    pool.insert({ bitsOf(x), bitsOf(y) });
                }
            }
        } else if (kw == "loguniform2" && tok.size() >= 6) {
            double lo0, hi0, lo1, hi1;
            long long n;
            bool neg0 = false, neg1 = false;
            for (std::size_t i = 6; i < tok.size(); ++i) {
                if (tok[i] == "neg0") neg0 = true;
                if (tok[i] == "neg1") neg1 = true;
            }
            if (parseValueToken(tok[1], lo0) && parseValueToken(tok[2], hi0) && parseValueToken(tok[3], lo1) &&
                parseValueToken(tok[4], hi1) && parseIntToken(tok[5], n) && lo0 > 0.0 && hi0 > lo0 && lo1 > 0.0 &&
                hi1 > lo1) {
                const double l0lo = std::log(lo0), l0hi = std::log(hi0);
                const double l1lo = std::log(lo1), l1hi = std::log(hi1);
                for (long long i = 0; i < n; ++i) {
                    double x = std::exp(l0lo + rng.uniform01() * (l0hi - l0lo));
                    double y = std::exp(l1lo + rng.uniform01() * (l1hi - l1lo));
                    if (neg0) x = -x;
                    if (neg1) y = -y;
                    pool.insert({ bitsOf(x), bitsOf(y) });
                }
            }
        } else if (kw == "points2") {
            for (std::size_t i = 1; i + 1 < tok.size(); i += 2) {
                double x, y;
                if (parseValueToken(tok[i], x) && parseValueToken(tok[i + 1], y))
                    pool.insert({ bitsOf(x), bitsOf(y) });
            }
        } else if (kw == "reduction2" && tok.size() >= 2) {
            std::vector<std::string> args(tok.begin() + 2, tok.end());
            std::string err;
            std::vector<double> boundary;
            applyReductionKind(
                tok[1], args, [&](double v) { boundary.push_back(v); }, err);
            for (double b : boundary) {
                pool.insert({ bitsOf(b), bitsOf(b) });
                pool.insert({ bitsOf(b), bitsOf(1.0) });
                pool.insert({ bitsOf(1.0), bitsOf(b) });
            }
        }
    }
}

} // namespace

// =====================================================================
//  MPFR evaluation
// =====================================================================

namespace {

int classOf(double v) { return static_cast<int>(aether_tools::ulp::classify(v)); }

double evalUnaryMPFR(FuncId f, double x)
{
    mpfr_t a, r;
    mpfr_init2(a, kOraclePrecBits);
    mpfr_init2(r, kOraclePrecBits);
    mpfr_set_d(a, x, MPFR_RNDN);
    switch (f) {
        case FuncId::EXP: mpfr_exp(r, a, MPFR_RNDN); break;
        case FuncId::EXP2: mpfr_exp2(r, a, MPFR_RNDN); break;
        case FuncId::LOG: mpfr_log(r, a, MPFR_RNDN); break;
        case FuncId::LOG2: mpfr_log2(r, a, MPFR_RNDN); break;
        case FuncId::LOG1P: mpfr_log1p(r, a, MPFR_RNDN); break;
        case FuncId::EXPM1: mpfr_expm1(r, a, MPFR_RNDN); break;
        case FuncId::SQRT: mpfr_sqrt(r, a, MPFR_RNDN); break;
        case FuncId::RSQRT: mpfr_rec_sqrt(r, a, MPFR_RNDN); break;
        case FuncId::CBRT: mpfr_cbrt(r, a, MPFR_RNDN); break;
        case FuncId::SIN: mpfr_sin(r, a, MPFR_RNDN); break;
        case FuncId::COS: mpfr_cos(r, a, MPFR_RNDN); break;
        case FuncId::TAN: mpfr_tan(r, a, MPFR_RNDN); break;
        case FuncId::ATAN: mpfr_atan(r, a, MPFR_RNDN); break;
        case FuncId::ASIN: mpfr_asin(r, a, MPFR_RNDN); break;
        case FuncId::ACOS: mpfr_acos(r, a, MPFR_RNDN); break;
        case FuncId::SINH: mpfr_sinh(r, a, MPFR_RNDN); break;
        case FuncId::COSH: mpfr_cosh(r, a, MPFR_RNDN); break;
        case FuncId::TANH: mpfr_tanh(r, a, MPFR_RNDN); break;
        default: mpfr_set_nan(r); break;
    }
    const double out = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(a);
    mpfr_clear(r);
    return out;
}

void evalSincosMPFR(double x, double& outSin, double& outCos)
{
    mpfr_t a, rs, rc;
    mpfr_init2(a, kOraclePrecBits);
    mpfr_init2(rs, kOraclePrecBits);
    mpfr_init2(rc, kOraclePrecBits);
    mpfr_set_d(a, x, MPFR_RNDN);
    mpfr_sin_cos(rs, rc, a, MPFR_RNDN);
    outSin = mpfr_get_d(rs, MPFR_RNDN);
    outCos = mpfr_get_d(rc, MPFR_RNDN);
    mpfr_clear(a);
    mpfr_clear(rs);
    mpfr_clear(rc);
}

// atan2's MPFR signature is atan2(rop, y, x, rnd) — same (y, x) argument
// order as libm's atan2, which is the order this harness's `in0, in1`
// carry for FuncId::ATAN2 throughout (see README.md).
double evalBinaryMPFR(FuncId f, double x0, double x1)
{
    mpfr_t a, b, r;
    mpfr_init2(a, kOraclePrecBits);
    mpfr_init2(b, kOraclePrecBits);
    mpfr_init2(r, kOraclePrecBits);
    mpfr_set_d(a, x0, MPFR_RNDN);
    mpfr_set_d(b, x1, MPFR_RNDN);
    switch (f) {
        case FuncId::POW: mpfr_pow(r, a, b, MPFR_RNDN); break;
        case FuncId::HYPOT: mpfr_hypot(r, a, b, MPFR_RNDN); break;
        case FuncId::ATAN2: mpfr_atan2(r, a, b, MPFR_RNDN); break;
        case FuncId::FMOD: mpfr_fmod(r, a, b, MPFR_RNDN); break;
        default: mpfr_set_nan(r); break;
    }
    const double out = mpfr_get_d(r, MPFR_RNDN);
    mpfr_clear(a);
    mpfr_clear(b);
    mpfr_clear(r);
    return out;
}

} // namespace

// =====================================================================
//  buildCorpus
// =====================================================================

std::vector<Row> buildCorpus(FuncId f, const CorpusSpec& spec)
{
    std::vector<Row> rows;
    if (funcArity(f) == 1) {
        std::set<std::uint64_t> pool;
        for (auto b : unarySpecialsFor(f)) pool.insert(b);
        generateUnaryPool(spec.rawText, spec.seed, pool);

        rows.reserve(pool.size());
        for (std::uint64_t b : pool) {
            const double x = doubleOf(b);
            Row row;
            row.in0 = b;
            if (funcIsDualOutput(f)) {
                double s, c;
                evalSincosMPFR(x, s, c);
                row.ref       = bitsOf(s);
                row.refClass  = classOf(s);
                row.ref2      = bitsOf(c);
                row.refClass2 = classOf(c);
            } else {
                const double y = evalUnaryMPFR(f, x);
                row.ref        = bitsOf(y);
                row.refClass   = classOf(y);
            }
            rows.push_back(row);
        }
    } else {
        std::set<std::pair<std::uint64_t, std::uint64_t>> pool;
        for (auto pr : binarySpecialsFor(f)) pool.insert(pr);
        generateBinaryPool(spec.rawText, spec.seed, pool);

        rows.reserve(pool.size());
        for (auto& pr : pool) {
            const double x0 = doubleOf(pr.first);
            const double x1 = doubleOf(pr.second);
            const double y  = evalBinaryMPFR(f, x0, x1);
            Row row;
            row.in0      = pr.first;
            row.in1      = pr.second;
            row.ref      = bitsOf(y);
            row.refClass = classOf(y);
            rows.push_back(row);
        }
    }
    return rows;
}

// =====================================================================
//  MD5 (RFC 1321) — self-contained, no external crypto dependency.
//  Verified against the two canonical test vectors in selftest.cpp:
//    md5("")    == d41d8cd98f00b204e9800998ecf8427e
//    md5("abc") == 900150983cd24fb0d6963f7d28e17f72
// =====================================================================

namespace {

inline std::uint32_t leftRotate(std::uint32_t x, std::uint32_t c) { return (x << c) | (x >> (32 - c)); }

const std::uint32_t kMd5K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};
const std::uint32_t kMd5S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
};

} // namespace

std::string md5Hex(const std::string& data)
{
    std::uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;

    std::vector<std::uint8_t> msg(data.begin(), data.end());
    const std::uint64_t origBitLen = static_cast<std::uint64_t>(data.size()) * 8ULL;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0x00);
    for (int i = 0; i < 8; ++i) msg.push_back(static_cast<std::uint8_t>((origBitLen >> (8 * i)) & 0xFF));

    for (std::size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        std::uint32_t M[16];
        for (int i = 0; i < 16; ++i) {
            M[i] = static_cast<std::uint32_t>(msg[chunk + 4 * i]) |
                (static_cast<std::uint32_t>(msg[chunk + 4 * i + 1]) << 8) |
                (static_cast<std::uint32_t>(msg[chunk + 4 * i + 2]) << 16) |
                (static_cast<std::uint32_t>(msg[chunk + 4 * i + 3]) << 24);
        }
        std::uint32_t A = a0, B = b0, C = c0, D = d0;
        for (int i = 0; i < 64; ++i) {
            std::uint32_t F;
            int g;
            if (i < 16) {
                F = (B & C) | ((~B) & D);
                g = i;
            } else if (i < 32) {
                F = (D & B) | ((~D) & C);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                F = B ^ C ^ D;
                g = (3 * i + 5) % 16;
            } else {
                F = C ^ (B | (~D));
                g = (7 * i) % 16;
            }
            F         = F + A + kMd5K[i] + M[g];
            A         = D;
            D         = C;
            C         = B;
            B         = B + leftRotate(F, kMd5S[i]);
        }
        a0 += A;
        b0 += B;
        c0 += C;
        d0 += D;
    }

    std::uint8_t digest[16];
    std::uint32_t words[4] = { a0, b0, c0, d0 };
    for (int w = 0; w < 4; ++w)
        for (int i = 0; i < 4; ++i) digest[4 * w + i] = static_cast<std::uint8_t>((words[w] >> (8 * i)) & 0xFF);

    static const char* hexd = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[2 * i]     = hexd[(digest[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[digest[i] & 0xF];
    }
    return out;
}

// =====================================================================
//  Header emission
// =====================================================================

namespace {

std::string upperName(FuncId f)
{
    std::string s = funcName(f);
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

void appendCommentBlock(std::string& out, const std::string& text)
{
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) out += "// " + line + "\n";
}

} // namespace

bool writeHeader(const std::string& outPath, FuncId f, const std::string& specPath, const CorpusSpec& spec,
    const std::vector<Row>& rows, const std::string& argvEcho)
{
    const bool binary = (funcArity(f) == 2);
    const bool dual   = funcIsDualOutput(f);
    const std::string fn = funcName(f);
    const std::string FN = upperName(f);

    std::string head;
    head += "// GENERATED by tools/ulp_oracle/ulp_oracle.cpp — golden reference table for\n";
    head += "// `" + fn + "`, DO NOT EDIT BY HAND. Regenerate with the mint command line below and\n";
    head += "// commit the diff in the SAME commit as whatever test consuming this\n";
    head += "// header changed — a moved row here is a moved oracle answer.\n";
    head += "//\n";
    head += "// mint command line: " + argvEcho + "\n";
    head += "// corpus-spec file:  " + specPath + "\n";
    head += "// seed:              0x";
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llX", static_cast<unsigned long long>(spec.seed));
        head += buf;
    }
    head += "ULL (deterministic — reproduces this exact corpus)\n";
    head += "// MPFR precision:    " + std::to_string(kOraclePrecBits) +
        " bits, correctly rounded to double (MPFR_RNDN)\n";
    head += "//\n";
    head += "// corpus-spec source, verbatim (see tools/ulp_oracle/README.md for the grammar):\n";
    head += "// ---8<---\n";
    appendCommentBlock(head, spec.rawText.empty() ? std::string("(empty — specials-only corpus)\n") : spec.rawText);
    head += "// ---8<---\n";
    head += "//\n";
    head += "// refClass encoding (aether_tools::ulp::Class in ulp.h): 0=zero 1=subnormal\n";
    head += "// 2=normal 3=inf 4=nan.\n";
    head += "//\n";
    head += "// The row block below is md5-FENCED: recompute (BEGIN_RE anchored on the\n";
    head += "// `---8<---` prefix so it does not also match prose sentences that mention the\n";
    head += "// marker words, such as this one):\n";
    head += "//   BEGIN_RE='^// ---8<--- GOLDEN ROWS BEGIN' ; END_RE='^// ---8<--- GOLDEN ROWS END'\n";
    head += "//   sed -n \"/$BEGIN_RE/,/$END_RE/p\" <this file> | sed '1d;$d' | md5sum\n";
    head += "// and compare against the value on the BEGIN line — see README.md \"fence check\".\n";

    std::string body;
    body += "#pragma once\n\n";
    body += head + "\n";
    body += "#include <cstdint>\n#include <cstddef>\n\n";
    body += "namespace aether_tests {\nnamespace ulp_golden {\nnamespace " + fn + " {\n\n";

    if (dual) {
        body += "struct Row {\n";
        body += "    std::uint64_t in0;       ///< input bit pattern\n";
        body += "    std::uint64_t refSin;    ///< MPFR sin(in0), correctly rounded\n";
        body += "    std::uint8_t  refSinClass;\n";
        body += "    std::uint64_t refCos;    ///< MPFR cos(in0), correctly rounded\n";
        body += "    std::uint8_t  refCosClass;\n";
        body += "};\n\n";
    } else if (binary) {
        body += "struct Row {\n";
        body += "    std::uint64_t in[2];     ///< input bit patterns" +
            std::string(f == FuncId::ATAN2 ? " (in[0]=y, in[1]=x — libm atan2(y,x) order)" : "") + "\n";
        body += "    std::uint64_t ref;       ///< MPFR reference, correctly rounded to double\n";
        body += "    std::uint8_t  refClass;\n";
        body += "};\n\n";
    } else {
        body += "struct Row {\n";
        body += "    std::uint64_t in[1];     ///< input bit pattern\n";
        body += "    std::uint64_t ref;       ///< MPFR reference, correctly rounded to double\n";
        body += "    std::uint8_t  refClass;\n";
        body += "};\n\n";
    }

    std::string rowsText; // the exact text that will sit between BEGIN/END, for md5ing
    rowsText += "inline constexpr Row kRows[] = {\n";
    for (const Row& r : rows) {
        char buf[256];
        if (dual) {
            std::snprintf(buf, sizeof(buf), "    { 0x%016llXULL, 0x%016llXULL, %d, 0x%016llXULL, %d },\n",
                static_cast<unsigned long long>(r.in0), static_cast<unsigned long long>(r.ref), r.refClass,
                static_cast<unsigned long long>(r.ref2), r.refClass2);
        } else if (binary) {
            std::snprintf(buf, sizeof(buf), "    { { 0x%016llXULL, 0x%016llXULL }, 0x%016llXULL, %d },\n",
                static_cast<unsigned long long>(r.in0), static_cast<unsigned long long>(r.in1),
                static_cast<unsigned long long>(r.ref), r.refClass);
        } else {
            std::snprintf(buf, sizeof(buf), "    { { 0x%016llXULL }, 0x%016llXULL, %d },\n",
                static_cast<unsigned long long>(r.in0), static_cast<unsigned long long>(r.ref), r.refClass);
        }
        rowsText += buf;
    }
    rowsText += "};\n";

    const std::string md5 = md5Hex(rowsText);

    body += "// ---8<--- GOLDEN ROWS BEGIN (md5 = " + md5 + ") ---8<---\n";
    body += rowsText;
    body += "// ---8<--- GOLDEN ROWS END ---8<---\n\n";
    body += "inline constexpr std::size_t kCorpusSize = " + std::to_string(rows.size()) + ";\n\n";
    body += "} // namespace " + fn + "\n";
    body += "} // namespace ulp_golden\n";
    body += "} // namespace aether_tests\n";

    std::FILE* fp = std::fopen(outPath.c_str(), "w");
    if (!fp) return false;
    const std::size_t n = std::fwrite(body.data(), 1, body.size(), fp);
    std::fclose(fp);
    return n == body.size();
}

} // namespace oracle
} // namespace aether_tools
