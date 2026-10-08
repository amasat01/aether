// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/ulp_oracle/oracle_core.h — shared corpus/MPFR/header-emission logic
// behind both the `ulp_oracle` CLI (ulp_oracle.cpp) and the `selftest`
// target (selftest.cpp). See README.md for the corpus-spec grammar and the
// golden-header format this emits.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aether_tools {
namespace oracle {

/// @brief The 23 functions this harness oracles. Binary = takes two inputs
///        (POW, HYPOT, ATAN2, FMOD); SINCOS is unary-input but dual-output;
///        every other entry is unary-input, single-output.
enum class FuncId {
    EXP, EXP2, LOG, LOG2, LOG1P, EXPM1,
    SQRT, RSQRT, CBRT,
    SIN, COS, TAN, SINCOS, ATAN, ASIN, ACOS,
    SINH, COSH, TANH,
    POW, HYPOT, ATAN2, FMOD,
};

bool lookupFunc(const std::string& name, FuncId& out);
const char* funcName(FuncId f);
int funcArity(FuncId f);           ///< 1 or 2 inputs
bool funcIsDualOutput(FuncId f);   ///< true only for SINCOS

/// @brief The default seed constant (see golden_random.h), reused here
///        rather than inventing a fresh one — one fewer arbitrary constant
///        in the ecosystem. Overridable per-corpus with a `seed` directive;
///        whichever value is actually used is printed into the minted
///        header.
inline constexpr std::uint64_t kDefaultSeed = 0xFE7A5EED0C0FFEE5ULL;

/// @brief MPFR working precision, bits. >= 200 gives correctly-rounded
///        results at double precision with margin; 256 leaves comfortable
///        headroom so `mpfr_get_d`'s final round-to-double is
///        correctly-rounded for anything but an astronomically improbable
///        halfway coincidence.
inline constexpr int kOraclePrecBits = 256;

/// @brief One minted corpus row. `in1`/`ref2`/`refClass2` are only
///        meaningful for binary functions / SINCOS respectively — the
///        header emitter writes only the fields that apply to `f`.
struct Row {
    std::uint64_t in0 = 0;
    std::uint64_t in1 = 0;   // unused (0) for unary, non-SINCOS functions
    std::uint64_t ref = 0;   // sin() for SINCOS, else the sole result
    int refClass = 0;        // aether_tools::ulp::Class, as an int (ulp.h)
    std::uint64_t ref2 = 0;      // cos() — SINCOS only
    int refClass2 = 0;           // SINCOS only
};

/// @brief Parsed corpus-spec: the resolved seed plus enough of the raw
///        source to embed as header provenance.
struct CorpusSpec {
    std::uint64_t seed = kDefaultSeed;
    std::string rawText;   // verbatim spec file content
};

/// @brief Parses corpus-spec text for function `f`. Directives whose arity
///        does not match `f` (e.g. `uniform2` for a unary function) are
///        reported in `errors` and make the overall parse fail. An empty or
///        all-comment spec parses OK with just the default seed — the
///        always-on specials block still applies on top of it.
bool parseCorpusSpec(const std::string& specText, FuncId f, CorpusSpec& outSpec,
    std::vector<std::string>& errors);

/// @brief Builds the full corpus for `f`: the always-on enumerated specials
///        (see README.md) UNION every point/pair `spec`'s directives
///        produce, each evaluated against the MPFR oracle at
///        kOraclePrecBits and correctly rounded to double (RNDN).
///        Deduplicated and returned in a deterministic (sorted) order that
///        does not depend on directive order in the spec file.
std::vector<Row> buildCorpus(FuncId f, const CorpusSpec& spec);

/// @brief Self-contained MD5 (RFC 1321) of `data`, lowercase 32-hex string.
///        No external crypto dependency — see oracle_core.cpp for the
///        two-vector self-check this is built against.
std::string md5Hex(const std::string& data);

/// @brief Writes the golden header to `outPath` (mint-golden.h style:
///        provenance comment, seed + corpus-spec echoed as a comment, an
///        md5-fenced row block, `kCorpusSize`). Returns false on I/O
///        failure.
bool writeHeader(const std::string& outPath, FuncId f, const std::string& specPath,
    const CorpusSpec& spec, const std::vector<Row>& rows, const std::string& argvEcho);

} // namespace oracle
} // namespace aether_tools
