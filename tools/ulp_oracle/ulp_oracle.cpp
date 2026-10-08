// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/ulp_oracle/ulp_oracle.cpp — CLI front end.
//
//   ulp_oracle <function> <corpus-spec> <out-header>
//
// See README.md for the corpus-spec grammar and the emitted header format.
// Unlike aether's random-golden pattern (tools/random/mint_golden.cpp +
// mint_golden.sh), this is a SINGLE binary: it takes the corpus spec as an
// argument rather than having one hardcoded, so there is no separate
// "regenerate" wrapper script to keep in sync — the md5 fence is computed
// in-process (oracle_core.cpp's writeHeader) on every invocation.
#include "oracle_core.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace aether_tools::oracle;

namespace {

void printUsage(const char* prog)
{
    std::fprintf(stderr, "usage: %s <function> <corpus-spec> <out-header>\n\n", prog);
    std::fprintf(stderr, "functions:\n");
    std::fprintf(stderr,
        "  unary:  exp exp2 log log2 log1p expm1 sqrt rsqrt cbrt\n"
        "          sin cos tan sincos atan asin acos sinh cosh tanh\n"
        "  binary: pow hypot atan2 fmod   (corpus-spec uses the *2 directives)\n\n");
    std::fprintf(stderr, "corpus-spec grammar and header format: tools/ulp_oracle/README.md\n");
}

bool readFile(const std::string& path, std::string& out)
{
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::string joinArgv(int argc, char** argv)
{
    std::string s;
    for (int i = 0; i < argc; ++i) {
        if (i) s += " ";
        s += argv[i];
    }
    return s;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        printUsage(argv[0]);
        return 2;
    }
    const std::string funcArg = argv[1];
    const std::string specPath = argv[2];
    const std::string outPath = argv[3];

    FuncId f;
    if (!lookupFunc(funcArg, f)) {
        std::fprintf(stderr, "ulp_oracle: unknown function '%s'\n\n", funcArg.c_str());
        printUsage(argv[0]);
        return 2;
    }

    std::string specText;
    if (!readFile(specPath, specText)) {
        std::fprintf(stderr, "ulp_oracle: cannot read corpus-spec file: %s\n", specPath.c_str());
        return 2;
    }

    CorpusSpec spec;
    std::vector<std::string> errors;
    if (!parseCorpusSpec(specText, f, spec, errors)) {
        for (const auto& e : errors) std::fprintf(stderr, "ulp_oracle: corpus-spec error: %s\n", e.c_str());
        return 2;
    }

    std::vector<Row> rows = buildCorpus(f, spec);
    if (rows.empty()) {
        std::fprintf(stderr, "ulp_oracle: corpus came back empty for '%s' — refusing to mint\n", funcArg.c_str());
        return 1;
    }

    const std::string argvEcho = joinArgv(argc, argv);
    if (!writeHeader(outPath, f, specPath, spec, rows, argvEcho)) {
        std::fprintf(stderr, "ulp_oracle: failed to write %s\n", outPath.c_str());
        return 1;
    }

    std::fprintf(stdout, "ulp_oracle: wrote %s — %zu entries (function=%s, seed=0x%016llX)\n", outPath.c_str(),
        rows.size(), funcArg.c_str(), static_cast<unsigned long long>(spec.seed));
    return 0;
}
