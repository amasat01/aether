// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Full-corpus faithful-rounding gate for the CPU packet math: scores every
// function at every width (1, 2, 4, 8 — width 8 only on CPUs with AVX-512F)
// over the golden files written by tools/packet_math_ref/gen_golden.py and
// exits non-zero unless every result is within 1 ULP of the exact value
// (faithfully rounded) and every file was present.
//
//   aether_faithful_check <golden-dir> [--only fn,fn] [--plant] [--worst K <out.bin>]
//
// --plant   non-vacuity probe: plants a 2-ULP error in one row of every
//           function and exits non-zero unless EVERY plant is caught.
// --worst   also appends, per function, the K rows with the largest error
//           (any width) to <out.bin> — input for the committed hard-case set.
//
// `make faithful-gate` regenerates or reuses the corpus, checks it against
// the committed md5 list, then runs this binary with and without --plant.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "tests/packetmath/FaithfulCheck.h"

using namespace aether_pm_test;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <golden-dir> [--only fn,fn] [--plant] [--worst K out.bin]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    std::set<std::string> only;
    bool plant = false;
    std::size_t worstK = 0;
    std::string worstPath;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--plant") {
            plant = true;
        } else if (a == "--only" && i + 1 < argc) {
            std::string s = argv[++i];
            for (std::size_t p = 0; p <= s.size();) {
                const std::size_t q = std::min(s.find(',', p), s.size());
                if (q > p)
                    only.insert(s.substr(p, q - p));
                p = q + 1;
            }
        } else if (a == "--worst" && i + 2 < argc) {
            worstK = static_cast<std::size_t>(std::atol(argv[++i]));
            worstPath = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const Kernels* widths[] = { &kernelsW1(), &kernelsW2(), &kernelsW4(), &kernelsW8() };
    std::FILE* wf = worstK ? std::fopen(worstPath.c_str(), "wb") : nullptr;
    int failures = 0, functions = 0, caught = 0;
    for (const FaithfulFn& f : faithfulFns()) {
        if (!only.empty() && !only.count(f.name) && !only.count(f.file))
            continue;
        GoldenRows g;
        const std::string path = dir + "/" + f.file + ".bin";
        if (!loadGolden(path, f.binary, g)) {
            std::printf("[faithful] %-11s MISSING %s\n", f.name, path.c_str());
            ++failures;
            continue;
        }
        ++functions;
        std::vector<double> rowWorst(wf ? g.size() : 0, 0.0);
        bool fnCaught = true;
        for (const Kernels* k : widths) {
            if (!k->available()) {
                std::printf("[faithful] W=%zu unavailable on this CPU (needs AVX-512F)\n", k->width);
                continue;
            }
            if (plant) {
                const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(g.size() / 2);
                const FaithfulResult r = faithfulScore(*k, f, g, at);
                fnCaught = fnCaught && r.worst >= 1.0;
                continue;
            }
            const FaithfulResult r = faithfulScore(*k, f, g);
            const bool ok = r.worst < 1.0;
            std::printf("[faithful] W=%zu %-11s max %.4f ULP  bad=%zu  n=%zu  %s", k->width, f.name, r.worst, r.bad,
                g.size(), ok ? "ok" : "FAIL");
            if (f.binary)
                std::printf("  worst x=%.17g y=%.17g got=%.17g want=%.17g (frac %.3f)\n", g.x[r.at], g.y[r.at], r.got,
                    g.hi[r.at], g.frac[r.at]);
            else
                std::printf("  worst x=%.17g got=%.17g want=%.17g (frac %.3f)\n", g.x[r.at], r.got, g.hi[r.at],
                    g.frac[r.at]);
            if (!ok)
                ++failures;
            if (wf) {
                // Recompute per-row errors for the worst-K dump.
                const std::size_t n = g.size(), padded = (n + 7) / 8 * 8;
                std::vector<double> x(padded, 1.25), y(padded, 0.75), out(padded);
                std::copy(g.x.begin(), g.x.end(), x.begin());
                if (f.binary) {
                    std::copy(g.y.begin(), g.y.end(), y.begin());
                    k->binary(f.fn, x.data(), y.data(), out.data(), padded);
                } else {
                    k->unary(f.fn, x.data(), out.data(), padded);
                }
                for (std::size_t i = 0; i < n; ++i)
                    rowWorst[i] = std::max(rowWorst[i], faithfulError(out[i], g.hi[i], g.frac[i]));
            }
        }
        if (plant) {
            std::printf("[faithful] planted 2-ULP error in %-11s %s\n", f.name, fnCaught ? "caught" : "MISSED");
            caught += fnCaught ? 1 : 0;
        }
        if (wf) {
            std::vector<std::size_t> idx(g.size());
            for (std::size_t i = 0; i < idx.size(); ++i)
                idx[i] = i;
            const std::size_t k = std::min(worstK, idx.size());
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                [&](std::size_t a, std::size_t b) { return rowWorst[a] > rowWorst[b]; });
            std::fprintf(wf, "%s %zu\n", f.name, k);
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t i = idx[j];
                std::fprintf(wf, "%a %a %a %a %.4f\n", g.x[i], f.binary ? g.y[i] : 0.0, g.hi[i], g.frac[i], rowWorst[i]);
            }
        }
    }
    if (wf)
        std::fclose(wf);
    if (plant) {
        std::printf("[faithful] non-vacuity: %d/%d planted errors caught\n", caught, functions);
        return (caught == functions && functions > 0 && failures == 0) ? 0 : 1;
    }
    std::printf("[faithful] %s: %d function(s), %d failure(s)\n", failures ? "RED" : "GREEN", functions, failures);
    return failures || functions == 0 ? 1 : 0;
}
