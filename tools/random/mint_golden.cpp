// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/random/mint_golden.cpp — mints tests/random/golden_random.h.
//
// WHY A GOLDEN HEADER EXISTS AT ALL. Purely statistical random tests
// (means, variances, k-sigma bands) can tell you a stream LOOKS right,
// and nothing whatever about whether it IS the same stream as
// yesterday's, or as the other arm's. aether's random module carries a
// much stronger claim — one seed names ONE stream, host and device
// alike, bit for bit — and a k-sigma band cannot certify that. This
// header is the committed, machine-checkable statement of what that
// stream IS:
//
//   * `tests/test_Random.cpp` checks the HOST arm against it (exact for
//     `randomBits64`/`uniform01`, TIER-TOL for `standardNormal`);
//   * `tests/test_Random.cu` checks the DEVICE arm against the SAME rows,
//     computed by a kernel — so cross-mode bit-identity is a RUN, not a
//     claim;
//   * the row values themselves were certified against libcurand by
//     `tools/random/curand_host_probe.cpp`.
//
// The header is regenerated ONLY by a deliberate `tools/random/mint_golden.sh`
// (which refuses to run under $CI — a gate that regenerates its own
// expectation is a tautology) and is md5-fenced so an ad-hoc edit to the row
// block is visible. See that script.
//
// This program writes the header to STDOUT with the literal placeholder
// `@@GOLDEN_MD5@@` where the fence value goes; the script substitutes it.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "aether/random/Generator.h"
#include "aether/random/detail/Backend.h"

namespace {

using aether::random::Generator;
namespace rnd = aether::random::detail;

// Seeds: zero, a small fixed test seed, the shared DEFAULT_SEED constant,
// a bit-alternating value, and all-ones — the key halves cover
// 0x00000000 and 0xFFFFFFFF.
const std::uint64_t kSeeds[] = {
    0x0000000000000000ULL,
    0x0000000000C0FFEEULL,
    0xFE7A5EED0C0FFEE5ULL,
    0x0123456789ABCDEFULL,
    0xFFFFFFFFFFFFFFFFULL,
};

// Sample ids: the low end, plus a mid value and the two `offset_t` extremes
// (sample addressing is 32-bit, so 0xFFFFFFFF is a real, reachable id).
const std::uint32_t kGlobals[] = { 0u, 1u, 3u, 17u, 1000u, 65535u, 4294967295u };

// Sub-counters: 0..3 walk the intra-quad output positions, 4/7 straddle the
// first quad boundary, 9/64/1000 exercise the low-counter bump.
const std::uint64_t kCounters[] = { 0ull, 1ull, 2ull, 3ull, 4ull, 7ull, 9ull, 64ull, 1000ull };

// Keying-chain coverage: stream ids and substream selectors.
const std::uint64_t kStreamIds[] = { 0ull, 1ull, 7ull };
const std::uint64_t kSubstreamKs[] = { 0ull, 1ull, 42ull };

std::uint64_t bitsOf(double v)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

std::uint32_t bitsOf(float v)
{
    std::uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

} // namespace

int main()
{
    std::printf("#pragma once\n\n");
    std::printf("/**\n");
    std::printf(" * @file golden_random.h\n");
    std::printf(" * @brief GENERATED — committed golden values for `aether::random`'s core\n");
    std::printf(" *        stream. DO NOT EDIT BY HAND.\n");
    std::printf(" *\n");
    std::printf(" * Minted by `tools/random/mint_golden.sh` (which drives\n");
    std::printf(" * `tools/random/mint_golden.cpp`). Regenerate ONLY deliberately, and commit\n");
    std::printf(" * the diff in the SAME commit as whatever changed the stream — a moved value\n");
    std::printf(" * here is a moved stream, which is a breaking change for every consumer that\n");
    std::printf(" * has ever recorded a seed.\n");
    std::printf(" *\n");
    std::printf(" * The row block below is md5-FENCED: `tools/random/mint_golden.sh --check`\n");
    std::printf(" * re-derives the digest of everything between the BEGIN and END markers and\n");
    std::printf(" * compares it with the recorded value, so a hand-edit to a single constant is\n");
    std::printf(" * visible without regenerating anything.\n");
    std::printf(" *\n");
    std::printf(" * PROVENANCE. Every `bits64`/`uniformDoubleBits`/`uniformFloatBits` value is\n");
    std::printf(" * cuRAND's Philox4x32-10 stream at `(seed, subsequence = global, offset =\n");
    std::printf(" * counter)` — certified BIT-EXACT against a linked `libcurand` host\n");
    std::printf(" * generator by `tools/random/curand_host_probe.cpp`. The `normalDouble`/\n");
    std::printf(" * `normalFloat` values are aether's Box-Muller over `aether::math`, which is\n");
    std::printf(" * TIER-TOL against cuRAND's `sincospi`/`__sincosf` — the tests compare them\n");
    std::printf(" * within a documented tolerance, never exactly.\n");
    std::printf(" */\n\n");
    std::printf("#include <cstddef>\n#include <cstdint>\n\n");
    std::printf("namespace aether_tests {\n");
    std::printf("namespace golden {\n\n");

    std::printf("/** @brief One golden draw of the three core primitives at `(seed, global, counter)`. */\n");
    std::printf("struct CoreRow {\n");
    std::printf("    std::uint64_t seed;      ///< generator seed fed straight to the backend\n");
    std::printf("    std::uint32_t global;    ///< global sample id (Philox subsequence)\n");
    std::printf("    std::uint64_t counter;   ///< sub-counter (Philox offset)\n");
    std::printf("    std::uint64_t bits64;            ///< randomBits64 — EXACT\n");
    std::printf("    std::uint64_t uniformDoubleBits; ///< uniform01<double> bit pattern — EXACT\n");
    std::printf("    std::uint32_t uniformFloatBits;  ///< uniform01<float>  bit pattern — EXACT\n");
    std::printf("    double normalDouble;             ///< standardNormal<double> — TIER-TOL\n");
    std::printf("    float normalFloat;               ///< standardNormal<float>  — TIER-TOL\n");
    std::printf("};\n\n");

    std::printf("/** @brief One golden step of the KEYING chain: the mixed key a `Generator`\n");
    std::printf(" *         carries, and the key its `substream(k)` child carries. */\n");
    std::printf("struct KeyRow {\n");
    std::printf("    std::uint64_t seed;\n");
    std::printf("    std::uint64_t streamId;\n");
    std::printf("    std::uint64_t substreamK;\n");
    std::printf("    std::uint64_t key;          ///< Generator(seed, streamId).key() — EXACT\n");
    std::printf("    std::uint64_t substreamKey; ///< .substream(substreamK).key() — EXACT\n");
    std::printf("};\n\n");

    std::printf("// ---8<--- GOLDEN ROWS BEGIN (md5 = @@GOLDEN_MD5@@) ---8<---\n");

    std::printf("/** @brief Core-primitive golden rows (seeds x sample ids x sub-counters). */\n");
    std::printf("inline constexpr CoreRow kCoreRows[] = {\n");
    std::size_t coreCount = 0;
    for (const std::uint64_t seed : kSeeds) {
        for (const std::uint32_t global : kGlobals) {
            for (const std::uint64_t counter : kCounters) {
                const std::uint64_t bits = rnd::randomBits64(seed, global, counter);
                const double ud          = rnd::uniform01<double>(seed, global, counter);
                const float uf           = rnd::uniform01<float>(seed, global, counter);
                const double nd          = rnd::standardNormal<double>(seed, global, counter);
                const float nf           = rnd::standardNormal<float>(seed, global, counter);
                std::printf("    { 0x%016llXULL, %10uu, %6lluull, 0x%016llXULL, 0x%016llXULL, 0x%08XU, %.17g, %.9gf },\n",
                    static_cast<unsigned long long>(seed), global, static_cast<unsigned long long>(counter),
                    static_cast<unsigned long long>(bits), static_cast<unsigned long long>(bitsOf(ud)), bitsOf(uf), nd,
                    static_cast<double>(nf));
                ++coreCount;
            }
        }
    }
    std::printf("};\n\n");

    std::printf("/** @brief Keying-chain golden rows (SplitMix64 mixing + the substream salt). */\n");
    std::printf("inline constexpr KeyRow kKeyRows[] = {\n");
    std::size_t keyCount = 0;
    for (const std::uint64_t seed : kSeeds) {
        for (const std::uint64_t streamId : kStreamIds) {
            for (const std::uint64_t k : kSubstreamKs) {
                const Generator g(seed, streamId);
                std::printf("    { 0x%016llXULL, %lluull, %3lluull, 0x%016llXULL, 0x%016llXULL },\n",
                    static_cast<unsigned long long>(seed), static_cast<unsigned long long>(streamId),
                    static_cast<unsigned long long>(k), static_cast<unsigned long long>(g.key()),
                    static_cast<unsigned long long>(g.substream(k).key()));
                ++keyCount;
            }
        }
    }
    std::printf("};\n");
    std::printf("// ---8<--- GOLDEN ROWS END ---8<---\n\n");

    std::printf("/** @brief Number of core-primitive golden rows. */\n");
    std::printf("inline constexpr std::size_t kCoreRowCount = %zu;\n", coreCount);
    std::printf("/** @brief Number of keying-chain golden rows. */\n");
    std::printf("inline constexpr std::size_t kKeyRowCount = %zu;\n\n", keyCount);

    std::printf("/** @brief TIER-TOL band for `standardNormal<double>` (absolute AND relative).\n");
    std::printf(" *         aether's Box-Muller routes `sqrt`/`log`/`sincos` through\n");
    std::printf(" *         `aether::math`, so it is not bit-reproducible across the\n");
    std::printf(" *         host/device transcendental implementations the way the integer and\n");
    std::printf(" *         uniform streams are. */\n");
    std::printf("inline constexpr double kNormalDoubleTol = 1.0e-12;\n");
    std::printf("/** @brief TIER-TOL band for `standardNormal<float>` — looser, because the\n");
    std::printf(" *         device FP32 route reaches `sincosf` where cuRAND reaches the fast\n");
    std::printf(" *         `__sincosf` approximation. */\n");
    std::printf("inline constexpr float kNormalFloatTol = 1.0e-5f;\n\n");

    std::printf("} // namespace golden\n");
    std::printf("} // namespace aether_tests\n");
    return 0;
}
