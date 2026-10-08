// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/random/curand_host_probe.cpp — de-risk probe for aether's random module.
//
// THROWAWAY TOOL, NOT PART OF THE TEST SUITE. It answers one question with an
// INDEPENDENT instrument: does aether's own Philox4x32-10
// (aether/random/detail/{Philox,Backend}.h) reproduce cuRAND's stream
// BIT-EXACTLY -- key/counter layout, intra-quad output ordering, subsequence
// placement and the word->real conversions included?
//
// The instrument is deliberately NOT the header aether was written from: it
// links the real `libcurand` HOST generator (`curandCreateGeneratorHost` +
// `CURAND_RNG_PSEUDO_PHILOX4_32_10`), a separate NVIDIA-compiled
// implementation of the same generator running on the CPU. Agreement with it
// cannot be an artifact of having transcribed a header.
//
// -------------------------------------------------------------------------
// MEASURED ENUMERATION OF THE HOST GENERATOR (this probe's own finding, and
// the reason its coverage goes beyond a literal per-subsequence sweep):
// the cuRAND host generator does NOT walk one subsequence linearly. Output
// word k comes from (subsequence = k/4, offset = k%4) -- i.e. it flattens a
// grid of virtual threads, four words each. `curandSetGeneratorOffset(O)`
// positions at FLATTENED index O, so "subsequence 0, offsets 0..63" is not
// reachable through this API at all (verified for orderings DEFAULT / BEST /
// LEGACY; SEEDED and DYNAMIC are rejected outright for this generator).
// The double path reaches further INTO a subsequence: `curand_uniform4_double`
// consumes 8 words per thread, so double i uses words (t = i/4, j = i%4)
// -> hq(word(t, 2j), word(t, 2j+1)), i.e. offsets 0..7.
//
// COVERAGE ACTUALLY DELIVERED (per seed, 5 seeds, 256 values per call):
//   A  raw words        : subsequence 0..63 x offset 0..3   (Philox core, key
//                         layout, subsequence placement, intra-quad order)
//   B  uniform floats   : same grid, through `_curand_uniform`
//   C  uniform doubles  : subsequence 0..63 x offset 0..7   (also exercises
//                         the low-counter bump for offset >= 4), through
//                         `_curand_uniform_double_hq` -- the conversion
//                         aether's double Box-Muller uses
//   D  uniform01<double>: aether's ONE-word `_curand_uniform_double` applied
//                         at every (subsequence, offset) whose word cuRAND
//                         itself produced in A. NOTE: libcurand's HOST
//                         `curandGenerateUniformDouble` uses the two-word hq
//                         conversion (check C), while the DEVICE entry point
//                         `curand_uniform_double(curandStatePhilox4_32_10_t*)`
//                         -- the one aether's device kernel calls -- uses
//                         the one-word conversion. The
//                         two are different cuRAND functions, so no host call
//                         can produce that stream; D therefore certifies the
//                         POSITION (cuRAND's own words) and the twin
//                         certifies the conversion against an independent
//                         device-side reference.
// That is a strict superset of the structural intent: it pins the
// subsequence axis, which was otherwise expected to be covered only by
// the device twin.
// -------------------------------------------------------------------------
//
// Prints `PROBE_RC=0` only on full equality; any mismatch prints the first
// divergent (seed, subsequence, offset, aether, curand) tuple and PROBE_RC=1.
//
// Build+run (no CMake wiring by design -- throwaway):
//   CP=$CONDA_PREFIX   # a conda env carrying the CUDA toolkit
//   g++ -std=c++23 -O2 -DAETHER_CPP_MODE=1 -I<aether-root>
//       -I$CP/targets/x86_64-linux/include
//       tools/random/curand_host_probe.cpp -o $TMPDIR/curand_host_probe
//       -L$CP/lib -lcurand -Wl,-rpath,$CP/lib
//   $TMPDIR/curand_host_probe

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <curand.h>

#include "aether/random/detail/Backend.h"
#include "aether/random/detail/Philox.h"

namespace {

using aether::random::detail::PhiloxStream;

/// Values requested per cuRAND call: 256 = 64 virtual threads x 4 words.
constexpr unsigned kN = 256;

const std::uint64_t kSeeds[] = {
    0x0000000000000000ULL,
    0x0000000000C0FFEEULL,
    0xFE7A5EED0C0FFEE5ULL,
    0x0123456789ABCDEFULL,
    0xFFFFFFFFFFFFFFFFULL,
};

/// One 32-bit word of aether's stream at (seed, subsequence, offset).
std::uint32_t word(std::uint64_t seed, std::uint64_t sub, std::uint64_t off)
{
    PhiloxStream st(seed, sub, off);
    return st.next();
}

bool ok(curandStatus_t s, const char* what)
{
    if (s != CURAND_STATUS_SUCCESS) {
        std::printf("PROBE: curand call failed (%s), status=%d\n", what, static_cast<int>(s));
        return false;
    }
    return true;
}

int failures = 0;
long long compared = 0;

void reportFirst(const char* check, std::uint64_t seed, std::uint64_t sub, std::uint64_t off, const char* mineStr, const char* refStr)
{
    if (failures == 0)
        std::printf("PROBE: FIRST DIVERGENCE [%s] seed=0x%016" PRIx64 " subsequence=%" PRIu64 " offset=%" PRIu64
                    " aether=%s curand=%s\n",
            check, seed, sub, off, mineStr, refStr);
    ++failures;
}

} // namespace

int main()
{
    for (const std::uint64_t seed : kSeeds) {
        std::vector<unsigned int> refWords(kN, 0u);
        std::vector<float> refFloats(kN, 0.0f);
        std::vector<double> refDoubles(kN, 0.0);

        curandGenerator_t gen = nullptr;
#define PROBE_STEP(call, label)                     \
    if (!ok(call, label)) {                         \
        std::printf("PROBE_RC=1\n");                \
        return 1;                                   \
    }
        PROBE_STEP(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10), "createHost/words")
        PROBE_STEP(curandSetPseudoRandomGeneratorSeed(gen, seed), "setSeed/words")
        PROBE_STEP(curandSetGeneratorOffset(gen, 0ULL), "setOffset/words")
        PROBE_STEP(curandGenerate(gen, refWords.data(), kN), "curandGenerate")
        curandDestroyGenerator(gen);

        gen = nullptr;
        PROBE_STEP(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10), "createHost/floats")
        PROBE_STEP(curandSetPseudoRandomGeneratorSeed(gen, seed), "setSeed/floats")
        PROBE_STEP(curandSetGeneratorOffset(gen, 0ULL), "setOffset/floats")
        PROBE_STEP(curandGenerateUniform(gen, refFloats.data(), kN), "curandGenerateUniform")
        curandDestroyGenerator(gen);

        gen = nullptr;
        PROBE_STEP(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10), "createHost/doubles")
        PROBE_STEP(curandSetPseudoRandomGeneratorSeed(gen, seed), "setSeed/doubles")
        PROBE_STEP(curandSetGeneratorOffset(gen, 0ULL), "setOffset/doubles")
        PROBE_STEP(curandGenerateUniformDouble(gen, refDoubles.data(), kN), "curandGenerateUniformDouble")
        curandDestroyGenerator(gen);
#undef PROBE_STEP

        char mine[64];
        char ref[64];

        for (unsigned k = 0; k < kN; ++k) {
            const std::uint64_t sub = k / 4u;
            const std::uint64_t off = k % 4u;

            // A -- raw 32-bit words.
            const std::uint32_t w = word(seed, sub, off);
            ++compared;
            if (w != refWords[k]) {
                std::snprintf(mine, sizeof(mine), "0x%08" PRIx32, w);
                std::snprintf(ref, sizeof(ref), "0x%08x", refWords[k]);
                reportFirst("words", seed, sub, off, mine, ref);
            }

            // B -- uniform floats, cuRAND's `_curand_uniform`.
            const float f = aether::random::detail::uniform01<float>(
                seed, static_cast<aether::offset_t>(sub), off);
            ++compared;
            if (f != refFloats[k]) {
                std::snprintf(mine, sizeof(mine), "%.9g", static_cast<double>(f));
                std::snprintf(ref, sizeof(ref), "%.9g", static_cast<double>(refFloats[k]));
                reportFirst("uniform01<float>", seed, sub, off, mine, ref);
            }

            // D -- aether's one-word `uniform01<double>` conversion applied at
            // the position whose word cuRAND itself produced (see the header).
            const double d = aether::random::detail::uniform01<double>(
                seed, static_cast<aether::offset_t>(sub), off);
            const double dRef = aether::random::detail::uniformDoubleFromWord(refWords[k]);
            ++compared;
            if (d != dRef) {
                std::snprintf(mine, sizeof(mine), "%.17g", d);
                std::snprintf(ref, sizeof(ref), "%.17g", dRef);
                reportFirst("uniform01<double>", seed, sub, off, mine, ref);
            }

            // C -- uniform doubles through the two-word hq conversion; double
            // k comes from thread k/4's words 2*(k%4) and 2*(k%4)+1, so this
            // reaches offsets 0..7 within a subsequence.
            const std::uint64_t dsub = k / 4u;
            const std::uint64_t j    = k % 4u;
            const double hq          = aether::random::detail::uniformDoubleHq(
                word(seed, dsub, 2u * j), word(seed, dsub, 2u * j + 1u));
            ++compared;
            if (hq != refDoubles[k]) {
                std::snprintf(mine, sizeof(mine), "%.17g", hq);
                std::snprintf(ref, sizeof(ref), "%.17g", refDoubles[k]);
                reportFirst("uniformDoubleHq", seed, dsub, 2u * j, mine, ref);
            }
        }
    }

    std::printf("PROBE: seeds=%zu subsequences=0..63 offsets=0..3 (words/float/double1w) and 0..7 (double-hq); "
                "comparisons=%lld mismatches=%d\n",
        sizeof(kSeeds) / sizeof(kSeeds[0]), compared, failures);
    std::printf("PROBE_RC=%d\n", failures == 0 ? 0 : 1);
    return failures == 0 ? 0 : 1;
}
