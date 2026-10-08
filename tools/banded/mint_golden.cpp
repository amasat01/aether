// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/banded/mint_golden.cpp — mints tests/banded/golden_band.h.
//
// WHY A GOLDEN HEADER EXISTS AT ALL. The conformance batteries
// (`tests/test_BandCell8_common.h`, `tests/test_BandedReal_common.h`) certify
// the codec and the value type against ORACLES — an exact 128-bit
// reconstruction, an independent probe codec, the format's own constants. What
// no oracle in those files can answer is "does the DEVICE arm compute the same
// bits as the host arm", because each arm can only ever ask its own compiler.
// A committed golden turns that into a RUN:
//
//   * `tests/test_BandGolden.cpp` evaluates the rows on the HOST and compares;
//   * `tests/test_BandGolden.cu` evaluates the SAME rows inside a __global__
//     kernel and compares against the same committed numbers.
//
// Both register `BandGoldenTest.CrossModeBitExact`, so cross-mode bit-identity
// is two runs against one committed statement rather than two runs against each
// other. (It is also the only instrument that survives a change to BOTH arms:
// two arms that move together are invisible to a host-vs-device comparison and
// loud here.)
//
// WHAT IS IN THE ROWS. Two blocks, both ENUMERATED and never sampled:
//   * CODEC rows — one `double` in, the stored 64-bit word and the three decoded
//     limbs out. The corpus enumerates both signed zeros, both infinities, a
//     canonical NaN and five payload variants, the tier-1 edges 2^-94 and 2^125
//     and one binade inside each, a subnormal (which must flush to canonical
//     zero, sign-destroying), and a sweep of binades x mantissa patterns.
//   * CHAIN rows — two stored words in, the four certified primitives
//     (add/sub/mul/div), a three-op chain `((a-b)*a)/b` and one facade
//     composite `copysign(fmax(abs a, abs b), b)` out, each as its THREE RAW
//     LIMBS rather than as a packed word. Limbs, deliberately: they carry the
//     carrier's whole ~72-bit content instead of the codec's 56, so the claim is
//     stronger, and nothing here has to survive a pack whose precondition
//     (`cell8BandIsStorable`) a near-cancelling pair can legitimately violate.
//
// The header is regenerated ONLY by a deliberate `tools/banded/mint_golden.sh`
// (which refuses to run under $CI — a gate that regenerates its own expectation
// is a tautology) and is md5-fenced so an ad-hoc edit to the row block is
// visible without regenerating anything. See that script.
//
// This program writes the header to STDOUT with the literal placeholder
// `@@GOLDEN_MD5@@` where the fence value goes; the script substitutes it.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "aether/banded/banded.h"

namespace {

using aether::banded::Band;
using aether::banded::BandCell8;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;

std::uint64_t dbits(double x)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

double dfrom(std::uint64_t b)
{
    double x = 0.0;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}

std::uint32_t fbits(float f)
{
    std::uint32_t b = 0;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

double build(int e, std::uint64_t mant52, bool neg)
{
    return dfrom((neg ? (1ull << 63) : 0ull)
        | (static_cast<std::uint64_t>(e + 1023) << 52) | mant52);
}

/// The ENUMERATED codec corpus, as raw `double` bit patterns. Never sampled and
/// never filtered: the specials are listed, the tier edges are listed, and the
/// sweep walks every 7th binade of the window against six mantissa patterns.
std::vector<std::uint64_t> codecCorpus()
{
    std::vector<std::uint64_t> v;
    // The specials, enumerated (both zeros, both infinities, the canonical NaN
    // and five payload variants — every one of which must collapse onto the
    // single reserved NaN code point).
    v.push_back(0x0000000000000000ull); // +0
    v.push_back(0x8000000000000000ull); // -0
    v.push_back(0x7FF0000000000000ull); // +Inf
    v.push_back(0xFFF0000000000000ull); // -Inf
    v.push_back(0x7FF8000000000000ull); // canonical qNaN
    v.push_back(0x7FF8000000000001ull);
    v.push_back(0x7FFFFFFFFFFFFFFFull);
    v.push_back(0xFFF8000000000000ull);
    v.push_back(0x7FF0000000000001ull); // an sNaN pattern
    v.push_back(0xFFF4000000000000ull);
    // Subnormals: must flush to CANONICAL zero, sign-destroying.
    v.push_back(0x0000000000000001ull);
    v.push_back(0x800FFFFFFFFFFFFFull);
    // The tier-1 edges, and one binade inside each.
    for (int e : { -94, -93, 124, 125 })
        for (bool neg : { false, true }) {
            v.push_back(dbits(build(e, 0x0000000000000ull, neg)));
            v.push_back(dbits(build(e, 0xFFFFFFFFFFFFFull, neg)));
        }
    // The sweep: every 7th binade of the window x six mantissa patterns x both
    // signs.
    const std::uint64_t mants[] = { 0x0000000000000ull, 0x8000000000000ull,
        0x0000000000001ull, 0xFFFFFFFFFFFFFull, 0x0000001000000ull,
        0x5555555555555ull };
    for (int e = -94; e <= 125; e += 7)
        for (std::uint64_t m : mants)
            for (bool neg : { false, true })
                v.push_back(dbits(build(e, m, neg)));
    return v;
}

/// The ENUMERATED chain corpus: pairs of STORED WORDS. Built from a short list
/// of magnitudes plus the reserved code points, all pairs — so the arithmetic
/// meets `Inf/Inf`, `0/0`, `NaN op x` and `x op -0` by construction rather than
/// by luck.
std::vector<std::pair<std::uint64_t, std::uint64_t>> chainCorpus()
{
    std::vector<BandedReal> ops;
    const double mags[] = { 1.0, -1.0, 3.0, -0.5, 1.0 / 3.0, 1e-9, 1e9,
        0.9999999999999999, 1.0000000000000002 };
    for (double m : mags)
        ops.push_back(BandedReal::fromDouble(m));
    // The tier edges, as values.
    ops.push_back(BandedReal::fromDouble(build(-94, 0, false)));
    ops.push_back(BandedReal::fromDouble(build(125, 0, false)));
    // The reserved code points and both zeros.
    ops.push_back(BandedReal::fromBits(0ull));
    ops.push_back(BandedReal::fromBits(0x8000000000000000ull));
    ops.push_back(BandedReal::fromBits(bd::kBandCell8PosInfWord));
    ops.push_back(BandedReal::fromBits(bd::kBandCell8NegInfWord));
    ops.push_back(BandedReal::fromBits(bd::kBandCell8NanWord));

    std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
    for (const BandedReal& a : ops)
        for (const BandedReal& b : ops)
            pairs.emplace_back(a.toBits(), b.toBits());
    return pairs;
}

void emitLimbs(Band b)
{
    std::printf(" 0x%08XU, 0x%08XU, 0x%08XU,", fbits(b.hi), fbits(b.lo), fbits(b.tail));
}

} // namespace

int main()
{
    std::printf("#pragma once\n\n");
    std::printf("/**\n");
    std::printf(" * @file golden_band.h\n");
    std::printf(" * @brief GENERATED — committed golden values for the banded\n");
    std::printf(" *        codec and the certified arithmetic core. DO NOT EDIT BY HAND.\n");
    std::printf(" *\n");
    std::printf(" * Minted by `tools/banded/mint_golden.sh` (which drives\n");
    std::printf(" * `tools/banded/mint_golden.cpp`). Regenerate ONLY deliberately, and commit\n");
    std::printf(" * the diff in the SAME commit as whatever moved the numbers — a moved value\n");
    std::printf(" * here is a moved CODEC or a moved ROUNDING SCHEDULE, which is a breaking\n");
    std::printf(" * change for every consumer that has ever stored a banded array.\n");
    std::printf(" *\n");
    std::printf(" * The row block below is md5-FENCED: `tools/banded/mint_golden.sh --check`\n");
    std::printf(" * re-derives the digest of everything between the BEGIN and END markers and\n");
    std::printf(" * compares it with the recorded value, so a hand-edit to a single constant is\n");
    std::printf(" * visible without regenerating anything.\n");
    std::printf(" *\n");
    std::printf(" * WHAT THE TWO BLOCKS CLAIM. `kCodecRows` carries one `double` in, the stored\n");
    std::printf(" * 64-bit word and the three decoded limbs out. `kChainRows` carries two stored\n");
    std::printf(" * words in and the RAW LIMBS of `add`, `sub`, `mul`, `div`, a three-op chain\n");
    std::printf(" * `((a-b)*a)/b` and the facade composite `copysign(fmax(|a|,|b|), b)` out.\n");
    std::printf(" * Both corpora are ENUMERATED — specials, tier edges 2^-94 / 2^125, subnormals\n");
    std::printf(" * and a binade sweep — never sampled.\n");
    std::printf(" *\n");
    std::printf(" * `tests/test_BandGolden.cpp` checks the HOST arm against these rows and\n");
    std::printf(" * `tests/test_BandGolden.cu` checks the DEVICE arm against the same rows from\n");
    std::printf(" * inside a kernel, both under the name `BandGoldenTest.CrossModeBitExact` —\n");
    std::printf(" * which is how cross-mode bit-identity becomes a RUN rather than a claim.\n");
    std::printf(" */\n\n");
    std::printf("#include <cstddef>\n#include <cstdint>\n\n");
    std::printf("namespace aether_tests {\nnamespace golden {\n\n");

    std::printf("/** @brief One golden codec row: a `double` in, the stored word and the\n");
    std::printf(" *         decoded carrier's three limbs out. */\n");
    std::printf("struct BandCodecRow {\n");
    std::printf("    std::uint64_t inBits;   ///< the ingested double's bit pattern\n");
    std::printf("    std::uint64_t word;     ///< BandedReal::fromDouble(x).toBits()\n");
    std::printf("    std::uint32_t hi;       ///< the decoded Band's leading limb, as bits\n");
    std::printf("    std::uint32_t lo;       ///< …its second limb\n");
    std::printf("    std::uint32_t tail;     ///< …its residue limb\n");
    std::printf("};\n\n");

    std::printf("/** @brief One golden arithmetic row: two stored words in, six results out,\n");
    std::printf(" *         each as the THREE RAW LIMBS of the working carrier. */\n");
    std::printf("struct BandChainRow {\n");
    std::printf("    std::uint64_t aWord;    ///< left operand, as a stored codec word\n");
    std::printf("    std::uint64_t bWord;    ///< right operand\n");
    std::printf("    std::uint32_t add[3];   ///< a + b\n");
    std::printf("    std::uint32_t sub[3];   ///< a - b\n");
    std::printf("    std::uint32_t mul[3];   ///< a * b\n");
    std::printf("    std::uint32_t div[3];   ///< a / b\n");
    std::printf("    std::uint32_t chain[3]; ///< ((a - b) * a) / b\n");
    std::printf("    std::uint32_t facade[3];///< copysign(fmax(|a|, |b|), b)\n");
    std::printf("};\n\n");

    std::printf("// ---8<--- GOLDEN ROWS BEGIN (md5 = @@GOLDEN_MD5@@) ---8<---\n");

    const std::vector<std::uint64_t> cc = codecCorpus();
    std::printf("/** @brief Codec golden rows (specials, tier edges, subnormals, sweep). */\n");
    std::printf("inline constexpr BandCodecRow kBandCodecRows[] = {\n");
    for (std::uint64_t ub : cc) {
        const BandedReal r = BandedReal::fromDouble(dfrom(ub));
        const Band b       = static_cast<Band>(r);
        std::printf("    { 0x%016llXULL, 0x%016llXULL, 0x%08XU, 0x%08XU, 0x%08XU },\n",
            static_cast<unsigned long long>(ub),
            static_cast<unsigned long long>(r.toBits()), fbits(b.hi), fbits(b.lo),
            fbits(b.tail));
    }
    std::printf("};\n\n");

    const auto pc = chainCorpus();
    std::printf("/** @brief Arithmetic golden rows (all pairs over the enumerated operand set). */\n");
    std::printf("inline constexpr BandChainRow kBandChainRows[] = {\n");
    for (const auto& p : pc) {
        const Band a = static_cast<Band>(BandedReal::fromBits(p.first));
        const Band b = static_cast<Band>(BandedReal::fromBits(p.second));
        std::printf("    { 0x%016llXULL, 0x%016llXULL, {",
            static_cast<unsigned long long>(p.first),
            static_cast<unsigned long long>(p.second));
        emitLimbs(bd::add(a, b));
        std::printf(" }, {");
        emitLimbs(bd::sub(a, b));
        std::printf(" }, {");
        emitLimbs(bd::mul(a, b));
        std::printf(" }, {");
        emitLimbs(bd::div(a, b));
        std::printf(" }, {");
        emitLimbs(bd::div(bd::mul(bd::sub(a, b), a), b));
        std::printf(" }, {");
        emitLimbs(bd::copysign(bd::fmax(bd::abs(a), bd::abs(b)), b));
        std::printf(" } },\n");
    }
    std::printf("};\n");
    std::printf("// ---8<--- GOLDEN ROWS END ---8<---\n\n");

    std::printf("/** @brief Number of codec golden rows. */\n");
    std::printf("inline constexpr std::size_t kBandCodecRowCount = %zu;\n", cc.size());
    std::printf("/** @brief Number of arithmetic golden rows. */\n");
    std::printf("inline constexpr std::size_t kBandChainRowCount = %zu;\n\n", pc.size());

    std::printf("} // namespace golden\n} // namespace aether_tests\n");
    return 0;
}
