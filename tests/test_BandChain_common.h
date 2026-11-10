// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_BandChain_common.h
 * @brief `BandChain` suite — shared by `test_BandChain.cpp` and
 *        `test_BandChain.cu`. Rows for `aether/banded/Chain.h`: the lazy
 *        (compile-time tracked) chain against the per-op normalised one and a
 *        `long double` reference, the specials table re-run through lazy
 *        chains, the type rows and a RED-first mutant of the terminal.
 *
 * The accuracy bound. Every banded add/sub/mul claims <= 1 ULP53 of ITS result.
 * A chain of n such ops whose result keeps at least half of the magnitude scale
 * `M` of its intermediates (the corpus REJECTS chains that cancel harder, since
 * an ULP of the result is not a meaningful unit there) therefore owes
 * `2n + 1` ULP of the final value (the +1 is the store rounding to `double`).
 * The per-op chain and the lazy chain are held to the SAME bound.
 *
 * That bound is loose by about five orders of magnitude (a Band carries ~70
 * bits, the reference is `__float128`), so the rows ALSO hold the lazy error to
 * the per-op error plus the derived `bd::bandChainErrorBound` (ULP53 units), and `HopsEvidence`
 * prints how the unrestored error grows with the hop count under it; that
 * derived bound, pinned by `static_assert` in `Chain.h`, is what sets
 * `kBandChainMaxHops`.
 */

#include <gtest/gtest.h>

#include "tests/WideReference.h"

#include "aether/banded/banded.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace aether_tests {
namespace BandChainTest {

using aether::banded::Band;
using aether::banded::BandedReal;
namespace bd = aether::banded::detail;

constexpr int kMax = bd::kBandChainMaxHops;

// ---- reference and conversion helpers ------------------------------------

inline double bitsToDouble(std::uint64_t b)
{
    double x;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}
inline Band bandOf(double x)
{
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return bd::bandFromIEEE(static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(b >> 32));
}
inline double doubleOf(Band v)
{
    std::uint32_t lo = 0, hi = 0;
    bd::bandToIEEE(v, lo, hi);
    return bitsToDouble((static_cast<std::uint64_t>(hi) << 32) | lo);
}
// The reference is `__float128` (113 bits): a Band carries ~70 certified bits, so
// a 64-bit `long double` reference could not resolve its error and a rounded
// `double` read would hide it behind the 0.5 ULP of the store.
#if defined(__aarch64__)
// aarch64 has no __float128; long double is the same 113-bit format on Linux
using Q = long double;
#else
using Q = __float128;
#endif
inline Q qabs(Q x) { return x < 0 ? -x : x; }
inline Q qOf(Band b) { return static_cast<Q>(b.hi) + b.lo + b.tail; }
/// Error of `got` against `ref` in ULP53 of `ref` (no rounding of `got`).
inline double ulpErrQ(Q ref, Q got)
{
    if (ref == 0)
        return got == 0 ? 0.0 : 1e30;
    int e = 0;
    std::frexp(static_cast<double>(ref), &e); // ref = m 2^e, m in [0.5,1): ulp53 = 2^(e-53)
    return static_cast<double>(qabs(got - ref) / static_cast<Q>(std::ldexp(1.0, e - 53)));
}
inline double ulpErr(Q ref, double got) { return ulpErrQ(ref, static_cast<Q>(got)); }

// ---- the lazy path with a runtime state tag ------------------------------
// The typed operators of Chain.h cannot be driven by a runtime op sequence
// directly (their result TYPE changes), so the state is type-erased here and
// every step visits the exact typed overload for the tags it holds.

struct L {
    int kind; // -4: C-N (sum of normalised operands), -3: N (Band), -1: C (cancel-capable), k >= 0: R<k>
    float hi, lo, tail;
};

template<typename T>
inline L pack(T t)
{
    if constexpr (std::is_same_v<T, Band>)
        return L{ -3, t.hi, t.lo, t.tail };
    else if constexpr (std::is_same_v<T, bd::BandRawCancelN>)
        return L{ -4, t.hi, t.lo, t.tail };
    else if constexpr (bd::IsChainCancel<T>)
        return L{ -1, t.hi, t.lo, t.tail };
    else
        return L{ T::hops, t.hi, t.lo, t.tail };
}

template<typename F, int... Ks>
inline L visitR(const L& s, F& f, std::integer_sequence<int, Ks...>)
{
    L out{ -2, 0, 0, 0 };
    ((s.kind == Ks ? (out = f(bd::BandRawT<Ks>{ s.hi, s.lo, s.tail }), 0) : 0), ...);
    return out;
}
template<typename F>
inline L visit(const L& s, F f)
{
    if (s.kind == -3)
        return f(Band{ s.hi, s.lo, s.tail });
    if (s.kind == -4)
        return f(bd::BandRawCancelN{ s.hi, s.lo, s.tail });
    if (s.kind < 0)
        return f(bd::BandRawCancel{ s.hi, s.lo, s.tail });
    return visitR(s, f, std::make_integer_sequence<int, kMax + 1>{});
}
inline L lz(Band b) { return pack(b); }

enum Op { kMul = 0, kAdd = 1, kSub = 2 };
inline L lazyApply(Op op, const L& a, const L& b)
{
    return visit(a, [&](auto ta) {
        return visit(b, [&](auto tb) {
            if (op == kMul)
                return pack(ta * tb);
            if (op == kAdd)
                return pack(ta + tb);
            return pack(ta - tb);
        });
    });
}
inline Band lazyFinish(const L& s)
{
    if (s.kind == -3)
        return Band{ s.hi, s.lo, s.tail };
    if (s.kind == -4)
        return bd::normalize(bd::BandRawCancelN{ s.hi, s.lo, s.tail });
    if (s.kind < 0)
        return bd::normalizeSafe(bd::BandRawCancel{ s.hi, s.lo, s.tail });
    return bd::normalize(bd::BandRaw{ s.hi, s.lo, s.tail });
}
/// The RED-first mutant terminal: a bare limb read, no normalise.
inline Band lazyFinishMutant(const L& s) { return Band{ s.hi, s.lo, s.tail }; }

inline Band bandApply(Op op, Band a, Band b)
{
    return op == kMul ? Band(a * b) : (op == kAdd ? Band(a + b) : Band(a - b));
}

// ---- random chains --------------------------------------------------------

struct Step {
    Op op;
    bool pair;     // operand is the product of two leaves
    double x, y;   // leaf(s)
};

inline double randLeaf(std::mt19937_64& g)
{
    const std::uint64_t m = g() & ((1ull << 52) - 1);
    const int e = static_cast<int>(g() % 9) - 4;
    const double v = std::ldexp(1.0 + static_cast<double>(m) * 0x1p-52, e);
    return (g() & 1) ? -v : v;
}

struct Outcome {
    double errPer, errLazy, mutDelta;
    int nOps;
    bool kept; // magnitude filter
};

inline Outcome runChain(const std::vector<Step>& steps, double x0)
{
    Band bp   = bandOf(x0);
    L lazyAcc = lz(bp);
    Q r = x0, M = qabs(x0);
    int nOps = 0;
    for (const Step& s : steps) {
        Band yb = bandOf(s.x);
        L yl    = lz(yb);
        Q yr = s.x, yM = qabs(s.x);
        if (s.pair) {
            const Band y2 = bandOf(s.y);
            yb = yb * y2;
            yl = lazyApply(kMul, yl, lz(y2));
            yr *= s.y;
            yM *= qabs(static_cast<Q>(s.y));
            ++nOps;
        }
        bp       = bandApply(s.op, bp, yb);
        lazyAcc  = lazyApply(s.op, lazyAcc, yl);
        ++nOps;
        if (s.op == kMul) {
            r *= yr;
            M *= yM;
        } else {
            r = (s.op == kAdd) ? r + yr : r - yr;
            M += yM;
        }
    }
    Outcome o{};
    const Band lazyBand = lazyFinish(lazyAcc);
    o.errPer   = ulpErrQ(r, qOf(bp));
    o.errLazy  = ulpErrQ(r, qOf(lazyBand));
    // The mutant terminal: the unrestored limbs fed to the store codec as they
    // stand. `mutDelta` is how far (ULP53) that read lands from the normalised one.
    // A bare read the store codec itself refuses (it asserts in debug) counts as over.
    const Band mutBand = lazyFinishMutant(lazyAcc);
    o.mutDelta = bd::cell8BandIsStorable(mutBand) ? ulpErr(qOf(lazyBand), BandedReal(mutBand).toDouble()) : 2.0;
    o.nOps     = nOps;
    o.kept     = qabs(r) >= M / 2;
    return o;
}

struct Stats {
    double maxPer = 0, maxLazy = 0, sumPer = 0, sumLazy = 0;
    std::size_t n = 0, overPer = 0, overLazy = 0, mutOver = 0, rejected = 0;
};

inline Stats corpus(int len, std::size_t count, std::uint64_t seed)
{
    std::mt19937_64 g(seed + static_cast<std::uint64_t>(len) * 7919u);
    Stats st;
    while (st.n < count) {
        std::vector<Step> steps(static_cast<std::size_t>(len));
        for (Step& s : steps) {
            s.op   = static_cast<Op>(g() % 3);
            s.pair = (g() % 3) == 0;
            s.x    = randLeaf(g);
            s.y    = randLeaf(g);
        }
        const Outcome o = runChain(steps, randLeaf(g));
        if (!o.kept) {
            ++st.rejected;
            continue;
        }
        const double bound = 2.0 * o.nOps + 1.0;
        ++st.n;
        st.maxPer = std::max(st.maxPer, o.errPer);
        st.maxLazy = std::max(st.maxLazy, o.errLazy);
        st.sumPer += o.errPer;
        st.sumLazy += o.errLazy;
        st.overPer += o.errPer > bound;
        st.overLazy += o.errLazy > bound;
        st.mutOver += o.mutDelta > 1.0;
    }
    return st;
}

// ---- rows: accuracy ---------------------------------------------------------

TEST(BandChain, AccuracyLazyWithinPerOpBound)
{
    std::size_t totalMutOver = 0, total = 0;
    for (int len = 1; len <= kMax + 3; ++len) {
        const Stats st = corpus(len, 8000, 0xC4A1u);
        std::fprintf(stderr,
            "BandChain len %d  (a) per-op max %.3e mean %.3e  (b) lazy max %.3e mean %.3e  over-bound a %zu b %zu  mutant-over %zu  rejected %zu\n",
            len, st.maxPer, st.sumPer / static_cast<double>(st.n), st.maxLazy,
            st.sumLazy / static_cast<double>(st.n), st.overPer, st.overLazy, st.mutOver, st.rejected);
        EXPECT_EQ(st.overPer, 0u) << "the per-op path breaks the calibration bound: the bound is wrong";
        EXPECT_EQ(st.overLazy, 0u);
        // The per-op error (every op rounds) plus the derived raw-hop growth: a length-`len` chain holds at most
        // `2*len` raw multiplies (a pair operand adds one), split by the budget into segments of <= kMax hops.
        const double segments = static_cast<double>((2 * len + kMax - 1) / kMax);
        EXPECT_LE(st.maxLazy, st.maxPer + segments * bd::bandChainErrorBound(kMax) / 0x1p-52)
            << "lazy error is above the per-op error plus the derived raw-hop bound (ULP53)";
        totalMutOver += st.mutOver;
        total += st.n;
    }
    // RED-first: the same rows with the terminal normalise replaced by a bare
    // limb read must go red, or the rows above prove nothing.
    std::fprintf(stderr, "BandChain RED-first(bare-limb terminal): %zu/%zu chains read more than 1 ULP from the normalised terminal\n",
        totalMutOver, total);
    EXPECT_GT(totalMutOver, total / 100);
}

/// Pure raw product chains of n muls with NO intervening normalise: where does
/// the unrestored form leave the per-op bound? Evidence for `kBandChainMaxHops`.
TEST(BandChain, HopsEvidence)
{
    AETHER_SKIP_WITHOUT_WIDE_REFERENCE();
    std::mt19937_64 g(0x40B5u);
    for (int n = 1; n <= kMax + 3; ++n) {
        double maxRaw = 0, maxPer = 0;
        std::size_t over = 0;
        const std::size_t count = 8000;
        for (std::size_t it = 0; it < count; ++it) {
            const double x0 = randLeaf(g);
            Q r             = x0;
            Band per        = bandOf(x0);
            bd::BandRaw raw = bd::BandRaw(per);
            for (int i = 0; i < n; ++i) {
                const double x = randLeaf(g);
                r *= x;
                per = per * bandOf(x);
                raw = bd::mulRaw(raw, bd::BandRaw(bandOf(x)));
            }
            const double ep = ulpErrQ(r, qOf(per));
            const double er = ulpErrQ(r, qOf(bd::normalize(raw)));
            maxPer = std::max(maxPer, ep);
            maxRaw = std::max(maxRaw, er);
            over += er > 2.0 * n + 1.0;
        }
        std::fprintf(stderr, "BandChain hops %d%s  per-op max %.3e ULP  unrestored max %.3e ULP  over-bound %zu\n", n,
            n <= kMax ? " (within budget)" : " (BEYOND budget, evidence only)", maxPer, maxRaw, over);
        if (n <= kMax) {
            EXPECT_EQ(over, 0u);
            EXPECT_LE(maxRaw, bd::bandChainErrorBound(n) / 0x1p-52);
        }
    }
}

// ---- rows: specials -----------------------------------------------------------

inline bool sameOutcome(Band a, Band b, int nOps)
{
    if (bd::bandHiIsSpecial(a.hi))
        return std::memcmp(&a, &b, sizeof(Band)) == 0; // canonical special carrier, bit-identical
    if (bd::bandHiIsSpecial(b.hi))
        return false;
    const Q va = qOf(a), vb = qOf(b);
    return ulpErrQ(va, vb) <= 2.0 * nOps + 1.0;
}

TEST(BandChain, SpecialsThroughLazyChains)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 8> sp   = { inf, -inf, nan, 0.0f, -0.0f, 1.5f, -2.5f, 3.0e38f };
    const std::array<float, 4> fill = { 1.5f, -2.5f, 0.0f, 3.0e30f };
    std::size_t rows = 0, bad = 0, specialOut = 0;
    for (int n = 2; n <= 3; ++n) {
        int combos = 1;
        for (int i = 0; i < n; ++i)
            combos *= 3;
        for (int oc = 0; oc < combos; ++oc) {
            std::array<Op, 3> ops{};
            int t = oc;
            for (int i = 0; i < n; ++i, t /= 3)
                ops[static_cast<std::size_t>(i)] = static_cast<Op>(t % 3);
            for (int pos = 0; pos <= n; ++pos) { // the special enters at leaf `pos` (first ... last)
                for (float s : sp) {
                    int fills = 1;
                    for (int i = 0; i < n; ++i)
                        fills *= 4;
                    for (int fi = 0; fi < fills; ++fi) {
                        std::array<Band, 4> leaf{};
                        int f = fi;
                        for (int i = 0; i <= n; ++i) {
                            float v = fill[static_cast<std::size_t>(f % 4)];
                            if (i == pos)
                                v = s;
                            else
                                f /= 4;
                            leaf[static_cast<std::size_t>(i)] = Band{ v, 0.0f, 0.0f };
                        }
                        Band per = leaf[0];
                        L lazyAcc = lz(leaf[0]);
                        for (int i = 0; i < n; ++i) {
                            per     = bandApply(ops[static_cast<std::size_t>(i)], per, leaf[static_cast<std::size_t>(i) + 1]);
                            lazyAcc = lazyApply(ops[static_cast<std::size_t>(i)], lazyAcc, lz(leaf[static_cast<std::size_t>(i) + 1]));
                        }
                        const Band lazy = lazyFinish(lazyAcc);
                        ++rows;
                        specialOut += bd::bandHiIsSpecial(per.hi);
                        if (!sameOutcome(per, lazy, 2 * n))
                            ++bad;
                    }
                }
            }
        }
    }
    std::fprintf(stderr, "BandChain specials: %zu rows, %zu with a special outcome, %zu mismatches\n", rows,
        specialOut, bad);
    EXPECT_GT(specialOut, rows / 10); // the table really reaches the specials
    EXPECT_EQ(bad, 0u);
}

// ---- rows: types --------------------------------------------------------------

TEST(BandChain, TypeRows)
{
    const Band a{ 1.5f, 0, 0 }, b{ 2.25f, 0, 0 }, c{ 0.7f, 0, 0 };
    static_assert(std::is_same_v<decltype(a * b), bd::BandRawT<1>>);
    static_assert(std::is_same_v<decltype((a * b) * c), bd::BandRawT<2>>);
    static_assert(std::is_same_v<decltype(a + b), bd::BandRawCancelN>, "N + N: C of normalised operands");
    static_assert(std::is_same_v<decltype(a - b), bd::BandRawCancelN>);
    static_assert(std::is_same_v<decltype((a * b) + c), bd::BandRawCancel>, "an unrestored operand makes a general C");
    static_assert(std::is_same_v<decltype((a + b) * c), bd::BandRawT<1>>);
    static_assert(std::is_same_v<decltype((a + b) + c), bd::BandRawCancelN>, "C-N + N stays C-N (the add-after-add)");
    static_assert(std::is_same_v<decltype(((a * b) + c) + c), bd::BandRawCancel>);
    static_assert(std::is_same_v<decltype(a * b), bd::BandRawT<1>>, "N * N is R<1>: Band's own operator");
    static_assert(std::is_same_v<decltype(bd::chainSqr(a)), bd::BandRawT<1>>);
    static_assert(std::is_same_v<decltype(-(a * b)), bd::BandRawT<1>>);
    // The budget: an operand at kBandChainMaxHops is normalised first.
    static_assert(std::is_same_v<decltype(bd::chainMul(bd::BandRawT<kMax>{}, bd::BandRawT<kMax>{})), bd::BandRawT<1>>);
    static_assert(std::is_same_v<decltype(bd::chainMul(bd::BandRawT<kMax>{}, bd::BandRawT<1>{})), bd::BandRawT<2>>
        || kMax == 1);
    static_assert(std::is_convertible_v<bd::BandRawT<1>, Band>, "the normalising constructor is implicit");
    static_assert(std::is_convertible_v<bd::BandRawCancel, Band>);
    static_assert(std::is_convertible_v<bd::BandRawCancelN, Band>);
    static_assert(std::is_convertible_v<bd::BandRawT<1>, bd::BandRaw>);
    static_assert(!std::is_convertible_v<bd::BandRawCancel, bd::BandRaw>);
    static_assert(!std::is_convertible_v<bd::BandRawCancelN, bd::BandRaw>);
    static_assert(std::is_constructible_v<BandedReal, bd::BandRawT<2>>);
    static_assert(std::is_constructible_v<BandedReal, bd::BandRawCancel>);
    static_assert(std::is_convertible_v<bd::BandRawT<2>, BandedReal>, "the store terminal is implicit, as Band's");
    static_assert(std::is_convertible_v<bd::BandRawCancelN, BandedReal>);
    static_assert(std::is_convertible_v<bd::BandRawCancel, BandedReal>);

    // The store terminal agrees with the explicit bridge.
    const BandedReal viaCtor((a * b) * c + a);
    const BandedReal viaBridge(bd::normalizeSafe((a * b) * c + a));
    EXPECT_EQ(viaCtor.toBits(), viaBridge.toBits());
    const BandedReal viaR(a * b);
    EXPECT_EQ(viaR.toBits(), BandedReal(bd::normalize(bd::BandRaw(a * b))).toBits());
}

// ---- rows: the implicit bridges are the explicit terminals ---------------------------
//
// An R, C-N or C value converts to `Band` through its own terminal (`normalize`, `normalize`,
// `normalizeSafe`). Each row feeds the three states into a consumer that takes `Band` (or a store of it)
// and requires the result bit-identical to the same consumer on the explicit terminal.

inline bool sameBand(Band x, Band y)
{
    return std::memcmp(&x, &y, sizeof(Band)) == 0;
}

// Calls `f(implicitValue, explicitBand)` for R<1>, C-N and C over a fixed corpus of normalised leaves.
template<typename F>
inline void forEachBridge(F f)
{
    std::mt19937_64 g(0x1ceb00daULL);
    for (int i = 0; i < 400; ++i) {
        const Band a = bandOf(randLeaf(g)), b = bandOf(randLeaf(g)), c = bandOf(randLeaf(g));
        f(a * b, bd::normalize(bd::BandRaw(a * b)));
        f(a + b, bd::normalize(a + b));
        f(a - b, bd::normalize(a - b));
        f((a * b) + c, bd::normalizeSafe((a * b) + c));
    }
}

TEST(BandChain, ImplicitRawToBand)
{
    forEachBridge([](auto v, Band e) {
        const Band got = v;
        EXPECT_TRUE(sameBand(got, e));
    });
    const Band a{ 1.5f, 0, 0 }, b{ 2.25f, 0, 0 };
    const Band viaCtor(a * b);
    EXPECT_TRUE(sameBand(viaCtor, bd::normalize(bd::BandRaw(a * b))));
}

TEST(BandChain, ImplicitCancelToBand)
{
    const Band a{ 1.5f, 0, 0 }, b{ -1.5f, 0x1p-30f, 0 };
    const Band s = a + b;
    EXPECT_TRUE(sameBand(s, bd::normalize(a + b)));
    const Band t = (a * a) + b;
    EXPECT_TRUE(sameBand(t, bd::normalizeSafe((a * a) + b)));
}

TEST(BandChain, ImplicitCodec)
{
    forEachBridge([](auto v, Band e) {
        EXPECT_EQ(bd::cell8FromBand(v).w, bd::cell8FromBand(e).w);
    });
}

TEST(BandChain, ImplicitStore)
{
    forEachBridge([](auto v, Band e) {
        BandedReal s = v;
        EXPECT_EQ(s.toBits(), BandedReal(e).toBits());
        s = v;
        EXPECT_EQ(s.toBits(), BandedReal(e).toBits());
    });
}

TEST(BandChain, ImplicitAccumAddTerm)
{
    forEachBridge([](auto v, Band e) {
        aether::banded::BandAccum x, y;
        x.addTerm(v);
        y.addTerm(e);
        EXPECT_TRUE(sameBand(x.s, y.s));
    });
}

TEST(BandChain, ImplicitAccumAddTermCancel)
{
    const Band a{ 1.5f, 0, 0 }, b{ -1.5f, 0x1p-30f, 0 };
    aether::banded::BandAccum x, y;
    x.addTerm((a * a) + b);
    y.addTerm(bd::normalizeSafe((a * a) + b));
    EXPECT_TRUE(sameBand(x.s, y.s));
}

TEST(BandChain, SqrtRawConsumedUnnormalised)
{
    // sqrtRaw feeds a product with no normalise in between; the value must match the normalised-sqrt route within
    // the carrier's own rounding, and the exit of the twin alone must be bit-identical to sqrt.
    const float his[] = { 0.5f, 2.0f, 3.0f, 1e-30f, 7.25e20f, 1.0f };
    for (float h : his) {
        const Band x{ h, h * 0x1p-25f, h * 0x1p-50f };
        const Band y{ 1.5f * h, 0.0f, 0.0f };
        const bd::BandRawT<1> r = bd::sqrtRaw(x);
        EXPECT_TRUE(sameBand(bd::normalize(bd::BandRaw{ r.hi, r.lo, r.tail }), bd::sqrt_(x))) << h;
        const Band viaRaw  = bd::normalize(bd::mulRaw(bd::BandRaw{ r.hi, r.lo, r.tail }, bd::BandRaw{ y.hi, y.lo, y.tail }));
        const Band viaNorm = bd::normalize(bd::mulRaw(bd::BandRaw{ bd::sqrt_(x).hi, bd::sqrt_(x).lo, bd::sqrt_(x).tail },
            bd::BandRaw{ y.hi, y.lo, y.tail }));
        const double a = static_cast<double>(viaRaw.hi) + static_cast<double>(viaRaw.lo);
        const double b = static_cast<double>(viaNorm.hi) + static_cast<double>(viaNorm.lo);
        EXPECT_LE(std::fabs(a - b), std::ldexp(std::fabs(b), -52)) << h;
    }
    const bd::BandRawT<1> z = bd::sqrtRaw(Band{ 0.0f, 0.0f, 0.0f });
    EXPECT_EQ(z.hi, 0.0f);
    EXPECT_TRUE(std::isnan(bd::sqrtRaw(Band{ -1.0f, 0.0f, 0.0f }).hi));
    EXPECT_TRUE(std::isinf(bd::sqrtRaw(Band{ INFINITY, 0.0f, 0.0f }).hi));
}

TEST(BandChain, ImplicitOrdering)
{
    forEachBridge([](auto v, Band e) {
        const BandedReal ra(1.5);
        EXPECT_EQ(ra < v, ra < BandedReal(e));
        EXPECT_EQ(ra <= v, ra <= BandedReal(e));
        EXPECT_EQ(ra == v, ra == BandedReal(e));
    });
}

TEST(BandChain, ImplicitToIeee)
{
    forEachBridge([](auto v, Band e) {
        std::uint32_t l1 = 0, h1 = 0, l2 = 0, h2 = 0;
        bd::bandToIEEE(v, l1, h1);
        bd::bandToIEEE(e, l2, h2);
        EXPECT_EQ(l1, l2);
        EXPECT_EQ(h1, h2);
    });
}

TEST(BandChain, ImplicitFloor)
{
    forEachBridge([](auto v, Band e) { EXPECT_TRUE(sameBand(bd::floor(v), bd::floor(e))); });
}

TEST(BandChain, ImplicitFmod)
{
    const Band m{ 0.75f, 0, 0 };
    forEachBridge([&](auto v, Band e) { EXPECT_TRUE(sameBand(bd::fmod(v, m), bd::fmod(e, m))); });
}

// The facade doors take a chain state and normalise once on entry: bit-identical to the same call on the
// explicit `Band(...)`.
TEST(BandChain, FacadeDoorsAdmitChainStates)
{
    forEachBridge([](auto v, Band e) {
        const Band m{ 0.75f, 0, 0 };
        EXPECT_TRUE(sameBand(aether::math::exp(v), aether::math::exp(e)));
        EXPECT_TRUE(sameBand(aether::math::sin(v), aether::math::sin(e)));
        EXPECT_TRUE(sameBand(aether::math::atan(v), aether::math::atan(e)));
        EXPECT_TRUE(sameBand(aether::math::floor(v), aether::math::floor(e)));
        EXPECT_TRUE(sameBand(aether::math::fmod(v, m), aether::math::fmod(e, m)));
        EXPECT_TRUE(sameBand(aether::math::atan2(v, m), aether::math::atan2(e, m)));
        EXPECT_TRUE(sameBand(aether::math::fma(v, m, m), aether::math::fma(e, m, m)));
        EXPECT_TRUE(sameBand(aether::math::sqrt(v), aether::math::sqrt(e)));
    });
    const Band a{ 1.5f, 0, 0 }, b{ 2.25f, 0, 0 }, c{ 0.7f, 0, 0 };
    EXPECT_TRUE(sameBand(aether::math::exp(a * b + c), aether::math::exp(Band(a * b + c))));
    EXPECT_TRUE(sameBand(aether::math::sqrt(a * b), aether::math::sqrt(Band(a * b))));
    EXPECT_TRUE(sameBand(aether::math::sin(a - b), aether::math::sin(Band(a - b))));
    EXPECT_TRUE(sameBand((a * b) / (c + a), Band(a * b) / Band(c + a)));
    EXPECT_TRUE(sameBand((a - b) / (c * a), Band(a - b) / Band(c * a)));
    EXPECT_TRUE(sameBand((a * b) / c, Band(a * b) / c));
    EXPECT_TRUE(sameBand(c / (a * b), c / Band(a * b)));
}

TEST(BandChain, FmaHopsWithinBound)
{
    AETHER_SKIP_WITHOUT_WIDE_REFERENCE();
    // The bound is relative to the magnitude the hop works at (the product and the addend), not to a result that
    // a cancelling addend has shrunk: the error is measured against the largest intermediate of the chain.
    std::mt19937_64 g(0xF3A5u);
    double maxErr[kMax + 1] = {};
    const auto rel          = [](Q ref, Band got, Q scale) {
        const Q d = qOf(got) - ref;
        return static_cast<double>((d < 0 ? -d : d) / (qabs(scale) * 0x1p-52));
    };
    for (int it = 0; it < 8000; ++it) {
        const double x0 = randLeaf(g), x1 = randLeaf(g), x2 = randLeaf(g), x3 = randLeaf(g), c2 = randLeaf(g),
                     c3 = randLeaf(g);
        Q r         = static_cast<Q>(x0) * x1;
        Q scale     = qabs(r);
        const auto s1 = bandOf(x0) * bandOf(x1); // R<1>
        maxErr[1]     = std::max(maxErr[1], rel(r, bd::normalize(bd::BandRaw(s1)), scale));
        scale         = std::max(std::max(scale, qabs(r * x2)), qabs(static_cast<Q>(c2)));
        r             = r * x2 + c2;
        const auto s2 = bd::fma(s1, bandOf(x2), bandOf(c2)); // R<2>
        maxErr[2]     = std::max(maxErr[2], rel(r, bd::normalize(bd::BandRaw(s2)), scale));
        scale         = std::max(std::max(scale, qabs(r * x3)), qabs(static_cast<Q>(c3)));
        r             = r * x3 + c3;
        const auto s3 = bd::fma(s2, bandOf(x3), bandOf(c3)); // R<3>
        static_assert(std::is_same_v<std::remove_cv_t<decltype(s3)>, bd::BandRawT<3>>);
        maxErr[3] = std::max(maxErr[3], rel(r, bd::normalize(bd::BandRaw(s3)), scale));
    }
    for (int n = 1; n <= kMax; ++n) {
        std::fprintf(stderr, "BandChain fma hops %d  max %.3e ULP  bound %.3e\n", n, maxErr[n],
            bd::bandChainFmaErrorBound(n) / 0x1p-52);
        EXPECT_LE(maxErr[n], bd::bandChainFmaErrorBound(n) / 0x1p-52);
    }
}

TEST(BandChain, FmaChainIsExactWhereTheOperatorPathIs)
{
    // Small integers: every limb is exact, so the fused chain and the operator chain land on the same value.
    for (int a = -5; a <= 5; ++a)
        for (int b = -4; b <= 4; ++b)
            for (int c = -3; c <= 3; ++c) {
                const Band A{ static_cast<float>(a), 0, 0 }, B{ static_cast<float>(b), 0, 0 },
                    C{ static_cast<float>(c), 0, 0 };
                const Band fused = bd::normalize(bd::BandRaw(bd::fma(A * B, B, C)));
                const Band ops   = Band((A * B) * B + C);
                EXPECT_TRUE(sameBand(fused, ops));
            }
}

// ---- rows: single-limb operands, exact scaling, raw reciprocal ------------------------------

inline bool sameLimbs(float a, float b) { return a == b || (a != a && b != b); }
inline bool sameRaw(bd::BandRaw x, bd::BandRaw y)
{
    return sameLimbs(x.hi, y.hi) && sameLimbs(x.lo, y.lo) && sameLimbs(x.tail, y.tail);
}

TEST(BandChain, SingleLimbProductMatchesTheFullProduct)
{
    AETHER_SKIP_WITHOUT_WIDE_REFERENCE();
    std::mt19937_64 g(0x5A17u);
    double maxErr[kMax + 2] = {};
    for (int it = 0; it < 20000; ++it) {
        const double x0 = randLeaf(g), x1 = randLeaf(g);
        const float f   = static_cast<float>(randLeaf(g));
        const Band n    = bandOf(x0);
        const auto r1   = n * bandOf(x1); // R<1>
        // bit-identical to the full product with a one-limb multiplier (zero signs aside)
        EXPECT_TRUE(sameRaw(bd::mulRawS(bd::BandRaw(n), f), bd::mulRaw(bd::BandRaw(n), bd::BandRaw{ f, 0.0f, 0.0f })));
        EXPECT_TRUE(sameRaw(bd::mulRawS(bd::BandRaw(r1), f), bd::mulRaw(bd::BandRaw(r1), bd::BandRaw{ f, 0.0f, 0.0f })));
        const auto sN = bd::mulRawS(n, f);
        const auto sR = bd::mulRawS(r1, f);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(sN)>, bd::BandRawT<1>>);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(sR)>, bd::BandRawT<2>>);
        const Q refN = static_cast<Q>(x0) * f;
        const Q refR = static_cast<Q>(x0) * x1 * f;
        maxErr[1]    = std::max(maxErr[1], ulpErrQ(refN, qOf(bd::normalize(bd::BandRaw(sN)))));
        maxErr[2]    = std::max(maxErr[2], ulpErrQ(refR, qOf(bd::normalize(bd::BandRaw(sR)))));
    }
    for (int k = 1; k <= 2; ++k)
        EXPECT_LE(maxErr[k], bd::bandChainErrorBound(k) / 0x1p-52);
}

TEST(BandChain, SingleLimbSumMatchesTheFullSum)
{
    std::mt19937_64 g(0xADD5u);
    for (int it = 0; it < 20000; ++it) {
        const Band n  = bandOf(randLeaf(g));
        const auto r1 = n * bandOf(randLeaf(g));
        const float f = static_cast<float>(randLeaf(g));
        const bd::BandRaw one{ f, 0.0f, 0.0f };
        for (const bd::BandRaw x : { bd::BandRaw(n), bd::BandRaw(r1) }) {
            const bd::BandRawCancel s = bd::addRawS(x, f), w = bd::addRaw(x, one);
            EXPECT_TRUE(sameRaw(bd::BandRaw{ s.hi, s.lo, s.tail }, bd::BandRaw{ w.hi, w.lo, w.tail }));
        }
    }
    const Band a{ 1.5f, 0, 0 };
    static_assert(std::is_same_v<decltype(bd::addRawS(a, 1.0f)), bd::BandRawCancelN>);
    static_assert(std::is_same_v<decltype(bd::addRawS(a * a, 1.0f)), bd::BandRawCancel>);
}

TEST(BandChain, ScalePow2OnRawIsExactAndKeepsTheState)
{
    std::mt19937_64 g(0x5CA1Eu);
    for (int it = 0; it < 20000; ++it) {
        const auto r = bandOf(randLeaf(g)) * bandOf(randLeaf(g)); // R<1>
        const float s = std::ldexp(1.0f, static_cast<int>(g() % 9) - 4);
        const auto t  = bd::scalePow2f(r, s);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(t)>, bd::BandRawT<1>>);
        EXPECT_EQ(t.hi, r.hi * s);
        EXPECT_EQ(t.lo, r.lo * s);
        EXPECT_EQ(t.tail, r.tail * s);
        // the scaled value is exactly the value times s
        EXPECT_TRUE(qOf(Band{ t.hi, t.lo, t.tail }) == qOf(Band{ r.hi, r.lo, r.tail }) * s);
    }
}

TEST(BandChain, RecipRawAcceptsRawOperands)
{
    AETHER_SKIP_WITHOUT_WIDE_REFERENCE();
    std::mt19937_64 g(0x4EC1Bu);
    double maxN = 0, maxR = 0;
    for (int it = 0; it < 20000; ++it) {
        const double x0 = randLeaf(g), x1 = randLeaf(g);
        const Band n    = bandOf(x0);
        const auto r1   = n * bandOf(x1);
        maxN = std::max(maxN, ulpErrQ(1 / static_cast<Q>(x0), qOf(bd::normalize(bd::recipRaw(bd::BandRaw(n))))));
        maxR = std::max(maxR,
            ulpErrQ(1 / (static_cast<Q>(x0) * x1), qOf(bd::normalize(bd::recipRaw(bd::BandRaw(r1))))));
    }
    std::fprintf(stderr, "BandChain recipRaw max ULP53: N %.3e  R<1> %.3e\n", maxN, maxR);
    EXPECT_LE(maxN, 1e-3);
    EXPECT_LE(maxR, 1e-3);
}

// ---- raw-exit twins: the barrier-free op, normalised by its consumer, is the certified op -----------------

// bit-identical, except that two NaN carriers, or two zeros of either sign, count as equal: the barrier does not
// keep the sign of a zero limb set, and the payload of an invalid result is not a value
inline bool sameOrBothNaN(Band x, Band y)
{
    return sameBand(x, y) || (x.hi != x.hi && y.hi != y.hi)
        || (x.hi == 0.0f && y.hi == 0.0f && x.lo == 0.0f && y.lo == 0.0f && x.tail == 0.0f && y.tail == 0.0f);
}

TEST(BandChain, RawExitTwinsNormaliseToTheFacadeResult)
{
    std::mt19937_64 g(0x7A17u);
    for (int it = 0; it < 4000; ++it) {
        const Band x = bandOf(randLeaf(g) * 0.7), y = bandOf(randLeaf(g));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::sinRaw(x))), bd::sin(x)));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::cosRaw(x))), bd::cos(x)));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::expRaw(x))), bd::exp(x)));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::divRaw(bd::BandRaw(x), y))), bd::div(x, y)));
        EXPECT_TRUE(sameOrBothNaN(bd::div(bd::BandRaw(x), y), bd::div(x, y)));
    }
    // the specials and the exact points come through the twins unchanged
    const float sp[] = { 0.0f, -0.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), 1.0f, -1.0f, 100.0f, -100.0f };
    for (float a : sp) {
        const Band x{ a, 0.0f, 0.0f };
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::sinRaw(x))), bd::sin(x)));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::cosRaw(x))), bd::cos(x)));
        EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::expRaw(x))), bd::exp(x)));
        for (float b : sp) {
            const Band y{ b, 0.0f, 0.0f };
            EXPECT_TRUE(sameOrBothNaN(bd::normalize(bd::BandRaw(bd::divRaw(bd::BandRaw(x), y))), bd::div(x, y)));
        }
    }
}

// A raw twin keeps the sign of an exact zero (a sign reader can see it before any barrier): sin(+-0) = +-0, and
// 0 / b has the sign of 0 times the sign of b.
TEST(BandChain, RawExitTwinsKeepTheSignOfAnExactZero)
{
    const float zs[] = { 0.0f, -0.0f };
    const float bs[] = { 1.0f, -1.0f, 3.5f, -0.25f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity() };
    for (float z : zs) {
        EXPECT_EQ(std::signbit(bd::sinRaw(Band{ z, 0.0f, 0.0f }).hi), std::signbit(z));
        EXPECT_EQ(bd::sinRaw(Band{ z, 0.0f, 0.0f }).hi, 0.0f);
        EXPECT_FALSE(std::signbit(bd::cosRaw(Band{ z, 0.0f, 0.0f }).hi));
        for (float b : bs) {
            const auto q = bd::divRaw(bd::BandRaw{ z, 0.0f, 0.0f }, Band{ b, 0.0f, 0.0f });
            EXPECT_EQ(q.hi, 0.0f);
            EXPECT_EQ(std::signbit(q.hi), std::signbit(z) != std::signbit(b)) << z << " / " << b;
        }
    }
}

TEST(BandChain, RawExitTwinsFeedTheChainWithinTheBound)
{
    AETHER_SKIP_WITHOUT_WIDE_REFERENCE();
    std::mt19937_64 g(0x7B17u);
    double worst[4] = {};
    const auto rel = [](Q ref, const bd::BandRaw& got) {
        const Q d = qOf(Band{ got.hi, got.lo, got.tail }) - ref;
        return static_cast<double>((d < 0 ? -d : d) / (qabs(ref) * 0x1p-52));
    };
    for (int it = 0; it < 3000; ++it) {
        const double xv = randLeaf(g) * 0.5, yv = randLeaf(g), cv = randLeaf(g);
        const Band x = bandOf(xv), y = bandOf(yv), c = bandOf(cv);
        // twin -> `-` , `*` and scalePow2f; the reference is the facade value (a normalised Band) taken exactly
        const Q sinq = qOf(bd::sin(x)), expq = qOf(bd::exp(x)), divq = qOf(bd::div(x, y));
        const auto s = bd::sinRaw(x) - c;                  // C
        worst[0]     = std::max(worst[0], rel(sinq - qOf(c), bd::BandRaw{ s.hi, s.lo, s.tail }));
        const auto e = bd::expRaw(x) * c;                  // R<2>
        worst[1]     = std::max(worst[1], rel(expq * qOf(c), bd::BandRaw{ e.hi, e.lo, e.tail }));
        const auto d = bd::scalePow2f(bd::divRaw(bd::BandRaw(x), y), 0.25f); // R<1>
        worst[2]     = std::max(worst[2], rel(divq * 0.25, bd::BandRaw{ d.hi, d.lo, d.tail }));
        const auto q = (x * y) / c;                        // chain division: R<1>
        static_assert(std::is_same_v<std::remove_cv_t<decltype(q)>, bd::BandRawT<1>>);
        worst[3] = std::max(worst[3], rel(qOf(x) * qOf(y) / qOf(c), bd::BandRaw{ q.hi, q.lo, q.tail }));
    }
    for (int i = 0; i < 4; ++i) {
        std::fprintf(stderr, "BandChain raw-exit chain %d: max %.3e ULP53\n", i, worst[i]);
        EXPECT_LE(worst[i], 0.5 + bd::bandChainErrorBound(kMax) / 0x1p-52 * 4);
    }
}

// ---- rows: host / device ----------------------------------------------------------

AETHER_DEVICEHOST() inline Band lazyExpr(Band a, Band b, Band c, Band d, Band e)
{
    // ((a*b)*c + d) * e - a*b: R2, C, R1, C
    const auto p = a * b;
    const auto s = ((p * c) + d) * e;
    return bd::normalizeSafe(s - p);
}

#ifdef __CUDACC__
__global__ void lazyKernel(const Band* in, Band* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = lazyExpr(in[5 * i], in[5 * i + 1], in[5 * i + 2], in[5 * i + 3], in[5 * i + 4]);
}
#endif

TEST(BandChain, HostDeviceAgree)
{
    constexpr int n = 4096;
    std::mt19937_64 g(0xD3Fu);
    std::vector<Band> in(5 * n), host(n);
    for (Band& v : in)
        v = bandOf(randLeaf(g));
    for (int i = 0; i < n; ++i)
        host[static_cast<std::size_t>(i)] = lazyExpr(in[5 * i], in[5 * i + 1], in[5 * i + 2], in[5 * i + 3], in[5 * i + 4]);
#ifdef __CUDACC__
    std::vector<Band> dev(n);
    Band *din = nullptr, *dout = nullptr;
    ASSERT_EQ(cudaMalloc(&din, in.size() * sizeof(Band)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, dev.size() * sizeof(Band)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(din, in.data(), in.size() * sizeof(Band), cudaMemcpyHostToDevice), cudaSuccess);
    lazyKernel<<<(n + 127) / 128, 128>>>(din, dout, n);
    ASSERT_EQ(cudaMemcpy(dev.data(), dout, dev.size() * sizeof(Band), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(din);
    cudaFree(dout);
    std::size_t identical = 0;
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        const Band h = host[static_cast<std::size_t>(i)], d = dev[static_cast<std::size_t>(i)];
        identical += std::memcmp(&h, &d, sizeof(Band)) == 0;
        const double e = ulpErrQ(qOf(h), qOf(d));
        worst = std::max(worst, e);
        EXPECT_LE(e, 1.0);
    }
    std::fprintf(stderr, "BandChain host/device: %zu/%d bit-identical, worst %.3e ULP\n", identical, n, worst);
#else
    // CPP_MODE: the host twin against the long double reference of the same expression.
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        auto v = [&](int k) { return qOf(in[5 * i + k]); };
        const Q ab = v(0) * v(1);
        const Q r  = (ab * v(2) + v(3)) * v(4) - ab;
        const Q M  = qabs(ab * v(2) * v(4)) + qabs(v(3) * v(4)) + qabs(ab);
        if (qabs(r) >= M / 2)
            worst = std::max(worst, ulpErrQ(r, qOf(host[static_cast<std::size_t>(i)])));
    }
    std::fprintf(stderr, "BandChain host twin vs long double: worst %.3e ULP\n", worst);
    EXPECT_LE(worst, 2.0 * 5 + 1.0);
#endif
}

} // namespace BandChainTest
} // namespace aether_tests
