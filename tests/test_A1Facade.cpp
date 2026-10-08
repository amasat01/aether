// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A1 tests (host / AETHER_CPP_MODE
// build). Paired with test_A1Facade.cu (identical content, house convention —
// see tests/test_Math.cpp/.cu). RED-FIRST: previously, `aether::math::
// tan`/`tanh` did not exist (a probe TU calling `aether::math::tan(1.0)` did
// not compile),
// `aether::cwiseSqrt`/... etc. and `aether::select` did not exist, and
// `aether::Item<T,1>{v}`/`aether::Item<T>{v}` had no viable constructor (the
// N-scalar ctor in `aether/view/Item.h` is guarded `Size >= 2`).
//
// Covers, one `TEST_F` group per HAWK requirement (`hawk/emit/aether.py`):
//   (1) — the two NEW `aether::math` facade entries, `tan`/`tanh`
//         (`aether/math/trig.h`).
//   (2) — rank>=1 ELEMENTWISE forms: the unary transcendentals HAWK's own
//         `_MATH1` table recognizes (`sqrt`,`rsqrt`,`exp`,`log`,`sin`,
//         `cos`,`asin`,`acos`,`atan`) plus the new `tan`/`tanh`;
//         two-tensor `div`/`pow`/`min`/`max`; and the scalar-exponent
//         `pow` overload (`aether/expr/nodes/Elementwise.h`).
//   (3) — `aether::select(cond, a, b)`, both the per-SAMPLE (rank-0
//         `cond`) and per-ELEMENT (matching-shape `cond`) forms, plus a
//         a rank-generic sanity check at rank-0.
//   (4) — the 1-component `Item<T,1>`/rank-0 `Item<T>` constructor
//         (`aether/view/Item.h`).
//
// TOLERANCE NOTE (mirrors test_MathExpLog.cpp's own note for exactly this
// situation): every new/exercised facade entry is a THIN WRAPPER over
// `std::` on the host route (no alternate formula, no emulation arm — this
// this suite adds no SoftDouble/Banded backend), so every check below is BIT-EXACT
// (`EXPECT_DOUBLE_EQ`/`EXPECT_FLOAT_EQ`, NaN compared via `std::isnan` since
// NaN != NaN), never a tolerance — a stronger check than any epsilon, and it
// is also what lets a single corpus safely include INVALID-DOMAIN entries
// (e.g. `sqrt(-1)`, `asin(2)`, `log(-1)`): whatever `std::` produces for such
// an input (typically NaN) is exactly what aether's thin wrapper produces
// too, so no domain restriction is needed per function — one corpus serves
// every unary entry point.
//
// The rank>=1 (ET) checks double as a "lazy-evaluation identity" check:
// `expected` is built by calling the SAME `std::`/aether
// scalar function on each component DIRECTLY (never through the ET
// machinery under test), mirroring test_ExprElementwise.cpp's own
// established convention (`EXPECT_DOUBLE_EQ(out(0), std::fmin(a(0), 1.0))`).

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Mat33d;
using aether::Vec3d;

class A1FacadeTest : public ::testing::Test { };

// -- shared corpus + comparator ---------------------------------------------

// The special-value corpus every unary check below sweeps: "0, ±denormal,
// ±1, large, ±inf, nan" verbatim (9 entries).
template<class T>
std::array<T, 9> Corpus()
{
    return { T(0), std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(), T(1), T(-1),
        std::numeric_limits<T>::max() / T(4), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::quiet_NaN() };
}

// Nine hand-picked BINARY special-value pairs for div/pow/min/max: normal,
// positive/negative div-by-zero, 0/0, inf/finite, finite/inf, inf/inf, NaN
// propagation, and a denormal operand.
template<class T>
std::array<T, 9> BinaryCorpusA()
{
    return { T(2), T(1), T(-1), T(0), std::numeric_limits<T>::infinity(), T(1), std::numeric_limits<T>::infinity(),
        std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::denorm_min() };
}
template<class T>
std::array<T, 9> BinaryCorpusB()
{
    return { T(3), T(0), T(0), T(0), T(1), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), T(1),
        T(2) };
}

template<class T>
void ExpectEq(T got, T want)
{
    if (std::isnan(want)) {
        EXPECT_TRUE(std::isnan(got));
        return;
    }
    if constexpr (std::is_same_v<T, double>) {
        EXPECT_DOUBLE_EQ(got, want);
    } else {
        static_assert(std::is_same_v<T, float>, "ExpectEq: T must be float or double");
        EXPECT_FLOAT_EQ(got, want);
    }
}

template<class T, class AFn, class SFn>
void ExpectScalarCorpusMatchesStd(AFn afn, SFn sfn)
{
    for (T x : Corpus<T>())
        ExpectEq<T>(afn(x), sfn(x));
}

// Sweeps the corpus THREE elements at a time (9 / 3 == 0 remainder) building
// an `Item<T,3>`, applies `cwiseFn`, and checks every component against
// `sfn` computed DIRECTLY per component (the lazy-evaluation identity).
template<class T, class CwiseFn, class SFn>
void ExpectCwiseUnaryCorpusMatchesStd(CwiseFn cwiseFn, SFn sfn)
{
    const std::array<T, 9> corpus = Corpus<T>();
    for (std::size_t base = 0; base < corpus.size(); base += 3) {
        aether::Item<T, 3> in{ corpus[base], corpus[base + 1], corpus[base + 2] };
        aether::Item<T, 3> out = cwiseFn(in);
        aether::Item<T, 3> expected{ sfn(in(0)), sfn(in(1)), sfn(in(2)) };
        for (std::size_t c = 0; c < 3; ++c)
            ExpectEq<T>(out(c), expected(c));
    }
}

template<class T, class CwiseFn, class SFn>
void ExpectCwiseBinaryCorpusMatchesStd(CwiseFn cwiseFn, SFn sfn)
{
    const std::array<T, 9> a = BinaryCorpusA<T>();
    const std::array<T, 9> b = BinaryCorpusB<T>();
    for (std::size_t base = 0; base < a.size(); base += 3) {
        aether::Item<T, 3> va{ a[base], a[base + 1], a[base + 2] };
        aether::Item<T, 3> vb{ b[base], b[base + 1], b[base + 2] };
        aether::Item<T, 3> out = cwiseFn(va, vb);
        aether::Item<T, 3> expected{ sfn(va(0), vb(0)), sfn(va(1), vb(1)), sfn(va(2), vb(2)) };
        for (std::size_t c = 0; c < 3; ++c)
            ExpectEq<T>(out(c), expected(c));
    }
}

// =============================================================================
// (1) — aether::math::tan / tanh (aether/math/trig.h)
// =============================================================================

TEST_F(A1FacadeTest, TanMatchesStdIncludingSpecialValues)
{
    ExpectScalarCorpusMatchesStd<double>(
        [](double x) { return aether::math::tan(x); }, [](double x) { return std::tan(x); });
    ExpectScalarCorpusMatchesStd<float>(
        [](float x) { return aether::math::tan(x); }, [](float x) { return std::tan(x); });
}

TEST_F(A1FacadeTest, TanhMatchesStdIncludingSpecialValues)
{
    ExpectScalarCorpusMatchesStd<double>(
        [](double x) { return aether::math::tanh(x); }, [](double x) { return std::tanh(x); });
    ExpectScalarCorpusMatchesStd<float>(
        [](float x) { return aether::math::tanh(x); }, [](float x) { return std::tanh(x); });
}

// =============================================================================
// (2) — rank>=1 unary transcendentals (aether/expr/nodes/Elementwise.h)
// =============================================================================

TEST_F(A1FacadeTest, CwiseSqrtMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseSqrt(e); }, [](double x) { return std::sqrt(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseSqrt(e); }, [](float x) { return std::sqrt(x); });
}

TEST_F(A1FacadeTest, CwiseRsqrtMatchesStdAndElementwiseLoop)
{
    // No `std::rsqrt` — mirrors `aether::math::rsqrt`'s own host route
    // (`aether/math/root.h`): `T(1) / std::sqrt(x)`.
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseRsqrt(e); }, [](double x) { return double(1) / std::sqrt(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseRsqrt(e); }, [](float x) { return float(1) / std::sqrt(x); });
}

TEST_F(A1FacadeTest, CwiseExpMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseExp(e); }, [](double x) { return std::exp(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseExp(e); }, [](float x) { return std::exp(x); });
}

TEST_F(A1FacadeTest, CwiseLogMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseLog(e); }, [](double x) { return std::log(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseLog(e); }, [](float x) { return std::log(x); });
}

TEST_F(A1FacadeTest, CwiseSinMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseSin(e); }, [](double x) { return std::sin(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseSin(e); }, [](float x) { return std::sin(x); });
}

TEST_F(A1FacadeTest, CwiseCosMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseCos(e); }, [](double x) { return std::cos(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseCos(e); }, [](float x) { return std::cos(x); });
}

TEST_F(A1FacadeTest, CwiseAsinMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseAsin(e); }, [](double x) { return std::asin(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseAsin(e); }, [](float x) { return std::asin(x); });
}

TEST_F(A1FacadeTest, CwiseAcosMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseAcos(e); }, [](double x) { return std::acos(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseAcos(e); }, [](float x) { return std::acos(x); });
}

TEST_F(A1FacadeTest, CwiseAtanMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseAtan(e); }, [](double x) { return std::atan(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseAtan(e); }, [](float x) { return std::atan(x); });
}

TEST_F(A1FacadeTest, CwiseTanMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseTan(e); }, [](double x) { return std::tan(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseTan(e); }, [](float x) { return std::tan(x); });
}

TEST_F(A1FacadeTest, CwiseTanhMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwiseTanh(e); }, [](double x) { return std::tanh(x); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwiseTanh(e); }, [](float x) { return std::tanh(x); });
}

// =============================================================================
// (2b) — two-tensor div/pow/min/max + scalar-exponent pow
// =============================================================================

TEST_F(A1FacadeTest, CwiseDivMatchesStdAndElementwiseLoopIncludingDivByZero)
{
    ExpectCwiseBinaryCorpusMatchesStd<double>(
        [](const auto& l, const auto& r) { return aether::cwiseDiv(l, r); }, [](double a, double b) { return a / b; });
    ExpectCwiseBinaryCorpusMatchesStd<float>(
        [](const auto& l, const auto& r) { return aether::cwiseDiv(l, r); }, [](float a, float b) { return a / b; });
}

TEST_F(A1FacadeTest, CwisePowScalarExponentMatchesStdAndElementwiseLoop)
{
    ExpectCwiseUnaryCorpusMatchesStd<double>(
        [](const auto& e) { return aether::cwisePow(e, 3.0); }, [](double x) { return std::pow(x, 3.0); });
    ExpectCwiseUnaryCorpusMatchesStd<float>(
        [](const auto& e) { return aether::cwisePow(e, 3.0f); }, [](float x) { return std::pow(x, 3.0f); });
}

TEST_F(A1FacadeTest, CwisePowTwoTensorMatchesStdAndElementwiseLoop)
{
    ExpectCwiseBinaryCorpusMatchesStd<double>(
        [](const auto& l, const auto& r) { return aether::cwisePow(l, r); },
        [](double a, double b) { return std::pow(a, b); });
    ExpectCwiseBinaryCorpusMatchesStd<float>(
        [](const auto& l, const auto& r) { return aether::cwisePow(l, r); },
        [](float a, float b) { return std::pow(a, b); });
}

TEST_F(A1FacadeTest, CwiseMinTwoTensorMatchesStdAndElementwiseLoop)
{
    ExpectCwiseBinaryCorpusMatchesStd<double>(
        [](const auto& l, const auto& r) { return aether::cwiseMin(l, r); },
        [](double a, double b) { return std::fmin(a, b); });
    ExpectCwiseBinaryCorpusMatchesStd<float>(
        [](const auto& l, const auto& r) { return aether::cwiseMin(l, r); },
        [](float a, float b) { return std::fmin(a, b); });
}

TEST_F(A1FacadeTest, CwiseMaxTwoTensorMatchesStdAndElementwiseLoop)
{
    ExpectCwiseBinaryCorpusMatchesStd<double>(
        [](const auto& l, const auto& r) { return aether::cwiseMax(l, r); },
        [](double a, double b) { return std::fmax(a, b); });
    ExpectCwiseBinaryCorpusMatchesStd<float>(
        [](const auto& l, const auto& r) { return aether::cwiseMax(l, r); },
        [](float a, float b) { return std::fmax(a, b); });
}

TEST_F(A1FacadeTest, CwiseMinMaxScalarAndTwoTensorOverloadsCoexistUnambiguously)
{
    // The two-tensor `cwiseMin`/`cwiseMax` OVERLOAD the pre-existing
    // scalar-bound free functions (aether/expr/nodes/Elementwise.h) —
    // regression check that both remain independently reachable.
    Vec3d a{ -2.0, 0.5, 5.0 };
    Vec3d b{ 1.0, 1.0, 1.0 };
    Vec3d minScalar = aether::cwiseMin(a, 1.0);
    Vec3d minTensor = aether::cwiseMin(a, b);
    Vec3d maxScalar = aether::cwiseMax(a, 1.0);
    Vec3d maxTensor = aether::cwiseMax(a, b);
    for (std::size_t c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(minScalar(c), std::fmin(a(c), 1.0));
        EXPECT_DOUBLE_EQ(minTensor(c), std::fmin(a(c), b(c)));
        EXPECT_DOUBLE_EQ(maxScalar(c), std::fmax(a(c), 1.0));
        EXPECT_DOUBLE_EQ(maxTensor(c), std::fmax(a(c), b(c)));
    }
}

// =============================================================================
// (3) — aether::select(cond, a, b)
// =============================================================================

TEST_F(A1FacadeTest, SelectPerSampleChoosesWholeBranch)
{
    Vec3d a{ 1.0, 2.0, 3.0 };
    Vec3d b{ -1.0, -2.0, -3.0 };
    aether::Item<bool> condTrue{ true };
    aether::Item<bool> condFalse{ false };

    Vec3d selTrue  = aether::select(condTrue, a, b);
    Vec3d selFalse = aether::select(condFalse, a, b);
    for (std::size_t c = 0; c < 3; ++c) {
        EXPECT_DOUBLE_EQ(selTrue(c), a(c));
        EXPECT_DOUBLE_EQ(selFalse(c), b(c));
    }
}

TEST_F(A1FacadeTest, SelectPerElementChoosesComponentwise)
{
    Vec3d a{ 1.0, 2.0, 3.0 };
    Vec3d b{ -1.0, -2.0, -3.0 };
    aether::Item<bool, 3> mask{ true, false, true };
    Vec3d out = aether::select(mask, a, b);
    EXPECT_DOUBLE_EQ(out(0), a(0));
    EXPECT_DOUBLE_EQ(out(1), b(1));
    EXPECT_DOUBLE_EQ(out(2), a(2));
}

TEST_F(A1FacadeTest, SelectWorksOnARank0ScalarToo)
{
    // ONE rank-generic operator set: the SAME `select` used above on a
    // rank-1 Vec3d must also work, unmodified, on a rank-0 scalar Item.
    aether::Item<double> a{ 5.0 };
    aether::Item<double> b{ -5.0 };
    aether::Item<bool> condTrue{ true };
    aether::Item<double> out = aether::select(condTrue, a, b);
    EXPECT_DOUBLE_EQ(out(), 5.0);
}

// =============================================================================
// (4) — the 1-component Item<T,1> / rank-0 Item<T> constructor
// =============================================================================

TEST_F(A1FacadeTest, Item1ComponentVecConstructorRoundTripsSpecialValues)
{
    for (double x : Corpus<double>()) {
        aether::Item<double, 1> v{ x };
        if (std::isnan(x)) {
            EXPECT_TRUE(std::isnan(v(0)));
        } else {
            EXPECT_DOUBLE_EQ(v(0), x);
        }
    }
    for (float x : Corpus<float>()) {
        aether::Item<float, 1> v{ x };
        if (std::isnan(x)) {
            EXPECT_TRUE(std::isnan(v(0)));
        } else {
            EXPECT_FLOAT_EQ(v(0), x);
        }
    }
}

TEST_F(A1FacadeTest, ItemRank0SingleScalarConstructorRoundTripsSpecialValues)
{
    for (double x : Corpus<double>()) {
        aether::Item<double> v{ x };
        if (std::isnan(x)) {
            EXPECT_TRUE(std::isnan(v()));
        } else {
            EXPECT_DOUBLE_EQ(v(), x);
        }
    }
}

TEST_F(A1FacadeTest, Item1ComponentVecIsAFirstClassExpression)
{
    // Not just constructible — genuinely usable as an `aether_expression`:
    // the SAME `cwiseAbs` used elsewhere on `Vec3d` also works, unmodified,
    // on a 1-component `Item`.
    aether::Item<double, 1> v{ -5.0 };
    aether::Item<double, 1> out = aether::cwiseAbs(v);
    EXPECT_DOUBLE_EQ(out(0), 5.0);
}

// =============================================================================
// Rank-generic sanity: the SAME new ops used above on Vec3d also work,
// unmodified, on a rank-2 Mat33d (mirrors test_ExprElementwise.cpp's own
// RankGenericWorksOnAMatrixItemToo).
// =============================================================================

TEST_F(A1FacadeTest, RankGenericA1OpsWorkOnAMatrixItemToo)
{
    Mat33d a = Mat33d::Zeros();
    Mat33d b = Mat33d::Zeros();
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            a(r, c) = static_cast<double>(r) - static_cast<double>(c) * 2.0 + 1.0; // never zero
            b(r, c) = static_cast<double>(r + c) + 1.0;                             // never zero
        }
    }
    Mat33d expOut = aether::cwiseExp(a);
    Mat33d tanOut = aether::cwiseTan(a);
    Mat33d divOut = aether::cwiseDiv(a, b);
    Mat33d minOut = aether::cwiseMin(a, b);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(expOut(r, c), std::exp(a(r, c)));
            EXPECT_DOUBLE_EQ(tanOut(r, c), std::tan(a(r, c)));
            EXPECT_DOUBLE_EQ(divOut(r, c), a(r, c) / b(r, c));
            EXPECT_DOUBLE_EQ(minOut(r, c), std::fmin(a(r, c), b(r, c)));
        }
    }
}

} // namespace
} // namespace aether_tests
