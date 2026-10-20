// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file test_CCore_common.h
 * @brief `CCore` suite: aether's C reference core
 *        (`dsc/aether_dsc/c_core.py`, minted into
 *        `tests/c_core/aether_c_corpus.h` by `tools/c_core/mint_golden.py`)
 *        pinned bit for bit against the expression templates it transcribes.
 *
 * Every corpus function of both widths (`aether_c_f64_*`, `aether_c_f32_*`)
 * is called on C structs converted from `aether::Item`s, and its result is
 * compared with the aether expression of the same kind evaluated at
 * `SampleIndex::make(0)` into an `Item` (or a scalar for the folds and the
 * rank-0 functions). The aether spellings are the ones hawk emits (the
 * "Bit-identity formula table" of
 * docs/superpowers/specs/2026-10-16-embedded-c-target-design.md): `neg` is
 * `x * T(-1)`, the rank-2 `sum` is `m.matDot(ones)`, `div_s` is `x / s`.
 * The comparison is per component, `std::memcmp` exact or both NaN (aether
 * may flip a NaN's sign where the C multiplies by -1). Out-of-domain and
 * FP-exception inputs are compared too, never skipped.
 *
 * The corpus (`Corpus`): the specials `NaN, +inf, -inf, +0, -0, the smallest
 * denormal (`denorm_min()`), the smallest normal (`min()`), `max()`, 1, -1,
 * 0.5, 3`, plus in-domain values `+-0.1, +-0.75, +-2.5, +-1e-3` (so
 * asin/acos/tanh/log/sqrt/exp/pow see inputs the spread, uniform in
 * [-1e3, 1e3], never lands on), or 64 values from
 * `std::mt19937_64(20261004)` uniform in `[-1e3, 1e3]`. An input is a pick
 * `(i, j, k)` of three start indices: operand `a` is the sliding window of
 * the values starting at `i` (one value per component, row-major, wrapping),
 * `b` the window at `j`, the auxiliary vector `v` (`mat_mv`'s vector,
 * `quat_rotate`'s 3-vector, `quat_as_pure`'s source) the window at `k`, and
 * the scalar `s` (`scale`/`div_s`'s scalar, `clip`'s `hi`) is `values[k]`.
 * The spread picks `(i, i+1, i+2)` for its 64 windows (window i paired with
 * window i+1); the specials pick every triple, so every special meets every
 * special as a scalar binary (and `clip` sees every `(x, lo, hi)`).
 *
 * The C function names are pasted by `AC(w, s, k)` / `AS(w, k)` from the
 * per-family kind tables (`CCORE_*_KINDS`, one row per kind: C arguments and
 * the aether expression); one generic checker (`checkKind`) runs every row.
 * Per-file `-ffp-contract=off` (tests/CMakeLists.txt) keeps the compiler from
 * fusing either side.
 *
 * Identity holds only for a C translation unit compiled with
 * `-ffp-contract=off` and against the templates' default `std::` host route
 * (no `AETHER_HOST_VECTOR_MATH`, no fast-math).
 */

#include <math.h>

#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <aether/expr/nodes/Geometric.h>
#include <aether/expr/nodes/Inverse.h>
#include <aether/expr/nodes/Product.h>
#include <aether/expr/nodes/Quaternion.h>
#include <aether/expr/nodes/Structural.h>
#include <aether/expr/Reduce.h>
#include <aether/math/math.h>
#include <aether/view/Item.h>

#include "c_core/aether_c_corpus.h"

/// A shaped corpus function: `aether_c_<w>_<s>_<k>` (e.g. `AC(f64, vec3, dot)`,
/// `AC(f32, mat2x3, mm_mat3x2)`, `AC(f64, quat, rotate)`).
#define AC(w, s, k) aether_c_##w##_##s##_##k
/// A rank-0 corpus function: `aether_c_<w>_<k>`.
#define AS(w, k) aether_c_##w##_##k

namespace aether_tests {
namespace ccore {

/// Same bits, or both NaN (aether's `-x` and the C's `x * -1` may differ in a NaN's sign).
template<class T>
bool sameBits(T a, T b)
{
    static_assert(std::is_floating_point_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0 || (std::isnan(a) && std::isnan(b));
}

// ---- corpus ---------------------------------------------------------------

template<class T>
struct Corpus {
    const char* name;
    std::vector<T> values;
    std::vector<std::array<std::size_t, 3>> picks; // (i, j, k) window starts
};

template<class T>
Corpus<T> specials()
{
    Corpus<T> c{ "specials", { std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
                                 -std::numeric_limits<T>::infinity(), T(0), -T(0), std::numeric_limits<T>::denorm_min(),
                                 std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), T(1), T(-1),
                                 T(0.5), T(3), T(0.1), T(-0.1), T(0.75), T(-0.75), T(2.5), T(-2.5), T(1e-3),
                                 T(-1e-3) },
        {} };
    const std::size_t n = c.values.size();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                c.picks.push_back({ i, j, k });
    return c;
}

template<class T>
Corpus<T> spread()
{
    constexpr std::size_t n = 64;
    Corpus<T> c{ "spread", {}, {} };
    std::mt19937_64 g(20261004);
    std::uniform_real_distribution<T> u(T(-1e3), T(1e3));
    for (std::size_t i = 0; i < n; ++i)
        c.values.push_back(u(g));
    for (std::size_t i = 0; i < n; ++i)
        c.picks.push_back({ i, (i + 1) % n, (i + 2) % n });
    return c;
}

/// The sliding window of `values` starting at `start`: a scalar, or one value
/// per component in row-major order, wrapping.
template<class X, class T>
X window(const std::vector<T>& values, std::size_t start)
{
    if constexpr (std::is_arithmetic_v<X>) {
        return values[start % values.size()];
    } else {
        X x{};
        for (std::size_t c = 0; c < X::Size; ++c)
            x.data()[c] = values[(start + c) % values.size()];
        return x;
    }
}

/// One input of a kind: operands `a`, `b`, auxiliary vector `v`, scalar `s`.
template<class T, class A, class B, class V>
struct In {
    A a;
    B b;
    V v;
    T s;
    std::array<std::size_t, 3> pick;
};

template<class A, class B, class V, class T>
std::vector<In<T, A, B, V>> inputs(const Corpus<T>& corpus)
{
    std::vector<In<T, A, B, V>> out;
    for (const auto& p : corpus.picks)
        out.push_back({ window<A>(corpus.values, p[0]), window<B>(corpus.values, p[1]),
            window<V>(corpus.values, p[2]), corpus.values[p[2]], p });
    return out;
}

// ---- Item <-> C struct ----------------------------------------------------

template<class CS>
concept CVector = requires(CS& c) { c.v; };
template<class CS>
concept CMatrix = requires(CS& c) { c.m; };

/// The C struct's shape must be the Item's: checked at compile time.
template<class X, class CS>
constexpr void checkShape()
{
    using Ext = typename X::element_extents;
    if constexpr (CVector<CS>) {
        static_assert(Ext::Rank == 1 && Ext::static_extent(0) == std::extent_v<decltype(CS::v)>, "vector shape mismatch");
    } else {
        static_assert(CMatrix<CS>, "not an aether C struct");
        static_assert(Ext::Rank == 2 && Ext::static_extent(0) == std::extent_v<decltype(CS::m), 0>
                && Ext::static_extent(1) == std::extent_v<decltype(CS::m), 1>,
            "matrix shape mismatch");
    }
}

/// Converts to whichever C struct the C function's parameter asks for.
template<class X>
struct ToC {
    const X& x;
    template<class CS>
    operator CS() const
    {
        checkShape<X, CS>();
        CS c{};
        if constexpr (CVector<CS>) {
            for (std::size_t k = 0; k < X::Size; ++k)
                c.v[k] = x.data()[k];
        } else {
            for (std::size_t r = 0; r < std::extent_v<decltype(CS::m), 0>; ++r)
                for (std::size_t k = 0; k < std::extent_v<decltype(CS::m), 1>; ++k)
                    c.m[r][k] = x(r, k);
        }
        return c;
    }
};

/// A C argument: a scalar as is, an `Item` as its C struct.
template<class X>
auto cArg(const X& x)
{
    if constexpr (std::is_arithmetic_v<X>)
        return x;
    else
        return ToC<X>{ x };
}

template<class X, class CS>
X fromC(const CS& c)
{
    checkShape<X, CS>();
    X x{};
    if constexpr (CVector<CS>) {
        for (std::size_t k = 0; k < X::Size; ++k)
            x.data()[k] = c.v[k];
    } else {
        for (std::size_t r = 0; r < std::extent_v<decltype(CS::m), 0>; ++r)
            for (std::size_t k = 0; k < std::extent_v<decltype(CS::m), 1>; ++k)
                x(r, k) = c.m[r][k];
    }
    return x;
}

// ---- aether side ----------------------------------------------------------

template<class E, std::size_t... Is>
auto itemOf(const E& e, std::index_sequence<Is...>)
{
    return aether::Item<typename E::element_type, E::element_extents::static_extent(Is)...>(e);
}

/// Materialise an aether result: a scalar as is, an expression into an
/// `Item` (evaluated at `SampleIndex::make(0)`).
template<class E>
auto materialise(const E& e)
{
    if constexpr (std::is_arithmetic_v<E>)
        return e;
    else
        return itemOf(e, std::make_index_sequence<E::element_extents::Rank>{});
}

/// `aether::constant<extents of X>(1)`: hawk's rank-2 `sum` is `m.matDot(ones)`.
template<class X>
auto onesLike(const X&)
{
    return aether::constant<typename X::element_extents>(typename X::element_type(1));
}

// ---- comparison -----------------------------------------------------------

template<class W, class G>
bool sameResult(const W& want, const G& got)
{
    if constexpr (std::is_arithmetic_v<W>) {
        static_assert(std::is_same_v<W, G>, "scalar result type mismatch");
        return sameBits(want, got);
    } else {
        const W g = fromC<W>(got);
        for (std::size_t k = 0; k < W::Size; ++k)
            if (!sameBits(want.data()[k], g.data()[k]))
                return false;
        return true;
    }
}

template<class X>
std::string show(const X& x)
{
    std::ostringstream os;
    os << std::hexfloat;
    if constexpr (std::is_arithmetic_v<X>) {
        os << x;
    } else {
        os << "{";
        for (std::size_t k = 0; k < X::Size; ++k)
            os << (k ? ", " : "") << x.data()[k];
        os << "}";
    }
    return os.str();
}

/// The C result as the aether result's type (for the failure message).
template<class W, class G>
W asWant(const W&, const G& got)
{
    if constexpr (std::is_arithmetic_v<W>)
        return got;
    else
        return fromC<W>(got);
}

/// The one checker: every input of `ins` through `f`, which returns
/// `(aether result, C result)` for that input.
template<class T, class Ins, class F>
void checkKind(const Corpus<T>& corpus, const char* w, const char* shape, const char* kind, const Ins& ins, F f)
{
    for (std::size_t n = 0; n < ins.size(); ++n) {
        const auto& in = ins[n];
        const auto [want, got] = f(in);
        ASSERT_TRUE(sameResult(want, got))
            << "kind " << kind << ", width " << w << ", shape " << shape << ", corpus " << corpus.name << " input #"
            << n << " (i=" << in.pick[0] << ", j=" << in.pick[1] << ", k=" << in.pick[2] << "): a=" << show(in.a)
            << " b=" << show(in.b) << " v=" << show(in.v) << " s=" << show(in.s) << "; aether " << show(want)
            << " vs C " << show(asWant(want, got));
    }
}

// ---- kind tables: X(w, sh, kind, (C arguments), aether expression) ---------
// In each row `a`, `b`, `v`, `s` are the input's operands (see the file
// comment) and `T` is the width's real type.

/// Every elementwise kind; a vector shape and its matrix mirror share them.
#define CCORE_CWISE_KINDS(X, w, sh) \
    X(w, sh, add, (cArg(a), cArg(b)), a + b) \
    X(w, sh, sub, (cArg(a), cArg(b)), a - b) \
    X(w, sh, mul, (cArg(a), cArg(b)), a.cwiseMul(b)) \
    X(w, sh, cwise_div, (cArg(a), cArg(b)), aether::cwiseDiv(a, b)) \
    X(w, sh, scale, (cArg(a), s), s * a) \
    X(w, sh, div_s, (cArg(a), s), a / s) \
    X(w, sh, min, (cArg(a), cArg(b)), aether::cwiseMin(a, b)) \
    X(w, sh, max, (cArg(a), cArg(b)), aether::cwiseMax(a, b)) \
    X(w, sh, pow, (cArg(a), cArg(b)), aether::cwisePow(a, b)) \
    X(w, sh, neg, (cArg(a)), a * T(-1)) \
    X(w, sh, abs, (cArg(a)), aether::cwiseAbs(a)) \
    X(w, sh, sqrt, (cArg(a)), aether::cwiseSqrt(a)) \
    X(w, sh, rsqrt, (cArg(a)), aether::cwiseRsqrt(a)) \
    X(w, sh, exp, (cArg(a)), aether::cwiseExp(a)) \
    X(w, sh, log, (cArg(a)), aether::cwiseLog(a)) \
    X(w, sh, sin, (cArg(a)), aether::cwiseSin(a)) \
    X(w, sh, cos, (cArg(a)), aether::cwiseCos(a)) \
    X(w, sh, tan, (cArg(a)), aether::cwiseTan(a)) \
    X(w, sh, tanh, (cArg(a)), aether::cwiseTanh(a)) \
    X(w, sh, asin, (cArg(a)), aether::cwiseAsin(a)) \
    X(w, sh, acos, (cArg(a)), aether::cwiseAcos(a)) \
    X(w, sh, atan, (cArg(a)), aether::cwiseAtan(a))

/// The vector folds.
#define CCORE_FOLD_KINDS(X, w, sh) \
    X(w, sh, dot, (cArg(a), cArg(b)), a.dot(b)) \
    X(w, sh, norm, (cArg(a)), a.norm()) \
    X(w, sh, sum, (cArg(a)), a.sum())

/// The matrix-only kinds (`v` is a vector of the column count).
#define CCORE_MAT_KINDS(X, w, sh) \
    X(w, sh, sum, (cArg(a)), a.matDot(onesLike(a))) \
    X(w, sh, mv, (cArg(a), cArg(v)), a * v) \
    X(w, sh, transpose, (cArg(a)), a.transpose())

/// Quaternions are 4-vectors `(w, x, y, z)`; `v` is a 3-vector.
#define CCORE_QUAT_KINDS(X, w, sh) \
    X(w, sh, mul, (cArg(a), cArg(b)), a.quatMul(b)) \
    X(w, sh, conj, (cArg(a)), a.quatConj()) \
    X(w, sh, recip, (cArg(a)), a.quatReciprocal()) \
    X(w, sh, rotate, (cArg(a), cArg(v)), a.quatRotate(v)) \
    X(w, sh, as_pure, (cArg(v)), v.asPureQuaternion()) \
    X(w, sh, as_vec3, (cArg(a)), a.asBack3DVector())

/// Rank 0: `a`, `b`, `s` are scalars (`clip(x = a, lo = b, hi = s)`).
#define CCORE_SCALAR_KINDS(X, w, sh) \
    X(w, sh, rsqrt, (a), aether::math::rsqrt(a)) \
    X(w, sh, sign, (a), aether::math::sign(a)) \
    X(w, sh, clip, (a, b, s), aether::math::clip(a, b, s)) \
    X(w, sh, remainder, (a, b), aether::math::remainder(a, b))

/// One row checked over `ins` against the C function `fn`; the first
/// mismatch stops the family (ASSERT_NO_FATAL_FAILURE needs a line of its own).
#define CCORE_CHECK(fn, w, shape, kind, cargs, e) \
    checkKind(corpus, w, shape, kind, ins, [](const auto& in) { \
        [[maybe_unused]] const auto& a = in.a; \
        [[maybe_unused]] const auto& b = in.b; \
        [[maybe_unused]] const auto& v = in.v; \
        [[maybe_unused]] const T s = in.s; \
        return std::pair{ materialise(e), fn cargs }; \
    }); \
    if (::testing::Test::HasFatalFailure()) \
        return;
#define CCORE_SHAPED(w, sh, k, cargs, e) CCORE_CHECK(AC(w, sh, k), #w, #sh, #k, cargs, e)
#define CCORE_RANK0(w, sh, k, cargs, e) CCORE_CHECK(AS(w, k), #w, #sh, #k, cargs, e)

/// `ins` for operands `A`, `B` and auxiliary vector `V` (all `aether::Item<T, ...>` or `T`).
#define CCORE_INPUTS(A, B, V) const auto ins = inputs<A, B, V>(corpus)
#define CCORE_ITEM(...) aether::Item<T, __VA_ARGS__>

#define CCORE_VEC(w, n) \
    { \
        CCORE_INPUTS(CCORE_ITEM(n), CCORE_ITEM(n), CCORE_ITEM(n)); \
        CCORE_FOLD_KINDS(CCORE_SHAPED, w, vec##n) \
        CCORE_CWISE_KINDS(CCORE_SHAPED, w, vec##n) \
    }
#define CCORE_MAT(w, r, c) \
    { \
        CCORE_INPUTS(CCORE_ITEM(r, c), CCORE_ITEM(r, c), CCORE_ITEM(c)); \
        CCORE_CWISE_KINDS(CCORE_SHAPED, w, mat##r##x##c) \
        CCORE_MAT_KINDS(CCORE_SHAPED, w, mat##r##x##c) \
    }
#define CCORE_MM(w, r, k, c) \
    { \
        CCORE_INPUTS(CCORE_ITEM(r, k), CCORE_ITEM(k, c), T); \
        CCORE_SHAPED(w, mat##r##x##k, mm_mat##k##x##c, (cArg(a), cArg(b)), a * b) \
    }
#define CCORE_OUTER(w, m, n) \
    { \
        CCORE_INPUTS(CCORE_ITEM(m), CCORE_ITEM(n), T); \
        CCORE_SHAPED(w, vec##m, outer_vec##n, (cArg(a), cArg(b)), aether::outer(a, b)) \
    }

/// One checker per family and width, each covering every corpus shape of
/// the family (the shapes of tools/c_core/mint_golden.py).
#define CCORE_FAMILIES(w, Real) \
    inline void checkScalars_##w(const Corpus<Real>& corpus) \
    { \
        using T = Real; \
        CCORE_INPUTS(T, T, T); \
        CCORE_SCALAR_KINDS(CCORE_RANK0, w, scalar) \
    } \
    inline void checkVec_##w(const Corpus<Real>& corpus) \
    { \
        using T = Real; \
        CCORE_VEC(w, 1) \
        CCORE_VEC(w, 2) \
        CCORE_VEC(w, 3) \
        CCORE_VEC(w, 4) \
        CCORE_VEC(w, 6) \
        CCORE_INPUTS(CCORE_ITEM(3), CCORE_ITEM(3), CCORE_ITEM(3)); \
        CCORE_SHAPED(w, vec3, cross, (cArg(a), cArg(b)), a.cross(b)) \
    } \
    inline void checkMat_##w(const Corpus<Real>& corpus) \
    { \
        using T = Real; \
        CCORE_MAT(w, 2, 2) \
        CCORE_MAT(w, 3, 3) \
        CCORE_MAT(w, 4, 4) \
        CCORE_MAT(w, 2, 3) \
        CCORE_MAT(w, 3, 2) \
        CCORE_MM(w, 2, 3, 2) \
        CCORE_MM(w, 3, 3, 3) \
        CCORE_MM(w, 2, 2, 2) \
        CCORE_OUTER(w, 2, 3) \
        CCORE_OUTER(w, 3, 3) \
    } \
    inline void checkQuat_##w(const Corpus<Real>& corpus) \
    { \
        using T = Real; \
        CCORE_INPUTS(CCORE_ITEM(4), CCORE_ITEM(4), CCORE_ITEM(3)); \
        CCORE_QUAT_KINDS(CCORE_SHAPED, w, quat) \
    }

CCORE_FAMILIES(f64, double)
CCORE_FAMILIES(f32, float)

} // namespace ccore
} // namespace aether_tests

namespace {

using namespace aether_tests::ccore;

TEST(CCore, Scalars_f64) { ASSERT_NO_FATAL_FAILURE(checkScalars_f64(spread<double>())); }
TEST(CCore, Scalars_f32) { ASSERT_NO_FATAL_FAILURE(checkScalars_f32(spread<float>())); }
TEST(CCore, Vec_f64) { ASSERT_NO_FATAL_FAILURE(checkVec_f64(spread<double>())); }
TEST(CCore, Vec_f32) { ASSERT_NO_FATAL_FAILURE(checkVec_f32(spread<float>())); }
TEST(CCore, Mat_f64) { ASSERT_NO_FATAL_FAILURE(checkMat_f64(spread<double>())); }
TEST(CCore, Mat_f32) { ASSERT_NO_FATAL_FAILURE(checkMat_f32(spread<float>())); }
TEST(CCore, Quat_f64) { ASSERT_NO_FATAL_FAILURE(checkQuat_f64(spread<double>())); }
TEST(CCore, Quat_f32) { ASSERT_NO_FATAL_FAILURE(checkQuat_f32(spread<float>())); }

/// Every kind of every shape of both widths over the specials, every triple.
TEST(CCore, SpecialValues)
{
    const auto d = specials<double>();
    const auto f = specials<float>();
    ASSERT_NO_FATAL_FAILURE(checkScalars_f64(d));
    ASSERT_NO_FATAL_FAILURE(checkScalars_f32(f));
    ASSERT_NO_FATAL_FAILURE(checkVec_f64(d));
    ASSERT_NO_FATAL_FAILURE(checkVec_f32(f));
    ASSERT_NO_FATAL_FAILURE(checkMat_f64(d));
    ASSERT_NO_FATAL_FAILURE(checkMat_f32(f));
    ASSERT_NO_FATAL_FAILURE(checkQuat_f64(d));
    ASSERT_NO_FATAL_FAILURE(checkQuat_f32(f));
}

} // namespace
