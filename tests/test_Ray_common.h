// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::geometry (aether/geometry/Ray.h): the host battery shared by
// test_Ray.cpp and test_Ray.cu: hand-checked hits, misses and degenerate
// inputs (parallel ray, zero-area triangle, zero direction, tangent sphere,
// negative zero, origin inside) and finiteness of every result. The lanes are
// in test_Ray_corpus.h.
#pragma once

#include "test_Ray_corpus.h"

namespace aether_tests {
namespace ray {

template<class T>
bool finite4(const ag::Vec4<T>& r)
{
    for (T x : r.v)
        if (!std::isfinite(x))
            return false;
    return true;
}

template<class T>
void checkHandCases()
{
    const auto c = corpus<T>();
    const T tol = T(8) * std::numeric_limits<T>::epsilon();
    // triangle
    auto r = tri(c[0]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_NEAR(r.v[1], T(1), tol);
    EXPECT_NEAR(r.v[2], T(0.25), tol);
    EXPECT_NEAR(r.v[3], T(0.25), tol);
    EXPECT_EQ(tri(c[1]).v[0], T(0));
    r = tri(c[2]);
    EXPECT_EQ(r.v[0], T(0));
    EXPECT_EQ(r.v[1], T(0));
    EXPECT_EQ(r.v[2], T(0));
    EXPECT_EQ(r.v[3], T(0));
    EXPECT_EQ(tri(c[3]).v[0], T(1)) << "a ray on an edge counts as a hit";
    EXPECT_EQ(tri(c[4]).v[0], T(0));
    EXPECT_LT(tri(c[4]).v[1], T(0));
    EXPECT_EQ(tri(c[5]).v[0], T(0));
    EXPECT_EQ(tri(c[6]).v[0], T(1));
    EXPECT_EQ(tri(c[7]).v[0], T(0));
    r = tri(c[21]);
    EXPECT_EQ(r.v[0], T(0)) << "zero-area triangle";
    EXPECT_EQ(r.v[1], T(0));
    EXPECT_EQ(r.v[2], T(0));
    EXPECT_EQ(r.v[3], T(0));
    // box
    r = box(c[8]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_NEAR(r.v[1], T(2), tol);
    EXPECT_NEAR(r.v[2], T(4), tol);
    EXPECT_EQ(r.v[3], T(0));
    EXPECT_EQ(box(c[9]).v[0], T(0)) << "d_y = 0 outside the slab is a miss";
    EXPECT_EQ(box(c[10]).v[0], T(0)) << "d_y = -0 outside the slab is a miss";
    r = box(c[11]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_NEAR(r.v[1], T(-1), tol);
    EXPECT_NEAR(r.v[2], T(1), tol);
    EXPECT_EQ(box(c[12]).v[0], T(0));
    EXPECT_EQ(box(c[13]).v[0], T(0));
    EXPECT_EQ(box(c[14]).v[0], T(0));
    // sphere
    r = sph(c[15]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_NEAR(r.v[1], T(2), tol);
    EXPECT_NEAR(r.v[2], T(4), tol);
    r = sph(c[16]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_EQ(r.v[1], r.v[2]) << "tangent: the two roots coincide";
    r = sph(c[17]);
    EXPECT_EQ(r.v[0], T(0));
    EXPECT_EQ(r.v[1], T(0));
    EXPECT_EQ(r.v[2], T(0));
    r = sph(c[18]);
    EXPECT_EQ(r.v[0], T(1));
    EXPECT_NEAR(r.v[1], T(-1), tol);
    EXPECT_NEAR(r.v[2], T(1), tol);
    EXPECT_EQ(sph(c[19]).v[0], T(0));
    EXPECT_EQ(sph(c[20]).v[0], T(0));
}

template<class T>
void checkAllFinite()
{
    std::size_t hits[3] = { 0, 0, 0 };
    const auto c = corpus<T>();
    for (std::size_t i = 0; i < c.size(); ++i) {
        const ag::Vec4<T> r[3] = { tri(c[i]), box(c[i]), sph(c[i]) };
        for (int k = 0; k < 3; ++k) {
            EXPECT_TRUE(finite4(r[k])) << "primitive " << k << " lane " << i;
            hits[k] += r[k].v[0] == T(1);
        }
    }
    for (int k = 0; k < 3; ++k) {
        EXPECT_GT(hits[k], std::size_t(3)) << "primitive " << k << " sees hits and misses";
        EXPECT_LT(hits[k], c.size() - 3);
    }
}

/** @brief Operands may be raw pointers or `Item`s; both read the same values. */
template<class T>
void checkItemOperands()
{
    const Lane<T> l = unitLane<T>({ T(0.25), T(0.25), T(1) }, { T(0), T(0), T(-1) });
    aether::Item<T, 3> o, d, a, b, c;
    for (std::size_t k = 0; k < 3; ++k) {
        o(k) = l.o[k];
        d(k) = l.d[k];
        a(k) = l.a[k];
        b(k) = l.b[k];
        c(k) = l.c[k];
    }
    const ag::Vec4<T> viaItem = ag::rayTriangle<T>(o, d, a, b, c, l.tmax), viaPtr = tri(l);
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(viaItem.v[k], viaPtr.v[k]);
}

TEST(Ray, HandCasesF64) { checkHandCases<double>(); }
TEST(Ray, HandCasesF32) { checkHandCases<float>(); }
TEST(Ray, EveryResultIsFiniteF64) { checkAllFinite<double>(); }
TEST(Ray, EveryResultIsFiniteF32) { checkAllFinite<float>(); }
TEST(Ray, ItemOperandsF64) { checkItemOperands<double>(); }
TEST(Ray, ItemOperandsF32) { checkItemOperands<float>(); }

} // namespace ray
} // namespace aether_tests
