// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::geometry (aether/geometry/Ray.h): the lane corpus (hand-checked hits,
// misses and degenerate inputs, then random lanes) and the one-lane wrappers
// shared by test_Ray_common.h (host battery), test_Ray.cu (device twin) and
// test_CCore_common.h (bit identity with the C99 twin). No tests live here.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace ray {

namespace ag = aether::geometry;

template<class T>
using V3 = std::array<T, 3>;

/** @brief One lane: a ray, the operands of the three primitives (`a`, `b`, `c` = triangle, `a`, `b` = box, `a`, `r` = sphere) and `tmax`. */
template<class T>
struct Lane {
    T o[3], d[3], a[3], b[3], c[3];
    T r, tmax;
};

template<class T>
void put(T (&dst)[3], const V3<T>& src)
{
    for (std::size_t k = 0; k < 3; ++k)
        dst[k] = src[k];
}

/** @brief splitmix64 stream, uniform in [-1, 1). */
struct Stream {
    std::uint64_t s;
    explicit Stream(std::uint64_t seed) : s(seed) {}
    double next()
    {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * 0x1.0p-52 - 1.0;
    }
};

template<class T>
constexpr T inf() { return std::numeric_limits<T>::infinity(); }

/** @brief The unit right triangle in the z = 0 plane and the box `[-1, 1]^3`, unit sphere at the origin. */
template<class T>
Lane<T> unitLane(V3<T> o, V3<T> d, T tmax = inf<T>())
{
    Lane<T> l;
    put(l.o, o);
    put(l.d, d);
    put(l.a, { T(0), T(0), T(0) });
    put(l.b, { T(1), T(0), T(0) });
    put(l.c, { T(0), T(1), T(0) });
    l.r    = T(1);
    l.tmax = tmax;
    return l;
}

template<class T>
Lane<T> boxLane(V3<T> o, V3<T> d, T tmax = inf<T>())
{
    Lane<T> l = unitLane(o, d, tmax);
    put(l.a, { T(-1), T(-1), T(-1) });
    put(l.b, { T(1), T(1), T(1) });
    return l;
}

template<class T>
Lane<T> sphereLane(V3<T> o, V3<T> d, T tmax = inf<T>())
{
    Lane<T> l = unitLane(o, d, tmax);
    put(l.a, { T(0), T(0), T(0) });
    return l;
}

template<class T>
ag::Vec4<T> tri(const Lane<T>& l) { return ag::rayTriangle<T>(l.o, l.d, l.a, l.b, l.c, l.tmax); }
template<class T>
ag::Vec4<T> box(const Lane<T>& l) { return ag::rayAabb<T>(l.o, l.d, l.a, l.b, l.tmax); }
template<class T>
ag::Vec4<T> sph(const Lane<T>& l) { return ag::raySphere<T>(l.o, l.d, l.a, l.r, l.tmax); }

/** @brief Every case the host battery names, plus 256 random lanes; the device test replays all of them. */
template<class T>
std::vector<Lane<T>> corpus()
{
    const T z = T(0), nz = -T(0), one = T(1);
    std::vector<Lane<T>> v = {
        unitLane<T>({ T(0.25), T(0.25), one }, { z, z, -one }),                 // triangle hit
        unitLane<T>({ T(2), T(2), one }, { z, z, -one }),                       // triangle miss
        unitLane<T>({ T(0.25), T(0.25), one }, { one, z, z }),                  // parallel to the plane
        unitLane<T>({ T(0.5), T(0.5), one }, { z, z, -one }),                   // exactly on the hypotenuse
        unitLane<T>({ T(0.25), T(0.25), -one }, { z, z, -one }),                // plane behind the origin
        unitLane<T>({ T(0.25), T(0.25), one }, { z, z, -one }, T(0.5)),         // beyond tmax
        unitLane<T>({ T(0.25), T(0.25), one }, { nz, nz, -one }),               // negative zeros in the direction
        unitLane<T>({ T(0.25), T(0.25), one }, { z, z, z }),                    // zero direction
        boxLane<T>({ T(-3), z, z }, { one, z, z }),                             // box hit, d_y = d_z = 0 inside both slabs
        boxLane<T>({ T(-3), T(2), z }, { one, z, z }),                          // box miss, d_y = 0 outside the slab
        boxLane<T>({ T(-3), T(2), z }, { one, nz, z }),                         // same with d_y = -0
        boxLane<T>({ z, z, z }, { one, z, z }),                                 // origin inside
        boxLane<T>({ T(3), z, z }, { one, z, z }),                              // box behind
        boxLane<T>({ T(-3), z, z }, { one, z, z }, T(1)),                       // beyond tmax
        boxLane<T>({ T(-3), z, z }, { z, z, z }),                               // zero direction, outside
        sphereLane<T>({ T(-3), z, z }, { one, z, z }),                          // sphere hit
        sphereLane<T>({ T(-3), one, z }, { one, z, z }),                        // tangent
        sphereLane<T>({ T(-3), T(2), z }, { one, z, z }),                       // miss
        sphereLane<T>({ z, z, z }, { one, z, z }),                              // origin inside
        sphereLane<T>({ T(3), z, z }, { one, z, z }),                           // sphere behind
        sphereLane<T>({ T(-3), z, z }, { z, z, z }),                            // zero direction
    };
    Lane<T> flat = unitLane<T>({ T(0.25), T(0.25), one }, { z, z, -one });
    for (std::size_t k = 0; k < 3; ++k)
        flat.b[k] = flat.c[k] = flat.a[k];
    v.push_back(flat); // zero-area triangle
    Stream rng(20261027);
    const auto vec = [&](double scale) {
        return V3<T>{ static_cast<T>(scale * rng.next()), static_cast<T>(scale * rng.next()), static_cast<T>(scale * rng.next()) };
    };
    for (int i = 0; i < 256; ++i) {
        Lane<T> l;
        put(l.o, vec(3.0));
        put(l.d, vec(1.0));
        put(l.a, vec(1.0));
        put(l.b, vec(1.0));
        put(l.c, vec(1.0));
        for (std::size_t k = 0; k < 3; ++k) {
            const T lo = std::min(l.a[k], l.b[k]);
            l.b[k]     = std::max(l.a[k], l.b[k]);
            l.a[k]     = lo;
        }
        l.r    = static_cast<T>(0.25 + 0.5 * (rng.next() + 1.0));
        l.tmax = (i % 4 == 0) ? static_cast<T>(2.0) : inf<T>();
        v.push_back(l);
    }
    return v;
}


} // namespace ray
} // namespace aether_tests
