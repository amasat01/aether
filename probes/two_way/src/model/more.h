// The constructs the first probe left out, each as an author would write it.
#pragma once
#include "aether_probe/real.h"

namespace model {

using aether::Int;
using aether::Mutable;
using aether::PerSample;
using aether::Real;
using aether::Uniform;

/// Zero inside the band of half-width w, the distance to its edge outside. Early returns of a value.
inline Real deadband(Real v, Real w) {
    if (v > w) {
        return v - w;
    }
    if (-v > w) {
        return v + w;
    }
    return 0.0;
}

inline void band(PerSample<Real> v, Uniform<Real> w, Mutable<Real> y) {
    y = deadband(v, w);
}

/// A plane written on both sides of a branch.
inline void sides(PerSample<Real> v, Mutable<Real> y) {
    if (v > 1.0) {
        y = v * v;
    } else {
        y = aether::math::sqrt(v);
    }
}

inline constexpr double kCoefficients[4] = {0.5, -1.25, 2.0, 0.75};

/// A loop whose index is a plain integer: it subscripts a table of constants.
inline void horner(PerSample<Real> v, Mutable<Real> y) {
    Real acc = 0.0;
    for (int i = 0; i < 4; ++i) {
        acc = acc * v + kCoefficients[i];
    }
    y = acc;
}

/// An integer that changes with the data, carried through a loop with a break.
inline void counted(PerSample<Real> v, Mutable<Real> x, Mutable<Int> iters) {
    Real root = v;
    Int n = 0;
    for (int i = 0; i < 8; ++i) {
        root = 0.5 * (root + v / root);
        n = n + 1;
        if (aether::math::abs(root * root - v) < 1e-12) {
            break;
        }
    }
    x = root;
    iters = n;
}

/// A `while` on a computed value, with its bound stated.
inline void whileRoot(PerSample<Real> v, Mutable<Real> x) {
    Real root = v;
    while (aether::bounded(aether::math::abs(root * root - v) > 1e-12, 8)) {
        root = 0.5 * (root + v / root);
    }
    x = root;
}

} // namespace model
