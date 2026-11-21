// Two kernels with the other control forms: a bounded while, and early returns of a value.
#pragma once
#include "aether_probe/real.h"

namespace physics {

using aether::Mutable;
using aether::PerSample;
using aether::Real;
using aether::Uniform;

/// Newton's iteration for the root of v, at most eight steps.
inline void relax(PerSample<Real> v, Mutable<Real> x) {
    Real root = v;
    while (aether::bounded(aether::math::abs(root * root - v) > 1e-12, 8)) {
        root = 0.5 * (root + v / root);
    }
    x = root;
}

/// Zero inside the band of half-width w, the distance to its edge outside.
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

} // namespace physics
