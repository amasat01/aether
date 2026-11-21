// Three kernels in plain C++ against the aether scalar: arithmetic, a branch through a helper of
// another header, and a bounded loop with a break.
#pragma once
#include "aether_probe/real.h"
#include "model/helper.h"

namespace model {

using aether::Mutable;
using aether::PerSample;
using aether::Real;
using aether::Uniform;

inline void drag(PerSample<Real> v, Uniform<Real> cd, Mutable<Real> out) {
    out = -cd * v * aether::math::abs(v);
}

inline void clip(PerSample<Real> v, Uniform<Real> lim, Mutable<Real> y) {
    y = softClip(v, lim);
}

inline void newton(PerSample<Real> v, Mutable<Real> x) {
    Real root = v;
    for (int i = 0; i < 8; ++i) {
        root = 0.5 * (root + v / root);
        if (aether::math::abs(root * root - v) < 1e-12) {
            break;
        }
    }
    x = root;
}

} // namespace model
