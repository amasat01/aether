// A kernel the reader cannot raise: a `while` on a computed value with no bound stated.
#pragma once
#include "aether_probe/real.h"

namespace bad {

inline void unbounded(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> x) {
    aether::Real root = v;
    while (aether::math::abs(root * root - v) > 1e-12) {
        root = 0.5 * (root + v / root);
    }
    x = root;
}

} // namespace bad
