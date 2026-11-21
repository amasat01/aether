// A kernel whose helper picks its result with a switch on an integer that changes with the data.
#pragma once
#include "aether_probe/real.h"

namespace physics {

using aether::Int;
using aether::Mutable;
using aether::PerSample;
using aether::Real;

/// The speed after gear k: doubled in gear 0, raised by one in gears 1 and 2, reversed otherwise.
inline Real geared(Int k, Real v) {
    switch (k) {
    case 0:
        return v * 2.0;
    case 1:
    case 2:
        return v + 1.0;
    default:
        return -v;
    }
}

inline void gear(PerSample<Int> k, PerSample<Real> v, Mutable<Real> out) {
    out = geared(k, v);
}

} // namespace physics
