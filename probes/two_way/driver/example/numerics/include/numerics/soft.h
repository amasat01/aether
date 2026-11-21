// A soft clip: the polynomial below the limit, the limit plus the root of the excess above it.
#pragma once
#include "aether_probe/real.h"
#include "poly.h"

namespace numerics {

inline aether::Real softClip(aether::Real v, aether::Real lim) {
    aether::Real y = polyEval(v);
    if (v > lim) {
        y = lim + aether::math::sqrt(v - lim);
    }
    return y;
}

} // namespace numerics
