// A helper in its own header, with a branch on a computed value.
#pragma once
#include "aether_probe/real.h"

namespace model {

/// Twice the value below the limit; above it, the limit plus the root of the excess.
inline aether::Real softClip(aether::Real v, aether::Real lim) {
    aether::Real y = v * 2.0;
    if (v > lim) {
        y = lim + aether::math::sqrt(v - lim);
    }
    return y;
}

} // namespace model
