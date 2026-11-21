// The drag kernel: a quadratic drag on the soft-clipped speed.
#pragma once
#include <numerics/soft.h>

#include "aether_probe/real.h"
#include "detail/limit.h"

namespace physics {

using aether::Mutable;
using aether::PerSample;
using aether::Real;
using aether::Uniform;

inline void drag(PerSample<Real> v, Uniform<Real> cd, Mutable<Real> out) {
    Real speed = numerics::softClip(v, detail::kLimit);
    out = -cd * speed * aether::math::abs(speed);
}

} // namespace physics
