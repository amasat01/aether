// A kernel whose result depends on a global counter: the second recording differs from the first.
#pragma once
#include "aether_probe/real.h"

namespace bad {

inline int gCalls = 0;

inline void counted(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> x) {
    x = v * aether::Real(static_cast<double>(++gCalls));
}

} // namespace bad
