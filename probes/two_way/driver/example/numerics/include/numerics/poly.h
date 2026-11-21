// A polynomial in Horner form, evaluated by a bounded loop.
#pragma once
#include "aether_probe/real.h"

namespace numerics {

inline constexpr double kCoefficients[4] = {0.5, -1.25, 2.0, 0.75};

inline aether::Real polyEval(aether::Real x) {
    aether::Real acc = 0.0;
    for (int i = 0; i < 4; ++i) {
        acc = acc * x + kCoefficients[i];
    }
    return acc;
}

} // namespace numerics
