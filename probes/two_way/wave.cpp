// A kernel that calls aether's REAL math headers, built in either arm: the normal arm prints
// values, the trace arm (-DAETHER_TRACE) writes the IR.
#include <cstdio>
#include <fstream>
#include <string>

#ifdef AETHER_TRACE
#include "aether_probe/trace.h"
namespace aether::trace {
// what aether's math calls for a traced scalar: one node of the kind the function is named after
inline Real aetherTraceUnary(const char* kind, const Real& x) { return Real(makeOp(kind, {x.node()})); }
inline Real aetherTraceBinary(const char* kind, const Real& a, const Real& b) {
    return Real(makeOp(kind, {a.node(), b.node()}));
}
} // namespace aether::trace
namespace aether { using Real = trace::Real; }
#else
namespace aether { using Real = double; }
#endif

#include "aether/math/math.h"

namespace aether {
template <class T> struct PerSample {
    using Scalar = T;
    T value;
    operator T() const { return value; } // NOLINT(google-explicit-constructor)
};
template <class T> struct Uniform {
    using Scalar = T;
    T value;
    operator T() const { return value; } // NOLINT(google-explicit-constructor)
};
} // namespace aether

using aether::Real;

/// Real aether math on planes, on a mix of a plane and a literal, and on computed values.
Real wave(aether::PerSample<Real> v, aether::Uniform<Real> k) {
    Real r = aether::math::hypot(v, k);
    return aether::math::sinh(v) * aether::math::atan2(v, k) + aether::math::fmod(r, 1.5) +
           aether::math::cbrt(r * r);
}

int main() {
#ifdef AETHER_TRACE
    namespace tr = aether::trace;
    auto leaf = [](const char* kind, const char* role, const char* name) {
        return Real(tr::makeNode("Leaf", "f64", {}, std::string("(('kind', '") + kind + "'), ('role', '" + role +
            "'), ('name', '" + name + "'), ('quantity', None), ('wire_index', 0), ('sorted', False))"));
    };
    Real y = wave({leaf("vocab_read", "per_sample", "v")}, {leaf("uniform", "uniform", "k")});
    rx_node root = tr::makeNode("Assign", "f64", {y.node()}, "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))");
    std::size_t length = 0;
    rx_dump(tr::tape().ctx, &root, 1, nullptr, 0, &length);
    std::string text(length + 1, '\0');
    tr::check(rx_dump(tr::tape().ctx, &root, 1, text.data(), text.size(), &length), "dump");
    text.resize(length);
    std::ofstream("out/cpp_wave.ir") << text;
    std::printf("raised wave: %zu characters\n", length);
#else
    for (double v : {-2.0, -0.3, 0.4, 1.7, 5.0}) std::printf("%.17g %.17g\n", v, wave({v}, {0.8}));
#endif
}
