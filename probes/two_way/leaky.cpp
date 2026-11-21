// A kernel that breaks the rule: a plain counter assigned under a traced condition. Written as
// the driver would rewrite it; the raise must refuse it by name.
#include "aether_probe/real.h"
using aether::Real;

Real leaky(Real v) {
    int hits = 0;
    Real r = v;
    aether::trace::detail::branch(v > 1.0, [&] {
        r = v * 2.0;
        hits = hits + 1;
    }, [&] {});
    return r + static_cast<double>(hits);
}

int main() {
    char entry = 0;
    aether::trace::tape().stackBase = &entry;
    Real v(aether::trace::makeNode("Leaf", "f64", {}, "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', 'v'), ('quantity', None), ('wire_index', 0), ('sorted', False))"));
    Real y = leaky(v);
    std::printf("NOT REFUSED: node %u\n", y.node());
}
