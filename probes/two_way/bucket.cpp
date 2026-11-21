// A kernel on INTEGERS that calls the library's integer header, built in either arm: the normal
// arm prints values, the trace arm (-DAETHER_TRACE) writes the IR and the plane report.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <string>

#include "aether/integer/integer.h"

#ifdef AETHER_TRACE
#include "aether_probe/trace.h"
namespace aether {
using Int = trace::Int;
namespace integer {
// what the integer header's names mean for a traced integer: one node of the operation's kind
inline Int floorDiv(const Int& a, const Int& b) { return Int(trace::makeOp("floordiv", {a.node(), b.node()})); }
inline Int mod(const Int& a, const Int& b) { return Int(trace::makeOp("mod", {a.node(), b.node()})); }
inline Int bitAnd(const Int& a, const Int& b) { return Int(trace::makeOp("band", {a.node(), b.node()})); }
} // namespace integer
} // namespace aether
#else
namespace aether { using Int = std::int32_t; }
#endif

using aether::Int;

/// The bucket of a key: floor division, a modulus that follows the divisor's sign, a bit mask.
Int bucket(Int key) {
    namespace in = aether::integer;
    return in::mod(in::floorDiv(key, Int(3)), Int(5)) + in::bitAnd(key, Int(7));
}

int main() {
#ifdef AETHER_TRACE
    namespace tr = aether::trace;
    Int key(tr::makeNode("Leaf", "i32", {},
                         "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', 'key'), ('quantity', None), "
                         "('wire_index', 0), ('sorted', False))"));
    Int y = bucket(key);
    rx_node root = tr::makeNode("Assign", "i32", {y.node()}, "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))");
    std::size_t length = 0;
    rx_dump(tr::tape().ctx, &root, 1, nullptr, 0, &length);
    std::string text(length + 1, '\0');
    tr::check(rx_dump(tr::tape().ctx, &root, 1, text.data(), text.size(), &length), "dump");
    text.resize(length);
    std::ofstream("out/cpp_bucket.ir") << text;
    std::ofstream("out/cpp_bucket.planes") << "per_sample key i32\nmutable y i32\n";
    std::printf("raised bucket: %zu characters\n", length);
#else
    for (std::int32_t key : {-100, -17, -8, -1, 0, 1, 5, 14, 15, 1000003, -2147483647})
        std::printf("%d %d\n", key, bucket(key));
#endif
}
