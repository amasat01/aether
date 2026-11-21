// The stub the raise driver would generate: it calls each named kernel once against the trace arm,
// with the plane names it read from the signature, and writes the IR text of each.
#include <deque>
#include <fstream>
#include <string>

#include "model/kernels.h"

namespace raise {

using aether::Real;
namespace tr = aether::trace;

std::deque<Real> storage;                       // the values the written planes end with
std::vector<std::pair<std::string, Real*>> written;

std::string leafFields(const char* kind, const char* role, const char* name) {
    return std::string("(('kind', '") + kind + "'), ('role', '" + role + "'), ('name', '" + name +
           "'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
}

std::vector<std::string> declared;   // one line per parameter: role, name, scalar type

template <class T> struct Arg;
template <> struct Arg<aether::PerSample<Real>> {
    static aether::PerSample<Real> make(const char* name) {
        declared.push_back(std::string("per_sample ") + name + " f64");
        return {Real(tr::makeNode("Leaf", "f64", {}, leafFields("vocab_read", "per_sample", name)))};
    }
};
template <> struct Arg<aether::Uniform<Real>> {
    static aether::Uniform<Real> make(const char* name) {
        declared.push_back(std::string("uniform ") + name + " f64");
        return {Real(tr::makeNode("Leaf", "f64", {}, leafFields("uniform", "uniform", name)))};
    }
};
template <> struct Arg<aether::Mutable<Real>> {
    static aether::Mutable<Real> make(const char* name) {
        storage.emplace_back();
        declared.push_back(std::string("mutable ") + name + " f64");
        written.emplace_back(name, &storage.back());
        return {&storage.back()};
    }
};

/// Raises one kernel: its roles come from the parameter types, its plane names from `names`.
template <class... A>
void kernel(const char* name, void (*body)(A...), std::array<const char*, sizeof...(A)> names) {
    written.clear();
    declared.clear();
    std::size_t next = 0;
    // braces fix the order the arguments are made in, so the leaves follow the parameters
    std::tuple<A...> args{Arg<A>::make(names[next++])...};
    char entry = 0;                      // the kernel's frames lie below this local
    tr::tape().stackBase = &entry;
    std::apply(body, args);
    tr::tape().stackBase = nullptr;
    std::vector<rx_node> roots;
    for (auto& [plane, value] : written) {
        roots.push_back(tr::makeNode(
            "Assign", "f64", {value->node()},
            "(('kind', 'assign'), ('role', 'mutable'), ('name', '" + plane + "'))"));
    }
    std::size_t length = 0;
    rx_dump(tr::tape().ctx, roots.data(), roots.size(), nullptr, 0, &length);
    std::string text(length + 1, '\0');
    tr::check(rx_dump(tr::tape().ctx, roots.data(), roots.size(), text.data(), text.size(), &length),
              "dump");
    text.resize(length);
    std::ofstream(std::string("out/cpp_") + name + ".ir") << text;
    std::ofstream planes(std::string("out/cpp_") + name + ".planes");
    for (const auto& line : declared) planes << line << "\n";
    std::printf("raised %s: %zu characters, %zu live traced variables left\n", name, length,
                tr::tape().live.size() - storage.size());
}

} // namespace raise

int main() {
    raise::kernel("drag", &model::drag, {"v", "cd", "out"});
    raise::kernel("clip", &model::clip, {"v", "lim", "y"});
    raise::kernel("newton", &model::newton, {"v", "x"});
}
