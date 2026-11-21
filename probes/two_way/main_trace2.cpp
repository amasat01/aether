// The stub the raise driver would generate for the second set of kernels.
#include <deque>
#include <fstream>
#include <string>

#include "model/more.h"

namespace raise {

using aether::Int;
using aether::Real;
namespace tr = aether::trace;

template <class T> std::deque<T>& storage() {
    static auto* kept = new std::deque<T>;
    return *kept;
}
std::vector<std::pair<std::string, tr::Var*>> written;

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
template <class T> struct Arg<aether::Mutable<T>> {
    static aether::Mutable<T> make(const char* name) {
        storage<T>().emplace_back();
        declared.push_back(std::string("mutable ") + name + " " + storage<T>().back().dtype());
        written.emplace_back(name, &storage<T>().back());
        return {&storage<T>().back()};
    }
};

template <class... A>
void kernel(const char* name, void (*body)(A...), std::array<const char*, sizeof...(A)> names) {
    written.clear();
    declared.clear();
    std::size_t next = 0;
    std::tuple<A...> args{Arg<A>::make(names[next++])...};
    char entry = 0;                      // the kernel's frames lie below this local
    tr::tape().stackBase = &entry;
    std::apply(body, args);
    tr::tape().stackBase = nullptr;
    std::vector<rx_node> roots;
    for (auto& [plane, value] : written) {
        roots.push_back(tr::makeNode(
            "Assign", value->dtype(), {value->node()},
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
    std::printf("raised %s: %zu lines\n", name, (size_t)std::count(text.begin(), text.end(), '\n'));
}

} // namespace raise

int main() {
    raise::kernel("band", &model::band, {"v", "w", "y"});
    raise::kernel("sides", &model::sides, {"v", "y"});
    raise::kernel("horner", &model::horner, {"v", "y"});
    raise::kernel("counted", &model::counted, {"v", "x", "iters"});
    raise::kernel("whileRoot", &model::whileRoot, {"v", "x"});
}
