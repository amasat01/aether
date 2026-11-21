// The stub the raise driver would generate: each kernel is called with arguments made from its
// declaration (the scalar type from the compiler, the role and the name from the source text).
#include <array>
#include <deque>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "model/kernels.h"
#include "model/more.h"

namespace raise {

namespace tr = aether::trace;

/// One parameter as the reader reports it.
struct Param {
    const char* name;
    const char* role;
};

template <class T> std::deque<T>& storage() {
    static auto* kept = new std::deque<T>;
    return *kept;
}
std::vector<std::pair<std::string, tr::Var*>> written;
std::vector<std::string> declared;   // one line per parameter: role, name, scalar type

/// A parameter taken by value is read: a leaf of its role.
template <class T> struct Arg {
    static T make(const Param& p) {
        const std::string role = p.role, dtype = T().dtype();
        if (role != "per_sample" && role != "uniform") {
            std::fprintf(stderr, "raise: parameter %s is taken by value and has the role %s\n", p.name, p.role);
            std::exit(4);
        }
        declared.push_back(role + " " + p.name + " " + dtype);
        const char* kind = role == "uniform" ? "uniform" : "vocab_read";
        return T(tr::makeNode("Leaf", dtype.c_str(), {},
                              std::string("(('kind', '") + kind + "'), ('role', '" + role + "'), ('name', '" + p.name +
                                  "'), ('quantity', None), ('wire_index', 0), ('sorted', False))"));
    }
};
/// A parameter taken by reference is written: the value it ends with becomes an assign.
template <class T> struct Arg<T&> {
    static T& make(const Param& p) {
        if (std::string(p.role) != "mutable") {
            std::fprintf(stderr, "raise: parameter %s is taken by reference and has the role %s\n", p.name, p.role);
            std::exit(4);
        }
        storage<T>().emplace_back();
        declared.push_back(std::string("mutable ") + p.name + " " + storage<T>().back().dtype());
        written.emplace_back(p.name, &storage<T>().back());
        return storage<T>().back();
    }
};

/// The arguments of a kernel; braces fix the order they are made in, so the leaves follow the parameters.
template <class... A, std::size_t... I>
std::tuple<A...> arguments(const std::array<Param, sizeof...(A)>& params, std::index_sequence<I...>) {
    return std::tuple<A...>{Arg<A>::make(params[I])...};
}

/// One recording of a kernel: the IR text of what it assigns.
template <class... A>
std::string record(void (*body)(A...), const std::array<Param, sizeof...(A)>& params) {
    written.clear();
    declared.clear();
    std::tuple<A...> args = arguments<A...>(params, std::index_sequence_for<A...>{});
    char entry = 0;                      // the kernel's frames lie below this local
    tr::tape().stackBase = &entry;
    std::apply(body, args);
    tr::tape().stackBase = nullptr;
    std::vector<rx_node> roots;
    for (auto& [plane, value] : written) {
        roots.push_back(tr::makeNode("Assign", value->dtype(), {value->node()},
                                     "(('kind', 'assign'), ('role', 'mutable'), ('name', '" + plane + "'))"));
    }
    std::size_t length = 0;
    rx_dump(tr::tape().ctx, roots.data(), roots.size(), nullptr, 0, &length);
    std::string text(length + 1, '\0');
    tr::check(rx_dump(tr::tape().ctx, roots.data(), roots.size(), text.data(), text.size(), &length), "dump");
    text.resize(length);
    return text;
}

/// Raises one kernel. It is recorded TWICE: a kernel whose result depends on state that outlives a
/// call (a global, a static, the heap) records differently the second time and is refused.
template <class... A>
void kernel(const char* name, void (*body)(A...), std::array<Param, sizeof...(A)> params) {
    const std::string text = record(body, params);
    if (record(body, params) != text) {
        std::fprintf(stderr, "raise: kernel %s records differently when recorded again: it reads or writes "
                             "state that outlives a call (a global, a static, the heap)\n", name);
        std::exit(5);
    }
    std::ofstream(std::string("out/cpp_") + name + ".ir") << text;
    std::ofstream planes(std::string("out/cpp_") + name + ".planes");
    for (const auto& line : declared) planes << line << "\n";
    std::printf("raised %s\n", name);
}

} // namespace raise

namespace model {
/// A condition that only converts to `bool` explicitly, as a callable holder does.
struct Flag {
    bool set;
    explicit operator bool() const { return set; }
};
/// Breaks no rule: the plain condition is known when the kernel is recorded.
inline void flagged(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> y) {
    const Flag doubled{true};
    aether::trace::detail::branch(doubled, [&] { y = v * 2.0; }, [&] { y = v; });
}
int calls = 0;
/// Breaks the rule: a global counter changes what the kernel computes.
inline void counting(aether::PerSample<aether::Real> v, aether::Mutable<aether::Real> y) {
    calls = calls + 1;
    y = v * static_cast<double>(calls);
}
} // namespace model

int main() {
    using raise::kernel;
    kernel("flagged", &model::flagged, {{{"v", "per_sample"}, {"y", "mutable"}}});
    kernel("counting", &model::counting, {{{"v", "per_sample"}, {"y", "mutable"}}});
    std::printf("NOT REFUSED\n");
}
