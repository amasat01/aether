// The entry the raise driver's generated main calls: it makes the arguments of a kernel from its
// signature, runs the kernel against the trace arm and writes `<name>.ir` and `<name>.planes`.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "aether_probe/real.h"

namespace aether::trace::raising {

template <class T>
std::deque<T>& storage() {
    static auto* kept = new std::deque<T>;
    return *kept;
}

/// What one kernel run collects: the planes it writes and one line per parameter.
struct Run {
    std::vector<std::pair<std::string, Var*>> written;
    std::vector<std::string> declared;
};

inline Run& run() {
    static Run current;
    return current;
}

inline std::string leafFields(const char* kind, const char* role, const char* name) {
    return std::string("(('kind', '") + kind + "'), ('role', '" + role + "'), ('name', '" + name +
           "'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
}

/// The value a parameter of role type `T` is called with.
template <class T>
struct Arg;

template <>
struct Arg<aether::PerSample<aether::Real>> {
    static aether::PerSample<aether::Real> make(const char* name) {
        run().declared.push_back(std::string("per_sample ") + name + " f64");
        return {aether::Real(makeNode("Leaf", "f64", {}, leafFields("vocab_read", "per_sample", name)))};
    }
};

template <>
struct Arg<aether::Uniform<aether::Real>> {
    static aether::Uniform<aether::Real> make(const char* name) {
        run().declared.push_back(std::string("uniform ") + name + " f64");
        return {aether::Real(makeNode("Leaf", "f64", {}, leafFields("uniform", "uniform", name)))};
    }
};

template <class T>
struct Arg<aether::Mutable<T>> {
    static aether::Mutable<T> make(const char* name) {
        storage<T>().emplace_back();
        run().declared.push_back(std::string("mutable ") + name + " " + storage<T>().back().dtype());
        run().written.emplace_back(name, &storage<T>().back());
        return {&storage<T>().back()};
    }
};

template <class... A, std::size_t... I>
std::tuple<A...> arguments(const std::array<const char*, sizeof...(A)>& names,
                           std::index_sequence<I...>) {
    return std::tuple<A...>{Arg<A>::make(names[I])...};
}

/// A kernel name as a file name: `a::b` becomes `a_b`.
inline std::string fileName(std::string name) {
    for (std::size_t at; (at = name.find("::")) != std::string::npos;) name.replace(at, 2, "_");
    return name;
}

} // namespace aether::trace::raising

namespace aether::trace {

/// Raises `body`, a plain function of role-typed parameters named `names`, and writes the IR and
/// the plane list of its written planes into `directory`.
template <class... A>
void raise(const char* name, void (*body)(A...), std::array<const char*, sizeof...(A)> names,
           const char* directory) {
    using namespace raising;
    run() = Run{};
    std::tuple<A...> args = arguments<A...>(names, std::index_sequence_for<A...>{});
    char entry = 0; // the kernel's frames lie below this local
    tape().stackBase = &entry;
    std::apply(body, args);
    tape().stackBase = nullptr;
    std::vector<rx_node> roots;
    for (auto& [plane, value] : run().written) {
        roots.push_back(makeNode("Assign", value->dtype(), {value->node()},
                                 "(('kind', 'assign'), ('role', 'mutable'), ('name', '" + plane + "'))"));
    }
    std::size_t length = 0;
    rx_dump(tape().ctx, roots.data(), roots.size(), nullptr, 0, &length);
    std::string text(length + 1, '\0');
    check(rx_dump(tape().ctx, roots.data(), roots.size(), text.data(), text.size(), &length), "dump");
    text.resize(length);
    const std::string base = std::string(directory) + "/" + fileName(name);
    std::ofstream(base + ".ir") << text;
    std::ofstream planes(base + ".planes");
    for (const auto& line : run().declared) planes << line << "\n";
    std::printf("raised %s: %zu lines\n", name,
                static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')));
}

} // namespace aether::trace
