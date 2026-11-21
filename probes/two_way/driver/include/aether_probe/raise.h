// The entry the raise driver's generated main calls: it makes the arguments of a kernel from its
// declaration (the scalar type from the compiler, the role and the name from the source text),
// records the kernel twice and writes `<name>.ir` and `<name>.planes`.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "aether_probe/real.h"
#include "aether_probe/views.h"

namespace aether::trace::raising {

/// One parameter as the reader reports it.
struct Param {
    const char* name;
    const char* role;
};

inline std::deque<planes::ViewPlane>& viewStorage() {
    static auto* kept = new std::deque<planes::ViewPlane>;
    return *kept;
}

template <class T>
std::deque<T>& storage() {
    static auto* kept = new std::deque<T>;
    return *kept;
}

/// What one recording collects: the planes the kernel writes and one line per parameter.
struct Run {
    std::vector<std::pair<std::string, Var*>> written;
    std::vector<std::string> declared;
    /// The view planes, and where in `declared` the line of each goes once its use is known.
    std::vector<std::pair<planes::ViewPlane*, std::size_t>> views;
};

inline Run& run() {
    static Run current;
    return current;
}

/// A parameter taken by value is read: a leaf of its role.
template <class T>
struct Arg {
    static T make(const Param& p) {
        const std::string role = p.role;
        const std::string dtype = T().dtype();
        if (role != "per_sample" && role != "uniform") {
            std::fprintf(stderr, "raise: parameter %s is taken by value and has the role %s\n",
                         p.name, p.role);
            std::exit(4);
        }
        run().declared.push_back(role + " " + p.name + " " + dtype);
        const char* kind = role == "uniform" ? "uniform" : "vocab_read";
        return T(makeNode("Leaf", dtype.c_str(), {},
                          planes::leafFields(kind, role.c_str(), p.name)));
    }
};

/// A parameter taken by reference is written: the value it ends with becomes an assign.
template <class T>
struct Arg<T&> {
    static T& make(const Param& p) {
        if (std::string(p.role) != "mutable") {
            std::fprintf(stderr, "raise: parameter %s is taken by reference and has the role %s\n",
                         p.name, p.role);
            std::exit(4);
        }
        storage<T>().emplace_back();
        run().declared.push_back(std::string("mutable ") + p.name + " " +
                                 storage<T>().back().dtype());
        run().written.emplace_back(p.name, &storage<T>().back());
        return storage<T>().back();
    }
};

/// A view parameter: a plane of the kernel, read or written by what its declaration says.
template <class T, class E, class L, bool Volatile, bool ReadOnly>
struct Arg<aether::View<T, E, L, Volatile, ReadOnly>> {
    static aether::View<T, E, L, Volatile, ReadOnly> make(const Param& p) {
        const std::string role = p.role;
        if (role != "view_read" && role != "view_write") {
            std::fprintf(stderr, "raise: parameter %s is a view and has the role %s\n", p.name, p.role);
            std::exit(4);
        }
        using Kind = planes::Element<std::remove_const_t<T>>;
        viewStorage().emplace_back(p.name, Kind::dtype, Kind::tag, planes::ElementShape<E>::get(),
                                   role == "view_read");
        run().views.emplace_back(&viewStorage().back(), run().declared.size());
        run().declared.emplace_back();
        return aether::View<T, E, L, Volatile, ReadOnly>(&viewStorage().back());
    }
};

/// The sample index parameter: no plane, the kernel runs for the one sample traced.
template <>
struct Arg<aether::SampleIndex> {
    static aether::SampleIndex make(const Param& p) {
        if (std::string(p.role) != "sample_index") {
            std::fprintf(stderr, "raise: parameter %s is a sample index and has the role %s\n", p.name, p.role);
            std::exit(4);
        }
        return {};
    }
};

/// The arguments of a kernel; braces fix the order they are made in, so the leaves follow the
/// parameters.
template <class... A, std::size_t... I>
std::tuple<A...> arguments(const std::array<Param, sizeof...(A)>& params,
                           std::index_sequence<I...>) {
    return std::tuple<A...>{Arg<A>::make(params[I])...};
}

/// One recording of a kernel: the IR text of what it assigns.
template <class... A>
std::string record(void (*body)(A...), const std::array<Param, sizeof...(A)>& params) {
    run() = Run{};
    tape().sinks.clear();
    std::tuple<A...> args = arguments<A...>(params, std::index_sequence_for<A...>{});
    char entry = 0; // the kernel's frames lie below this local
    tape().stackBase = &entry;
    std::apply(body, args);
    tape().stackBase = nullptr;
    std::vector<rx_node> roots;
    for (auto& [plane, value] : run().written) {
        roots.push_back(makeNode("Assign", value->dtype(), {value->node()},
                                 "(('kind', 'assign'), ('role', 'mutable'), ('name', '" + plane +
                                     "'))"));
    }
    for (auto& [plane, at] : run().views) {
        run().declared[at] = plane->report();
        if (const rx_node root = plane->assignRoot(); root != kNone) roots.push_back(root);
    }
    roots.insert(roots.end(), tape().sinks.begin(), tape().sinks.end());
    std::size_t length = 0;
    rx_dump(tape().ctx, roots.data(), roots.size(), nullptr, 0, &length);
    std::string text(length + 1, '\0');
    check(rx_dump(tape().ctx, roots.data(), roots.size(), text.data(), text.size(), &length),
          "dump");
    text.resize(length);
    return text;
}

/// A kernel name as a file name: `a::b` becomes `a_b`.
inline std::string fileName(std::string name) {
    for (std::size_t at; (at = name.find("::")) != std::string::npos;) name.replace(at, 2, "_");
    return name;
}

} // namespace aether::trace::raising

namespace aether::trace {

/// Raises `body`, a plain function whose parameters have the roles in `params`, and writes the IR
/// and the plane list into `directory`. The kernel is recorded TWICE: one whose result depends on
/// state that outlives a call (a global, a static, the heap) records differently the second time
/// and is refused.
template <class... A>
void raise(const char* name, void (*body)(A...),
           std::array<raising::Param, sizeof...(A)> params, const char* directory) {
    using namespace raising;
    const std::string text = record(body, params);
    if (record(body, params) != text) {
        std::fprintf(stderr,
                     "raise: kernel %s records differently when recorded again: it reads or writes "
                     "state that outlives a call (a global, a static, the heap)\n",
                     name);
        std::exit(5);
    }
    const std::string base = std::string(directory) + "/" + fileName(name);
    std::ofstream(base + ".ir") << text;
    std::ofstream planes(base + ".planes");
    for (const auto& line : run().declared) planes << line << "\n";
    std::printf("raised %s: %zu lines\n", name,
                static_cast<std::size_t>(std::count(text.begin(), text.end(), '\n')));
}

} // namespace aether::trace
