// The trace arm of the probe: a scalar type whose operations build IR nodes through the IR's C
// interface, and the recording calls a control statement on a traced value is rewritten into.
#pragma once
#include <concepts>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "raptor_ir.h"

namespace aether::trace {

class Var;

/// The state of one raise: the IR context, the traced variables alive now, the sinks made so far.
struct Tape {
    rx_ctx* ctx = rx_ctx_new();
    std::vector<Var*> live;
    std::vector<rx_node> sinks;
    int loopDepth = 0;
    const char* stackBase = nullptr;   // above the kernel's frames: set by whoever calls the kernel
};

inline Tape& tape() {
    static Tape* t = new Tape;   // never destroyed: traced variables of static lifetime outlive it
    return *t;
}

/// Stops the raise with the library's own message when a call is refused.
inline void check(int status, const char* what) {
    if (status != RX_OK) {
        std::fprintf(stderr, "raise: %s: %s\n", what, rx_last_error(tape().ctx));
        std::exit(2);
    }
}

inline rx_node makeNode(const char* cls, const char* dtype, const std::vector<rx_node>& ops,
                        const std::string& fields) {
    rx_node out = 0;
    check(rx_make(tape().ctx, cls, dtype, nullptr, 0, nullptr, ops.data(), ops.size(),
                  fields.c_str(), fields.size(), &out), cls);
    return out;
}

inline rx_node makeOp(const char* kind, const std::vector<rx_node>& ops) {
    rx_node out = 0;
    check(rx_make_op(tape().ctx, kind, ops.data(), ops.size(), "", 0, &out), kind);
    return out;
}

/// The text of a real literal as the IR's value text spells it (shortest form that reads back).
inline std::string literalText(double x) {
    char buf[40];
    auto end = std::to_chars(buf, buf + sizeof buf, x).ptr;
    std::string text(buf, end);
    if (text.find_first_of(".en") == std::string::npos) text += ".0";
    return text;
}

constexpr rx_node kNone = 0xffffffffu;

/// A traced boolean: the node of a comparison.
struct Bool {
    rx_node node;
};

/// A traced variable of some scalar type. Every live one is known to the tape, so that a recorded
/// branch or loop can see which variables its body assigned.
class Var {
public:
    Var(const Var& other) : node_(other.node_), dtype_(other.dtype_) { enter(); }
    Var& operator=(const Var& other) {
        node_ = other.node_;
        return *this;
    }
    ~Var() {
        auto& live = tape().live;
        live.erase(std::find(live.rbegin(), live.rend(), this).base() - 1);
    }
    rx_node node() const { return node_; }
    const char* dtype() const { return dtype_; }
    void rebind(rx_node node) { node_ = node; }

protected:
    Var(rx_node node, const char* dtype) : node_(node), dtype_(dtype) { enter(); }

private:
    void enter() { tape().live.push_back(this); }
    rx_node node_;
    const char* dtype_;
};

/// A traced real value.
class Real : public Var {
public:
    Real() : Var(kNone, "f64") {}
    Real(double literal) // NOLINT(google-explicit-constructor): a literal is a value
        : Var(makeNode("Const", "f64", {}, "(('literal', " + literalText(literal) + "),)"), "f64") {}
    explicit Real(rx_node node) : Var(node, "f64") {}
};

/// A traced integer value: a kernel integer that changes with the data.
class Int : public Var {
public:
    Int() : Var(kNone, "i32") {}
    Int(int literal) // NOLINT(google-explicit-constructor): a literal is a value
        : Var(makeNode("Const", "i32", {}, "(('literal', " + std::to_string(literal) + "),)"), "i32") {}
    explicit Int(rx_node node) : Var(node, "i32") {}
};
inline Int operator+(const Int& a, const Int& b) { return Int(makeOp("add", {a.node(), b.node()})); }
inline Bool operator!(Bool b) { return {makeOp("lnot", {b.node})}; }

inline Real binary(const char* kind, const Real& a, const Real& b) {
    return Real(makeOp(kind, {a.node(), b.node()}));
}
inline Real operator+(const Real& a, const Real& b) { return binary("add", a, b); }
inline Real operator-(const Real& a, const Real& b) { return binary("sub", a, b); }
inline Real operator*(const Real& a, const Real& b) { return binary("mul", a, b); }
inline Real operator/(const Real& a, const Real& b) { return binary("div", a, b); }
inline Real operator-(const Real& a) { return Real(makeOp("neg", {a.node()})); }
inline Bool operator>(const Real& a, const Real& b) { return {makeOp("gt", {a.node(), b.node()})}; }
inline Bool operator<(const Real& a, const Real& b) { return {makeOp("lt", {a.node(), b.node()})}; }

namespace detail {

/// The nodes of every traced variable alive now.
inline std::vector<rx_node> snapshot() {
    std::vector<rx_node> nodes;
    for (Var* v : tape().live) nodes.push_back(v->node());
    return nodes;
}

inline void restore(const std::vector<rx_node>& nodes) {
    for (std::size_t i = 0; i < nodes.size(); ++i) tape().live[i]->rebind(nodes[i]);
}

/// A condition that is not traced: `bool`, or anything an `if` accepts (a pointer, a type with an
/// explicit conversion to `bool`). The traced boolean is neither, so it never lands here.
template <class C>
concept PlainCondition = !std::same_as<std::remove_cvref_t<C>, Bool> && requires(C&& c) { static_cast<bool>(c); };

/// `if` on a plain condition: the ordinary statement.
template <PlainCondition C, class Then, class Else>
void branch(C&& condition, Then&& then, Else&& otherwise) {
    if (condition) then(); else otherwise();
}

/// The check that a recorded body wrote no PLAIN state: both sides of a traced branch run, and a
/// loop body runs more than once, so a plain variable assigned there would hold a wrong value
/// silently. It keeps the bytes of the caller frames (from the recording call up to the kernel's
/// entry) and compares them after the body ran; the storage of traced variables is exempt.
class PlainStateGuard {
public:
    explicit PlainStateGuard(const void* frame)
        : low_(static_cast<const char*>(frame)), high_(tape().stackBase) {
        if (high_ > low_) bytes_.assign(low_, high_);
    }
    void check(const char* where) const {
        if (bytes_.empty()) return;
        std::vector<char> exempt(bytes_.size(), 0);
        for (const Var* v : tape().live) {
            const char* at = reinterpret_cast<const char*>(v);
            if (at < low_ || at >= high_) continue;
            std::fill_n(exempt.begin() + (at - low_), sizeof(Var), 1);
        }
        for (std::size_t i = 0; i < bytes_.size(); ++i) {
            if (exempt[i] || low_[i] == bytes_[i]) continue;
            std::fprintf(stderr, "raise: %s assigns a plain variable (not a traced value) under a "
                                 "traced condition; make it a traced value (aether::Int, aether::Real)\n",
                         where);
            std::exit(3);
        }
    }

private:
    const char* low_;
    const char* high_;
    std::vector<char> bytes_;
};

/// Every variable either side of a branch assigned becomes the select of its two values.
inline void merge(Bool condition, const std::vector<rx_node>& before,
                  const std::vector<rx_node>& afterThen, const std::vector<rx_node>& afterElse) {
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (afterThen[i] == before[i] && afterElse[i] == before[i]) continue;
        if (afterThen[i] == kNone || afterElse[i] == kNone) {
            std::fprintf(stderr, "raise: a variable or plane gets its first value on one side of a "
                                 "branch only; give it a value before the branch or on both sides\n");
            std::exit(2);
        }
        Var* variable = tape().live[i];
        variable->rebind(makeNode("Select", variable->dtype(),
                                  {condition.node, afterThen[i], afterElse[i]}, "()"));
    }
}

/// `if` on a traced condition: both sides run and their assignments are merged.
template <class Then, class Else>
__attribute__((noinline)) void branch(Bool condition, Then&& then, Else&& otherwise) {
    const PlainStateGuard guard(__builtin_frame_address(0));
    const auto before = snapshot();
    then();
    guard.check("a branch");
    const auto afterThen = snapshot();
    restore(before);
    otherwise();
    guard.check("a branch");
    merge(condition, before, afterThen, snapshot());
}

/// A branch whose sides each return a value, on a plain condition: the ordinary choice.
template <PlainCondition C, class Then, class Else>
auto choose(C&& condition, Then&& then, Else&& otherwise)
    -> std::common_type_t<decltype(then()), decltype(otherwise())> {
    if (condition) return then();
    return otherwise();
}

/// A branch whose sides each return a value (an early `return`), on a traced condition: both
/// sides run, their assignments are merged, and the value is the select of the two.
template <class Then, class Else>
__attribute__((noinline)) auto choose(Bool condition, Then&& then, Else&& otherwise)
    -> std::common_type_t<decltype(then()), decltype(otherwise())> {
    using Value = std::common_type_t<decltype(then()), decltype(otherwise())>;
    const PlainStateGuard guard(__builtin_frame_address(0));
    const auto before = snapshot();
    Value a = then();
    guard.check("a branch that returns");
    const auto afterThen = snapshot();
    restore(before);
    Value b = otherwise();
    guard.check("a branch that returns");
    merge(condition, before, afterThen, snapshot());
    return Value(makeNode("Select", a.dtype(), {condition.node, a.node(), b.node()}, "()"));
}

/// How a loop body is being run: to find what it carries, to record it, or once per iteration.
enum class Pass { Discover, Record, Unroll };

/// The exit of a loop: the condition of its one `break` and the carried values there.
struct Exit {
    Pass pass = Pass::Discover;
    std::vector<Var*> carried;
    rx_node condition = kNone;
    std::vector<rx_node> values;

    /// A plain condition breaks for real; a traced one is recorded and the body goes on.
    template <PlainCondition C> bool when(C&& condition_) const { return static_cast<bool>(condition_); }
    bool when(Bool condition_) {
        if (pass == Pass::Unroll || (pass == Pass::Record && condition != kNone)) {
            std::fprintf(stderr, pass == Pass::Unroll
                ? "raise: a loop whose index is used as a plain integer may not break on a traced value\n"
                : "raise: a loop holds more than one break on a traced value\n");
            std::exit(2);
        }
        if (pass == Pass::Record) {
            condition = condition_.node;
            for (Var* v : carried) values.push_back(v->node());
        }
        return false;
    }
};

/// The index a loop body receives. Read as a plain integer (a subscript, a plain argument) it
/// makes the loop run once per iteration at trace time; read as a traced integer it stays one loop.
struct Index {
    rx_node node;
    long value;
    bool* readPlain;
    operator int() const { // NOLINT(google-explicit-constructor)
        *readPlain = true;
        return static_cast<int>(value);
    }
    operator Int() const { return Int(node); } // NOLINT(google-explicit-constructor)
};

/// A variable of the enclosing scope and the name it has in the source.
struct Named {
    const void* variable;   // any variable the statement text names; only traced ones are carried
    const char* name;
};

/// `for` with a trip count known at trace time, over a body that may assign traced variables: the
/// body runs once to find the variables it carries, then once more against carry nodes.
template <class Body>
__attribute__((noinline)) void loop(long start, long stop, long step, const char* indexName,
          std::initializer_list<Named> names, Body&& body) {
    auto& t = tape();
    const int depth = t.loopDepth++;
    const long trip = (stop - start + step - 1) / step;
    const std::string bounds = "(" + std::to_string(start) + ", " +
                               std::to_string(start + (trip - 1) * step) + ")";
    const rx_node index = makeNode(
        "LoopIndex", "i32", {},
        "(('depth', " + std::to_string(depth) + "), ('name', '" + indexName + "'), ('bounds', " +
            bounds + "))");

    const auto before = snapshot();
    bool readPlain = false;
    Exit discovery;
    const PlainStateGuard guard(__builtin_frame_address(0));
    body(Index{index, start, &readPlain}, discovery);
    if (!readPlain) guard.check("a loop body");
    const auto after = snapshot();
    restore(before);
    if (readPlain) {                    // the body needs the index as a number: one run per iteration
        Exit unrolled;
        unrolled.pass = Pass::Unroll;
        for (long at = start; at < stop; at += step) body(Index{index, at, &readPlain}, unrolled);
        --t.loopDepth;
        return;
    }

    Exit exit;
    exit.pass = Pass::Record;
    std::vector<rx_node> carries, inits;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (after[i] == before[i]) continue;
        Var* variable = t.live[i];
        std::string name = "c" + std::to_string(carries.size());
        for (const Named& n : names) if (n.variable == static_cast<const void*>(variable)) name = n.name;
        carries.push_back(makeNode(
            "LoopCarry", variable->dtype(), {},
            "(('depth', " + std::to_string(depth) + "), ('slot', " +
                std::to_string(carries.size()) + "), ('name', '" + name + "'))"));
        inits.push_back(before[i]);
        variable->rebind(carries.back());
        exit.carried.push_back(variable);
    }
    body(Index{index, start, &readPlain}, exit);
    std::vector<rx_node> nexts;
    for (Var* v : exit.carried) nexts.push_back(v->node());

    rx_node made = 0;
    check(rx_new_loop(t.ctx, index, carries.data(), carries.size(), inits.data(), nexts.data(),
                      start, stop, step, nullptr, 0,
                      exit.condition == kNone ? nullptr : &exit.condition, exit.values.data(),
                      exit.values.size(), nullptr, 0, nullptr, &made), "Loop");
    for (std::size_t slot = 0; slot < exit.carried.size(); ++slot) {
        const std::string params = "(('loop', #0), ('slot', " + std::to_string(slot) + "))";
        rx_node value = 0;
        check(rx_new(t.ctx, "LoopValue", params.c_str(), params.size(), &made, 1, &value),
              "LoopValue");
        exit.carried[slot]->rebind(value);
    }
    --t.loopDepth;
}

} // namespace detail
} // namespace aether::trace
