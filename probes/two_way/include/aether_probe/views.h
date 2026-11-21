// The traced stand-in for aether's view, sample index and element access, with the author-facing
// spelling of the library: `View<T, extents<..., dyn>>`, `plane[i]` with a `SampleIndex`, the
// proxy's `=`, `+=`, `-=`, `.get()`, `.eval<k>()` and `.at(k)`, and `plane(k)` at a computed index.
// Each access records an element the IR already has: a leaf read or an assign for the sample's own
// element, a component read or write for vectors, `At` for a read at a computed index, `WideWrite`
// for a store there and `AccumWrite` for an accumulate there. The role of a plane within read or
// written comes from the kernel's declaration; which kind of plane it is comes from how it is used.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "aether_probe/real.h"

namespace aether {

/// The runtime-extent marker of `extents`.
inline constexpr std::size_t dyn = static_cast<std::size_t>(-1);

/// The extents of a view: the element's static modes, then the sample mode (`dyn`).
template <std::size_t... Es>
struct extents {};

/// Layouts of a view; the trace arm does not tell them apart.
struct layout_right {};
struct layout_stride {};

/// The sample a kernel runs for; in the trace arm every access is to the sample being traced.
class SampleIndex {
public:
    static constexpr SampleIndex make(std::size_t) { return {}; }
};

namespace trace::planes {

/// Stops the raise with `message`; `code` tells the refusal apart in the exit status.
[[noreturn]] inline void refuse(int code, const std::string& message) {
    std::fprintf(stderr, "raise: %s\n", message.c_str());
    std::exit(code);
}

inline std::string leafFields(const char* kind, const char* role, const std::string& name) {
    return std::string("(('kind', '") + kind + "'), ('role', '" + role + "'), ('name', '" + name +
           "'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
}

/// The element type of a view: the traced value it reads as, and its leaf type.
template <class T>
struct Element;
template <>
struct Element<double> {
    using Value = Real;
    static constexpr const char* dtype = "f64";
    static constexpr const char* tag = nullptr;
};
template <>
struct Element<std::int32_t> {
    using Value = Int;
    static constexpr const char* dtype = "int32";
    static constexpr const char* tag = "index";
};

/// The static modes before the sample mode.
template <class E>
struct ElementShape;
template <std::size_t... Es>
struct ElementShape<extents<Es...>> {
    static Shape get() {
        const std::vector<std::size_t> all{Es...};
        Shape shape;
        for (std::size_t i = 0; i + 1 < all.size(); ++i) {
            if (all[i] == dyn) refuse(4, "a view has a runtime mode before its sample mode; only the last mode may be dyn");
            shape.push_back(static_cast<std::int64_t>(all[i]));
        }
        return shape;
    }
};

/// The value a written plane holds while the kernel runs: assigned as a whole or by component. A
/// branch that leaves one side without a value reads the plane's value from before the launch.
class PlaneValue final : public Var {
public:
    PlaneValue(std::string name, const char* dtype, Shape shape)
        : Var(kNone, dtype, std::move(shape)), name_(std::move(name)) {}

    rx_node priorNode() const override {
        if (prior_ == kNone) {
            prior_ = makeNode("Leaf", dtype(), shape(), {}, leafFields("prior_read", "mutable", name_));
        }
        return prior_;
    }

private:
    std::string name_;
    mutable rx_node prior_ = kNone;
};

/// A view parameter of the kernel: its declaration, and how the kernel used it.
struct ViewPlane {
    std::string name;
    const char* dtype;
    const char* tag;
    Shape shape;
    bool readOnly;
    bool sampleRead = false, computedRead = false;
    bool sampleWrite = false, computedStore = false, computedAccum = false;
    rx_node leaf = kNone;
    PlaneValue value;

    ViewPlane(std::string n, const char* d, const char* t, Shape s, bool ro)
        : name(std::move(n)), dtype(d), tag(t), shape(std::move(s)), readOnly(ro), value(name, d, shape) {}

    void requireWritable(const char* what) const {
        if (readOnly) refuse(4, "plane " + name + " is read-only (a const view) and the kernel " + what + " it");
    }
    void requireReadable(const char* what) const {
        if (!readOnly) refuse(4, "plane " + name + " is written (not a const view) and the kernel " + what + " it");
    }

    /// The read leaf of the plane at the sample being traced.
    rx_node sampleLeaf() {
        requireReadable("reads");
        sampleRead = true;
        if (leaf == kNone) {
            const char* role = shape.empty() ? "per_sample" : shape.size() == 1 ? "vec_in" : "mat_in";
            leaf = makeNode("Leaf", dtype, shape, {}, leafFields("vocab_read", role, name), tag);
        }
        return leaf;
    }

    /// The read leaf of the plane as a lookup table.
    rx_node tableLeaf() {
        requireReadable("reads");
        computedRead = true;
        if (leaf == kNone) leaf = makeNode("Leaf", dtype, {}, {}, leafFields("table_read", "lookup", name), tag);
        return leaf;
    }

    /// The traced value of a read node: an index plane is cast to the index type.
    rx_node valueOf(rx_node node) const {
        return tag == nullptr ? node : makeOp("cast", {node}, "'i32'");
    }

    /// The plane's role in the signature report, from how the kernel used it.
    std::string role() const {
        const int kinds = sampleRead + computedRead + sampleWrite + computedStore + computedAccum;
        if (kinds > 1) refuse(4, "plane " + name + " is used in more than one way (sample element, computed read, store, accumulate); use each plane one way");
        if (computedRead) return "lookup";
        if (computedStore) return "wide_out";
        if (computedAccum) return "accum_out";
        if (readOnly) return shape.empty() ? "per_sample" : shape.size() == 1 ? "vec_in" : "mat_in";
        return "mutable";
    }

    /// The report line: role, name, element type, shape and extent of a lookup plane.
    std::string report() const {
        std::string dims = "-";
        for (std::size_t i = 0; i < shape.size(); ++i) dims = (i ? dims + "x" : std::string()) + std::to_string(shape[i]);
        const std::string type = tag == nullptr ? dtype : "index";
        return role() + " " + name + " " + type + " " + dims + " " + (computedRead ? "dyn" : "-");
    }

    /// The assign of a written sample element that the kernel set, or `kNone`.
    rx_node assignRoot() const {
        if (value.node() == kNone) return kNone;
        return makeNode("Assign", dtype, shape, {value.node()},
                        "(('kind', 'assign'), ('role', 'mutable'), ('name', '" + name + "'))");
    }

    /// `plane[i] = v` (and `+=`, `-=` through `base`): the sample's own element.
    void storeOwn(rx_node node) {
        requireWritable("writes");
        if (!shape.empty()) refuse(4, "plane " + name + " holds a vector per sample: write it by component, plane[i].at(k) = v");
        sampleWrite = true;
        value.rebind(node);
    }

    /// The value written so far, for `+=` and `-=`.
    rx_node written() const {
        if (value.node() == kNone) refuse(4, "plane " + name + " is updated with += or -= before it was assigned in this kernel; assign it first");
        return value.node();
    }

    /// Component `index` of the vector element set to `node`.
    void storeComponent(rx_node index, rx_node node) {
        requireWritable("writes");
        if (shape.size() != 1) refuse(4, "plane " + name + " has no vector element to write by component");
        sampleWrite = true;
        rx_node base = value.node();
        if (base == kNone) base = tape().branchDepth > 0 ? value.priorNode() : zeros();
        value.rebind(makeNode("Op", dtype, shape, {base, index, node},
                              "(('kind', 'set_component_at'), ('literal', None))"));
    }

    /// A store or accumulate at a computed index commits at once, so it may not sit in a branch
    /// or a loop body.
    void requireUnconditional(const char* what) const {
        if (tape().branchDepth > 0 || tape().loopDepth > 0) {
            refuse(6, std::string(what) + " on plane " + name + " sits in a branch or a loop body; it would commit whether or not the branch is taken; write it outside");
        }
    }

    void scatter(rx_node value_, rx_node index) {
        requireWritable("stores to");
        requireUnconditional("a store at a computed index");
        computedStore = true;
        tape().sinks.push_back(makeNode("WideWrite", dtype, {value_, index},
            "(('kind', 'wide_write'), ('role', 'wide_out'), ('name', '" + name + "'))"));
    }

    void accumulate(rx_node value_, rx_node index) {
        requireWritable("accumulates into");
        requireUnconditional("an accumulate at a computed index");
        computedAccum = true;
        tape().sinks.push_back(makeNode("AccumWrite", dtype, {value_, index},
            "(('kind', 'accum_write'), ('role', 'accum_out'), ('name', '" + name +
                "'), ('contract', 'plain'), ('reduced', None))"));
    }

    /// The read at a computed index, clamped to the plane's extent.
    rx_node gather(rx_node index) {
        const rx_node table = tableLeaf();
        const rx_node extent = makeNode("PlaneExtent", "i32", {table}, "()");
        const rx_node clamped = makeOp("clamp_index", {index, extent});
        return makeNode("At", dtype, {table, clamped}, "(('kind', 'at'), ('literal', None))");
    }

    /// Component `k` of the sample's vector element.
    rx_node component(long k) { return makeOp("component", {sampleLeaf()}, std::to_string(k)); }
    rx_node componentAt(rx_node k) { return makeOp("component_at", {sampleLeaf(), k}); }

private:
    rx_node zeros() const {
        std::vector<rx_node> parts;
        for (std::int64_t i = 0; i < shape[0]; ++i) parts.push_back(Real(0.0).node());
        return makeOp("vec", parts);
    }
};

/// A component index: a plain integer is a constant of the IR, a traced one (a loop index) is a node.
inline rx_node indexNode(long k) { return Int(static_cast<int>(k)).node(); }
inline rx_node indexNode(const detail::Index& k) { return k.node; }
inline rx_node indexNode(const Int& k) { return k.node(); }

/// `plane[i].at(k)` and `plane(k)`: one element of a plane, read as a value or written.
template <class Value>
class Slot {
public:
    Slot(ViewPlane& plane, rx_node index, bool component, long literal, bool dynamic)
        : plane_(plane), index_(index), component_(component), literal_(literal), dynamic_(dynamic) {}

    /// The element's value.
    operator Value() const { // NOLINT(google-explicit-constructor): a slot reads as its value
        if (component_) {
            return Value(dynamic_ ? plane_.componentAt(index_) : plane_.component(literal_));
        }
        return Value(plane_.valueOf(plane_.gather(index_)));
    }
    Slot& operator=(const Value& v) {
        if (component_) plane_.storeComponent(dynamic_ ? index_ : indexNode(literal_), v.node());
        else plane_.scatter(v.node(), index_);
        return *this;
    }
    Slot& operator+=(const Value& v) {
        if (component_) refuse(4, "plane " + plane_.name + ": a component is stored with =, accumulate with a computed index");
        plane_.accumulate(v.node(), index_);
        return *this;
    }

private:
    ViewPlane& plane_;
    rx_node index_;
    bool component_;
    long literal_;
    bool dynamic_;
};

} // namespace trace::planes

/// The traced view of a plane: a handle the raise makes for each view parameter of a kernel.
template <class T, class Ext, class Layout = layout_right, bool Volatile = false, bool ReadOnly = false>
class View;

/// The proxy of one sample's element, as the library's `SampleRef` and `ConstSampleRef`.
template <class V>
class SampleRef {
public:
    using Value = typename std::remove_const_t<V>::Value;
    explicit SampleRef(trace::planes::ViewPlane& plane) : plane_(plane) {}

    /// The element, for a scalar element.
    Value get() const { return read(plane_.sampleLeaf()); }
    /// Component `K...` of a vector element.
    template <std::size_t... Ks>
    Value eval() const {
        if constexpr (sizeof...(Ks) == 0) return read(plane_.sampleLeaf());
        else return Value(plane_.component(static_cast<long>(Ks)...));
    }
    /// The slot of component `k`, read as a value or assigned.
    template <class K>
    trace::planes::Slot<Value> at(const K& k) const {
        if constexpr (std::is_integral_v<K>) {
            return {plane_, kNoneNode(), true, static_cast<long>(k), false};
        } else {
            return {plane_, trace::planes::indexNode(k), true, 0, true};
        }
    }
    SampleRef& operator=(const Value& v) { plane_.storeOwn(v.node()); return *this; }
    SampleRef& operator+=(const Value& v) {
        plane_.storeOwn(trace::makeOp("add", {plane_.written(), v.node()}));
        return *this;
    }
    SampleRef& operator-=(const Value& v) {
        plane_.storeOwn(trace::makeOp("sub", {plane_.written(), v.node()}));
        return *this;
    }

private:
    static constexpr rx_node kNoneNode() { return trace::kNone; }
    Value read(rx_node leaf) const { return Value(plane_.valueOf(leaf)); }
    trace::planes::ViewPlane& plane_;
};

template <class T, class Ext, class Layout, bool Volatile, bool ReadOnly>
class View {
public:
    using element_type = std::remove_const_t<T>;
    using Value = typename trace::planes::Element<element_type>::Value;
    using element_extents = Ext;

    View() = default;
    explicit View(trace::planes::ViewPlane* plane) : plane_(plane) {}

    SampleRef<View> operator[](const SampleIndex&) const { return SampleRef<View>(*plane_); }
    /// The element at a computed index: a read through the plane as a table, or a store or
    /// accumulate when assigned.
    trace::planes::Slot<Value> operator()(const Int& k) const {
        return {*plane_, k.node(), false, 0, true};
    }

private:
    trace::planes::ViewPlane* plane_ = nullptr;
};

} // namespace aether
